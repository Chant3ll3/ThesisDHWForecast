"""
model_training.py — Phase 2: Model Training (Distributed).

Trains the SFNO model for autoregressive SST forecasting using
PyTorch DistributedDataParallel (DDP) for multi-GPU / multi-node.

Launch methods
──────────────
  Single GPU:
    python model_training.py

  Multi-GPU, single node (e.g. 4 GPUs):
    torchrun --nproc_per_node=4 model_training.py

  Multi-node (e.g. 2 nodes × 4 GPUs via PBS — see run_training.pbs):
    torchrun --nnodes=2 --nproc_per_node=4 \\
             --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \\
             model_training.py
"""

import os
import csv
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from config import cfg
from sfno_model import SFNO, make_grid_features
from sst_dataset import SSTForecastDataset, load_split_data


# ── Distributed helpers ──────────────────────────────────────────────

def _setup_distributed():
    """
    Initialise the distributed process group.

    Supports two launch modes:
      1. torchrun / torch.distributed.launch — sets env vars automatically.
      2. PBS + manual env vars (MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK).

    Returns (rank, local_rank, world_size).  If distributed is disabled
    or only one GPU is available, returns (0, 0, 1).
    """
    conf = cfg.model_training

    if not conf.distributed:
        return 0, 0, 1

    # Check if we are inside a torchrun / distributed launch
    if "RANK" not in os.environ:
        # Not launched with torchrun — fall back to single-process
        return 0, 0, 1

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend=conf.dist_backend,
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    return rank, local_rank, world_size


def _cleanup_distributed():
    """Destroy the process group if it was initialised."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _is_main_process(rank: int) -> bool:
    return rank == 0


# ── Logging (rank-aware) ────────────────────────────────────────────

def _get_logger(rank: int):
    logger = logging.getLogger(f"Phase2_Training_R{rank}")
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            f"%(asctime)s | R{rank} | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
    return logger


# ── Loss functions ───────────────────────────────────────────────────

LOSS_FNS = {
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "huber": nn.HuberLoss,
}


# ── Rollout schedule ────────────────────────────────────────────────

def get_rollout_steps(epoch: int) -> int:
    """Linearly ramp rollout steps between configured epoch window."""
    conf = cfg.model_training
    if epoch < conf.rollout_epoch_start:
        return conf.initial_rollout_steps

    progress = (epoch - conf.rollout_epoch_start) / max(
        1, conf.rollout_epoch_end - conf.rollout_epoch_start
    )
    progress = min(progress, 1.0)
    steps = conf.initial_rollout_steps + progress * (
        conf.max_rollout_steps - conf.initial_rollout_steps
    )
    return int(round(steps))


# ── Learning-rate warmup wrapper ─────────────────────────────────────

class WarmupScheduler:
    """Wraps any LR scheduler with a linear warmup phase."""

    def __init__(self, optimizer, warmup_epochs: int, base_scheduler):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch, val_loss=None):
        if epoch < self.warmup_epochs:
            frac = (epoch + 1) / self.warmup_epochs
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg["lr"] = base_lr * frac
        else:
            if self.base_scheduler is not None:
                if isinstance(self.base_scheduler,
                              torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.base_scheduler.step(val_loss)
                else:
                    self.base_scheduler.step()

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ── Builder helpers ──────────────────────────────────────────────────

def _build_optimizer(model: nn.Module):
    conf = cfg.model_training
    opts = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
    }
    return opts[conf.optimizer](
        model.parameters(), lr=conf.learning_rate, weight_decay=conf.weight_decay
    )


def _build_scheduler(optimizer):
    conf = cfg.model_training
    if conf.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=conf.num_epochs - conf.warmup_epochs,
            eta_min=conf.min_learning_rate,
        )
    elif conf.scheduler == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=conf.scheduler_patience,
            factor=conf.scheduler_factor, min_lr=conf.min_learning_rate,
        )
    elif conf.scheduler == "step":
        sched = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=conf.scheduler_patience
        )
    else:
        sched = None
    return WarmupScheduler(optimizer, conf.warmup_epochs, sched)


# ── Autoregressive forward pass ─────────────────────────────────────

def _autoregressive_forward(model, x, grid_features, rollout_steps, device):
    """
    Roll the SFNO forward ``rollout_steps`` times autoregressively.

    Parameters
    ----------
    x : (B, history, H, W)
    grid_features : (1, 2, H, W) or None
    rollout_steps : int

    Returns
    -------
    preds : list of (B, 1, H, W) tensors
    """
    preds = []
    current_input = x

    for step in range(rollout_steps):
        if grid_features is not None:
            gf = grid_features.expand(current_input.size(0), -1, -1, -1).to(device)
            model_input = torch.cat([current_input, gf], dim=1)
        else:
            model_input = current_input

        pred = model(model_input)
        preds.append(pred)

        # Shift input window: drop oldest, append prediction
        current_input = torch.cat([current_input[:, 1:, :, :],
                                   pred.detach()], dim=1)

    return preds


# ── Train / validate one epoch ───────────────────────────────────────

def _train_one_epoch(model, loader, optimizer, criterion, grid_features,
                     rollout_steps, device):
    conf = cfg.model_training
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        preds = _autoregressive_forward(
            model, x, grid_features, rollout_steps, device
        )

        loss = torch.tensor(0.0, device=device)
        for step, pred in enumerate(preds):
            weight = conf.rollout_loss_decay ** step
            target = y[:, step : step + 1, :, :]
            loss = loss + weight * criterion(pred, target)

        loss.backward()

        if conf.gradient_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), conf.gradient_clip_norm)

        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _validate(model, loader, criterion, grid_features, rollout_steps, device):
    conf = cfg.model_training
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        preds = _autoregressive_forward(
            model, x, grid_features, rollout_steps, device
        )

        loss = torch.tensor(0.0, device=device)
        for step, pred in enumerate(preds):
            weight = conf.rollout_loss_decay ** step
            target = y[:, step : step + 1, :, :]
            loss = loss + weight * criterion(pred, target)

        total_loss += loss.item()
        n_batches += 1

    # Average across all ranks for consistent val_loss
    avg_loss = total_loss / max(n_batches, 1)
    if dist.is_initialized():
        loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()

    return avg_loss


# ── Main training pipeline ──────────────────────────────────────────

def run():
    """Execute the full SFNO training pipeline with DDP support."""
    conf = cfg.model_training

    # ── 0. Distributed setup ─────────────────────────────────────
    rank, local_rank, world_size = _setup_distributed()
    logger = _get_logger(rank)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    logger.info("=" * 60)
    logger.info("PHASE 2: MODEL TRAINING — START")
    logger.info("=" * 60)
    logger.info(f"Distributed: world_size={world_size}, rank={rank}, "
                f"local_rank={local_rank}, device={device}")
    logger.info(f"Effective batch size: {conf.batch_size} × {world_size} = "
                f"{conf.batch_size * world_size}")

    torch.manual_seed(conf.random_seed + rank)
    np.random.seed(conf.random_seed + rank)

    # ── 1. Load data ─────────────────────────────────────────────
    logger.info("Loading processed SST data")
    train_data, val_data, test_data, lat, lon = load_split_data()
    logger.info(f"  Train: {train_data.shape}  ({conf.train_years})")
    logger.info(f"  Val  : {val_data.shape}  ({conf.val_years})")
    logger.info(f"  Test : {test_data.shape}  ({conf.test_years})")

    # ── 2. Build datasets & distributed loaders ──────────────────
    max_rollout = conf.max_rollout_steps
    train_ds = SSTForecastDataset(train_data, max_rollout=max_rollout)
    val_ds = SSTForecastDataset(val_data, max_rollout=max_rollout)
    logger.info(f"  Train samples: {len(train_ds):,}")
    logger.info(f"  Val   samples: {len(val_ds):,}")

    # Use DistributedSampler when running with multiple processes
    use_ddp = dist.is_initialized() and world_size > 1
    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True
    ) if use_ddp else None
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=False
    ) if use_ddp else None

    train_loader = DataLoader(
        train_ds,
        batch_size=conf.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=conf.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=conf.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=conf.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=conf.num_workers,
        pin_memory=True,
        persistent_workers=conf.num_workers > 0,
    )

    # ── 3. Grid features ────────────────────────────────────────
    grid_features = None
    if conf.add_grid_features:
        grid_features = make_grid_features(
            torch.tensor(lat, dtype=torch.float32),
            torch.tensor(lon, dtype=torch.float32),
        ).to(device)
        logger.info(f"  Grid features: shape {list(grid_features.shape)}")

    # ── 4. Model ─────────────────────────────────────────────────
    model = SFNO().to(device)

    if conf.sync_batchnorm and use_ddp:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logger.info("  Converted to SyncBatchNorm")

    if use_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=conf.find_unused_parameters,
        )
        logger.info("  Model wrapped in DistributedDataParallel")

    # Access underlying model for parameter count / saving
    raw_model = model.module if use_ddp else model
    n_params = raw_model.count_parameters()
    logger.info(f"  SFNO model: {n_params:,} trainable parameters")
    logger.info(f"  Architecture: {conf.num_layers} blocks, "
                f"embed_dim={conf.embed_dim}, "
                f"modes=({conf.num_modes_lat}, {conf.num_modes_lon}), "
                f"mlp_ratio={conf.mlp_ratio}")

    # ── 5. Optimizer, scheduler, loss ────────────────────────────
    optimizer = _build_optimizer(model)
    scheduler = _build_scheduler(optimizer)
    criterion = LOSS_FNS[conf.loss_function]()
    logger.info(f"  Optimizer: {conf.optimizer}, LR={conf.learning_rate}")
    logger.info(f"  Scheduler: {conf.scheduler}, warmup={conf.warmup_epochs} epochs")
    logger.info(f"  Loss: {conf.loss_function}")

    # ── 6. Training loop ─────────────────────────────────────────
    if _is_main_process(rank):
        os.makedirs(conf.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.dirname(conf.training_log_path), exist_ok=True)

    # Barrier so all ranks wait for directories
    if dist.is_initialized():
        dist.barrier()

    best_val_loss = float("inf")
    patience_counter = 0
    log_rows = []

    logger.info(f"  Starting training for up to {conf.num_epochs} epochs")
    logger.info(f"  Rollout ramp: {conf.initial_rollout_steps} → "
                f"{conf.max_rollout_steps} steps "
                f"(epochs {conf.rollout_epoch_start}–{conf.rollout_epoch_end})")
    logger.info(f"  Early stopping patience: {conf.early_stopping_patience}")

    for epoch in range(1, conf.num_epochs + 1):
        rollout = get_rollout_steps(epoch)
        lr = scheduler.get_last_lr()[0]

        # Update sampler epoch for proper shuffling across ranks
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Train
        train_loss = _train_one_epoch(
            model, train_loader, optimizer, criterion,
            grid_features, rollout, device,
        )

        # Validate (all_reduced across ranks)
        val_loss = _validate(
            model, val_loader, criterion,
            grid_features, rollout, device,
        )

        # Scheduler step
        scheduler.step(epoch, val_loss=val_loss)

        # Logging (rank 0 only)
        if _is_main_process(rank):
            log_rows.append({
                "epoch": epoch, "train_loss": train_loss,
                "val_loss": val_loss, "lr": lr, "rollout_steps": rollout,
            })

            if epoch % 5 == 0 or epoch == 1 or epoch == conf.num_epochs:
                logger.info(
                    f"  Epoch {epoch:>4d}/{conf.num_epochs} | "
                    f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                    f"LR: {lr:.2e} | Rollout: {rollout}"
                )

            # Checkpointing (rank 0 only)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "world_size": world_size,
                    "config": {
                        "embed_dim": conf.embed_dim,
                        "num_layers": conf.num_layers,
                        "num_modes_lat": conf.num_modes_lat,
                        "num_modes_lon": conf.num_modes_lon,
                        "input_history": conf.input_history,
                        "total_input_channels": conf.total_input_channels,
                        "output_channels": conf.output_channels,
                    },
                }, conf.best_model_path)
                logger.info(f"  ✓ New best model saved (val_loss={val_loss:.6f})")
            else:
                patience_counter += 1
        else:
            # Non-rank-0 processes track patience via broadcast
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

        # Early stopping (synchronised across ranks)
        if patience_counter >= conf.early_stopping_patience:
            logger.info(f"  Early stopping at epoch {epoch} "
                        f"(no improvement for {patience_counter} epochs)")
            break

    # ── 7. Save training log (rank 0) ────────────────────────────
    if _is_main_process(rank) and log_rows:
        with open(conf.training_log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)
        logger.info(f"Training log saved to {conf.training_log_path}")

    # ── 8. Summary ───────────────────────────────────────────────
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Model checkpoint: {conf.best_model_path}")
    logger.info("=" * 60)
    logger.info("PHASE 2: MODEL TRAINING — COMPLETE")
    logger.info("=" * 60)

    _cleanup_distributed()
    return None if not _is_main_process(rank) else raw_model


if __name__ == "__main__":
    run()
