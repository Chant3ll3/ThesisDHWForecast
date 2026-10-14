"""
model_training.py — Phase 2: Model Training (Distributed).

Trains the SFNO model for autoregressive SST forecasting using
PyTorch DistributedDataParallel (DDP) for multi-GPU / multi-node.

Launch methods:
  Single GPU:     python model_training.py
  Multi-GPU:      torchrun --nproc_per_node=4 model_training.py
  Multi-node:     see run_phase2.pbs
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
from sfno_model import SFNO, make_grid_features, make_seasonal_encoding
from ConvLSTM import *
from sst_dataset import SSTForecastDataset, load_split_data


# ── Model Factory ────────────────────────────────────────────────────
def _build_model(device):
    conf = cfg.model_training

    if conf.model_name == "sfno":
        model = SFNO()

    elif conf.model_name == "convlstm":
        model = ConvLSTM(
            input_dim=conf.total_input_channels,
            hidden_dim=conf.embed_dim,
            num_layers=conf.num_layers,
            output_dim=conf.output_channels
        )

    else:
        raise ValueError(f"Unknown model: {conf.model_name}")

    return model.to(device)

# ── Distributed helpers ──────────────────────────────────────────────

def _setup_distributed():
    conf = cfg.model_training
    if not conf.distributed:
        return 0, 0, 1
    if "RANK" not in os.environ:
        return 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=conf.dist_backend, init_method="env://",
                            world_size=world_size, rank=rank)
    return rank, local_rank, world_size

def _cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def _is_main(rank):
    return rank == 0

def _get_logger(rank):
    lg = logging.getLogger(f"Phase2_R{rank}")
    lg.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            f"%(asctime)s | R{rank} | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
        lg.addHandler(h)
    return lg


# ── Loss / rollout / LR helpers ─────────────────────────────────────

LOSS_FNS = {"mse": nn.MSELoss, "mae": nn.L1Loss, "huber": nn.HuberLoss}

def get_rollout_steps(epoch):
    conf = cfg.model_training
    if epoch < conf.rollout_epoch_start:
        return conf.initial_rollout_steps
    progress = min((epoch - conf.rollout_epoch_start) /
                   max(1, conf.rollout_epoch_end - conf.rollout_epoch_start), 1.0)
    return int(round(conf.initial_rollout_steps +
                     progress * (conf.max_rollout_steps - conf.initial_rollout_steps)))


class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, base_scheduler):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch, val_loss=None):
        if epoch < self.warmup_epochs:
            frac = (epoch + 1) / self.warmup_epochs
            for pg, blr in zip(self.optimizer.param_groups, self.base_lrs):
                pg["lr"] = blr * frac
        elif self.base_scheduler is not None:
            if isinstance(self.base_scheduler,
                          torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.base_scheduler.step(val_loss)
            else:
                self.base_scheduler.step()

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


def _build_optimizer(model):
    conf = cfg.model_training
    opts = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD}
    return opts[conf.optimizer](model.parameters(), lr=conf.learning_rate,
                                weight_decay=conf.weight_decay)

def _build_scheduler(optimizer):
    conf = cfg.model_training
    if conf.scheduler == "cosine":
        s = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=conf.num_epochs - conf.warmup_epochs,
            eta_min=conf.min_learning_rate)
    elif conf.scheduler == "plateau":
        s = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=conf.scheduler_patience,
            factor=conf.scheduler_factor, min_lr=conf.min_learning_rate)
    elif conf.scheduler == "step":
        s = torch.optim.lr_scheduler.StepLR(optimizer, step_size=conf.scheduler_patience)
    else:
        s = None
    return WarmupScheduler(optimizer, conf.warmup_epochs, s)


# ── Build auxiliary input features ───────────────────────────────────

def _build_aux_features(doy, grid_features, device, lat_size, lon_size):
    """
    Build all auxiliary input channels (grid + seasonal) for a batch.

    Parameters
    ----------
    doy            : (B,) tensor of day-of-year or None
    grid_features  : (1, 2, H, W) tensor or None
    device         : torch device
    lat_size, lon_size : spatial dims

    Returns
    -------
    aux : (B, C_aux, H, W) or None
    """
    conf = cfg.model_training
    parts = []

    if conf.add_grid_features and grid_features is not None:
        B = doy.size(0) if doy is not None else 1
        parts.append(grid_features.expand(B, -1, -1, -1))

    if conf.add_seasonal_encoding and doy is not None:
        period = conf.seasonal_cycle_period
        se = make_seasonal_encoding(doy, period, lat_size, lon_size).to(device)
        parts.append(se)

    if not parts:
        return None
    return torch.cat(parts, dim=1)


# ── Autoregressive forward pass ─────────────────────────────────────

def _autoregressive_forward(model, x, doy, grid_features, rollout_steps, device):
    """
    Roll the model forward autoregressively (supports SFNO and ConvLSTM).

    Parameters
    ----------
    x              : (B, history, H, W)
    doy            : (B,) day-of-year of the most recent input frame
    grid_features  : (1, 2, H, W) or None
    rollout_steps  : int

    Returns
    -------
    preds : list of tensors, each (B, 1, H, W)
    """
    conf = cfg.model_training
    preds = []

    current_input = x
    current_doy = doy.clone() if doy is not None else None
    B, _, H, W = x.shape

    # Steps-per-day for advancing day_of_year during rollout
    ta = cfg.data_preparation.temporal_average
    doy_increment = {"none": 1.0, "weekly": 7.0, "monthly": 30.44}
    step_days = doy_increment.get(ta, 1.0)

    for step in range(rollout_steps):

        # ── Build auxiliary features (ONLY for SFNO) ──
        aux = None
        if conf.model_name == "sfno":
            aux = _build_aux_features(current_doy, grid_features, device, H, W)

        # ── Prepare model input ──
        if aux is not None:
            model_input = torch.cat([current_input, aux], dim=1)
        else:
            model_input = current_input

        # ── Forward pass depending on model ──
        if conf.model_name == "convlstm":
            # ConvLSTM expects (B, T, C, H, W)
            model_input_cl = model_input.unsqueeze(2)  # (B, T, 1, H, W)

            pred = model(model_input_cl)

            # Ensure output is (B, 1, H, W)
            if pred.dim() == 5:  # (B, T, C, H, W) → take last timestep
                pred = pred[:, -1, :, :, :]

            if pred.dim() == 4 and pred.shape[1] != 1:
                pred = pred.unsqueeze(1)

        else:  # SFNO
            pred = model(model_input)

        # ── Ensure correct shape (B, 1, H, W) ──
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)

        preds.append(pred)

        # ── Autoregressive update ──
        current_input = torch.cat(
            [current_input[:, 1:, :, :], pred.detach()],
            dim=1
        )

        # ── Advance day-of-year ──
        if current_doy is not None:
            current_doy = current_doy + step_days

    return preds


# ── Train / validate one epoch ───────────────────────────────────────

def _train_one_epoch(model, loader, optimizer, criterion, grid_features,
                     rollout_steps, device):
    conf = cfg.model_training
    model.train()
    total_loss = 0.0; n_batches = 0

    for x, y, doy in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        doy = doy.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        preds = _autoregressive_forward(model, x, doy, grid_features,
                                        rollout_steps, device)
        loss = torch.tensor(0.0, device=device)
        for step, pred in enumerate(preds):
            weight = conf.rollout_loss_decay ** step
            loss = loss + weight * criterion(pred, y[:, step:step+1, :, :])
        loss.backward()
        if conf.gradient_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), conf.gradient_clip_norm)
        optimizer.step()
        total_loss += loss.item(); n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _validate(model, loader, criterion, grid_features, rollout_steps, device):
    conf = cfg.model_training
    model.eval()
    total_loss = 0.0; n_batches = 0

    for x, y, doy in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        doy = doy.to(device, non_blocking=True)

        preds = _autoregressive_forward(model, x, doy, grid_features,
                                        rollout_steps, device)
        loss = torch.tensor(0.0, device=device)
        for step, pred in enumerate(preds):
            weight = conf.rollout_loss_decay ** step
            loss = loss + weight * criterion(pred, y[:, step:step+1, :, :])
        total_loss += loss.item(); n_batches += 1

    avg = total_loss / max(n_batches, 1)
    if dist.is_initialized():
        t = torch.tensor(avg, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        avg = t.item()
    return avg


# ── Main training pipeline ──────────────────────────────────────────

def run():
    conf = cfg.model_training
    rank, local_rank, world_size = _setup_distributed()
    logger = _get_logger(rank)

    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() \
        else torch.device("cpu")

    logger.info("=" * 60)
    logger.info("PHASE 2: MODEL TRAINING — START")
    logger.info("=" * 60)
    logger.info(f"Distributed: world_size={world_size}, rank={rank}, device={device}")
    logger.info(f"Effective batch size: {conf.batch_size} × {world_size}")

    torch.manual_seed(conf.random_seed + rank)
    np.random.seed(conf.random_seed + rank)

    # ── 1. Load data ─────────────────────────────────────────────
    logger.info("Loading processed SST data")
    (train_data, val_data, test_data,
     train_times, val_times, test_times, lat, lon) = load_split_data()
    logger.info(f"  Train: {train_data.shape}  ({conf.train_years})")
    logger.info(f"  Val  : {val_data.shape}  ({conf.val_years})")

    # ── 2. Datasets & loaders ────────────────────────────────────
    max_rollout = conf.max_rollout_steps
    train_ds = SSTForecastDataset(train_data, train_times, max_rollout)
    val_ds = SSTForecastDataset(val_data, val_times, max_rollout)
    logger.info(f"  Train samples: {len(train_ds):,}")
    logger.info(f"  Val   samples: {len(val_ds):,}")

    use_ddp = dist.is_initialized() and world_size > 1
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                       rank=rank, shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size,
                                     rank=rank, shuffle=False) if use_ddp else None

    train_loader = DataLoader(train_ds, batch_size=conf.batch_size,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=conf.num_workers, pin_memory=True,
                              drop_last=True, persistent_workers=conf.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=conf.batch_size, shuffle=False,
                            sampler=val_sampler, num_workers=conf.num_workers,
                            pin_memory=True, persistent_workers=conf.num_workers > 0)

    # ── 3. Grid features ────────────────────────────────────────
    grid_features = None
    if conf.add_grid_features:
        grid_features = make_grid_features(
            torch.tensor(lat, dtype=torch.float32),
            torch.tensor(lon, dtype=torch.float32),
        ).to(device)
        logger.info(f"  Grid features: {list(grid_features.shape)}")

    if conf.add_seasonal_encoding:
        logger.info(f"  Seasonal encoding: period={conf.seasonal_cycle_period:.2f} steps")

    # ── 4. Model ─────────────────────────────────────────────────
    # model = SFNO().to(device)
    model = _build_model(device)
    if conf.sync_batchnorm and use_ddp:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=conf.find_unused_parameters)

    raw_model = model.module if use_ddp else model

    # ── Model logging ─────────────────────────────────────────────

    logger.info(f"  Model: {conf.model_name}")

    # Parameter count (safe for any model)
    if hasattr(raw_model, "count_parameters"):
        logger.info(f"  Params: {raw_model.count_parameters():,}")
    else:
        total_params = sum(p.numel() for p in raw_model.parameters())
        logger.info(f"  Params: {total_params:,}")

    # Input channel breakdown (mainly relevant for SFNO)
    input_desc = f"{conf.input_history} history"
    if conf.add_grid_features:
        input_desc += " + grid"
    if conf.add_seasonal_encoding:
        input_desc += " + seasonal"

    logger.info(
        f"  Input channels: {conf.total_input_channels} ({input_desc})"
    )

    # Extra detail for SFNO (spectral info is useful)
    if conf.model_name == "sfno":
        logger.info(
            f"  Spectral modes: lat={conf.num_modes_lat}, lon={conf.num_modes_lon}"
        )


    # logger.info(f"  SFNO: {raw_model.count_parameters():,} params, "
    #             f"{conf.total_input_channels} input channels "
    #             f"({conf.input_history} history + "
    #             f"{'grid ' if conf.add_grid_features else ''}"
    #             f"{'seasonal' if conf.add_seasonal_encoding else ''})")

    # ── 5. Optimizer, scheduler, loss ────────────────────────────
    optimizer = _build_optimizer(model)
    scheduler = _build_scheduler(optimizer)
    criterion = LOSS_FNS[conf.loss_function]()

    logger.info(f"  Optimizer: {conf.optimizer}, LR={conf.learning_rate}")
    logger.info(f"  Scheduler: {conf.scheduler}, warmup={conf.warmup_epochs}")
    logger.info(f"  Loss: {conf.loss_function}")

    # ── 6. Training loop ─────────────────────────────────────────
    if _is_main(rank):
        os.makedirs(conf.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.dirname(conf.training_log_path), exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    best_val = float("inf"); patience = 0; log_rows = []
    prev_rollout = 0  # track rollout changes

    logger.info(f"  Training up to {conf.num_epochs} epochs")
    logger.info(f"  Rollout: {conf.initial_rollout_steps} → {conf.max_rollout_steps} "
                f"(epochs {conf.rollout_epoch_start}–{conf.rollout_epoch_end})")
    logger.info(f"  Early stopping patience: {conf.early_stopping_patience} "
                f"(resets on rollout change)")

    for epoch in range(1, conf.num_epochs + 1):
        rollout = get_rollout_steps(epoch)
        lr = scheduler.get_last_lr()[0]
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # ── Detect rollout change and reset early stopping ───
        if rollout != prev_rollout:
            if prev_rollout > 0:
                logger.info(f"  Rollout changed {prev_rollout} → {rollout} — "
                            f"resetting early stopping & best val loss")
            best_val = float("inf")
            patience = 0
            prev_rollout = rollout

        train_loss = _train_one_epoch(model, train_loader, optimizer, criterion,
                                      grid_features, rollout, device)
        val_loss = _validate(model, val_loader, criterion,
                             grid_features, rollout, device)

        # Normalize loss by rollout steps so values are comparable
        # (sum of weighted losses → mean per step)
        weight_sum = sum(conf.rollout_loss_decay ** s for s in range(rollout))
        train_loss_norm = train_loss / weight_sum
        val_loss_norm = val_loss / weight_sum

        scheduler.step(epoch, val_loss=val_loss_norm)

        if _is_main(rank):
            log_rows.append({"epoch": epoch, "train_loss": train_loss_norm,
                             "val_loss": val_loss_norm, "lr": lr, "rollout": rollout})
            if epoch % 5 == 0 or epoch == 1:
                logger.info(f"  Epoch {epoch:>4d}/{conf.num_epochs} | "
                            f"Train: {train_loss_norm:.6f} | Val: {val_loss_norm:.6f} | "
                            f"LR: {lr:.2e} | Rollout: {rollout}")
            if val_loss_norm < best_val:
                best_val = val_loss_norm; patience = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss_norm,
                    "rollout": rollout,
                    "world_size": world_size,
                    "config": {
                        "embed_dim": conf.embed_dim, "num_layers": conf.num_layers,
                        "num_modes_lat": conf.num_modes_lat,
                        "num_modes_lon": conf.num_modes_lon,
                        "input_history": conf.input_history,
                        "total_input_channels": conf.total_input_channels,
                        "output_channels": conf.output_channels,
                    },
                }, conf.best_model_path)
                logger.info(f"  ✓ Best model saved (val_loss={val_loss_norm:.6f}, "
                            f"rollout={rollout})")
            else:
                patience += 1
        else:
            if val_loss_norm < best_val: best_val = val_loss_norm; patience = 0
            else: patience += 1

        if patience >= conf.early_stopping_patience:
            logger.info(f"  Early stopping at epoch {epoch}"); break

    # ── 7. Save log ──────────────────────────────────────────────
    if _is_main(rank) and log_rows:
        with open(conf.training_log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            w.writeheader(); w.writerows(log_rows)
        logger.info(f"Training log: {conf.training_log_path}")

    logger.info(f"Best val loss: {best_val:.6f}")
    logger.info("=" * 60)
    logger.info("PHASE 2: MODEL TRAINING — COMPLETE")
    logger.info("=" * 60)

    _cleanup_distributed()
    return None if not _is_main(rank) else raw_model


if __name__ == "__main__":
    run()
