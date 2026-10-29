"""
model_inference.py — Phase 3: Model Inference.

Loads the trained SFNO and performs autoregressive inference on
validation and/or test splits.  Stores predictions alongside actuals
in NetCDF with dimensions (chain, lead_step, lat, lon).
"""

import os
import logging
import numpy as np
import xarray as xr
import torch

from config import cfg
from SST_Forecast_Pipeline.sfno_model import SFNO, make_grid_features, make_seasonal_encoding
from SST_Forecast_Pipeline.sst_dataset import load_split_data, _compute_fractional_day_of_year

logger = logging.getLogger("Phase3_Inference")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(h)


# ── Load model ───────────────────────────────────────────────────────

def load_model(device):
    conf = cfg.model_inference
    logger.info(f"Loading model from {conf.model_path}")
    ckpt = torch.load(conf.model_path, map_location=device, weights_only=False)
    model = SFNO().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"  Epoch: {ckpt.get('epoch','?')}, "
                f"val_loss: {ckpt.get('val_loss','?')}, "
                f"params: {model.count_parameters():,}")
    return model


# ── Chain start indices ──────────────────────────────────────────────

def get_chain_starts(n_time, history, lead_steps, stride_steps):
    first = history
    last = n_time - lead_steps
    return list(range(first, last, stride_steps))


# ── Build auxiliary features for a single sample ─────────────────────

def _build_aux_single(doy_value, grid_features, device, H, W):
    """Build (1, C_aux, H, W) auxiliary features for one sample."""
    conf = cfg.model_training
    parts = []

    if conf.add_grid_features and grid_features is not None:
        parts.append(grid_features)  # already (1, 2, H, W)

    if conf.add_seasonal_encoding:
        doy_tensor = torch.tensor([doy_value], dtype=torch.float32, device=device)
        se = make_seasonal_encoding(doy_tensor, conf.seasonal_cycle_period, H, W)
        parts.append(se.to(device))

    if not parts:
        return None
    return torch.cat(parts, dim=1)


# ── Run one forecast chain ───────────────────────────────────────────

@torch.no_grad()
def run_single_chain(model, data_np, times, t_start, lead_steps,
                     grid_features, device):
    """
    Run one autoregressive forecast chain.

    Returns predictions, actuals, init_field arrays.
    """
    conf = cfg.model_training
    history = conf.input_history

    context = data_np[t_start - history : t_start]
    current_input = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)
    init_field = data_np[t_start - 1].copy()

    _, _, H, W = current_input.shape

    # Day-of-year for seasonal encoding
    doy_values = _compute_fractional_day_of_year(times)
    current_doy = float(doy_values[t_start - 1])

    ta = cfg.data_preparation.temporal_average
    doy_increment = {"none": 1.0, "weekly": 7.0, "monthly": 30.44}.get(ta, 1.0)

    predictions = np.empty((lead_steps, H, W), dtype=np.float32)
    actuals = np.empty_like(predictions)

    for step in range(lead_steps):
        current_doy += doy_increment
        aux = _build_aux_single(current_doy, grid_features, device, H, W)

        if aux is not None:
            model_input = torch.cat([current_input, aux], dim=1)
        else:
            model_input = current_input

        pred = model(model_input)

        predictions[step] = pred[0, 0].cpu().numpy()
        actuals[step] = data_np[t_start + step]

        current_input = torch.cat([current_input[:, 1:, :, :],
                                   pred.detach()], dim=1)

    return predictions, actuals, init_field


# ── Run inference on a full split ────────────────────────────────────

def run_inference_split(model, data_np, time_coords, split_name,
                        grid_features, device, lat, lon):
    conf = cfg.model_inference
    train_conf = cfg.model_training
    lead_steps = conf.forecast_lead_steps
    stride = conf.chain_stride_steps
    history = train_conf.input_history

    starts = get_chain_starts(len(data_np), history, lead_steps, stride)
    n_chains = len(starts)
    logger.info(f"  {split_name}: {n_chains} chains "
                f"(lead={lead_steps}, stride={stride}, "
                f"from {len(data_np)} time steps)")

    if n_chains == 0:
        logger.warning(f"  No valid chains — skipping"); return None

    n_lat, n_lon = data_np.shape[1], data_np.shape[2]
    all_preds = np.empty((n_chains, lead_steps, n_lat, n_lon), dtype=np.float32)
    all_actuals = np.empty_like(all_preds)
    all_init_fields = np.empty((n_chains, n_lat, n_lon), dtype=np.float32)
    init_times = []
    valid_times = np.empty((n_chains, lead_steps), dtype="datetime64[ns]")

    for i, t_start in enumerate(starts):
        preds, acts, init_f = run_single_chain(
            model, data_np, time_coords, t_start, lead_steps,
            grid_features, device)
        all_preds[i] = preds
        all_actuals[i] = acts
        all_init_fields[i] = init_f
        init_times.append(time_coords[t_start])
        for s in range(lead_steps):
            valid_times[i, s] = time_coords[t_start + s]
        if (i+1) % 10 == 0 or i == n_chains - 1:
            logger.info(f"    Chain {i+1:>4d}/{n_chains} — "
                        f"init: {str(init_times[-1])[:10]}")

    ds = xr.Dataset(
        {
            "prediction":  (["chain", "lead_step", "lat", "lon"], all_preds),
            "actual":      (["chain", "lead_step", "lat", "lon"], all_actuals),
            "init_field":  (["chain", "lat", "lon"], all_init_fields),
            "init_time":   (["chain"], np.array(init_times)),
            "valid_time":  (["chain", "lead_step"], valid_times),
        },
        coords={
            "chain": np.arange(n_chains),
            "lead_step": np.arange(1, lead_steps + 1),
            "lat": lat, "lon": lon,
        },
        attrs={
            "split": split_name,
            "forecast_lead_steps": lead_steps,
            "chain_stride_steps": stride,
            "input_history": history,
            "temporal_average": cfg.data_preparation.temporal_average,
            "model_path": conf.model_path,
            "project": cfg.project_name,
        },
    )
    ds["prediction"].attrs = {"long_name": "Predicted SST (standardised)"}
    ds["actual"].attrs = {"long_name": "Actual SST (standardised)"}
    ds["init_field"].attrs = {"long_name": "Last observed SST (persistence anchor)"}
    ds["init_time"].attrs = {"long_name": "Forecast initialisation time"}
    ds["valid_time"].attrs = {"long_name": "Valid time for each lead step"}
    ds["lead_step"].attrs = {"long_name": "Forecast lead time in time steps"}

    return ds


# ── Save ─────────────────────────────────────────────────────────────

def save_predictions(ds, filename):
    conf = cfg.model_inference
    out_path = os.path.join(conf.output_dir, filename)
    os.makedirs(conf.output_dir, exist_ok=True)
    encoding = {
        "prediction": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "actual":     {"zlib": True, "complevel": 4, "dtype": "float32"},
        "init_field": {"zlib": True, "complevel": 4, "dtype": "float32"},
    }
    logger.info(f"  Saving {out_path}")
    ds.to_netcdf(out_path, encoding=encoding, mode="w")
    logger.info(f"  Saved — chains={ds.sizes['chain']}, "
                f"lead_steps={ds.sizes['lead_step']}")


# ── Main ─────────────────────────────────────────────────────────────

def run():
    print()
    logger.info("=" * 60)
    logger.info("PHASE 3: MODEL INFERENCE — START")
    logger.info("=" * 60)

    conf = cfg.model_inference
    train_conf = cfg.model_training
    device = conf.device

    model = load_model(device)

    logger.info("Loading processed SST data splits")
    (train_data, val_data, test_data,
     train_times, val_times, test_times, lat, lon) = load_split_data()
    logger.info(f"  Val : {val_data.shape}  ({train_conf.val_years})")
    logger.info(f"  Test: {test_data.shape}  ({train_conf.test_years})")

    grid_features = None
    if train_conf.add_grid_features:
        grid_features = make_grid_features(
            torch.tensor(lat, dtype=torch.float32),
            torch.tensor(lon, dtype=torch.float32),
        ).to(device)

    logger.info(f"Forecast: lead={conf.forecast_lead_steps} steps, "
                f"stride={conf.chain_stride_steps} steps")

    if conf.run_validation:
        logger.info("Running inference on VALIDATION split")
        ds = run_inference_split(model, val_data, val_times, "validation",
                                grid_features, device, lat, lon)
        if ds is not None:
            save_predictions(ds, conf.val_predictions_file)

    if conf.run_test:
        logger.info("Running inference on TEST split")
        ds = run_inference_split(model, test_data, test_times, "test",
                                grid_features, device, lat, lon)
        if ds is not None:
            save_predictions(ds, conf.test_predictions_file)

    logger.info("=" * 60)
    logger.info("PHASE 3: MODEL INFERENCE — COMPLETE")
    logger.info("=" * 60)
    print()


if __name__ == "__main__":
    run()
