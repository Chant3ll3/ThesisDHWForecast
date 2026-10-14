# MOST RECENT DHW CALC SO FAR

"""
Calc_DHW02.py — Phase 5.2: DHW calculation based on NOAA CRW Methodology
Reference: https://coralreefwatch.noaa.gov/product/5km/methodology.php

SLIDING WINDOW CONSTRUCTION (NOAA CRW standard)
------------------------------------------------
DHW at lead step L uses a blended SST window of 84 days:

  Lead L:  (84-L) days actual SST  +  L days predicted SST
  Lead  1: 83 actual + 1 predicted   → DHW nearly identical to actual
  Lead 30: 54 actual + 30 predicted  → increasing weight on predictions
  Lead 84:  0 actual + 84 predicted  → fully predicted DHW
  Lead 90: predicted days 7-90       → window shifted fully into predictions

This is the CORRECT approach.  The previous version applied rolling(time=84)
directly to the 90-day prediction chains, which only had 1 day in the window
at lead 1, producing near-zero predicted DHW and a large negative error.

THREE OUTPUT VARIABLES
----------------------
  predicted_dhw : blended window using predicted SST for the lead days
  actual_dhw    : 84-day sliding window on continuous actual SST (ground truth)
  persist_dhw   : blended window using init-day SST held constant (persistence
                  baseline — harder to beat than naive flat-line DHW)

INPUTS (from ConcatChainsUpdated.py)
--------------------------------------
  sst_predicted_{split}_chains.nc    — forecast chains (chain, lead, lat, lon)
  sst_actual_{split}_continuous.nc   — continuous actual SST (time, lat, lon)

CONFIGURATION
-------------
All paths and parameters are read from config.CalcDHWConfig via cfg.
Edit config.py only — do not hardcode values in this file.
"""

import numpy as np
import xarray as xr
from pathlib import Path

from config import cfg


# ============================================================================
# STEP 1: Load MMM Climatology
# ============================================================================

def load_mmm_climatology():
    """
    Load NOAA CRW ct5km MMM climatology (Maximum Monthly Mean).
    Path is read from cfg.calc_dhw.mmm_path.

    The MMM is the reference temperature used to compute HotSpot:
      HotSpot = SST - MMM
    Pixels where SST exceeds MMM by >= hotspot_threshold contribute to DHW.
    """
    conf = cfg.calc_dhw

    print("=" * 60)
    print("STEP 1: Load MMM Climatology")
    print("=" * 60)

    ds  = xr.open_dataset(conf.mmm_path, decode_cf=True)
    mmm = ds["sst_clim_mmm"].isel(time=0)

    if mmm.dtype != np.float64:
        mmm = mmm.astype(np.float64)

    print(f"  MMM range: {float(mmm.min()):.2f} to {float(mmm.max()):.2f} degC")
    print(f"  MMM mean:  {float(mmm.mean()):.2f} degC")
    return mmm


# ============================================================================
# STEP 2: Interpolate MMM to Model Grid
# ============================================================================

def interpolate_mmm_to_grid(mmm, ds_ref):
    """
    Bilinearly interpolate the NOAA 0.05 deg MMM onto the coarser model
    grid (64 x 128).

    The MMM file uses -180/+180 lon; the model grid uses 0-360.  We convert
    target lons to -180/+180 before interpolating, then restore the original
    column order afterwards.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Interpolate MMM to Model Grid")
    print("=" * 60)

    target_lat = ds_ref.lat.values
    target_lon = ds_ref.lon.values

    print(f"  Source grid: {len(mmm.lat)} lat x {len(mmm.lon)} lon")
    print(f"  Target grid: {len(target_lat)} lat x {len(target_lon)} lon")

    # Convert 0-360 to -180/+180 for interpolation
    target_lon_180 = np.where(target_lon > 180, target_lon - 360, target_lon)

    # Sort for monotone coords (required by xr.interp)
    sort_idx          = np.argsort(target_lon_180)
    target_lon_sorted = target_lon_180[sort_idx]

    mmm_interp = mmm.interp(lat=target_lat, lon=target_lon_sorted,
                            method="linear")

    # Restore original column order
    if not np.all(np.diff(target_lon_180) >= 0):
        unsort_idx = np.argsort(sort_idx)
        mmm_interp = mmm_interp.isel(lon=unsort_idx)

    mmm_interp = mmm_interp.assign_coords(lat=target_lat, lon=target_lon)

    print(f"  Interpolated MMM: {float(mmm_interp.min()):.2f} to "
          f"{float(mmm_interp.max()):.2f} degC")
    print(f"  Interpolated MMM mean: {float(mmm_interp.mean()):.2f} degC")
    return mmm_interp


# ============================================================================
# STEP 3: Create Ocean Mask
# ============================================================================

def create_ocean_mask(mmm):
    """
    Boolean (lat, lon) mask: True = ocean, False = land.

    Land pixels in the NOAA MMM climatology are stored as NaN.
    Only ocean pixels contribute to bleaching statistics.
    """
    print("\n" + "=" * 60)
    print("STEP 3: Create Ocean Mask")
    print("=" * 60)

    ocean_mask = ~np.isnan(mmm.values)
    n_ocean    = ocean_mask.sum()
    n_total    = ocean_mask.size

    print(f"  Total pixels: {n_total:,}")
    print(f"  Ocean pixels: {n_ocean:,} ({100 * n_ocean / n_total:.1f}%)")
    print(f"  Land pixels:  {n_total - n_ocean:,} "
          f"({100 * (n_total - n_ocean) / n_total:.1f}%)")
    return ocean_mask


# ============================================================================
# STEP 4+5: DHW via Blended Sliding Window (core fix)
# ============================================================================

def compute_dhw_blended(actual_sst_cont, forecast_sst,
                        chain_start_idx, mmm_np, ocean_mask):
    """
    Compute DHW for one forecast chain using the NOAA CRW blended window.

    For each lead step L (0-indexed, so lead step = L+1):
      window = actual_sst_cont[start : chain_start_idx+1]   (84-L-1 days)
             + forecast_sst[0 : L+1]                        (L+1 days)
      HotSpot = max(window - MMM, 0)
      DHW     = sum(HotSpot >= threshold) / 7

    This ensures:
      Lead 1:  83 actual + 1 predicted  — DHW almost identical to actual
      Lead 84:  0 actual + 84 predicted — fully forecast DHW
      Lead 90: days 7-90 of predictions — window shifted past actual history

    Parameters
    ----------
    actual_sst_cont : np.ndarray (T_actual, lat, lon)
        Continuous actual SST time series (real degC, trend added back).
    forecast_sst    : np.ndarray (n_leads, lat, lon)
        Predicted SST for one forecast chain (real degC).
    chain_start_idx : int
        Index in actual_sst_cont of the init day (day before lead 1).
    mmm_np          : np.ndarray (lat, lon)
        Interpolated MMM climatology.
    ocean_mask      : np.ndarray (lat, lon) bool

    Returns
    -------
    dhw : np.ndarray (n_leads, lat, lon) float32
    """
    conf      = cfg.calc_dhw
    window    = conf.window_days       # 84
    threshold = conf.hotspot_threshold  # 1.0

    n_leads, nlat, nlon = forecast_sst.shape
    dhw = np.full((n_leads, nlat, nlon), np.nan, dtype=np.float32)

    for L in range(n_leads):
        n_actual = window - (L + 1)   # actual days in window (decreases with L)
        n_pred   = L + 1              # predicted days in window (increases with L)

        # ── Actual SST slice (pre-forecast history) ───────────────────────────
        if n_actual > 0:
            start_idx = chain_start_idx - n_actual + 1
            if start_idx < 0:
                # Not enough actual history for this chain/lead — leave NaN
                continue
            actual_slice = actual_sst_cont[start_idx : chain_start_idx + 1]
        else:
            # Lead >= window: no actual SST in window, only predictions
            actual_slice = np.empty((0, nlat, nlon), dtype=np.float32)

        # ── Predicted SST slice (forecast days) ───────────────────────────────
        pred_slice = forecast_sst[:n_pred]   # (n_pred, lat, lon)

        # ── Blended 84-day window ─────────────────────────────────────────────
        window_sst = np.concatenate([actual_slice, pred_slice], axis=0)
        # shape: (window, lat, lon)

        # ── HotSpot and DHW ───────────────────────────────────────────────────
        hotspot       = window_sst - mmm_np[np.newaxis, :, :]
        hotspot_above = np.where(hotspot >= threshold, hotspot, 0.0)
        dhw_L         = np.nansum(hotspot_above, axis=0) / 7.0
        dhw_L[~ocean_mask] = np.nan

        dhw[L] = dhw_L.astype(np.float32)

    return dhw


# ============================================================================
# STEP 6: Compute DHW for All Chains
# ============================================================================

def compute_dhw_all_chains(ds_pred, ds_actual_cont, mmm_interp, ocean_mask):
    """
    Compute three DHW products for every forecast chain:

    predicted_dhw
        Blended window: (84-L) actual + L predicted SST.
        This is the model forecast of bleaching risk.

    actual_dhw
        Sliding 84-day window on continuous actual SST only.
        This is the ground truth — what DHW really was.

    persist_dhw
        Blended window: (84-L) actual + L init-day SST (held constant).
        Scientifically rigorous persistence baseline.  Harder to beat than
        naive flat-line DHW because DHW evolves naturally as old heat stress
        falls off the 84-day window and new stress (at constant SST anomaly)
        accumulates.

    Parameters
    ----------
    ds_pred         : xr.Dataset
        Output of ConcatChainsUpdated.py:
        sst_predicted (chain, lead_step, lat, lon), init_time (chain),
        valid_time (chain, lead_step).
    ds_actual_cont  : xr.Dataset
        Output of ConcatChainsUpdated.py:
        sst_actual (time, lat, lon), time coordinate.
    mmm_interp      : xr.DataArray (lat, lon)
    ocean_mask      : np.ndarray (lat, lon) bool

    Returns
    -------
    xr.Dataset with dims (chain, lead_step, lat, lon).
    """
    conf = cfg.calc_dhw

    n_chains = ds_pred.sizes["chain"]
    n_leads  = ds_pred.sizes["lead_step"]
    nlat     = ds_pred.sizes["lat"]
    nlon     = ds_pred.sizes["lon"]
    lat      = ds_pred.lat.values
    lon      = ds_pred.lon.values

    # Load continuous actual SST into memory for fast indexed access
    actual_cont  = ds_actual_cont["sst_actual"].values   # (T, lat, lon)
    actual_times = ds_actual_cont["time"].values          # datetime64

    predicted_dhw = np.full((n_chains, n_leads, nlat, nlon), np.nan,
                            dtype=np.float32)
    actual_dhw    = np.full_like(predicted_dhw, np.nan)
    persist_dhw   = np.full_like(predicted_dhw, np.nan)

    mmm_np = mmm_interp.values   # (lat, lon)

    print(f"\n  Processing {n_chains} chains x {n_leads} lead steps ...")

    for c in range(n_chains):
        init_time = ds_pred["init_time"].values[c]   # datetime64

        # Locate the init day in the continuous actual timeline.
        # chain_start_idx is the last actual day before lead step 1.
        time_diffs      = np.abs((actual_times - init_time)
                                 .astype("timedelta64[D]").astype(int))
        chain_start_idx = int(np.argmin(time_diffs))

        # ── predicted_dhw: blended actual + model forecast ────────────────────
        pred_sst = ds_pred["sst_predicted"].values[c]   # (lead, lat, lon)
        predicted_dhw[c] = compute_dhw_blended(
            actual_cont, pred_sst, chain_start_idx, mmm_np, ocean_mask)

        # ── actual_dhw: ground truth — pure actual SST sliding window ─────────
        for L in range(n_leads):
            window_end   = chain_start_idx + L + 1
            window_start = window_end - conf.window_days
            if window_start < 0:
                continue   # not enough history at the start of the record
            window_sst    = actual_cont[window_start : window_end]
            hotspot       = window_sst - mmm_np[np.newaxis]
            hotspot_above = np.where(hotspot >= conf.hotspot_threshold,
                                     hotspot, 0.0)
            dhw_L         = np.nansum(hotspot_above, axis=0) / 7.0
            dhw_L[~ocean_mask] = np.nan
            actual_dhw[c, L]   = dhw_L.astype(np.float32)

        # ── persist_dhw: init-day SST held constant ───────────────────────────
        # Tile the first predicted SST frame to create a constant "forecast".
        # This gives a blended persistence: actual history + flat SST anomaly.
        init_sst    = pred_sst[0:1]                          # (1, lat, lon)
        persist_sst = np.tile(init_sst, (n_leads, 1, 1))    # (n_leads, lat, lon)
        persist_dhw[c] = compute_dhw_blended(
            actual_cont, persist_sst, chain_start_idx, mmm_np, ocean_mask)

        if (c + 1) % 5 == 0 or c == n_chains - 1:
            print(f"    Chain {c + 1:>3d}/{n_chains} — "
                  f"init: {str(init_time)[:10]}")

    # ── Pack outputs into an xarray Dataset ──────────────────────────────────
    lead_steps  = np.arange(1, n_leads + 1)
    init_times  = ds_pred["init_time"].values
    valid_times = ds_pred["valid_time"].values

    window  = conf.window_days
    thr     = conf.hotspot_threshold

    ds_out = xr.Dataset(
        {
            "predicted_dhw": (["chain", "lead_step", "lat", "lon"],
                               predicted_dhw),
            "actual_dhw":    (["chain", "lead_step", "lat", "lon"],
                               actual_dhw),
            "persist_dhw":   (["chain", "lead_step", "lat", "lon"],
                               persist_dhw),
        },
        coords={
            "chain":      np.arange(n_chains),
            "lead_step":  lead_steps,
            "lat":        lat,
            "lon":        lon,
            "init_time":  (["chain"],              init_times),
            "valid_time": (["chain", "lead_step"], valid_times),
        },
    )

    ds_out["predicted_dhw"].attrs = {
        "long_name": "Predicted DHW — blended sliding window",
        "units":     "degC-weeks",
        "method":    f"(84-L) actual + L predicted SST, "
                     f"HotSpot >= {thr} degC, window = {window} days",
    }
    ds_out["actual_dhw"].attrs = {
        "long_name": "Actual DHW — ground truth (observed SST only)",
        "units":     "degC-weeks",
        "method":    f"84-day sliding window on continuous actual SST, "
                     f"HotSpot >= {thr} degC",
    }
    ds_out["persist_dhw"].attrs = {
        "long_name": "Persistence DHW — init-day SST held constant",
        "units":     "degC-weeks",
        "method":    f"(84-L) actual + L init-day SST, "
                     f"HotSpot >= {thr} degC, window = {window} days",
    }

    return ds_out


# ============================================================================
# STEP 7: Print Summary
# ============================================================================

def print_dhw_summary(ds_dhw, label=""):
    """Print bleaching statistics for all three DHW variables."""
    conf = cfg.calc_dhw
    print("\n" + "=" * 60)
    print(f"DHW Summary — {label}")
    print("=" * 60)
    print(f"  Window: {conf.window_days} days  |  "
          f"Threshold: {conf.hotspot_threshold} degC")

    for var, name in [
        ("predicted_dhw", "Predicted"),
        ("actual_dhw",    "Actual"),
        ("persist_dhw",   "Persistence"),
    ]:
        d      = ds_dhw[var].values
        finite = d[np.isfinite(d)]
        if len(finite) == 0:
            print(f"\n  {name}: all NaN")
            continue

        print(f"\n  {name}:")
        print(f"    Mean: {np.nanmean(finite):.4f}  "
              f"Max: {np.nanmax(finite):.4f} degC-weeks")
        print(f"    No Stress (<4):    {np.sum(finite < 4):>10,} "
              f"({100 * np.mean(finite < 4):.1f}%)")
        print(f"    Warning  (4-8):    {np.sum((finite >= 4) & (finite < 8)):>10,} "
              f"({100 * np.mean((finite >= 4) & (finite < 8)):.1f}%)")
        print(f"    Alert L2 (>=8):    {np.sum(finite >= 8):>10,} "
              f"({100 * np.mean(finite >= 8):.1f}%)")


# ============================================================================
# STEP 8: Save Results
# ============================================================================

def save_dhw_netcdf(ds_dhw, output_path):
    """Save DHW products to a compressed NetCDF file."""
    print("\n" + "=" * 60)
    print("STEP 8: Save DHW Results")
    print("=" * 60)
    print(f"  Output: {output_path}")

    encoding = {var: {"zlib": True, "complevel": 4, "dtype": "float32"}
                for var in ds_dhw.data_vars}
    ds_dhw.to_netcdf(output_path, encoding=encoding)
    print("  Saved successfully")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    conf        = cfg.calc_dhw
    results_dir = Path(conf.results_dir)

    # DHW output goes to a dedicated directory so it doesn't mix with
    # intermediate SST files in results_dir.
    dhw_output_dir = Path(conf.dhw_output_dir)
    dhw_output_dir.mkdir(parents=True, exist_ok=True)

    # Load MMM climatology once — reused for both splits
    mmm_global = load_mmm_climatology()

    for split, run_flag in [("test", conf.run_test),
                             ("val",  conf.run_validation)]:
        if not run_flag:
            continue

        print("\n" + "#" * 80)
        print("#" + f"  DHW CALCULATION — {split.upper()} SET".center(78) + "#")
        print("#" * 80)

        # ── Load ConcatChains outputs ─────────────────────────────────────────
        pred_path   = results_dir / conf.predicted_file_template.format(
                          split=split)
        actual_path = results_dir / conf.actual_file_template.format(
                          split=split)

        print(f"\nLoading predicted SST chains : {pred_path}")
        ds_pred = xr.open_dataset(pred_path)
        print(f"  Chains:    {ds_pred.sizes['chain']}")
        print(f"  Leads:     {ds_pred.sizes['lead_step']}")
        print(f"  Grid:      {ds_pred.sizes['lat']} lat x "
              f"{ds_pred.sizes['lon']} lon")

        print(f"\nLoading continuous actual SST: {actual_path}")
        ds_actual = xr.open_dataset(actual_path)
        print(f"  Timesteps: {ds_actual.sizes['time']}")

        # ── Interpolate MMM and build ocean mask ──────────────────────────────
        mmm_model  = interpolate_mmm_to_grid(mmm_global, ds_pred)
        ocean_mask = create_ocean_mask(mmm_model)

        # ── Compute DHW with the correct blended sliding window ───────────────
        print("\n" + "=" * 60)
        print("STEP 6: Compute DHW — blended sliding window")
        print("=" * 60)
        ds_dhw = compute_dhw_all_chains(ds_pred, ds_actual,
                                        mmm_model, ocean_mask)

        # ── Summary and save ──────────────────────────────────────────────────
        print_dhw_summary(ds_dhw, label=split.capitalize())

        out_path = dhw_output_dir / conf.output_file_template.format(
                       split=split)
        save_dhw_netcdf(ds_dhw, out_path)

        ds_pred.close()
        ds_actual.close()

    print("\n" + "#" * 80)
    print("#" + "  DHW CALCULATION COMPLETE".center(78) + "#")
    print("#" * 80)