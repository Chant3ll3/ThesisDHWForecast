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

THREE DHW OUTPUT VARIABLES
---------------------------
  predicted_dhw : blended window using predicted SST for the lead days
  actual_dhw    : 84-day sliding window on continuous actual SST (ground truth)
  persist_dhw   : blended window using init-day SST held constant (persistence
                  baseline — harder to beat than naive flat-line DHW)

THREE MATCHING HOTSPOT OUTPUT VARIABLES
-----------------------------------------
  predicted_hs / actual_hs / persist_hs : the HotSpot (SST - MMM) on the
  final day of each window, i.e. the same day the corresponding DHW value
  is reported for. HS is required alongside DHW to classify pixels into
  NOAA CRW's Bleaching Alert Area categories (Table 1.1), since every
  category above "Bleaching Watch" is gated on HS >= 1, not on DHW alone.

BLEACHING ALERT AREA CLASSIFICATION (NOAA CRW, Table 1.1)
------------------------------------------------------------
  No Stress               : HS <= 0
  Bleaching Watch          : 0 < HS < 1
  Bleaching Warning        : HS >= 1 and 0 < DHW < 4
  Bleaching Alert Level 1 : HS >= 1 and 4 <= DHW < 8
  Bleaching Alert Level 2 : HS >= 1 and 8 <= DHW < 12
  Bleaching Alert Level 3 : HS >= 1 and 12 <= DHW < 16
  Bleaching Alert Level 4 : HS >= 1 and 16 <= DHW < 20
  Bleaching Alert Level 5 : HS >= 1 and DHW >= 20

  Note: Table 1.1 does not explicitly define HS >= 1 combined with
  DHW <= 0 (Warning requires DHW > 0 strictly). This edge case is
  classified as "No Stress" here, since there is no accumulated heat
  stress despite an elevated instantaneous HS. This is a modelling
  choice, not a NOAA-defined rule — see thesis methodology for discussion.

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
# STEP 4+5: DHW and HotSpot via Blended Sliding Window (core fix)
# ============================================================================

def compute_dhw_blended(actual_sst_cont, forecast_sst,
                        chain_start_idx, mmm_np, ocean_mask):
    """
    Compute DHW and the final-day HotSpot for one forecast chain, using
    the NOAA CRW blended window.

    For each lead step L (0-indexed, so lead step = L+1):
      window = actual_sst_cont[start : chain_start_idx+1]   (84-L-1 days)
             + forecast_sst[0 : L+1]                        (L+1 days)
      HotSpot(t) = max(window(t) - MMM, 0)  for every day t in the window
      DHW        = sum(HotSpot(t) >= threshold) / 7
      HS_final   = HotSpot on the LAST day of the window
                   (i.e. the day the DHW value is reported for — needed
                   to classify each pixel into a Bleaching Alert Area
                   category alongside DHW, per Table 1.1).

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
    dhw      : np.ndarray (n_leads, lat, lon) float32
    hs_final : np.ndarray (n_leads, lat, lon) float32
        Final-day HotSpot (SST - MMM, not thresholded) for each lead step.
    """
    conf      = cfg.calc_dhw
    window    = conf.window_days       # 84
    threshold = conf.hotspot_threshold  # 1.0

    n_leads, nlat, nlon = forecast_sst.shape
    dhw      = np.full((n_leads, nlat, nlon), np.nan, dtype=np.float32)
    hs_final = np.full((n_leads, nlat, nlon), np.nan, dtype=np.float32)

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

        # ── Final-day HotSpot (for Bleaching Alert Area classification) ──────
        hs_L = hotspot[-1].copy()
        hs_L[~ocean_mask] = np.nan
        hs_final[L] = hs_L.astype(np.float32)

    return dhw, hs_final


# ============================================================================
# STEP 6: Compute DHW and HotSpot for All Chains
# ============================================================================

def compute_dhw_all_chains(ds_pred, ds_actual_cont, mmm_interp, ocean_mask):
    """
    Compute three DHW products and three matching HotSpot products for
    every forecast chain:

    predicted_dhw / predicted_hs
        Blended window: (84-L) actual + L predicted SST.
        This is the model forecast of bleaching risk.

    actual_dhw / actual_hs
        Sliding 84-day window on continuous actual SST only.
        This is the ground truth — what DHW/HS really was.

    persist_dhw / persist_hs
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
    xr.Dataset with dims (chain, lead_step, lat, lon), containing
    predicted_dhw, actual_dhw, persist_dhw, predicted_hs, actual_hs,
    persist_hs.
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

    predicted_hs  = np.full_like(predicted_dhw, np.nan)
    actual_hs     = np.full_like(predicted_dhw, np.nan)
    persist_hs    = np.full_like(predicted_dhw, np.nan)

    mmm_np = mmm_interp.values   # (lat, lon)
    threshold = conf.hotspot_threshold

    print(f"\n  Processing {n_chains} chains x {n_leads} lead steps ...")

    for c in range(n_chains):
        init_time = ds_pred["init_time"].values[c]   # datetime64

        # Locate the init day in the continuous actual timeline.
        # chain_start_idx is the last actual day before lead step 1.
        time_diffs      = np.abs((actual_times - init_time)
                                 .astype("timedelta64[D]").astype(int))
        chain_start_idx = int(np.argmin(time_diffs))

        # ── predicted_dhw / predicted_hs: blended actual + model forecast ─────
        pred_sst = ds_pred["sst_predicted"].values[c]   # (lead, lat, lon)
        predicted_dhw[c], predicted_hs[c] = compute_dhw_blended(
            actual_cont, pred_sst, chain_start_idx, mmm_np, ocean_mask)

        # ── actual_dhw / actual_hs: ground truth — pure actual SST window ─────
        for L in range(n_leads):
            window_end   = chain_start_idx + L + 1
            window_start = window_end - conf.window_days
            if window_start < 0:
                continue   # not enough history at the start of the record
            window_sst    = actual_cont[window_start : window_end]
            hotspot       = window_sst - mmm_np[np.newaxis]
            hotspot_above = np.where(hotspot >= threshold, hotspot, 0.0)
            dhw_L         = np.nansum(hotspot_above, axis=0) / 7.0
            dhw_L[~ocean_mask] = np.nan
            actual_dhw[c, L]   = dhw_L.astype(np.float32)

            hs_L = hotspot[-1].copy()
            hs_L[~ocean_mask] = np.nan
            actual_hs[c, L] = hs_L.astype(np.float32)

        # ── persist_dhw / persist_hs: init-day SST held constant ──────────────
        # Tile the first predicted SST frame to create a constant "forecast".
        # This gives a blended persistence: actual history + flat SST anomaly.
        init_sst    = pred_sst[0:1]                          # (1, lat, lon)
        persist_sst = np.tile(init_sst, (n_leads, 1, 1))    # (n_leads, lat, lon)
        persist_dhw[c], persist_hs[c] = compute_dhw_blended(
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
            "predicted_hs":  (["chain", "lead_step", "lat", "lon"],
                               predicted_hs),
            "actual_hs":     (["chain", "lead_step", "lat", "lon"],
                               actual_hs),
            "persist_hs":    (["chain", "lead_step", "lat", "lon"],
                               persist_hs),
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
    ds_out["predicted_hs"].attrs = {
        "long_name": "Predicted HotSpot (SST - MMM) on final day of window",
        "units":     "degC",
        "method":    "Instantaneous HotSpot on the last day of the "
                     "blended predicted window, used for Bleaching Alert "
                     "Area classification alongside predicted_dhw.",
    }
    ds_out["actual_hs"].attrs = {
        "long_name": "Actual HotSpot (SST - MMM) on final day of window",
        "units":     "degC",
        "method":    "Instantaneous HotSpot on the last day of the "
                     "actual SST window, used for Bleaching Alert Area "
                     "classification alongside actual_dhw.",
    }
    ds_out["persist_hs"].attrs = {
        "long_name": "Persistence HotSpot (SST - MMM) on final day of window",
        "units":     "degC",
        "method":    "Instantaneous HotSpot on the last day of the "
                     "persistence window, used for Bleaching Alert Area "
                     "classification alongside persist_dhw.",
    }

    return ds_out


# ============================================================================
# STEP 7: Bleaching Alert Area Classification (NOAA CRW, Table 1.1)
# ============================================================================

def classify_bleaching_alert(hs, dhw):
    """
    Classify each pixel into one of NOAA CRW's eight Bleaching Alert Area
    categories, using both HotSpot (HS) and DHW, per Table 1.1:

        No Stress               : HS <= 0
        Bleaching Watch          : 0 < HS < 1
        Bleaching Warning        : HS >= 1 and 0 < DHW < 4
        Bleaching Alert Level 1 : HS >= 1 and 4 <= DHW < 8
        Bleaching Alert Level 2 : HS >= 1 and 8 <= DHW < 12
        Bleaching Alert Level 3 : HS >= 1 and 12 <= DHW < 16
        Bleaching Alert Level 4 : HS >= 1 and 16 <= DHW < 20
        Bleaching Alert Level 5 : HS >= 1 and DHW >= 20

    Note: HS >= 1 combined with DHW <= 0 is not explicitly defined by
    Table 1.1 (Warning requires DHW > 0 strictly). This edge case is
    classified as "No Stress" here — a modelling choice, not a
    NOAA-defined rule.

    Parameters
    ----------
    hs  : np.ndarray
        HotSpot values (SST - MMM), any shape.
    dhw : np.ndarray
        DHW values (degC-weeks), same shape as hs.

    Returns
    -------
    labels : np.ndarray of str, same shape as hs.
        One of: "No Stress", "Bleaching Watch", "Bleaching Warning",
        "Alert Level 1", "Alert Level 2", "Alert Level 3",
        "Alert Level 4", "Alert Level 5", or "NaN" for missing pixels.
    """
    hs  = np.asarray(hs)
    dhw = np.asarray(dhw)

    labels = np.full(hs.shape, "NaN", dtype=object)

    valid = np.isfinite(hs) & np.isfinite(dhw)

    no_stress      = valid & (hs <= 0)
    watch          = valid & (hs > 0) & (hs < 1)
    hs_high_no_dhw = valid & (hs >= 1) & (dhw <= 0)
    warning        = valid & (hs >= 1) & (dhw > 0)  & (dhw < 4)
    alert_l1       = valid & (hs >= 1) & (dhw >= 4)  & (dhw < 8)
    alert_l2       = valid & (hs >= 1) & (dhw >= 8)  & (dhw < 12)
    alert_l3       = valid & (hs >= 1) & (dhw >= 12) & (dhw < 16)
    alert_l4       = valid & (hs >= 1) & (dhw >= 16) & (dhw < 20)
    alert_l5       = valid & (hs >= 1) & (dhw >= 20)

    labels[no_stress]      = "No Stress"
    labels[watch]          = "Bleaching Watch"
    labels[hs_high_no_dhw] = "No Stress"
    labels[warning]        = "Bleaching Warning"
    labels[alert_l1]       = "Alert Level 1"
    labels[alert_l2]       = "Alert Level 2"
    labels[alert_l3]       = "Alert Level 3"
    labels[alert_l4]       = "Alert Level 4"
    labels[alert_l5]       = "Alert Level 5"

    return labels


def print_bleaching_alert_summary(hs, dhw, label=""):
    """
    Print the distribution of pixels across all eight Bleaching Alert
    Area categories (NOAA CRW, Table 1.1), for one DHW/HS field.

    Parameters
    ----------
    hs    : np.ndarray — HotSpot values, any shape.
    dhw   : np.ndarray — DHW values, same shape as hs.
    label : str — descriptive label for the printed header.
    """
    labels = classify_bleaching_alert(hs, dhw)

    categories = [
        "No Stress", "Bleaching Watch", "Bleaching Warning",
        "Alert Level 1", "Alert Level 2", "Alert Level 3",
        "Alert Level 4", "Alert Level 5",
    ]

    n_valid = np.sum(labels != "NaN")

    print(f"\n  Bleaching Alert Area — {label}")
    print(f"  Total valid pixels: {n_valid:,}")

    for cat in categories:
        n_cat = np.sum(labels == cat)
        pct   = 100 * n_cat / n_valid if n_valid > 0 else 0.0
        print(f"    {cat:<20s}: {n_cat:>10,} ({pct:5.1f}%)")


# ============================================================================
# STEP 8: Print DHW Summary
# ============================================================================

def print_dhw_summary(ds_dhw, label=""):
    """Print DHW statistics (mean/max/simple bins) for all three DHW variables."""
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
# STEP 9: Save Results
# ============================================================================

def save_dhw_netcdf(ds_dhw, output_path):
    """Save DHW and HotSpot products to a compressed NetCDF file."""
    print("\n" + "=" * 60)
    print("STEP 9: Save DHW Results")
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

        # ── Compute DHW and HS with the correct blended sliding window ────────
        print("\n" + "=" * 60)
        print("STEP 6: Compute DHW and HotSpot — blended sliding window")
        print("=" * 60)
        ds_dhw = compute_dhw_all_chains(ds_pred, ds_actual,
                                        mmm_model, ocean_mask)

        # ── DHW summary (simple bins, retained for continuity) ────────────────
        print_dhw_summary(ds_dhw, label=split.capitalize())

        # ── Full Bleaching Alert Area classification (Table 1.1) ──────────────
        print("\n" + "=" * 60)
        print("STEP 7: Bleaching Alert Area Classification (Table 1.1)")
        print("=" * 60)
        print_bleaching_alert_summary(
            ds_dhw["predicted_hs"].values, ds_dhw["predicted_dhw"].values,
            label=f"{split.capitalize()} — Predicted")
        print_bleaching_alert_summary(
            ds_dhw["actual_hs"].values, ds_dhw["actual_dhw"].values,
            label=f"{split.capitalize()} — Actual")
        print_bleaching_alert_summary(
            ds_dhw["persist_hs"].values, ds_dhw["persist_dhw"].values,
            label=f"{split.capitalize()} — Persistence")

        # ── Save ────────────────────────────────────────────────────────────
        out_path = dhw_output_dir / conf.output_file_template.format(
                       split=split)
        save_dhw_netcdf(ds_dhw, out_path)

        ds_pred.close()
        ds_actual.close()

    print("\n" + "#" * 80)
    print("#" + "  DHW CALCULATION COMPLETE".center(78) + "#")
    print("#" * 80)