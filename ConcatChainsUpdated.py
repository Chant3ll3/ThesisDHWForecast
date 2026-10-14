"""
ConcatChainsUpdated.py — Phase 3.5: Chain Concatenation.

Prepares SST forecast output for coral bleaching / DHW analysis.

Prediction files contain 19 (test) or 16 (validation) chains of 90 days
each, with an ACTUAL SST field and a PREDICTED SST field per chain.

  ACTUAL SST:
      Independent of forecast lead time — can be concatenated into one
      continuous timeline using init_time / valid_time.

  PREDICTED SST:
      Built via the sliding-window method: given input_history = k, the
      first prediction uses k actual SST frames as input, the next uses
      (k-1) actual + 1 predicted, and so on. Because later chains reuse
      predictions from earlier steps, forecast chains overlap in time
      and cannot be concatenated into one continuous series — they must
      remain independent, sequential chains.

Pipeline:
  1. Load prediction files.
  2. Remove global standardisation.
  3. Add back global linear trend.
  4. Concatenate ACTUAL SST into one continuous timeline.
  5. Preserve PREDICTED SST as independent forecast chains.
  6. Check for duplicate timestamps.
  7. Save NetCDF with real datetime coordinates.

Configuration:
  All paths and tunable parameters are read from config.ConcatChainsConfig
  via the global cfg object. To change any setting, edit config.py only —
  do not hardcode values in this file.

Outputs:
  Actual SST:    sst_actual_{split}_continuous.nc
  Predicted SST: sst_predicted_{split}_chains.nc
"""

import os
import numpy as np
import xarray as xr
import pandas as pd

from config import cfg


# ── Load transformation parameters ───────────────────────────────────

def load_params():
    """
    Load the parameters needed to reverse the two preprocessing steps:
      1. Global standardisation (subtract mean, divide by std)
      2. Global linear trend removal (subtracted during data prep)

    Paths are read from cfg.concat_chains:
      std_params_path   -> global_mean, global_std
      trend_params_path -> per-pixel slope, intercept
    """
    conf = cfg.concat_chains

    std_params = xr.open_dataset(conf.std_params_path)

    global_mean = float(std_params['global_mean'].values)
    global_std  = float(std_params['global_std'].values)

    print("Standardisation params:")
    print(f"  global_mean: {global_mean:.4f}")
    print(f"  global_std:  {global_std:.4f}")

    trend_params = xr.open_dataset(conf.trend_params_path)

    print("\nTrend params variables:",
          list(trend_params.data_vars))

    print("Trend params dims:",
          dict(trend_params.sizes))

    slope     = trend_params['slope'].values
    intercept = trend_params['intercept'].values

    print(f"  slope shape: {np.shape(slope)}")
    print(f"  intercept shape: {np.shape(intercept)}")

    return global_mean, global_std, slope, intercept


# ── Reverse transformations ──────────────────────────────────────────

def reverse_transformations(
    data_standardised,
    time_days,
    global_mean,
    global_std,
    slope,
    intercept
):
    """
    Reverse preprocessing:
      1. Destandardise
      2. Add back trend
    """
    # Destandardise
    sst = data_standardised * global_std + global_mean

    # Add trend back
    trend = slope * time_days + intercept

    sst = sst + trend

    return sst


# ── Duplicate check ──────────────────────────────────────────────────

def check_duplicates(times):
    print("\n2. CHECKING FOR DUPLICATE TIMESTAMPS")
    print("-" * 50)

    total      = len(times)
    unique     = len(np.unique(times))
    duplicates = total - unique

    print(f"Total timestamps:      {total}")
    print(f"Unique timestamps:     {unique}")
    print(f"Duplicate timestamps:  {duplicates}")

    if duplicates > 0:
        unique_vals, counts = np.unique(times, return_counts=True)
        dupes = unique_vals[counts > 1]

        print("\nDuplicate dates found:")
        for d in dupes[:10]:
            print(" ", pd.Timestamp(d))

    return duplicates


# ── Fix lon coordinates ───────────────────────────────────────────────

def fix_lon_coordinates(lon_raw):
    """
    Reconstruct the correct T42 global lon grid from the grid size.

    The raw predictions file stores lon as values spanning 0->~179 degrees,
    which is half the correct 0->357 range for a 128-point global grid.
    The spacing is ~1.406 deg instead of the correct 2.8125 deg (= 360/128).

    Detection threshold is read from cfg.concat_chains.lon_correction_threshold
    (default 0.6). Set it to 0.0 in config.py to disable correction entirely.

    Parameters
    ----------
    lon_raw : np.ndarray
        Longitude values as stored in the predictions NetCDF.

    Returns
    -------
    np.ndarray
        Corrected longitude values on the standard T42 global grid,
        or the original array if no correction was needed.
    """
    conf = cfg.concat_chains

    nlon = len(lon_raw)
    if nlon < 2:
        return lon_raw

    stored_span   = float(lon_raw[-1] - lon_raw[0])
    expected_span = 360.0 * (1.0 - 1.0 / nlon)   # e.g. 357.1875 for nlon=128

    if stored_span < expected_span * conf.lon_correction_threshold:
        lon_correct = np.linspace(0, 360 - 360 / nlon, nlon)
        print(f"\n  WARNING: lon coordinate corrected.")
        print(f"    Stored:  {lon_raw[0]:.4f} to {lon_raw[-1]:.4f} "
              f"(span {stored_span:.2f} deg, spacing {lon_raw[1]-lon_raw[0]:.4f} deg)")
        print(f"    Correct: {lon_correct[0]:.4f} to {lon_correct[-1]:.4f} "
              f"(span {expected_span:.2f} deg, spacing {360/nlon:.4f} deg)")
        return lon_correct

    return lon_raw


# ── Process one split ────────────────────────────────────────────────

def process_predictions(split='test'):
    """
    Full pipeline for one data split.

    Input and output paths are built from cfg.concat_chains using the
    {split} placeholder in the file templates, so changing directories
    only requires editing config.py.
    """
    conf = cfg.concat_chains

    # ── Load predictions ─────────────────────────────────────────
    predictions_path = os.path.join(
        conf.predictions_dir,
        conf.predictions_file_template.format(split=split),
    )

    ds = xr.open_dataset(predictions_path, decode_times=True)

    global_mean, global_std, slope, intercept = load_params()

    n_chains = ds.sizes['chain']
    n_steps  = ds.sizes['lead_step']

    print("\n" + "=" * 60)
    print(f"PROCESSING {split.upper()}")
    print("=" * 60)

    print(f"Chains: {n_chains}")
    print(f"Lead steps: {n_steps}")

    print("Time range:")
    print(f"  {pd.Timestamp(ds.valid_time.min().values)}")
    print("  to")
    print(f"  {pd.Timestamp(ds.valid_time.max().values)}")

    # ── Fix lon coordinates ──────────────────────────────────────
    # The raw file stores lon over 0-179 instead of the correct 0-357 for a
    # 128-point T42 global grid. Threshold is set in cfg.concat_chains.
    lat = ds.lat.values                        # lat is correct, leave unchanged
    lon = fix_lon_coordinates(ds.lon.values)   # fix lon if halved

    print(f"\n  Grid: {len(lat)} lat x {len(lon)} lon")
    print(f"  Lat: {lat[0]:.3f} to {lat[-1]:.3f}")
    print(f"  Lon: {lon[0]:.4f} to {lon[-1]:.4f}")

    # ── Concatenate actual SST ───────────────────────────────────
    print("\n1. CONCATENATING ACTUAL SST")

    all_actuals     = []
    all_valid_times = []

    for i in range(n_chains):
        for j in range(n_steps):
            all_actuals.append(ds.actual.values[i, j])
            all_valid_times.append(ds.valid_time.values[i, j])

    all_actuals     = np.array(all_actuals)
    all_valid_times = np.array(all_valid_times)

    # ── Check duplicates ─────────────────────────────────────────
    check_duplicates(all_valid_times)

    # ── Sort + remove duplicates ─────────────────────────────────
    print("\n3. REMOVING DUPLICATES")

    sort_idx       = np.argsort(all_valid_times)
    sorted_times   = all_valid_times[sort_idx]
    sorted_actuals = all_actuals[sort_idx]

    _, unique_idx = np.unique(sorted_times, return_index=True)

    valid_unique  = sorted_times[unique_idx]
    actual_unique = sorted_actuals[unique_idx]

    print(f"Remaining unique timesteps: {len(valid_unique)}")

    # ── Compute time axis ────────────────────────────────────────
    print("\n4. COMPUTING TIME AXIS")

    ref_date = pd.Timestamp(valid_unique[0])

    time_days_actual = (
        pd.DatetimeIndex(valid_unique) - ref_date
    ).days.values

    print(f"Reference date: {ref_date}")
    print(f"Time days range: {time_days_actual.min()} to {time_days_actual.max()}")

    # ── Destandardise actual SST ─────────────────────────────────
    print("\n5. DESTANDARDISING + ADDING TREND")

    actual_real = np.empty_like(actual_unique, dtype=np.float32)

    for t in range(len(valid_unique)):
        actual_real[t] = reverse_transformations(
            actual_unique[t],
            time_days_actual[t],
            global_mean,
            global_std,
            slope,
            intercept,
        )

    print("Actual SST range:")
    print(f"  {np.nanmin(actual_real):.2f} to {np.nanmax(actual_real):.2f} °C")

    # ── Process predicted SST ────────────────────────────────────
    print("\n6. PROCESSING PREDICTED SST CHAINS")

    pred_real = np.empty_like(ds.prediction.values, dtype=np.float32)

    for i in range(n_chains):

        init_time = pd.Timestamp(ds.init_time.values[i])
        init_days = (init_time - ref_date).days

        for j in range(n_steps):

            time_days = init_days + j + 1

            pred_real[i, j] = reverse_transformations(
                ds.prediction.values[i, j],
                time_days,
                global_mean,
                global_std,
                slope,
                intercept,
            )

        if (i + 1) % 5 == 0 or i == n_chains - 1:
            print(f"  Processed chain {i+1}/{n_chains}")

    print("Predicted SST range:")
    print(f"  {np.nanmin(pred_real):.2f} to {np.nanmax(pred_real):.2f} °C")

    # ── Create output datasets ───────────────────────────────────
    print("\n7. CREATING OUTPUT DATASETS")

    # Actual SST dataset — uses corrected lon
    ds_actual = xr.Dataset(
        {'sst_actual': (['time', 'lat', 'lon'], actual_real)},
        coords={
            'time': valid_unique,
            'lat':  lat,
            'lon':  lon,    # corrected lon
        }
    )
    ds_actual['sst_actual'].attrs = {'units': 'degC', 'long_name': 'Actual SST'}

    # Predicted SST dataset — uses corrected lon
    ds_pred = xr.Dataset(
        {'sst_predicted': (['chain', 'lead_step', 'lat', 'lon'], pred_real)},
        coords={
            'chain':      np.arange(n_chains),
            'lead_step':  np.arange(1, n_steps + 1),
            'lat':        lat,
            'lon':        lon,    # corrected lon
            'init_time':  (['chain'], ds.init_time.values),
            'valid_time': (['chain', 'lead_step'], ds.valid_time.values),
        }
    )
    ds_pred['sst_predicted'].attrs = {'units': 'degC', 'long_name': 'Predicted SST'}

    # ── Show real dates ──────────────────────────────────────────
    print("\nREAL DATE CHECK")
    print("\nFirst 5 actual dates:");    print(ds_actual.time.values[:5])
    print("\nFirst 5 init dates:");      print(ds_pred.init_time.values[:5])
    print("\nFirst 5 forecast valid dates:"); print(ds_pred.valid_time.values[0, :5])

    # ── Save outputs ─────────────────────────────────────────────
    print("\n8. SAVING OUTPUTS")

    actual_path = os.path.join(
        conf.output_dir,
        conf.actual_file_template.format(split=split),
    )
    pred_path = os.path.join(
        conf.output_dir,
        conf.predicted_file_template.format(split=split),
    )

    # xarray handles datetime encoding automatically — do NOT manually specify
    # time encoding or datetime64 coordinates may be written incorrectly.
    ds_actual.to_netcdf(actual_path)
    ds_pred.to_netcdf(pred_path)

    print(f"\nSaved actual SST:    {actual_path}")
    print(f"Saved predicted SST: {pred_path}")
    print(f"\n  {split.upper()} COMPLETE")


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    conf = cfg.concat_chains

    # Splits to process come from cfg.concat_chains.splits — edit config.py
    # to run only one split (e.g. splits=["test"]) without touching this file.
    for split in conf.splits:

        try:
            process_predictions(split)

        except FileNotFoundError as e:
            print(f"\nCould not process {split}:")
            print(e)

    print("\n" + "=" * 60)
    print("READY FOR DHW CALCULATION")
    print("=" * 60)

    print("\nActual SST:")
    print("  - Continuous timeline")
    print("  - Duplicate timestamps removed")
    print("  - Real datetime coordinates")
    print("  - Corrected lon (0-357.19, T42 global grid)")

    print("\nPredicted SST:")
    print("  - Independent forecast chains")
    print("  - Overlapping forecasts preserved")
    print("  - Real datetime coordinates")
    print("  - Corrected lon (0-357.19, T42 global grid)")