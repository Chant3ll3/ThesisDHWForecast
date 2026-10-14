"""
data_preparation.py — Phase 1: Data Preparation.

Pipeline:
  1. Load yearly OISST NetCDF files and concatenate along time.
  2. Select the SST variable and squeeze the singleton zlev dimension.
  3. Downscale spatially FIRST (e.g. 1440×720 → 128×64) to reduce
     computation for all subsequent steps (~125× fewer grid points).
  4. Remove a global linear trend (single slope applied uniformly)
     to preserve spatial relationships.
  5. Standardise globally (one mean, one std across all valid values).
  6. Fill remaining missing values with zero.
  7. Optionally subset to a geographic region.
  8. Save processed data and fitted parameters to NetCDF.
  9. Generate a diagnostic PDF with 12 mid-month snapshots.
"""

import os
import logging
import numpy as np
import xarray as xr
import dask
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from config import cfg

# ── Suppress benign HDF5 "file not found" diagnostics ───────────────
# netCDF4/HDF5 checks H5Fis_accessible() before creating new files,
# which dumps a noisy error trace when the file doesn't exist yet.
import ctypes
try:
    libhdf5 = ctypes.CDLL("libhdf5.so")
    libhdf5.H5Eset_auto2(ctypes.c_int64(0), None, None)
except (OSError, AttributeError):
    pass  # not critical — just cosmetic

# ── Logging setup ────────────────────────────────────────────────────
logger = logging.getLogger("Phase1_DataPrep")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(handler)


# ── Analytics helper (batched for performance) ───────────────────────

def log_analytics(da: xr.DataArray, label: str):
    """
    Print summary statistics for a DataArray to verify each stage.

    Uses a single dask.compute() call to batch all statistics at once,
    avoiding multiple full-dataset reads.
    """
    logger.info(f"--- Analytics: {label} ---")
    logger.info(f"  Shape        : {dict(da.sizes)}")

    total_cells = int(np.prod(list(da.sizes.values())))

    # Batch all reductions into a single compute call
    logger.info(f"  Computing statistics (batched) …")
    nan_count_lazy = da.isnull().sum()
    min_lazy = da.min()
    max_lazy = da.max()
    mean_lazy = da.mean()
    std_lazy = da.std()

    results = dask.compute(nan_count_lazy, min_lazy, max_lazy,
                           mean_lazy, std_lazy)
    nan_count = int(results[0])
    vmin = float(results[1])
    vmax = float(results[2])
    vmean = float(results[3])
    vstd = float(results[4])

    valid_count = total_cells - nan_count
    nan_pct = 100.0 * nan_count / total_cells if total_cells > 0 else 0.0

    logger.info(f"  Total cells  : {total_cells:,}")
    logger.info(f"  Valid cells  : {valid_count:,}")
    logger.info(f"  NaN cells    : {nan_count:,}  ({nan_pct:.2f}%)")
    logger.info(f"  Min          : {vmin:.6f}")
    logger.info(f"  Max          : {vmax:.6f}")
    logger.info(f"  Mean         : {vmean:.6f}")
    logger.info(f"  Std          : {vstd:.6f}")
    logger.info(f"--- End analytics: {label} ---")


# ── Step 1: Load data ────────────────────────────────────────────────

def load_sst() -> xr.DataArray:
    """Open all yearly files lazily with dask and concatenate."""
    conf = cfg.data_preparation
    year_start, year_end = conf.year_range

    file_paths = []
    for year in range(year_start, year_end + 1):
        fp = os.path.join(conf.input_dir, conf.file_pattern.format(year=year))
        if os.path.exists(fp):
            file_paths.append(fp)
        else:
            logger.warning(f"File not found, skipping: {fp}")

    logger.info(f"Loading {len(file_paths)} files ({year_start}–{year_end})")

    ds = xr.open_mfdataset(
        file_paths,
        combine="by_coords",
        chunks=conf.chunks,
        parallel=True,
    )

    # Select SST and drop the singleton zlev dimension
    sst = ds[conf.variable].squeeze("zlev", drop=True)
    logger.info(f"SST loaded — shape: {dict(sst.sizes)}")
    logger.info(f"Time range: {str(sst.time.values[0])[:10]} to "
                f"{str(sst.time.values[-1])[:10]}")

    log_analytics(sst, "Raw SST")
    return sst


# ── Step 2: Global linear detrending ────────────────────────────────

def compute_area_weights(lat: xr.DataArray) -> xr.DataArray:
    """Cosine-of-latitude weights for area-weighted global mean."""
    weights = np.cos(np.deg2rad(lat))
    weights = weights / weights.sum()
    return weights


def detrend_global_linear(sst: xr.DataArray):
    """
    Fit a single linear trend to the area-weighted global-mean SST
    time series, then subtract that trend from every grid point.
    """
    conf = cfg.data_preparation
    logger.info("Computing area-weighted global-mean SST time series")

    weights = compute_area_weights(sst.lat)
    global_mean = sst.weighted(weights).mean(dim=["lat", "lon"])

    # Convert time to numeric (days since first time step)
    time_numeric = (sst.time - sst.time[0]).dt.total_seconds() / 86400.0
    t = time_numeric.values.astype(np.float64)

    logger.info("Loading global mean time series into memory for polyfit")
    gm = global_mean.values.astype(np.float64)

    # Mask out NaN time steps
    valid = np.isfinite(gm)
    slope, intercept = np.polyfit(t[valid], gm[valid], deg=1)
    logger.info(f"Global trend: slope = {slope:.6e} °C/day, "
                f"intercept = {intercept:.4f} °C")
    logger.info(f"  Over full record ({t[valid][-1]:.0f} days): "
                f"total trend = {slope * t[valid][-1]:.4f} °C")

    # Build the trend line and subtract
    trend_line = xr.DataArray(
        slope * time_numeric + intercept,
        dims=["time"],
        coords={"time": sst.time},
    )
    sst_detrended = sst - trend_line
    logger.info("Global linear trend removed from all grid points")

    # Save trend parameters
    params = xr.Dataset({
        "slope": xr.DataArray(slope),
        "intercept": xr.DataArray(intercept),
        "time_origin": sst.time[0],
    })
    params_path = os.path.join(conf.output_dir, conf.trend_params_file)
    params.to_netcdf(params_path, mode="w")
    logger.info(f"Trend parameters saved to {params_path}")

    log_analytics(sst_detrended, "Detrended SST")

    # Verification: check global mean trend is ~0
    logger.info("VERIFICATION: computing residual slope after detrending …")
    gm_detrended = sst_detrended.weighted(weights).mean(dim=["lat", "lon"])
    gm_dt_vals = gm_detrended.values.astype(np.float64)
    valid_dt = np.isfinite(gm_dt_vals)
    residual_slope, _ = np.polyfit(t[valid_dt], gm_dt_vals[valid_dt], deg=1)
    logger.info(f"VERIFICATION: residual global-mean slope = "
                f"{residual_slope:.2e} °C/day (should be ≈ 0)")

    return sst_detrended, slope, intercept


# ── Step 3: Global standardisation ──────────────────────────────────

def standardise_global(sst: xr.DataArray):
    """
    Compute a single mean and std over all valid (time, lat, lon) values
    and apply (x − mean) / std uniformly.
    """
    conf = cfg.data_preparation
    logger.info("Computing global mean and std for standardisation (batched)")

    # Batch the two reductions
    mean_lazy = sst.mean()
    std_lazy = sst.std()
    global_mean, global_std = dask.compute(mean_lazy, std_lazy)
    global_mean = float(global_mean)
    global_std = float(global_std)
    logger.info(f"Global mean = {global_mean:.6f}, std = {global_std:.6f}")

    sst_standardised = (sst - global_mean) / global_std

    # Save standardisation parameters
    params = xr.Dataset({
        "global_mean": xr.DataArray(global_mean),
        "global_std": xr.DataArray(global_std),
    })
    params_path = os.path.join(conf.output_dir, conf.standardise_params_file)
    params.to_netcdf(params_path, mode="w")
    logger.info(f"Standardisation parameters saved to {params_path}")

    log_analytics(sst_standardised, "Standardised SST")

    # Verification
    logger.info("VERIFICATION: computing post-standardisation mean & std (batched)")
    post_mean, post_std = dask.compute(
        sst_standardised.mean(), sst_standardised.std()
    )
    logger.info(f"VERIFICATION: post-standardisation mean = {float(post_mean):.6e} "
                f"(should be ≈ 0), std = {float(post_std):.6f} (should be ≈ 1)")

    return sst_standardised, global_mean, global_std


# ── Step 4: Fill land / missing values ───────────────────────────────

def _diffusion_fill_2d(field: np.ndarray, mask: np.ndarray,
                       n_iter: int) -> np.ndarray:
    """
    Fill NaN (land) regions of a 2-D field by iterative Laplacian
    relaxation (diffusion).  Ocean values are held fixed; land values
    are iteratively replaced by the mean of their neighbours.

    Parameters
    ----------
    field  : (lat, lon) — values to fill.  NaN on land.
    mask   : (lat, lon) — boolean, True = land (to be filled).
    n_iter : number of relaxation sweeps.

    Returns
    -------
    filled : (lat, lon) — no NaN remaining.
    """
    filled = field.copy()
    filled[mask] = 0.0  # seed land with zero for averaging

    for _ in range(n_iter):
        # 4-neighbour average (wraps in longitude, clamps in latitude)
        avg = np.zeros_like(filled)
        count = np.zeros_like(filled)

        # Shift up (lat-1)
        avg[1:, :] += filled[:-1, :]
        count[1:, :] += 1.0
        # Shift down (lat+1)
        avg[:-1, :] += filled[1:, :]
        count[:-1, :] += 1.0
        # Shift left (lon-1), wraps
        avg[:, 1:] += filled[:, :-1]
        count[:, 1:] += 1.0
        avg[:, 0] += filled[:, -1]
        count[:, 0] += 1.0
        # Shift right (lon+1), wraps
        avg[:, :-1] += filled[:, 1:]
        count[:, :-1] += 1.0
        avg[:, -1] += filled[:, 0]
        count[:, -1] += 1.0

        # Only update land points
        update = np.where(count > 0, avg / count, 0.0)
        filled[mask] = update[mask]

    return filled


def _nearest_fill_2d(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill NaN regions using nearest valid neighbour (scipy)."""
    from scipy.ndimage import distance_transform_edt

    filled = field.copy()
    # distance_transform_edt with return_indices gives nearest valid pixel
    _, indices = distance_transform_edt(mask, return_indices=True)
    filled[mask] = field[indices[0, mask], indices[1, mask]]
    return filled


def fill_missing(sst: xr.DataArray) -> xr.DataArray:
    """
    Fill land / missing values to create a complete field for the SFNO.

    Uses the method specified in config (diffusion, nearest, or zero).
    Operates per time step on in-memory numpy arrays.
    """
    conf = cfg.data_preparation
    method = conf.fill_method

    logger.info(f"Filling land/missing values — method: '{method}'")

    # Load into memory for per-timestep processing
    logger.info("  Loading data into memory for land fill …")
    sst_np = sst.values  # (time, lat, lon)
    n_time = sst_np.shape[0]

    # Build a static land mask from the first valid time step:
    # land = points that are NaN across all time steps.
    # This avoids filling occasional missing ocean pixels differently.
    logger.info("  Computing land mask …")
    all_nan = np.all(np.isnan(sst_np), axis=0)  # (lat, lon)
    n_land = int(all_nan.sum())
    n_total = all_nan.size
    logger.info(f"  Land mask: {n_land:,} land points / {n_total:,} total "
                f"({100.0 * n_land / n_total:.1f}%)")

    nan_before = int(np.isnan(sst_np).sum())
    logger.info(f"  Total NaN cells before fill: {nan_before:,}")

    if method == "diffusion":
        n_iter = conf.fill_diffusion_iterations
        logger.info(f"  Running diffusion fill ({n_iter} iterations per time step) …")
        for t in range(n_time):
            frame = sst_np[t]
            frame_mask = np.isnan(frame)
            if frame_mask.any():
                sst_np[t] = _diffusion_fill_2d(frame, frame_mask, n_iter)
            if (t + 1) % 1000 == 0 or t == n_time - 1:
                logger.info(f"    Filled {t + 1:>6d}/{n_time} time steps")

    elif method == "nearest":
        logger.info("  Running nearest-neighbour fill …")
        for t in range(n_time):
            frame = sst_np[t]
            frame_mask = np.isnan(frame)
            if frame_mask.any():
                sst_np[t] = _nearest_fill_2d(frame, frame_mask)
            if (t + 1) % 1000 == 0 or t == n_time - 1:
                logger.info(f"    Filled {t + 1:>6d}/{n_time} time steps")

    elif method == "zero":
        logger.info("  Zero fill")
        sst_np = np.nan_to_num(sst_np, nan=conf.fill_fallback_value)

    else:
        raise ValueError(f"Unknown fill_method: {method}")

    # Catch any remaining NaN (e.g. fully isolated land pixels)
    remaining_nan = int(np.isnan(sst_np).sum())
    if remaining_nan > 0:
        logger.warning(f"  {remaining_nan:,} NaN remaining after {method} — "
                       f"applying fallback fill ({conf.fill_fallback_value})")
        sst_np = np.nan_to_num(sst_np, nan=conf.fill_fallback_value)

    nan_after = int(np.isnan(sst_np).sum())
    logger.info(f"  Total NaN cells after fill: {nan_after:,} (should be 0)")

    # Wrap back into xarray
    sst_filled = xr.DataArray(
        sst_np,
        dims=sst.dims,
        coords=sst.coords,
        attrs=sst.attrs,
    )

    log_analytics(sst_filled, "After land fill")
    return sst_filled


# ── Step 5: Spatial downscaling ──────────────────────────────────────

def downscale(sst: xr.DataArray) -> xr.DataArray:
    """Coarsen/interpolate to target lon × lat grid."""
    conf = cfg.data_preparation
    if not conf.downscale:
        logger.info("Downscaling: DISABLED (keeping original resolution)")
        return sst

    target_lon = conf.target_lon_size
    target_lat = conf.target_lat_size
    orig_lon = sst.sizes["lon"]
    orig_lat = sst.sizes["lat"]

    logger.info(f"Downscaling: ({orig_lon} lon × {orig_lat} lat) → "
                f"({target_lon} lon × {target_lat} lat)")

    # Build new regular grids
    new_lat = np.linspace(float(sst.lat.min()), float(sst.lat.max()), target_lat)
    new_lon = np.linspace(float(sst.lon.min()), float(sst.lon.max()), target_lon)

    sst_downscaled = sst.interp(lat=new_lat, lon=new_lon, method="linear")
    logger.info(f"Downscaling complete — new shape: {dict(sst_downscaled.sizes)}")

    log_analytics(sst_downscaled, "Downscaled SST")
    return sst_downscaled


# ── Step 6: Region subsetting ────────────────────────────────────────

def subset_region(sst: xr.DataArray) -> xr.DataArray:
    """Slice to a lat/lon bounding box if configured."""
    conf = cfg.data_preparation
    if not conf.subset_region:
        logger.info("Region subsetting: DISABLED")
        return sst

    lat_min, lat_max = conf.lat_bounds
    lon_min, lon_max = conf.lon_bounds
    logger.info(f"Subsetting to lat=[{lat_min}, {lat_max}], lon=[{lon_min}, {lon_max}]")

    sst = sst.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    logger.info(f"Subset shape: {dict(sst.sizes)}")
    log_analytics(sst, "Region subset SST")
    return sst


# ── Step 7: Save to NetCDF ──────────────────────────────────────────

def save_netcdf(da: xr.DataArray, filename: str, description: str):
    """Write a DataArray to a compressed NetCDF-4 file."""
    conf = cfg.data_preparation
    out_path = os.path.join(conf.output_dir, filename)

    ds = da.to_dataset(name=conf.variable)
    ds.attrs["description"] = description
    ds.attrs["project"] = cfg.project_name

    encoding = {
        conf.variable: {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
        }
    }

    logger.info(f"Writing {out_path} …")
    ds.to_netcdf(out_path, encoding=encoding, mode="w")
    logger.info(f"Saved: {out_path}")


# ── Step 8: Diagnostic PDF ──────────────────────────────────────────

def _select_mid_month_snapshots(da: xr.DataArray, year: int):
    """
    Select 12 time slices, one near the 15th of each month for the
    given year.  Returns list of (month_label, 2-D DataArray) tuples.
    """
    snapshots = []
    for month in range(1, 13):
        target = f"{year}-{month:02d}-15"
        try:
            snap = da.sel(time=target, method="nearest")
            actual_date = str(snap.time.values)[:10]
        except Exception:
            logger.warning(f"  Could not find snapshot near {target}, skipping")
            continue
        label = f"{actual_date}"
        snapshots.append((label, snap))
    return snapshots


def generate_diagnostic_pdf(stages: dict):
    """
    Generate a multi-page PDF.  One page per transformation stage,
    each page showing 12 mid-month subplots (4 rows × 3 cols).

    Parameters
    ----------
    stages : dict[str, xr.DataArray]
        Ordered dict like {"1. Raw SST": da, "2. Detrended": da, ...}
    """
    conf = cfg.data_preparation
    year = conf.diagnostic_year

    pdf_path = os.path.join(conf.output_dir, conf.diagnostic_pdf)
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    logger.info(f"Generating diagnostic PDF for year {year}")

    with PdfPages(pdf_path) as pdf:
        for stage_name, da in stages.items():
            logger.info(f"  Plotting stage: {stage_name}")
            snapshots = _select_mid_month_snapshots(da, year)

            if not snapshots:
                logger.warning(f"  No snapshots found for {stage_name}, skipping page")
                continue

            fig, axes = plt.subplots(4, 3, figsize=(18, 22))
            fig.suptitle(stage_name, fontsize=18, fontweight="bold", y=0.98)

            for idx, (label, snap) in enumerate(snapshots):
                ax = axes.flat[idx]
                # Load single snapshot into memory (small: lat × lon)
                data = snap.values
                if data.ndim > 2:
                    data = data.squeeze()

                lat_vals = snap.lat.values
                lon_vals = snap.lon.values

                # Choose symmetric colour range for standardised data
                if "Standardised" in stage_name or "Final" in stage_name:
                    vabs = min(max(abs(np.nanmin(data)), abs(np.nanmax(data))), 5.0)
                    im = ax.pcolormesh(
                        lon_vals, lat_vals, data,
                        cmap="RdBu_r", vmin=-vabs, vmax=vabs, shading="auto",
                    )
                else:
                    im = ax.pcolormesh(
                        lon_vals, lat_vals, data,
                        cmap="RdYlBu_r", shading="auto",
                    )

                ax.set_title(label, fontsize=10)
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Hide unused axes
            for idx in range(len(snapshots), 12):
                axes.flat[idx].set_visible(False)

            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

        # ── Summary stats page ───────────────────────────────────
        logger.info("  Generating summary statistics page")
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis("off")
        summary_lines = ["DATA PREPARATION — SUMMARY STATISTICS", "=" * 55, ""]
        for stage_name, da in stages.items():
            # Batch compute for summary
            r = dask.compute(da.mean(), da.std(), da.min(), da.max(),
                             da.isnull().sum())
            vmean, vstd, vmin, vmax, nan_count = [float(x) for x in r]
            nan_count = int(nan_count)
            summary_lines.append(f"{stage_name}")
            summary_lines.append(f"  Shape : {dict(da.sizes)}")
            summary_lines.append(f"  Mean  : {vmean:>12.6f}    Std : {vstd:>12.6f}")
            summary_lines.append(f"  Min   : {vmin:>12.6f}    Max : {vmax:>12.6f}")
            summary_lines.append(f"  NaNs  : {nan_count:>12,}")
            summary_lines.append("")

        ax.text(0.05, 0.95, "\n".join(summary_lines),
                transform=ax.transAxes, fontsize=10,
                verticalalignment="top", fontfamily="monospace")
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    logger.info(f"Diagnostic PDF saved to {pdf_path}")
    return pdf_path


# ── Resume helper ────────────────────────────────────────────────────

def _load_intermediate(filename: str, label: str) -> xr.DataArray:
    """Load a previously saved intermediate NetCDF file."""
    conf = cfg.data_preparation
    path = os.path.join(conf.output_dir, filename)
    logger.info(f"RESUME: loading {label} from {path}")
    # Intermediates are on the downscaled grid — use simple chunking
    ds = xr.open_dataset(path, chunks={"time": 366})
    da = ds[conf.variable]
    logger.info(f"  → Shape: {dict(da.sizes)}")
    return da


# Stage numbers for reference:
#   1 = load
#   2 = downscale
#   3 = detrend
#   4 = standardise
#   5 = fill missing

STAGE_NAMES = {
    0: "start from scratch",
    1: "after load",
    2: "after downscale",
    3: "after detrend",
    4: "after standardise",
    5: "after fill",
}


# ── Pipeline ─────────────────────────────────────────────────────────

def run():
    """Execute the full data-preparation pipeline with resume support."""
    print()
    logger.info("=" * 60)
    logger.info("PHASE 1: DATA PREPARATION — START")
    logger.info("=" * 60)

    conf = cfg.data_preparation
    resume = conf.resume_after_stage

    if resume > 0:
        logger.info(f"RESUME MODE: skipping to {STAGE_NAMES.get(resume, '?')} "
                    f"(resume_after_stage={resume})")

    # ── Configure dask scheduler ─────────────────────────────────
    client = None
    cluster = None

    if conf.dask_scheduler == "distributed":
        from dask.distributed import Client, LocalCluster

        cluster = LocalCluster(
            n_workers=conf.dask_n_workers,
            threads_per_worker=conf.dask_threads_per_worker,
            memory_limit=conf.dask_memory_limit,
        )
        client = Client(cluster)
        logger.info(f"Dask distributed cluster: {conf.dask_n_workers} workers × "
                    f"{conf.dask_threads_per_worker} threads")
        logger.info(f"  Memory per worker: {conf.dask_memory_limit}")
        logger.info(f"  Dashboard: {client.dashboard_link}")

    elif conf.dask_scheduler == "threads":
        dask.config.set(scheduler="threads", num_workers=conf.dask_num_threads)
        logger.info(f"Dask threaded scheduler: {conf.dask_num_threads} threads")

    elif conf.dask_scheduler == "synchronous":
        dask.config.set(scheduler="synchronous")
        logger.info("Dask synchronous scheduler (single-threaded, for debugging)")

    else:
        raise ValueError(f"Unknown dask_scheduler: {conf.dask_scheduler}")

    # Create all output directories upfront
    os.makedirs(conf.output_dir, exist_ok=True)
    diag_dir = os.path.dirname(os.path.join(conf.output_dir, conf.diagnostic_pdf))
    os.makedirs(diag_dir, exist_ok=True)
    logger.info(f"Output directory: {os.path.abspath(conf.output_dir)}")

    # Collect intermediate stages for diagnostics
    stages = {}

    # ── Resume mapping: which file to reload at each resume point ─
    # When resuming, we only need the output of the last completed
    # stage — not every earlier intermediate.
    RESUME_FILE_MAP = {
        2: (conf.downscaled_file,   "downscaled SST"),
        3: (conf.detrended_file,    "detrended SST"),
        4: (conf.standardised_file, "standardised SST"),
        5: (conf.filled_file,       "filled SST"),
    }

    # ── Stage 1: Load ────────────────────────────────────────────
    if resume < 1:
        sst_raw = load_sst()
        stages["1. Raw SST (°C)"] = sst_raw
    else:
        logger.info("SKIP: Stage 1 (load) — already completed")
        sst_raw = None

    # ── Stage 2: Downscale ───────────────────────────────────────
    if resume < 2:
        sst_small = downscale(sst_raw)
        if conf.save_intermediates:
            save_netcdf(sst_small, conf.downscaled_file, "SST downscaled to target grid")
        label = "2. Downscaled SST (°C)" if conf.downscale else "2. SST (original res)"
        stages[label] = sst_small
    elif resume == 2:
        fname, label = RESUME_FILE_MAP[2]
        sst_small = _load_intermediate(fname, label)
        stages["2. Downscaled SST (°C)"] = sst_small
    else:
        logger.info("SKIP: Stage 2 (downscale) — already completed")
        sst_small = None

    # ── Stage 3: Detrend ─────────────────────────────────────────
    if resume < 3:
        sst_detrended, slope, intercept = detrend_global_linear(sst_small)
        if conf.save_intermediates:
            save_netcdf(sst_detrended, conf.detrended_file,
                        "SST downscaled with global linear trend removed")
        stages["3. Detrended SST (°C)"] = sst_detrended
    elif resume == 3:
        fname, label = RESUME_FILE_MAP[3]
        sst_detrended = _load_intermediate(fname, label)
        stages["3. Detrended SST (°C)"] = sst_detrended
    else:
        logger.info("SKIP: Stage 3 (detrend) — already completed")
        sst_detrended = None

    # ── Stage 4: Standardise ─────────────────────────────────────
    if resume < 4:
        sst_standardised, gmean, gstd = standardise_global(sst_detrended)
        if conf.save_intermediates:
            save_netcdf(sst_standardised, conf.standardised_file,
                        "SST downscaled, detrended and globally standardised")
        stages["4. Standardised SST"] = sst_standardised
    elif resume == 4:
        fname, label = RESUME_FILE_MAP[4]
        sst_standardised = _load_intermediate(fname, label)
        stages["4. Standardised SST"] = sst_standardised
    else:
        logger.info("SKIP: Stage 4 (standardise) — already completed")
        sst_standardised = None

    # ── Stage 5: Fill missing values ─────────────────────────────
    if resume < 5:
        sst_filled = fill_missing(sst_standardised)
        if conf.save_intermediates:
            save_netcdf(sst_filled, conf.filled_file,
                        "SST downscaled, detrended, standardised, and filled")
        stages["5. Final (land filled)"] = sst_filled
    elif resume == 5:
        fname, label = RESUME_FILE_MAP[5]
        sst_filled = _load_intermediate(fname, label)
        stages["5. Final (land filled)"] = sst_filled

    # ── Stage 6: Region subset (optional) ────────────────────────
    if conf.subset_region:
        sst_subset = subset_region(sst_filled)
        save_netcdf(sst_subset, conf.subset_file,
                    "SST fully processed and subsetted to region")
        stages["6. Region subset"] = sst_subset

    # ── Save final processed file ────────────────────────────────
    logger.info("Saving final processed dataset …")
    save_netcdf(sst_filled, conf.final_file,
                "SST downscaled, detrended, standardised, and filled")

    # ── Diagnostic PDF ───────────────────────────────────────────
    logger.info("Generating diagnostic plots …")
    pdf_path = generate_diagnostic_pdf(stages)

    # ── Shutdown dask ───────────────────────────────────────────
    if client is not None:
        logger.info("Shutting down dask distributed cluster")
        client.close()
        cluster.close()

    logger.info("=" * 60)
    logger.info("PHASE 1: DATA PREPARATION — COMPLETE")
    logger.info("=" * 60)
    print()

    return pdf_path


if __name__ == "__main__":
    run()
