"""
data_preparation.py — Phase 1: Data Preparation.

Pipeline:
  1. Load yearly OISST NetCDF files and concatenate along time.
  2. Downscale spatially FIRST (e.g. 1440×720 → 128×64).
  3. Optionally resample to weekly or monthly means.
  4. Remove a global linear trend (preserves spatial relationships).
  5. Standardise globally (one mean, one std).
  6. Fill land/missing values (diffusion, nearest, or zero).
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
import ctypes
try:
    libhdf5 = ctypes.CDLL("libhdf5.so")
    libhdf5.H5Eset_auto2(ctypes.c_int64(0), None, None)
except (OSError, AttributeError):
    pass

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


# ── Analytics helper ─────────────────────────────────────────────────

def log_analytics(da: xr.DataArray, label: str):
    logger.info(f"--- Analytics: {label} ---")
    logger.info(f"  Shape        : {dict(da.sizes)}")
    total_cells = int(np.prod(list(da.sizes.values())))
    logger.info(f"  Computing statistics (batched) …")
    nan_count_lazy = da.isnull().sum()
    min_lazy = da.min(); max_lazy = da.max()
    mean_lazy = da.mean(); std_lazy = da.std()
    results = dask.compute(nan_count_lazy, min_lazy, max_lazy, mean_lazy, std_lazy)
    nan_count, vmin, vmax, vmean, vstd = [float(x) for x in results]
    nan_count = int(nan_count)
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
    ds = xr.open_mfdataset(file_paths, combine="by_coords",
                           chunks=conf.chunks, parallel=False)
    sst = ds[conf.variable].squeeze("zlev", drop=True)
    logger.info(f"SST loaded — shape: {dict(sst.sizes)}")
    logger.info(f"Time range: {str(sst.time.values[0])[:10]} to "
                f"{str(sst.time.values[-1])[:10]}")
    log_analytics(sst, "Raw SST")
    return sst


# ── Step 2: Spatial downscaling ──────────────────────────────────────

def downscale(sst: xr.DataArray) -> xr.DataArray:
    conf = cfg.data_preparation
    if not conf.downscale:
        logger.info("Downscaling: DISABLED")
        return sst
    target_lon = conf.target_lon_size
    target_lat = conf.target_lat_size
    logger.info(f"Downscaling: ({sst.sizes['lon']} × {sst.sizes['lat']}) → "
                f"({target_lon} × {target_lat})")
    new_lat = np.linspace(float(sst.lat.min()), float(sst.lat.max()), target_lat)
    new_lon = np.linspace(float(sst.lon.min()), float(sst.lon.max()), target_lon)
    sst_ds = sst.interp(lat=new_lat, lon=new_lon, method="linear")
    logger.info(f"Downscaling complete — shape: {dict(sst_ds.sizes)}")
    log_analytics(sst_ds, "Downscaled SST")
    return sst_ds


# ── Step 3: Temporal averaging ───────────────────────────────────────

def temporal_average(sst: xr.DataArray) -> xr.DataArray:
    conf = cfg.data_preparation
    method = conf.temporal_average
    if method == "none":
        logger.info("Temporal averaging: DISABLED (keeping daily resolution)")
        return sst

    n_before = sst.sizes["time"]
    if method == "weekly":
        logger.info("Temporal averaging: daily → weekly means")
        sst_avg = sst.resample(time="7D").mean(skipna=True)
    elif method == "monthly":
        logger.info("Temporal averaging: daily → monthly means")
        sst_avg = sst.resample(time="MS").mean(skipna=True)
    else:
        raise ValueError(f"Unknown temporal_average: '{method}'")

    n_after = sst_avg.sizes["time"]
    logger.info(f"  Time steps: {n_before:,} → {n_after:,}")
    logger.info(f"  Time range: {str(sst_avg.time.values[0])[:10]} to "
                f"{str(sst_avg.time.values[-1])[:10]}")
    log_analytics(sst_avg, f"Temporally averaged SST ({method})")
    return sst_avg


# ── Step 4: Global linear detrending ────────────────────────────────

def compute_area_weights(lat: xr.DataArray) -> xr.DataArray:
    weights = np.cos(np.deg2rad(lat))
    return weights / weights.sum()


def detrend_global_linear(sst: xr.DataArray):
    conf = cfg.data_preparation
    logger.info("Computing area-weighted global-mean SST time series")
    weights = compute_area_weights(sst.lat)
    global_mean = sst.weighted(weights).mean(dim=["lat", "lon"])
    time_numeric = (sst.time - sst.time[0]).dt.total_seconds() / 86400.0
    t = time_numeric.values.astype(np.float64)
    logger.info("Loading global mean into memory for polyfit")
    gm = global_mean.values.astype(np.float64)
    valid = np.isfinite(gm)
    slope, intercept = np.polyfit(t[valid], gm[valid], deg=1)
    logger.info(f"Global trend: slope = {slope:.6e} °C/day, "
                f"intercept = {intercept:.4f} °C")
    logger.info(f"  Over full record ({t[valid][-1]:.0f} days): "
                f"total trend = {slope * t[valid][-1]:.4f} °C")

    trend_line = xr.DataArray(slope * time_numeric + intercept,
                              dims=["time"], coords={"time": sst.time})
    sst_detrended = sst - trend_line
    logger.info("Global linear trend removed")

    params = xr.Dataset({
        "slope": xr.DataArray(slope),
        "intercept": xr.DataArray(intercept),
        "time_origin": sst.time[0],
    })
    params_path = os.path.join(conf.output_dir, conf.trend_params_file)
    params.to_netcdf(params_path, mode="w")
    logger.info(f"Trend parameters saved to {params_path}")

    log_analytics(sst_detrended, "Detrended SST")

    logger.info("VERIFICATION: computing residual slope …")
    gm_dt = sst_detrended.weighted(weights).mean(dim=["lat", "lon"])
    gm_dt_vals = gm_dt.values.astype(np.float64)
    valid_dt = np.isfinite(gm_dt_vals)
    res_slope, _ = np.polyfit(t[valid_dt], gm_dt_vals[valid_dt], deg=1)
    logger.info(f"VERIFICATION: residual slope = {res_slope:.2e} (should be ≈ 0)")

    return sst_detrended, slope, intercept


# ── Step 5: Global standardisation ──────────────────────────────────

def standardise_global(sst: xr.DataArray):
    conf = cfg.data_preparation
    logger.info("Computing global mean and std (batched)")
    mean_lazy = sst.mean(); std_lazy = sst.std()
    global_mean, global_std = dask.compute(mean_lazy, std_lazy)
    global_mean = float(global_mean); global_std = float(global_std)
    logger.info(f"Global mean = {global_mean:.6f}, std = {global_std:.6f}")

    sst_std = (sst - global_mean) / global_std

    params = xr.Dataset({
        "global_mean": xr.DataArray(global_mean),
        "global_std": xr.DataArray(global_std),
    })
    params_path = os.path.join(conf.output_dir, conf.standardise_params_file)
    params.to_netcdf(params_path, mode="w")
    logger.info(f"Standardisation params saved to {params_path}")

    log_analytics(sst_std, "Standardised SST")

    logger.info("VERIFICATION: post-standardisation stats (batched)")
    pm, ps = dask.compute(sst_std.mean(), sst_std.std())
    logger.info(f"VERIFICATION: mean = {float(pm):.6e} (≈0), "
                f"std = {float(ps):.6f} (≈1)")

    return sst_std, global_mean, global_std


# ── Step 6: Fill land / missing values ───────────────────────────────

def _diffusion_fill_2d(field, mask, n_iter):
    filled = field.copy()
    filled[mask] = 0.0
    for _ in range(n_iter):
        avg = np.zeros_like(filled); count = np.zeros_like(filled)
        avg[1:, :] += filled[:-1, :]; count[1:, :] += 1.0
        avg[:-1, :] += filled[1:, :]; count[:-1, :] += 1.0
        avg[:, 1:] += filled[:, :-1]; count[:, 1:] += 1.0
        avg[:, 0] += filled[:, -1]; count[:, 0] += 1.0
        avg[:, :-1] += filled[:, 1:]; count[:, :-1] += 1.0
        avg[:, -1] += filled[:, 0]; count[:, -1] += 1.0
        update = np.where(count > 0, avg / count, 0.0)
        filled[mask] = update[mask]
    return filled


def _nearest_fill_2d(field, mask):
    from scipy.ndimage import distance_transform_edt
    filled = field.copy()
    _, indices = distance_transform_edt(mask, return_indices=True)
    filled[mask] = field[indices[0, mask], indices[1, mask]]
    return filled


def fill_missing(sst: xr.DataArray) -> xr.DataArray:
    conf = cfg.data_preparation
    method = conf.fill_method
    logger.info(f"Filling land/missing values — method: '{method}'")
    logger.info("  Loading data into memory …")
    sst_np = sst.values
    n_time = sst_np.shape[0]

    logger.info("  Computing land mask …")
    all_nan = np.all(np.isnan(sst_np), axis=0)
    n_land = int(all_nan.sum()); n_total = all_nan.size
    logger.info(f"  Land mask: {n_land:,} / {n_total:,} ({100.*n_land/n_total:.1f}%)")
    nan_before = int(np.isnan(sst_np).sum())
    logger.info(f"  NaN before fill: {nan_before:,}")

    if method == "diffusion":
        n_iter = conf.fill_diffusion_iterations
        logger.info(f"  Diffusion fill ({n_iter} iterations/timestep) …")
        for t in range(n_time):
            fm = np.isnan(sst_np[t])
            if fm.any():
                sst_np[t] = _diffusion_fill_2d(sst_np[t], fm, n_iter)
            if (t+1) % 1000 == 0 or t == n_time - 1:
                logger.info(f"    {t+1:>6d}/{n_time}")
    elif method == "nearest":
        logger.info("  Nearest-neighbour fill …")
        for t in range(n_time):
            fm = np.isnan(sst_np[t])
            if fm.any():
                sst_np[t] = _nearest_fill_2d(sst_np[t], fm)
            if (t+1) % 1000 == 0 or t == n_time - 1:
                logger.info(f"    {t+1:>6d}/{n_time}")
    elif method == "zero":
        sst_np = np.nan_to_num(sst_np, nan=conf.fill_fallback_value)
    else:
        raise ValueError(f"Unknown fill_method: {method}")

    remaining = int(np.isnan(sst_np).sum())
    if remaining > 0:
        logger.warning(f"  {remaining:,} NaN remaining — fallback fill")
        sst_np = np.nan_to_num(sst_np, nan=conf.fill_fallback_value)

    logger.info(f"  NaN after fill: {int(np.isnan(sst_np).sum()):,}")

    sst_filled = xr.DataArray(sst_np, dims=sst.dims, coords=sst.coords,
                              attrs=sst.attrs)
    log_analytics(sst_filled, "After land fill")
    return sst_filled


# ── Step 7: Region subsetting ────────────────────────────────────────

def subset_region(sst: xr.DataArray) -> xr.DataArray:
    conf = cfg.data_preparation
    if not conf.subset_region:
        logger.info("Region subsetting: DISABLED")
        return sst
    lat_min, lat_max = conf.lat_bounds
    lon_min, lon_max = conf.lon_bounds
    logger.info(f"Subsetting: lat=[{lat_min},{lat_max}], lon=[{lon_min},{lon_max}]")
    sst = sst.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    logger.info(f"Subset shape: {dict(sst.sizes)}")
    log_analytics(sst, "Region subset SST")
    return sst


# ── Save to NetCDF ───────────────────────────────────────────────────

def save_netcdf(da: xr.DataArray, filename: str, description: str):
    conf = cfg.data_preparation
    out_path = os.path.join(conf.output_dir, filename)
    ds = da.to_dataset(name=conf.variable)
    ds.attrs["description"] = description
    ds.attrs["project"] = cfg.project_name
    encoding = {conf.variable: {"zlib": True, "complevel": 4, "dtype": "float32"}}
    logger.info(f"Writing {out_path} …")
    ds.to_netcdf(out_path, encoding=encoding, mode="w")
    logger.info(f"Saved: {out_path}")


# ── Diagnostic PDF ───────────────────────────────────────────────────

def _select_mid_month_snapshots(da, year):
    snapshots = []
    for month in range(1, 13):
        target = f"{year}-{month:02d}-15"
        try:
            snap = da.sel(time=target, method="nearest")
            actual_date = str(snap.time.values)[:10]
        except Exception:
            continue
        snapshots.append((actual_date, snap))
    return snapshots


def generate_diagnostic_pdf(stages):
    conf = cfg.data_preparation
    year = conf.diagnostic_year
    pdf_path = os.path.join(conf.output_dir, conf.diagnostic_pdf)
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    logger.info(f"Generating diagnostic PDF for year {year}")

    with PdfPages(pdf_path) as pdf:
        for stage_name, da in stages.items():
            logger.info(f"  Plotting: {stage_name}")
            snapshots = _select_mid_month_snapshots(da, year)
            if not snapshots:
                continue

            fig, axes = plt.subplots(4, 3, figsize=(18, 22))
            fig.suptitle(stage_name, fontsize=18, fontweight="bold", y=0.98)

            for idx, (label, snap) in enumerate(snapshots):
                ax = axes.flat[idx]
                data = snap.values
                if data.ndim > 2: data = data.squeeze()
                if "Standardised" in stage_name or "Final" in stage_name or "filled" in stage_name.lower():
                    vabs = min(max(abs(np.nanmin(data)), abs(np.nanmax(data))), 5.0)
                    im = ax.pcolormesh(snap.lon.values, snap.lat.values, data,
                                       cmap="RdBu_r", vmin=-vabs, vmax=vabs, shading="auto")
                else:
                    im = ax.pcolormesh(snap.lon.values, snap.lat.values, data,
                                       cmap="RdYlBu_r", shading="auto")
                ax.set_title(label, fontsize=10)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            for idx in range(len(snapshots), 12):
                axes.flat[idx].set_visible(False)

            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, dpi=150); plt.close(fig)

        # Summary page
        logger.info("  Summary statistics page")
        fig, ax = plt.subplots(figsize=(12, 8)); ax.axis("off")
        lines = ["DATA PREPARATION — SUMMARY STATISTICS", "=" * 55, ""]
        for sn, da in stages.items():
            r = dask.compute(da.mean(), da.std(), da.min(), da.max(), da.isnull().sum())
            vm, vs, vn, vx, nc = [float(x) for x in r]; nc = int(nc)
            lines += [f"{sn}", f"  Shape: {dict(da.sizes)}",
                      f"  Mean: {vm:>12.6f}  Std: {vs:>12.6f}",
                      f"  Min:  {vn:>12.6f}  Max: {vx:>12.6f}",
                      f"  NaNs: {nc:>12,}", ""]
        ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
                fontsize=10, verticalalignment="top", fontfamily="monospace")
        fig.tight_layout(); pdf.savefig(fig, dpi=150); plt.close(fig)

    logger.info(f"Diagnostic PDF saved to {pdf_path}")
    return pdf_path


# ── Resume helper ────────────────────────────────────────────────────

def _load_intermediate(filename, label):
    conf = cfg.data_preparation
    path = os.path.join(conf.output_dir, filename)
    logger.info(f"RESUME: loading {label} from {path}")
    ds = xr.open_dataset(path, chunks={"time": 366})
    da = ds[conf.variable]
    logger.info(f"  → Shape: {dict(da.sizes)}")
    return da


# Stage numbers:
#   1 = load, 2 = downscale, 3 = temporal avg,
#   4 = detrend, 5 = standardise, 6 = fill

STAGE_NAMES = {
    0: "start from scratch",
    1: "after load",
    2: "after downscale",
    3: "after temporal avg",
    4: "after detrend",
    5: "after standardise",
    6: "after fill",
}


# ── Pipeline ─────────────────────────────────────────────────────────

def run():
    """Execute the data-preparation pipeline with resume support."""
    print()
    logger.info("=" * 60)
    logger.info("PHASE 1: DATA PREPARATION — START")
    logger.info("=" * 60)

    conf = cfg.data_preparation
    resume = conf.resume_after_stage

    if resume > 0:
        logger.info(f"RESUME MODE: skipping to {STAGE_NAMES.get(resume, '?')} "
                    f"(resume_after_stage={resume})")

    # ── Configure dask ───────────────────────────────────────────
    client = cluster = None
    if conf.dask_scheduler == "distributed":
        from dask.distributed import Client, LocalCluster
        cluster = LocalCluster(n_workers=conf.dask_n_workers,
                               threads_per_worker=conf.dask_threads_per_worker,
                               memory_limit=conf.dask_memory_limit)
        client = Client(cluster)
        logger.info(f"Dask distributed: {conf.dask_n_workers}×{conf.dask_threads_per_worker}")
        logger.info(f"  Dashboard: {client.dashboard_link}")
    elif conf.dask_scheduler == "threads":
        dask.config.set(scheduler="threads", num_workers=conf.dask_num_threads)
        logger.info(f"Dask threaded: {conf.dask_num_threads} threads")
    elif conf.dask_scheduler == "synchronous":
        dask.config.set(scheduler="synchronous")
        logger.info("Dask synchronous")

    os.makedirs(conf.output_dir, exist_ok=True)
    diag_dir = os.path.dirname(os.path.join(conf.output_dir, conf.diagnostic_pdf))
    os.makedirs(diag_dir, exist_ok=True)
    logger.info(f"Output directory: {os.path.abspath(conf.output_dir)}")

    stages = {}

    RESUME_FILE_MAP = {
        2: (conf.downscaled_file,          "downscaled SST"),
        3: (conf.temporally_averaged_file, "temporally averaged SST"),
        4: (conf.detrended_file,           "detrended SST"),
        5: (conf.standardised_file,        "standardised SST"),
        6: (conf.filled_file,              "filled SST"),
    }

    # ── Stage 1: Load ────────────────────────────────────────────
    if resume < 1:
        sst = load_sst()
        stages["1. Raw SST (°C)"] = sst
    else:
        logger.info("SKIP: Stage 1 (load)"); sst = None

    # ── Stage 2: Downscale ───────────────────────────────────────
    if resume < 2:
        sst = downscale(sst)
        if conf.save_intermediates:
            save_netcdf(sst, conf.downscaled_file, "SST downscaled")
        stages["2. Downscaled SST (°C)"] = sst
    elif resume == 2:
        f, l = RESUME_FILE_MAP[2]; sst = _load_intermediate(f, l)
        stages["2. Downscaled SST (°C)"] = sst
    else:
        logger.info("SKIP: Stage 2 (downscale)"); sst = None

    # ── Stage 3: Temporal averaging ──────────────────────────────
    if resume < 3:
        sst = temporal_average(sst)
        if conf.temporal_average != "none" and conf.save_intermediates:
            save_netcdf(sst, conf.temporally_averaged_file,
                        f"SST {conf.temporal_average} averaged")
        if conf.temporal_average != "none":
            stages[f"3. Temporal avg ({conf.temporal_average})"] = sst
    elif resume == 3:
        if conf.temporal_average != "none":
            f, l = RESUME_FILE_MAP[3]; sst = _load_intermediate(f, l)
            stages[f"3. Temporal avg ({conf.temporal_average})"] = sst
        else:
            f, l = RESUME_FILE_MAP[2]; sst = _load_intermediate(f, l)
    else:
        logger.info("SKIP: Stage 3 (temporal avg)"); sst = None

    # ── Stage 4: Detrend ─────────────────────────────────────────
    if resume < 4:
        sst, slope, intercept = detrend_global_linear(sst)
        if conf.save_intermediates:
            save_netcdf(sst, conf.detrended_file, "SST detrended")
        stages["4. Detrended SST (°C)"] = sst
    elif resume == 4:
        f, l = RESUME_FILE_MAP[4]; sst = _load_intermediate(f, l)
        stages["4. Detrended SST (°C)"] = sst
    else:
        logger.info("SKIP: Stage 4 (detrend)"); sst = None

    # ── Stage 5: Standardise ─────────────────────────────────────
    if resume < 5:
        sst, gmean, gstd = standardise_global(sst)
        if conf.save_intermediates:
            save_netcdf(sst, conf.standardised_file, "SST standardised")
        stages["5. Standardised SST"] = sst
    elif resume == 5:
        f, l = RESUME_FILE_MAP[5]; sst = _load_intermediate(f, l)
        stages["5. Standardised SST"] = sst
    else:
        logger.info("SKIP: Stage 5 (standardise)"); sst = None

    # ── Stage 6: Fill land ───────────────────────────────────────
    if resume < 6:
        sst = fill_missing(sst)
        if conf.save_intermediates:
            save_netcdf(sst, conf.filled_file, "SST filled")
        stages["6. Land filled"] = sst
    elif resume == 6:
        f, l = RESUME_FILE_MAP[6]; sst = _load_intermediate(f, l)
        stages["6. Land filled"] = sst

    # ── Region subset (optional) ─────────────────────────────────
    if conf.subset_region:
        sst = subset_region(sst)
        save_netcdf(sst, conf.subset_file, "SST subsetted")
        stages["7. Region subset"] = sst

    # ── Save final ───────────────────────────────────────────────
    desc = "SST downscaled, detrended, standardised, filled"
    if conf.temporal_average != "none":
        desc += f", {conf.temporal_average} averaged"
    logger.info("Saving final processed dataset …")
    save_netcdf(sst, conf.final_file, desc)

    # ── Diagnostic PDF ───────────────────────────────────────────
    logger.info("Generating diagnostic plots …")
    pdf_path = generate_diagnostic_pdf(stages)

    # ── Shutdown dask ────────────────────────────────────────────
    if client is not None:
        logger.info("Shutting down dask cluster")
        client.close(); cluster.close()

    logger.info("=" * 60)
    logger.info("PHASE 1: DATA PREPARATION — COMPLETE")
    logger.info("=" * 60)
    print()
    return pdf_path


if __name__ == "__main__":
    run()
