"""
results_analysis.py — Phase 4: Results Analysis.

Key analyses:
  1. Scalar metrics by lead step (RMSE, MAE, bias, ACC, R²).
  2. Persistence baseline and skill scores.
  3. Spatial maps with coastlines (all-grid and ocean-only).
  4. Sample forecast chain anomaly maps.
  5. ENSO 3.4 index analysis.
  6. CSV tables, text report, and multi-page PDF.
"""

import os
import csv
import logging
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    _cartopy_data = os.environ.get("CARTOPY_DATA_DIR", "")
    if _cartopy_data:
        os.environ["CARTOPY_DATA_DIR"] = _cartopy_data
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except (ImportError, Exception):
    HAS_CARTOPY = False

from config import cfg

logger = logging.getLogger("Phase4_Analysis")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(h)

if HAS_CARTOPY:
    logger.info("Cartopy available — coastlines will be drawn")


# ══════════════════════════════════════════════════════════════════════
# MAP HELPERS
# ══════════════════════════════════════════════════════════════════════

def _create_map_ax(fig, nrows, ncols, idx):
    if HAS_CARTOPY:
        ax = fig.add_subplot(nrows, ncols, idx, projection=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.5, color="black")
        return ax
    return fig.add_subplot(nrows, ncols, idx)

def _map_pcolormesh(ax, lon, lat, data, **kw):
    if HAS_CARTOPY:
        return ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(), **kw)
    return ax.pcolormesh(lon, lat, data, **kw)


# ══════════════════════════════════════════════════════════════════════
# OCEAN MASK (Natural Earth via regionmask, with fallbacks)
# ══════════════════════════════════════════════════════════════════════

try:
    import regionmask
    HAS_REGIONMASK = True
except ImportError:
    HAS_REGIONMASK = False

def compute_ocean_mask(ds):
    """
    Compute an ocean mask for the grid.  True = ocean, False = land.

    Method priority:
      1. regionmask + Natural Earth land polygons (most accurate)
      2. Cartopy Natural Earth geometries rasterised manually
      3. Fallback: variance-based heuristic from the data itself

    Returns (lat, lon) boolean array.
    """
    lat = ds["lat"].values
    lon = ds["lon"].values

    # ── Method 1: regionmask ─────────────────────────────────────
    if HAS_REGIONMASK:
        try:
            logger.info("  Computing ocean mask via regionmask (Natural Earth)")
            # regionmask API varies across versions
            try:
                land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
            except AttributeError:
                try:
                    land = regionmask.defined_regions.natural_earth.land_110
                except AttributeError:
                    land = regionmask.defined_regions.natural_earth_v4_1_0.land_110
            land_mask = land.mask(lon, lat)
            # land_mask: NaN = ocean, integer = land region
            ocean = np.isnan(land_mask.values)
            logger.info(f"  Ocean mask (regionmask): {ocean.sum():,} ocean / "
                        f"{(~ocean).sum():,} land of {ocean.size:,}")
            return ocean
        except Exception as e:
            logger.warning(f"  regionmask failed: {e}, trying cartopy fallback")

    # ── Method 2: Cartopy + matplotlib rasterisation ────────────
    if HAS_CARTOPY:
        try:
            from matplotlib.path import Path
            from cartopy.io.shapereader import natural_earth
            import cartopy.io.shapereader as shpreader

            logger.info("  Computing ocean mask via cartopy rasterisation")
            shpfile = natural_earth(resolution="110m", category="physical",
                                    name="land")
            reader = shpreader.Reader(shpfile)

            # Build (lat, lon) grid of points
            lon_grid, lat_grid = np.meshgrid(lon, lat)
            # Convert 0–360 → -180–180 for Natural Earth
            lon_180 = np.where(lon_grid > 180, lon_grid - 360, lon_grid)
            points = np.column_stack([lon_180.ravel(), lat_grid.ravel()])

            land = np.zeros(len(points), dtype=bool)
            for geom in reader.geometries():
                # Handle MultiPolygon
                if geom.geom_type == "MultiPolygon":
                    polys = list(geom.geoms)
                else:
                    polys = [geom]
                for poly in polys:
                    ext = np.array(poly.exterior.coords)
                    path = Path(ext)
                    land |= path.contains_points(points)

            ocean = ~land.reshape(len(lat), len(lon))
            logger.info(f"  Ocean mask (cartopy): {ocean.sum():,} ocean / "
                        f"{(~ocean).sum():,} land of {ocean.size:,}")
            return ocean
        except Exception as e:
            logger.warning(f"  Cartopy mask failed: {e}, using variance fallback")

    # ── Method 3: Variance-based fallback ────────────────────────
    logger.info("  Computing ocean mask via variance fallback (no regionmask/cartopy)")
    actual = ds["actual"].values  # (chain, lead, lat, lon)
    std = np.nanstd(actual, axis=(0, 1))
    ocean = std > 0.01
    logger.info(f"  Ocean mask (variance): {ocean.sum():,} ocean / "
                f"{(~ocean).sum():,} land of {ocean.size:,}")
    return ocean


def _apply_ocean_mask(data, ocean_mask):
    """Set land points to NaN for ocean-only plots."""
    masked = data.copy()
    masked[~ocean_mask] = np.nan
    return masked


# ══════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════

def rmse(pred, actual, dims=None):
    return np.sqrt(np.nanmean((pred - actual) ** 2, axis=dims))

def mae(pred, actual, dims=None):
    return np.nanmean(np.abs(pred - actual), axis=dims)

def bias_metric(pred, actual, dims=None):
    return np.nanmean(pred - actual, axis=dims)

def r_squared(pred, actual, dims=None):
    ss_res = np.nansum((actual - pred) ** 2, axis=dims)
    ss_tot = np.nansum((actual - np.nanmean(actual, axis=dims,
                                             keepdims=True)) ** 2, axis=dims)
    return np.where(ss_tot > 0, 1 - ss_res / ss_tot, 0.0)

def compute_area_weights(lat):
    """Cosine-latitude area weights, normalised, shape (lat,)."""
    w = np.cos(np.deg2rad(lat))
    return w / w.sum()


# ══════════════════════════════════════════════════════════════════════
# CLIMATOLOGY FOR ACC (daily, using netCDF4 for speed)
# ══════════════════════════════════════════════════════════════════════

def _load_processed_nc():
    """
    Load the processed SST file using netCDF4 directly (fast).

    Returns sst (T, lat, lon), times (datetime64), lat, lon arrays.
    """
    import netCDF4 as nc

    conf = cfg.results_analysis
    data_conf = cfg.data_preparation
    path = conf.processed_data_path

    logger.info(f"  Loading {path} via netCDF4")
    ds = nc.Dataset(path, "r")

    sst = ds.variables[data_conf.variable][:]    # (T, lat, lon)
    lat = ds.variables["lat"][:]
    lon = ds.variables["lon"][:]

    # Decode time
    time_var = ds.variables["time"]
    times = nc.num2date(time_var[:], time_var.units,
                        calendar=getattr(time_var, "calendar", "standard"))
    # Convert cftime to numpy datetime64
    times = np.array([np.datetime64(str(t)[:10]) for t in times])

    ds.close()
    logger.info(f"  Loaded: shape={sst.shape}, "
                f"time {str(times[0])[:10]} to {str(times[-1])[:10]}")
    return sst, times, lat, lon


def compute_daily_climatology():
    """
    Compute a day-of-year climatology (1–366) per grid point from the
    processed dataset over the configured baseline period (e.g. 1991–2020).

    Applies a rolling-mean smoother to reduce day-to-day noise.

    Returns
    -------
    clim : np.ndarray (366, lat, lon) — day-of-year index 0 = DOY 1
    """
    import pandas as pd

    conf = cfg.results_analysis
    yr_start, yr_end = conf.acc_clim_years
    smooth = conf.acc_clim_smoothing_window

    sst, times, lat, lon = _load_processed_nc()

    # Select baseline years
    years = pd.DatetimeIndex(times).year.values
    mask = (years >= yr_start) & (years <= yr_end)
    sst_base = sst[mask]
    times_base = times[mask]

    logger.info(f"  Baseline: {yr_start}–{yr_end}, "
                f"{mask.sum():,} time steps selected")

    # Day-of-year for each timestep (1–366)
    doy = pd.DatetimeIndex(times_base).dayofyear.values

    # Accumulate per-DOY sums and counts
    nlat, nlon = sst_base.shape[1], sst_base.shape[2]
    clim_sum = np.zeros((366, nlat, nlon), dtype=np.float64)
    clim_count = np.zeros(366, dtype=np.float64)

    for t in range(len(sst_base)):
        d = doy[t] - 1  # 0-indexed
        clim_sum[d] += sst_base[t]
        clim_count[d] += 1.0

    # Average
    valid = clim_count > 0
    clim = np.zeros_like(clim_sum, dtype=np.float32)
    for d in range(366):
        if valid[d]:
            clim[d] = clim_sum[d] / clim_count[d]

    # Fill any missing DOYs (e.g. DOY 366 if no leap years in baseline)
    for d in range(366):
        if not valid[d]:
            # Use nearest valid DOY
            for offset in range(1, 10):
                if valid[(d - offset) % 366]:
                    clim[d] = clim[(d - offset) % 366]; break
                if valid[(d + offset) % 366]:
                    clim[d] = clim[(d + offset) % 366]; break

    # Smooth with rolling mean (circular wrap for Jan 1 ↔ Dec 31)
    if smooth > 1:
        logger.info(f"  Smoothing climatology with {smooth}-day rolling mean")
        padded = np.concatenate([clim[-smooth:], clim, clim[:smooth]], axis=0)
        kernel = np.ones(smooth) / smooth
        smoothed = np.zeros_like(clim)
        for i in range(nlat):
            for j in range(nlon):
                smoothed[:, i, j] = np.convolve(
                    padded[:, i, j], kernel, mode="valid"
                )[:366]
        clim = smoothed

    logger.info(f"  Daily climatology: shape {clim.shape}, "
                f"range [{clim.min():.4f}, {clim.max():.4f}]")
    return clim


def get_climatology_fields(clim_daily, valid_times):
    """
    Look up the daily climatology for an array of valid_times.

    Parameters
    ----------
    clim_daily  : np.ndarray (366, lat, lon) from compute_daily_climatology()
    valid_times : np.ndarray of datetime64, any shape

    Returns
    -------
    clim_fields : np.ndarray, shape (*valid_times.shape, lat, lon)
    """
    import pandas as pd
    flat_times = valid_times.ravel()
    doy = pd.DatetimeIndex(flat_times).dayofyear.values  # 1–366

    nlat, nlon = clim_daily.shape[1], clim_daily.shape[2]
    clim_flat = np.empty((len(flat_times), nlat, nlon), dtype=np.float32)
    for i, d in enumerate(doy):
        clim_flat[i] = clim_daily[d - 1]  # 0-indexed

    return clim_flat.reshape(*valid_times.shape, nlat, nlon)


# ══════════════════════════════════════════════════════════════════════
# ACC — PROPER WMO FORMULATION
# ══════════════════════════════════════════════════════════════════════

def acc_scalar(pred, actual, clim, lat_weights):
    """
    Area-weighted Anomaly Correlation Coefficient (ACC).

    ACC = Σ_i w_i · f'_i · a'_i / sqrt(Σ_i w_i f'_i² · Σ_i w_i a'_i²)

    where f' = pred − clim, a' = actual − clim, w_i = cos(lat_i).

    Parameters
    ----------
    pred, actual : (n_chains, lat, lon)
    clim         : (n_chains, lat, lon) — climatology for each chain's valid time
    lat_weights  : (lat,) normalised area weights

    Returns
    -------
    acc_value : float — mean ACC across chains
    """
    f_anom = pred - clim       # forecast anomaly
    a_anom = actual - clim     # observed anomaly

    # Area weights: (lat,) → (1, lat, 1) for broadcasting
    w = lat_weights[np.newaxis, :, np.newaxis]

    # Per-chain weighted ACC
    num = np.nansum(w * f_anom * a_anom, axis=(-2, -1))    # (n_chains,)
    den_f = np.nansum(w * f_anom ** 2, axis=(-2, -1))      # (n_chains,)
    den_a = np.nansum(w * a_anom ** 2, axis=(-2, -1))      # (n_chains,)
    den = np.sqrt(den_f * den_a)
    per_chain = np.where(den > 0, num / den, 0.0)

    return float(np.nanmean(per_chain))


def acc_spatial(pred, actual, clim):
    """
    Per-grid-point ACC across chains.

    At each (lat, lon), computes the correlation of forecast anomalies
    vs observed anomalies across all chains.

    Parameters
    ----------
    pred, actual, clim : (n_chains, lat, lon)

    Returns
    -------
    acc_map : (lat, lon)
    """
    f_anom = pred - clim
    a_anom = actual - clim

    num = np.nansum(f_anom * a_anom, axis=0)
    den = np.sqrt(np.nansum(f_anom ** 2, axis=0) *
                  np.nansum(a_anom ** 2, axis=0))
    return np.where(den > 0, num / den, 0.0)


# ══════════════════════════════════════════════════════════════════════
# DE-STANDARDISATION
# ══════════════════════════════════════════════════════════════════════

class Destandardiser:
    def __init__(self):
        conf = cfg.results_analysis
        self.loaded = False
        try:
            std_ds = xr.open_dataset(conf.std_params_path)
            self.global_mean = float(std_ds["global_mean"].values)
            self.global_std = float(std_ds["global_std"].values)
            std_ds.close()
            trend_ds = xr.open_dataset(conf.trend_params_path)
            self.slope = float(trend_ds["slope"].values)
            self.intercept = float(trend_ds["intercept"].values)
            self.time_origin = trend_ds["time_origin"].values
            trend_ds.close()
            self.loaded = True
            logger.info(f"  De-standardisation: mean={self.global_mean:.4f}, "
                        f"std={self.global_std:.4f}, slope={self.slope:.6e}")
        except Exception as e:
            logger.warning(f"  De-standardisation params not loaded: {e}")

    def to_celsius(self, data_std, valid_times=None):
        if not self.loaded:
            return data_std
        data = data_std * self.global_std + self.global_mean
        if valid_times is not None:
            dt = (valid_times - self.time_origin).astype("timedelta64[s]").astype(float) / 86400.0
            trend = self.slope * dt + self.intercept
            for _ in range(data.ndim - trend.ndim):
                trend = trend[..., np.newaxis]
            data = data + trend
        return data

    @property
    def unit_label(self):
        return "°C" if self.loaded else "standardised"


# ══════════════════════════════════════════════════════════════════════
# BASELINES
# ══════════════════════════════════════════════════════════════════════

def compute_persistence_baseline(ds):
    actual = ds["actual"].values
    n_leads = actual.shape[1]
    if "init_field" in ds:
        init = ds["init_field"].values[:, np.newaxis, :, :]
        logger.info("    Using init_field as persistence anchor")
    else:
        init = actual[:, 0:1, :, :]
        logger.warning("    init_field not found — using actual at lead 1")
    return np.tile(init, (1, n_leads, 1, 1))


# ══════════════════════════════════════════════════════════════════════
# PER-LEAD METRICS
# ══════════════════════════════════════════════════════════════════════

def compute_metrics_by_lead(ds, clim_daily, lat_weights, persistence=None):
    pred = ds["prediction"].values       # (chain, lead, lat, lon)
    actual = ds["actual"].values
    valid_times = ds["valid_time"].values  # (chain, lead)
    lead_steps = ds["lead_step"].values

    # Get climatology fields for all (chain, lead) combos
    clim_all = get_climatology_fields(clim_daily, valid_times)  # (chain, lead, lat, lon)

    rows = []
    for i, ls in enumerate(lead_steps):
        p = pred[:, i, :, :]       # (chain, lat, lon)
        a = actual[:, i, :, :]
        c = clim_all[:, i, :, :]   # climatology for this lead step

        row = {
            "lead_step": int(ls),
            "fno_rmse": float(rmse(p, a)),
            "fno_mae": float(mae(p, a)),
            "fno_bias": float(bias_metric(p, a)),
            "fno_acc": float(acc_scalar(p, a, c, lat_weights)),
            "fno_r2": float(r_squared(p, a)),
        }
        if persistence is not None:
            pp = persistence[:, i, :, :]
            row["persist_rmse"] = float(rmse(pp, a))
            row["persist_mae"] = float(mae(pp, a))
            row["persist_acc"] = float(acc_scalar(pp, a, c, lat_weights))
            row["skill_score"] = (
                (1.0 - row["fno_rmse"] / row["persist_rmse"]) * 100.0
                if row["persist_rmse"] > 0 else 0.0
            )
        rows.append(row)
    return rows


# ══════════════════════════════════════════════════════════════════════
# SPATIAL METRICS
# ══════════════════════════════════════════════════════════════════════

def compute_spatial_metrics(ds, clim_daily, lead_steps_to_map):
    pred = ds["prediction"].values
    actual = ds["actual"].values
    valid_times = ds["valid_time"].values
    all_leads = ds["lead_step"].values

    clim_all = get_climatology_fields(clim_daily, valid_times)

    spatial = {}
    for ls in lead_steps_to_map:
        idx = np.where(all_leads == ls)[0]
        if len(idx) == 0:
            continue
        i = idx[0]
        p = pred[:, i, :, :]       # (chain, lat, lon)
        a = actual[:, i, :, :]
        c = clim_all[:, i, :, :]   # (chain, lat, lon)

        sp_rmse = np.sqrt(np.nanmean((p - a) ** 2, axis=0))
        sp_acc = acc_spatial(p, a, c)

        spatial[int(ls)] = {"rmse": sp_rmse, "acc": sp_acc}
    return spatial


# ══════════════════════════════════════════════════════════════════════
# ENSO 3.4
# ══════════════════════════════════════════════════════════════════════

def _extract_nino34_mean(data, lat, lon, lat_bounds, lon_bounds):
    lat_mask = (lat >= lat_bounds[0]) & (lat <= lat_bounds[1])
    lon_mask = (lon >= lon_bounds[0]) & (lon <= lon_bounds[1])
    weights = np.cos(np.deg2rad(lat[lat_mask]))
    weights = weights / weights.sum()
    region = data[..., lat_mask, :]
    region = region[..., :, lon_mask]
    weighted = np.tensordot(region, weights, axes=([-2], [0]))
    return np.nanmean(weighted, axis=-1)

def compute_enso34_climatology():
    """
    Compute ENSO 3.4 daily climatology: day-of-year average of the
    area-weighted Niño 3.4 region mean, using netCDF4 directly.

    Returns clim : np.ndarray (366,) — daily ENSO 3.4 climatology.
    """
    import pandas as pd

    conf = cfg.results_analysis
    yr_start, yr_end = conf.enso34_clim_years
    lb = conf.enso34_lat_bounds
    lob = conf.enso34_lon_bounds

    logger.info(f"  Computing ENSO 3.4 daily climatology "
                f"{yr_start}–{yr_end}")

    sst, times, lat, lon = _load_processed_nc()

    # Select baseline years
    years = pd.DatetimeIndex(times).year.values
    mask = (years >= yr_start) & (years <= yr_end)
    sst_base = sst[mask]
    times_base = times[mask]

    # Extract Niño 3.4 region
    lat_mask = (lat >= lb[0]) & (lat <= lb[1])
    lon_mask = (lon >= lob[0]) & (lon <= lob[1])
    sst_nino = sst_base[:, lat_mask, :][:, :, lon_mask]

    # Area-weighted mean
    weights = np.cos(np.deg2rad(lat[lat_mask]))
    weights = weights / weights.sum()
    nino_ts = np.average(sst_nino, axis=1, weights=weights)  # (T, lon_sub)
    nino_ts = np.nanmean(nino_ts, axis=1)                     # (T,)

    # Daily climatology
    doy = pd.DatetimeIndex(times_base).dayofyear.values
    clim = np.zeros(366, dtype=np.float64)
    count = np.zeros(366, dtype=np.float64)
    for t in range(len(nino_ts)):
        d = doy[t] - 1
        clim[d] += nino_ts[t]
        count[d] += 1.0
    valid = count > 0
    clim[valid] /= count[valid]

    # Fill missing DOYs
    for d in range(366):
        if not valid[d]:
            for off in range(1, 10):
                if valid[(d - off) % 366]: clim[d] = clim[(d - off) % 366]; break
                if valid[(d + off) % 366]: clim[d] = clim[(d + off) % 366]; break

    # Smooth
    smooth = conf.acc_clim_smoothing_window
    if smooth > 1:
        padded = np.concatenate([clim[-smooth:], clim, clim[:smooth]])
        kernel = np.ones(smooth) / smooth
        clim = np.convolve(padded, kernel, mode="valid")[:366]

    logger.info(f"  ENSO 3.4 daily clim range: [{clim.min():.4f}, {clim.max():.4f}]")
    return clim

def compute_enso34_for_chains(ds, enso_clim_daily, lat, lon):
    """
    Compute ENSO 3.4 anomaly index for predicted, actual, and persistence.

    Persistence = the ENSO 3.4 INDEX (anomaly) at initialisation,
    held constant for all lead steps.  This is the standard definition:
    "the current anomaly persists unchanged."

    Parameters
    ----------
    enso_clim_daily : np.ndarray (366,) — daily ENSO 3.4 climatology
    """
    import pandas as pd

    conf = cfg.results_analysis
    pred = ds["prediction"].values
    actual = ds["actual"].values
    valid_times = ds["valid_time"].values
    init_times = ds["init_time"].values
    lb = conf.enso34_lat_bounds; lob = conf.enso34_lon_bounds

    nino_pred = _extract_nino34_mean(pred, lat, lon, lb, lob)
    nino_actual = _extract_nino34_mean(actual, lat, lon, lb, lob)

    # Persistence anchor: raw Niño 3.4 mean from init_field
    if "init_field" in ds:
        nino_persist_raw = _extract_nino34_mean(
            ds["init_field"].values, lat, lon, lb, lob)  # (chain,)
    else:
        nino_persist_raw = nino_actual[:, 0]

    n_chains, n_leads = nino_pred.shape

    # Compute persistence ANOMALY at init time (constant for all leads)
    nino_persist_anom = np.empty(n_chains, dtype=np.float64)
    for c in range(n_chains):
        init_doy = int(pd.Timestamp(init_times[c]).dayofyear)
        nino_persist_anom[c] = nino_persist_raw[c] - enso_clim_daily[init_doy - 1]
    # Tile to (chain, lead) — constant across lead steps
    nino_persist = np.tile(nino_persist_anom[:, np.newaxis], (1, n_leads))

    # Subtract daily climatology from predicted and actual
    for c in range(n_chains):
        for l in range(n_leads):
            doy = int(pd.Timestamp(valid_times[c, l]).dayofyear)
            cv = enso_clim_daily[doy - 1]
            nino_pred[c, l] -= cv
            nino_actual[c, l] -= cv

    return nino_pred, nino_actual, nino_persist

def compute_enso34_metrics(enso_pred, enso_actual, enso_persist, lead_steps):
    rows = []
    for i, ls in enumerate(lead_steps):
        p = enso_pred[:, i]; a = enso_actual[:, i]; pe = enso_persist[:, i]
        corr = np.corrcoef(p, a)[0, 1] if len(p) > 2 else 0.0
        pcorr = np.corrcoef(pe, a)[0, 1] if len(pe) > 2 else 0.0
        rows.append({
            "lead_step": int(ls),
            "enso_rmse": float(rmse(p, a)),
            "enso_corr": float(corr),
            "enso_bias": float(np.mean(p - a)),
            "persist_enso_rmse": float(rmse(pe, a)),
            "persist_enso_corr": float(pcorr),
        })
    return rows


# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION SUMMARY PAGE
# ══════════════════════════════════════════════════════════════════════

def _plot_config_summary(pdf):
    """Generate a multi-page configuration summary at the start of the PDF."""
    from dataclasses import fields, asdict
    from datetime import datetime

    fig, ax = plt.subplots(figsize=(11, 16))
    ax.axis("off")

    lines = []
    lines.append(f"FORECAST ANALYSIS — CONFIGURATION SUMMARY")
    lines.append(f"Project: {cfg.project_name}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 72)
    lines.append("")

    # Iterate over all config dataclasses
    config_sections = [
        ("Data Preparation", cfg.data_preparation),
        ("Model Training", cfg.model_training),
        ("Model Inference", cfg.model_inference),
        ("Results Analysis", cfg.results_analysis),
    ]

    for section_name, section_cfg in config_sections:
        lines.append(f"── {section_name} ──")
        lines.append("")
        for f in fields(section_cfg):
            # Skip properties (they aren't dataclass fields with values)
            try:
                val = getattr(section_cfg, f.name)
            except Exception:
                continue
            # Format value
            if isinstance(val, float):
                val_str = f"{val:.6g}"
            elif isinstance(val, dict):
                val_str = str(val)[:60]
            else:
                val_str = str(val)
            # Truncate long values
            if len(val_str) > 60:
                val_str = val_str[:57] + "..."
            lines.append(f"  {f.name:.<40s} {val_str}")
        lines.append("")

    # Split into pages if needed (roughly 55 lines per page)
    max_lines_per_page = 55
    page_chunks = [lines[i:i + max_lines_per_page]
                   for i in range(0, len(lines), max_lines_per_page)]

    for page_idx, chunk in enumerate(page_chunks):
        if page_idx > 0:
            fig, ax = plt.subplots(figsize=(11, 16))
            ax.axis("off")

        text = "\n".join(chunk)
        ax.text(0.03, 0.97, text, transform=ax.transAxes,
                fontsize=8.5, verticalalignment="top", fontfamily="monospace",
                linespacing=1.3)

        if page_idx == 0:
            ax.set_title("Configuration Summary", fontsize=16,
                         fontweight="bold", pad=15)

        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    logger.info("  Configuration summary page(s) added to PDF")


# ══════════════════════════════════════════════════════════════════════
# PLOTTING — shared colorbar helper
# ══════════════════════════════════════════════════════════════════════

def _add_shared_colorbar(fig, im, label="", orientation="horizontal"):
    """Add a single shared colorbar at the bottom of the figure."""
    cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.018])
    fig.colorbar(im, cax=cbar_ax, orientation=orientation, label=label)


# ══════════════════════════════════════════════════════════════════════
# PLOTTING — metrics
# ══════════════════════════════════════════════════════════════════════

def _plot_metrics_vs_lead(rows, split, pdf):
    has_p = "persist_rmse" in rows[0]
    leads = [r["lead_step"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Forecast Skill vs Lead Step — {split}", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(leads, [r["fno_rmse"] for r in rows], "o-", ms=3, lw=2, label="fno")
    if has_p:
        ax.plot(leads, [r["persist_rmse"] for r in rows], "s--", ms=3, alpha=0.7, label="Persistence")
    ax.set(xlabel="Lead step", ylabel="RMSE", title="RMSE"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(leads, [r["fno_mae"] for r in rows], "o-", ms=3, lw=2, label="fno")
    if has_p:
        ax.plot(leads, [r["persist_mae"] for r in rows], "s--", ms=3, alpha=0.7, label="Persistence")
    ax.set(xlabel="Lead step", ylabel="MAE", title="MAE"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(leads, [r["fno_acc"] for r in rows], "o-", ms=3, lw=2, label="fno")
    if has_p:
        ax.plot(leads, [r["persist_acc"] for r in rows], "s--", ms=3, alpha=0.7, label="Persistence")
    ax.set(xlabel="Lead step", ylabel="ACC", title="ACC (centered)"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    if has_p:
        ax.bar(leads, [r["skill_score"] for r in rows], color="steelblue", edgecolor="black", alpha=0.8, width=0.8)
        ax.axhline(0, color="red", ls="--", lw=1)
        ax.set(xlabel="Lead step", ylabel="Skill (%)", title="RMSE Skill vs Persistence")
    else:
        ax.plot(leads, [r["fno_r2"] for r in rows], "o-", lw=2)
        ax.set(xlabel="Lead step", ylabel="R²", title="R²")
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95]); pdf.savefig(fig, dpi=150); plt.close(fig)


def _plot_bias_and_r2(rows, split, pdf):
    leads = [r["lead_step"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Bias and R² — {split}", fontsize=16, fontweight="bold")
    axes[0].plot(leads, [r["fno_bias"] for r in rows], "o-", color="darkorange", ms=3, lw=2)
    axes[0].axhline(0, color="gray", ls="--")
    axes[0].set(xlabel="Lead step", ylabel="Bias", title="Mean Bias"); axes[0].grid(alpha=0.3)
    axes[1].plot(leads, [r["fno_r2"] for r in rows], "o-", color="green", ms=3, lw=2)
    axes[1].set(xlabel="Lead step", ylabel="R²", title="R²"); axes[1].grid(alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); pdf.savefig(fig, dpi=150); plt.close(fig)


def _plot_metrics_table(rows, split, pdf):
    has_p = "persist_rmse" in rows[0]
    fig, ax = plt.subplots(figsize=(16, max(4, len(rows) * 0.4 + 2))); ax.axis("off")
    if has_p:
        headers = ["Lead\nStep", "fno\nRMSE", "Persist\nRMSE", "Skill\n%",
                    "fno\nMAE", "fno\nBias", "fno\nACC", "Persist\nACC", "fno\nR²"]
        cells = [[f"{r['lead_step']}", f"{r['fno_rmse']:.4f}", f"{r['persist_rmse']:.4f}",
                  f"{r['skill_score']:.1f}%", f"{r['fno_mae']:.4f}", f"{r['fno_bias']:.4f}",
                  f"{r['fno_acc']:.4f}", f"{r['persist_acc']:.4f}", f"{r['fno_r2']:.4f}"] for r in rows]
    else:
        headers = ["Lead\nStep", "fno\nRMSE", "fno\nMAE", "fno\nBias", "fno\nACC", "fno\nR²"]
        cells = [[f"{r['lead_step']}", f"{r['fno_rmse']:.4f}", f"{r['fno_mae']:.4f}",
                  f"{r['fno_bias']:.4f}", f"{r['fno_acc']:.4f}", f"{r['fno_r2']:.4f}"] for r in rows]
    table = ax.table(cellText=cells, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.4)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4472C4"); cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#D6E4F0")
    ax.set_title(f"Metrics by Lead Step — {split}", fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout(); pdf.savefig(fig, dpi=150); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# PLOTTING — spatial maps (with shared colorbars, ocean-only variant)
# ══════════════════════════════════════════════════════════════════════

def _plot_spatial_panel(spatial, lead_steps, field_key, cmap, title, lat, lon,
                        split, ocean_mask, pdf, vmin=None, vmax=None):
    """Generic spatial multi-panel plot in 3×N grid with shared colorbar below."""
    n = len(lead_steps)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(5.5 * ncols, 3.5 * nrows + 1.5))
    fig.suptitle(f"{title} — {split}", fontsize=16, fontweight="bold", y=0.98)

    # Auto-compute shared range if not provided
    if vmin is None or vmax is None:
        all_vals = [spatial[ls][field_key] for ls in lead_steps]
        if ocean_mask is not None:
            all_vals = [_apply_ocean_mask(v, ocean_mask) for v in all_vals]
        vmin = min(np.nanmin(v) for v in all_vals) if vmin is None else vmin
        vmax = max(np.nanmax(v) for v in all_vals) if vmax is None else vmax

    im = None
    for idx, ls in enumerate(lead_steps):
        ax = _create_map_ax(fig, nrows, ncols, idx + 1)
        data = spatial[ls][field_key]
        if ocean_mask is not None:
            data = _apply_ocean_mask(data, ocean_mask)
        im = _map_pcolormesh(ax, lon, lat, data, cmap=cmap,
                             vmin=vmin, vmax=vmax, shading="auto")
        ax.set_title(f"Step {ls}")

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        ax = fig.add_subplot(nrows, ncols, idx + 1)
        ax.set_visible(False)

    fig.subplots_adjust(left=0.03, right=0.97, top=0.94, bottom=0.12,
                        wspace=0.08, hspace=0.15)
    if im is not None:
        _add_shared_colorbar(fig, im)
    pdf.savefig(fig, dpi=150); plt.close(fig)


def _plot_spatial_maps(spatial, lat, lon, ocean_mask, split, pdf):
    """Spatial RMSE and ACC maps — full grid and ocean-only."""
    lead_steps = sorted(spatial.keys())
    if not lead_steps:
        return

    # Full grid
    _plot_spatial_panel(spatial, lead_steps, "rmse", "YlOrRd",
                        "Spatial RMSE (all grid)", lat, lon, split, None, pdf)
    _plot_spatial_panel(spatial, lead_steps, "acc", "RdYlGn",
                        "Spatial ACC (all grid)", lat, lon, split, None, pdf,
                        vmin=0, vmax=1)

    # Ocean only
    _plot_spatial_panel(spatial, lead_steps, "rmse", "YlOrRd",
                        "Spatial RMSE (ocean only)", lat, lon, split, ocean_mask, pdf)
    _plot_spatial_panel(spatial, lead_steps, "acc", "RdYlGn",
                        "Spatial ACC (ocean only)", lat, lon, split, ocean_mask, pdf,
                        vmin=0, vmax=1)


# ══════════════════════════════════════════════════════════════════════
# PLOTTING — sample chain anomaly maps (3×3 layout)
# ══════════════════════════════════════════════════════════════════════

def _plot_sample_chains(ds, split, n_samples, destd, ocean_mask, pdf):
    """For each sample chain: 3×3 grid of (actual, prediction, error)
    at 3 lead steps, converted to °C, with shared colorbars per row."""
    pred = ds["prediction"].values
    actual = ds["actual"].values
    init_times = ds["init_time"].values
    lead_steps = ds["lead_step"].values
    valid_times = ds["valid_time"].values
    lat = ds["lat"].values; lon = ds["lon"].values
    n_chains = pred.shape[0]
    unit = destd.unit_label

    indices = np.linspace(0, n_chains - 1, n_samples, dtype=int)

    for ci in indices:
        init_str = str(init_times[ci])[:10]

        # Show 3 lead steps: first, middle, last
        show = [0]
        if len(lead_steps) > 2:
            show.append(len(lead_steps) // 2)
        show.append(len(lead_steps) - 1)

        fig = plt.figure(figsize=(18, 4 * len(show) + 1.5))
        fig.suptitle(f"Chain init: {init_str} — {split} ({unit})",
                     fontsize=16, fontweight="bold", y=0.98)

        for row, li in enumerate(show):
            ls = lead_steps[li]
            vt = valid_times[ci, li]

            # Convert to °C
            p = destd.to_celsius(pred[ci, li], vt)
            a = destd.to_celsius(actual[ci, li], vt)
            if ocean_mask is not None:
                p = _apply_ocean_mask(p, ocean_mask)
                a = _apply_ocean_mask(a, ocean_mask)
            diff = p - a

            # Shared range for actual/prediction
            vmin_sst = np.nanmin(a); vmax_sst = np.nanmax(a)
            vpad = max((vmax_sst - vmin_sst) * 0.05, 0.5)
            vmin_sst -= vpad; vmax_sst += vpad

            # Error range (symmetric)
            eabs = max(abs(np.nanpercentile(diff[np.isfinite(diff)], 2)),
                       abs(np.nanpercentile(diff[np.isfinite(diff)], 98))) \
                   if np.any(np.isfinite(diff)) else 1.0

            # Actual
            ax = _create_map_ax(fig, len(show), 3, row * 3 + 1)
            im_sst = _map_pcolormesh(ax, lon, lat, a, cmap="RdYlBu_r",
                                     vmin=vmin_sst, vmax=vmax_sst, shading="auto")
            ax.set_title(f"Actual — Step {ls} ({unit})")

            # Prediction (same colour range)
            ax = _create_map_ax(fig, len(show), 3, row * 3 + 2)
            _map_pcolormesh(ax, lon, lat, p, cmap="RdYlBu_r",
                            vmin=vmin_sst, vmax=vmax_sst, shading="auto")
            ax.set_title(f"fno — Step {ls} ({unit})")

            # Error
            ax = _create_map_ax(fig, len(show), 3, row * 3 + 3)
            im_err = _map_pcolormesh(ax, lon, lat, diff, cmap="bwr",
                                     vmin=-eabs, vmax=eabs, shading="auto")
            ax.set_title(f"Error — Step {ls} ({unit})")

            # Colorbars after last row
            if row == len(show) - 1:
                cax1 = fig.add_axes([0.05, 0.06, 0.58, 0.018])
                fig.colorbar(im_sst, cax=cax1, orientation="horizontal",
                             label=f"SST ({unit})")
                cax2 = fig.add_axes([0.68, 0.06, 0.27, 0.018])
                fig.colorbar(im_err, cax=cax2, orientation="horizontal",
                             label=f"Error ({unit})")

        fig.subplots_adjust(left=0.03, right=0.97, top=0.94, bottom=0.12,
                            wspace=0.08, hspace=0.15)
        pdf.savefig(fig, dpi=150); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# PLOTTING — global mean chains
# ══════════════════════════════════════════════════════════════════════

def _plot_global_mean_chains(ds, persistence, split, n_samples, destd, pdf):
    """Global-mean SST time series in 3×N grid layout."""
    pred = ds["prediction"].values; actual = ds["actual"].values
    init_times = ds["init_time"].values; lead_steps = ds["lead_step"].values
    valid_times = ds["valid_time"].values
    n_chains = pred.shape[0]; unit = destd.unit_label
    indices = np.linspace(0, n_chains - 1, n_samples, dtype=int)

    ncols = 3
    nrows = (n_samples + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.5 * nrows + 1))
    fig.suptitle(f"Global-Mean SST — {split} ({unit})", fontsize=16, fontweight="bold")
    axes = np.atleast_2d(axes)

    for i, ci in enumerate(indices):
        r, c = divmod(i, ncols)
        ax = axes[r, c]; vt = valid_times[ci]
        gm_p = destd.to_celsius(np.nanmean(pred[ci], axis=(1, 2)), vt)
        gm_a = destd.to_celsius(np.nanmean(actual[ci], axis=(1, 2)), vt)
        ax.plot(lead_steps, gm_a, "k-", lw=2, label="Actual")
        ax.plot(lead_steps, gm_p, "r--", lw=2, label="fno")
        if persistence is not None:
            gm_pe = destd.to_celsius(np.nanmean(persistence[ci], axis=(1, 2)), vt)
            ax.plot(lead_steps, gm_pe, "b:", lw=1.5, label="Persistence")
        ax.set_title(f"Init: {str(init_times[ci])[:10]}", fontsize=10)
        if c == 0:
            ax.set_ylabel(f"SST ({unit})")
        if r == nrows - 1:
            ax.set_xlabel("Lead step")
        ax.legend(loc="upper right", fontsize=7); ax.grid(alpha=0.3)

    for i in range(n_samples, nrows * ncols):
        r, c = divmod(i, ncols); axes[r, c].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.0, w_pad=2.0)
    pdf.savefig(fig, dpi=150); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# PLOTTING — ENSO 3.4
# ══════════════════════════════════════════════════════════════════════

def _plot_enso34_skill(rows, split, pdf):
    leads = [r["lead_step"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"ENSO 3.4 Forecast Skill — {split}", fontsize=16, fontweight="bold")
    axes[0].plot(leads, [r["enso_rmse"] for r in rows], "o-", ms=3, lw=2, label="fno")
    axes[0].plot(leads, [r["persist_enso_rmse"] for r in rows], "s--", ms=3, alpha=0.7, label="Persistence")
    axes[0].set(xlabel="Lead step", ylabel="RMSE", title="ENSO 3.4 RMSE"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(leads, [r["enso_corr"] for r in rows], "o-", ms=3, lw=2, label="fno")
    axes[1].plot(leads, [r["persist_enso_corr"] for r in rows], "s--", ms=3, alpha=0.7, label="Persistence")
    axes[1].set(xlabel="Lead step", ylabel="Corr", title="ENSO 3.4 Correlation"); axes[1].legend()
    axes[1].set_ylim(-0.2, 1.05); axes[1].grid(alpha=0.3)
    axes[2].plot(leads, [r["enso_bias"] for r in rows], "o-", ms=3, lw=2, color="darkorange")
    axes[2].axhline(0, color="gray", ls="--")
    axes[2].set(xlabel="Lead step", ylabel="Bias", title="ENSO 3.4 Bias"); axes[2].grid(alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); pdf.savefig(fig, dpi=150); plt.close(fig)

def _plot_enso34_timeseries(ep, ea, epe, ds, split, n_samples, pdf):
    """ENSO 3.4 time series in 3×N grid, y-axis zoomed to data, with r annotated."""
    init_times = ds["init_time"].values; lead_steps = ds["lead_step"].values
    n_chains = ep.shape[0]; indices = np.linspace(0, n_chains - 1, n_samples, dtype=int)

    ncols = 3
    nrows = (n_samples + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.5 * nrows + 1))
    fig.suptitle(f"ENSO 3.4 Index — {split}", fontsize=16, fontweight="bold")
    axes = np.atleast_2d(axes)

    for i, ci in enumerate(indices):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        ax.plot(lead_steps, ea[ci], "k-", lw=2, label="Actual")
        ax.plot(lead_steps, ep[ci], "r--", lw=2, label="fno")
        ax.plot(lead_steps, epe[ci], "b:", lw=1.5, label="Persistence")
        ax.fill_between(lead_steps, -0.5, 0.5, alpha=0.1, color="gray")
        ax.axhline(0, color="gray", lw=0.5)

        # Correlation between fno and actual
        corr = np.corrcoef(ep[ci], ea[ci])[0, 1] if len(ep[ci]) > 2 else 0.0
        ax.set_title(f"Init: {str(init_times[ci])[:10]}  (r={corr:.3f})",
                     fontsize=9)

        # Zoom y-axis to data range with small padding
        all_vals = np.concatenate([ea[ci], ep[ci], epe[ci]])
        ymin, ymax = np.nanmin(all_vals), np.nanmax(all_vals)
        ypad = max((ymax - ymin) * 0.15, 0.1)
        ax.set_ylim(ymin - ypad, ymax + ypad)

        if c == 0:
            ax.set_ylabel("ENSO 3.4")
        if r == nrows - 1:
            ax.set_xlabel("Lead step")
        ax.legend(loc="upper right", fontsize=7); ax.grid(alpha=0.3)

    for i in range(n_samples, nrows * ncols):
        r, c = divmod(i, ncols); axes[r, c].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.0, w_pad=2.0)
    pdf.savefig(fig, dpi=150); plt.close(fig)

def _plot_enso34_scatter(ep, ea, ds, split, pdf):
    """ENSO 3.4 scatter in 3×N grid layout."""
    lead_steps = ds["lead_step"].values
    conf = cfg.results_analysis
    show = [ls for ls in conf.spatial_lead_steps if ls <= len(lead_steps)]
    n = len(show)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    fig.suptitle(f"ENSO 3.4 Scatter — {split}", fontsize=16, fontweight="bold")
    axes = np.atleast_2d(axes)
    for idx, ls in enumerate(show):
        r, c = divmod(idx, ncols); ax = axes[r, c]; li = ls - 1
        p = ep[:, li]; a = ea[:, li]
        ax.scatter(a, p, alpha=0.5, s=15, edgecolors="none")
        lims = [min(a.min(), p.min()) - 0.1, max(a.max(), p.max()) + 0.1]
        ax.plot(lims, lims, "r--", lw=1)
        corr = np.corrcoef(p, a)[0, 1] if len(p) > 2 else 0
        ax.set(title=f"Step {ls} (r={corr:.3f})", xlabel="Actual", ylabel="Predicted")
        ax.grid(alpha=0.3); ax.set_aspect("equal")
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols); axes[r, c].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=1.5, w_pad=1.5)
    pdf.savefig(fig, dpi=150); plt.close(fig)

def _plot_enso34_table(rows, split, pdf):
    fig, ax = plt.subplots(figsize=(14, max(4, len(rows) * 0.4 + 2))); ax.axis("off")
    headers = ["Lead\nStep", "fno\nRMSE", "Persist\nRMSE", "fno\nCorr", "Persist\nCorr", "fno\nBias"]
    cells = [[f"{r['lead_step']}", f"{r['enso_rmse']:.4f}", f"{r['persist_enso_rmse']:.4f}",
              f"{r['enso_corr']:.4f}", f"{r['persist_enso_corr']:.4f}", f"{r['enso_bias']:.4f}"] for r in rows]
    table = ax.table(cellText=cells, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.4)
    for (row, col), cell in table.get_celld().items():
        if row == 0: cell.set_facecolor("#2E7D32"); cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0: cell.set_facecolor("#C8E6C9")
    ax.set_title(f"ENSO 3.4 Metrics — {split}", fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout(); pdf.savefig(fig, dpi=150); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# TEXT REPORT & CSV
# ══════════════════════════════════════════════════════════════════════

def write_text_report(all_results, all_enso, report_path):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    lines = ["=" * 70, f"  FORECAST ANALYSIS REPORT — {cfg.project_name}", "=" * 70, ""]
    for split, rows in all_results.items():
        has_p = "persist_rmse" in rows[0]
        lines.append(f"── {split.upper()} — GLOBAL METRICS ──"); lines.append("")
        if has_p:
            hdr = f"  {'Lead':>4s}  {'fno RMSE':>10s}  {'Pers RMSE':>10s}  {'Skill%':>7s}  {'MAE':>9s}  {'Bias':>8s}  {'ACC':>9s}  {'Pers ACC':>9s}  {'R²':>7s}"
        else:
            hdr = f"  {'Lead':>4s}  {'RMSE':>10s}  {'MAE':>9s}  {'Bias':>8s}  {'ACC':>9s}  {'R²':>7s}"
        lines.append(hdr); lines.append("  " + "-" * (len(hdr) - 2))
        for r in rows:
            if has_p:
                lines.append(f"  {r['lead_step']:>4d}  {r['fno_rmse']:>10.4f}  {r['persist_rmse']:>10.4f}  {r['skill_score']:>6.1f}%  {r['fno_mae']:>9.4f}  {r['fno_bias']:>8.4f}  {r['fno_acc']:>9.4f}  {r['persist_acc']:>9.4f}  {r['fno_r2']:>7.4f}")
            else:
                lines.append(f"  {r['lead_step']:>4d}  {r['fno_rmse']:>10.4f}  {r['fno_mae']:>9.4f}  {r['fno_bias']:>8.4f}  {r['fno_acc']:>9.4f}  {r['fno_r2']:>7.4f}")
        lines.extend(["", ""])
    for split, rows in all_enso.items():
        lines.append(f"── {split.upper()} — ENSO 3.4 ──"); lines.append("")
        hdr = f"  {'Lead':>4s}  {'fno RMSE':>10s}  {'Pers RMSE':>10s}  {'fno Corr':>10s}  {'Pers Corr':>10s}  {'Bias':>8s}"
        lines.append(hdr); lines.append("  " + "-" * (len(hdr) - 2))
        for r in rows:
            lines.append(f"  {r['lead_step']:>4d}  {r['enso_rmse']:>10.4f}  {r['persist_enso_rmse']:>10.4f}  {r['enso_corr']:>10.4f}  {r['persist_enso_corr']:>10.4f}  {r['enso_bias']:>8.4f}")
        lines.extend(["", ""])
    report = "\n".join(lines)
    with open(report_path, "w") as f: f.write(report)
    logger.info(f"Report saved to {report_path}"); print(report)

def write_metrics_csv(all_results, all_enso, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    all_rows = []
    for split, rows in all_results.items():
        for r in rows: all_rows.append({"split": split, "type": "global", **r})
    for split, rows in all_enso.items():
        for r in rows: all_rows.append({"split": split, "type": "enso34", **r})
    if not all_rows: return
    keys = list(dict.fromkeys(k for r in all_rows for k in r))
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(all_rows)
    logger.info(f"CSV saved to {csv_path}")


# ══════════════════════════════════════════════════════════════════════
# ANALYSE ONE SPLIT
# ══════════════════════════════════════════════════════════════════════

def analyse_split(ds, split, enso_clim, acc_clim, destd, pdf):
    conf = cfg.results_analysis
    lat = ds["lat"].values; lon = ds["lon"].values

    logger.info(f"Analysing {split}: chains={ds.sizes['chain']}, "
                f"leads={ds.sizes['lead_step']}, grid={ds.sizes['lat']}x{ds.sizes['lon']}")

    ocean_mask = compute_ocean_mask(ds)
    lat_weights = compute_area_weights(lat)

    persistence = None
    if conf.compute_persistence:
        logger.info("  Computing persistence baseline")
        persistence = compute_persistence_baseline(ds)

    logger.info("  Computing metrics by lead step (with climatology-based ACC)")
    metrics = compute_metrics_by_lead(ds, acc_clim, lat_weights, persistence)

    valid_leads = [ls for ls in conf.spatial_lead_steps if ls <= ds.sizes["lead_step"]]
    logger.info(f"  Spatial metrics at: {valid_leads}")
    spatial = compute_spatial_metrics(ds, acc_clim, valid_leads)

    enso_rows = None; enso_pred = enso_actual = enso_persist = None
    if conf.compute_enso34 and enso_clim is not None:
        logger.info("  Computing ENSO 3.4")
        enso_pred, enso_actual, enso_persist = compute_enso34_for_chains(
            ds, enso_clim, lat, lon)
        enso_rows = compute_enso34_metrics(
            enso_pred, enso_actual, enso_persist, ds["lead_step"].values)

    if conf.generate_plots:
        logger.info("  Generating plots")
        _plot_metrics_table(metrics, split, pdf)
        _plot_metrics_vs_lead(metrics, split, pdf)
        _plot_bias_and_r2(metrics, split, pdf)
        if spatial:
            _plot_spatial_maps(spatial, lat, lon, ocean_mask, split, pdf)
        _plot_sample_chains(ds, split, conf.n_sample_chains, destd, ocean_mask, pdf)
        _plot_global_mean_chains(ds, persistence, split, conf.n_sample_chains, destd, pdf)
        if enso_rows is not None:
            _plot_enso34_table(enso_rows, split, pdf)
            _plot_enso34_skill(enso_rows, split, pdf)
            _plot_enso34_timeseries(enso_pred, enso_actual, enso_persist,
                                    ds, split, conf.enso34_n_sample_chains, pdf)
            _plot_enso34_scatter(enso_pred, enso_actual, ds, split, pdf)

    return metrics, enso_rows


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def run():
    print()
    logger.info("=" * 60)
    logger.info("PHASE 4: RESULTS ANALYSIS — START")
    logger.info("=" * 60)

    conf = cfg.results_analysis
    os.makedirs(conf.figures_dir, exist_ok=True)
    os.makedirs(conf.output_dir, exist_ok=True)

    all_results = {}; all_enso = {}

    logger.info("Loading de-standardisation parameters")
    destd = Destandardiser()

    # Daily climatology for ACC computation (shared across splits)
    logger.info("Computing daily climatology for ACC")
    try:
        acc_clim = compute_daily_climatology()
    except Exception as e:
        logger.warning(f"ACC climatology failed: {e}")
        acc_clim = None

    enso_clim = None
    if conf.compute_enso34:
        try:
            enso_clim = compute_enso34_climatology()
        except Exception as e:
            logger.warning(f"ENSO clim failed: {e}")

    pdf_path = conf.analysis_pdf
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        _plot_config_summary(pdf)

        if conf.analyse_validation and os.path.exists(conf.val_predictions_path):
            logger.info(f"Loading {conf.val_predictions_path}")
            ds = xr.open_dataset(conf.val_predictions_path, decode_timedelta=False)
            m, e = analyse_split(ds, "Validation", enso_clim, acc_clim, destd, pdf)
            all_results["Validation"] = m
            if e: all_enso["Validation"] = e
            ds.close()

        if conf.analyse_test and os.path.exists(conf.test_predictions_path):
            logger.info(f"Loading {conf.test_predictions_path}")
            ds = xr.open_dataset(conf.test_predictions_path, decode_timedelta=False)
            m, e = analyse_split(ds, "Test", enso_clim, acc_clim, destd, pdf)
            all_results["Test"] = m
            if e: all_enso["Test"] = e
            ds.close()

    logger.info(f"PDF saved to {pdf_path}")

    if all_results:
        write_text_report(all_results, all_enso, conf.report_file)
        write_metrics_csv(all_results, all_enso, conf.metrics_csv)

    logger.info("=" * 60)
    logger.info("PHASE 4: RESULTS ANALYSIS — COMPLETE")
    logger.info("=" * 60)
    print()


if __name__ == "__main__":
    run()
