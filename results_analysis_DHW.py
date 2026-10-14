"""
dhw_plots.py - DHW Prediction Analysis & Plotting
==================================================

Reads dhw_products_{split}.nc for both test and val splits,
writes a single combined PDF.

MAP RENDERING
-------------
Data uses 0-360 lon (T42 global grid, 128 points).
We avoid all Cartopy transform machinery for data rendering:
  - Plain Matplotlib axes with xlim=[-180,180], ylim=[lat_min,lat_max]
  - Data rolled to -180/+180 via _roll_to_180, rendered with imshow (no transform)
  - Coastlines drawn as plain Matplotlib lines from Cartopy Natural Earth geometry

CONFIGURATION
-------------
All paths and tunable parameters are read from config.DHWPlotsConfig via
the global cfg object.  To change any setting, edit config.py only -
do not hardcode values in this file.
"""

import os
import csv
import logging
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

from config import cfg

try:
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except (ImportError, Exception):
    HAS_CARTOPY = False

try:
    import regionmask
    HAS_REGIONMASK = True
except ImportError:
    HAS_REGIONMASK = False

logger = logging.getLogger("DHW_Plots")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(h)

if HAS_CARTOPY:
    logger.info("Cartopy available - coastlines will be drawn")


# =============================================================================
# LONGITUDE UTILITIES
# =============================================================================

def _wrap_lon(lon):
    """Convert 0-360 to -180/+180. Used for shapefile/mask calls only."""
    return np.where(lon > 180.0, lon - 360.0, lon)


def _roll_to_180(lon, data):
    """
    Roll a 0-360 lon array and the last axis of data so the grid runs
    -180 to +180.  Pivot is the first index where lon >= 180.
    Works correctly even when no point lands exactly on 180.
    """
    pivot  = int(np.searchsorted(lon, 180.0))
    lon_r  = np.concatenate([lon[pivot:] - 360.0, lon[:pivot]])
    data_r = np.concatenate([data[..., pivot:], data[..., :pivot]], axis=-1)
    return lon_r, data_r


def _roll_to_360(lon, data):
    """
    Keep lon in 0-360 order (no roll needed - data is already 0-360).
    Returns lon and data unchanged so _map_pcolormesh can use them
    with an axes whose xlim=(0, 360), matching NOAA's Pacific-centred view.
    """
    return lon.copy(), data.copy()


# =============================================================================
# MAP AXES + RENDERING
# =============================================================================

def _draw_geom_lines(ax, geom, **kw):
    """Recursively draw a Shapely geometry as plain Matplotlib lines."""
    gtype = geom.geom_type
    if gtype in ("GeometryCollection", "MultiLineString", "MultiPolygon"):
        for part in geom.geoms:
            _draw_geom_lines(ax, part, **kw)
    elif gtype == "LineString":
        ax.plot(*geom.xy, **kw)
    elif gtype == "Polygon":
        ax.plot(*geom.exterior.xy, **kw)


def _draw_geom_lines_360(ax, geom, **kw):
    """
    Draw a Shapely geometry (in -180/180 lon) onto axes with xlim=(0,360).
    Negative longitudes are shifted by +360 so they appear on the correct
    side of the Pacific-centred map.  Lines that cross the antimeridian are
    split so they don't draw a horizontal streak across the map.
    """
    gtype = geom.geom_type
    if gtype in ("GeometryCollection", "MultiLineString", "MultiPolygon"):
        for part in geom.geoms:
            _draw_geom_lines_360(ax, part, **kw)
        return

    if gtype == "LineString":
        coords = list(geom.coords)
    elif gtype == "Polygon":
        coords = list(geom.exterior.coords)
    else:
        return

    # Convert -180/180 → 0-360
    xs = [x + 360.0 if x < 0 else x for x, y in coords]
    ys = [y for x, y in coords]

    # Split at antimeridian jumps (>180 deg gap between consecutive points)
    segments_x, segments_y = [[]], [[]]
    for i in range(len(xs)):
        if i > 0 and abs(xs[i] - xs[i-1]) > 180:
            segments_x.append([])
            segments_y.append([])
        segments_x[-1].append(xs[i])
        segments_y[-1].append(ys[i])

    for sx, sy in zip(segments_x, segments_y):
        if len(sx) > 1:
            ax.plot(sx, sy, **kw)


def _create_map_ax(fig, nrows, ncols, idx, xlim=(0, 360), ylim=(-90, 90)):
    """
    Plain Matplotlib axes configured as a geographic map.

    xlim/ylim control the displayed window (pass data extent to zoom in).
    Coastlines are drawn as plain Matplotlib lines - no Cartopy transform -
    so they are clipped naturally by the axes limits.
    """
    ax = fig.add_subplot(nrows, ncols, idx)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    if HAS_CARTOPY:
        coast = cfeature.NaturalEarthFeature(
            "physical", "coastline", "110m",
            edgecolor="black", facecolor="none",
        )
        for geom in coast.geometries():
            # Natural Earth geometries use -180/180 lon; shift to 0-360
            # so they align with the Pacific-centred axes (xlim 0-360).
            _draw_geom_lines_360(ax, geom, color="black", linewidth=0.5, zorder=2)

    # Reef boxes are drawn after coastlines so they appear on top.
    # _draw_reef_boxes returns legend handles but we discard them here;
    # callers that need the legend call it separately on their axes.
    _draw_reef_boxes(ax)

    return ax


def _map_pcolormesh(ax, lon, lat, data, **kw):
    """
    Render 0-360 gridded data as an image on a plain Matplotlib axes.

    The axes xlim is (0, 360) so the map is centred on 180 deg (the Pacific),
    matching NOAA Coral Reef Watch's standard projection.  No lon roll is
    needed - the data is already in 0-360 order.

    Lat is handled so row 0 = south regardless of storage order.
    """
    cmap = kw.pop("cmap",    None)
    vmin = kw.pop("vmin",    None)
    vmax = kw.pop("vmax",    None)
    kw.pop("shading", None)

    # Keep data in native 0-360 lon order - axes xlim=(0,360) handles display
    lon_r  = lon.copy()
    data_r = data.copy()

    # Half-cell offsets so extent covers cell edges, not centres
    dlon = float(abs(lon_r[1] - lon_r[0])) if len(lon_r) > 1 else 1.0
    dlat = float(abs(lat[1]  - lat[0]))    if len(lat)   > 1 else 1.0
    x0 = float(lon_r[0])  - dlon / 2.0
    x1 = float(lon_r[-1]) + dlon / 2.0
    y0 = float(min(lat[0], lat[-1])) - dlat / 2.0
    y1 = float(max(lat[0], lat[-1])) + dlat / 2.0

    # Ensure row 0 = southernmost lat for imshow(origin="lower")
    lat_south_to_north = lat[0] < lat[-1]
    if not lat_south_to_north:
        data_r = data_r[::-1, :]

    return ax.imshow(
        data_r,
        origin="lower",
        extent=[x0, x1, y0, y1],
        cmap=cmap, vmin=vmin, vmax=vmax,
        interpolation="nearest",
        aspect="auto",
        zorder=0,
    )


def _data_map_extent(lon, lat, pad=2.0):
    """
    Compute (xlim, ylim) in 0-360 lon space that fits the data domain
    with padding, clamped to [0, 360] x [-90, 90].
    This matches NOAA's Pacific-centred (0-360) projection.
    """
    dlon = float(abs(lon[1] - lon[0])) if len(lon) > 1 else 1.0
    dlat = float(abs(lat[1] - lat[0])) if len(lat) > 1 else 1.0
    x0 = max(float(lon[0])  - dlon / 2.0 - pad,   0.0)
    x1 = min(float(lon[-1]) + dlon / 2.0 + pad,  360.0)
    y0 = max(float(min(lat[0], lat[-1])) - dlat / 2.0 - pad, -90.0)
    y1 = min(float(max(lat[0], lat[-1])) + dlat / 2.0 + pad,  90.0)
    return (x0, x1), (y0, y1)


def _add_shared_colorbar(fig, im, label="", orientation="horizontal"):
    cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.018])
    fig.colorbar(im, cax=cbar_ax, orientation=orientation, label=label)


def _draw_reef_boxes(ax):
    """
    Draw labelled bounding boxes for major coral reef systems on a map axes
    whose xlim is in 0-360 lon space.

    Box coordinates, colours, and labels are defined here and in
    cfg.dhw_plots.reef_boxes so they can be overridden from config.py.
    Returns a list of (patch, label) tuples suitable for building a legend.
    """
    import matplotlib.patches as mpatches
    conf = cfg.dhw_plots

    legend_handles = []
    for reef in conf.reef_boxes:
        name   = reef["name"]
        lat0   = reef["lat"][0]
        lat1   = reef["lat"][1]
        lon0   = reef["lon"][0]
        lon1   = reef["lon"][1]
        colour = reef["colour"]

        width  = lon1 - lon0
        height = lat1 - lat0

        rect = mpatches.FancyBboxPatch(
            (lon0, lat0), width, height,
            boxstyle="square,pad=0",
            linewidth=1.8,
            edgecolor=colour,
            facecolor="none",
            zorder=5,
        )
        ax.add_patch(rect)

        # Proxy artist for legend (filled square, same colour)
        proxy = mpatches.Patch(facecolor=colour, edgecolor=colour,
                               alpha=0.7, label=name)
        legend_handles.append(proxy)

    return legend_handles


def _make_cmap(cmap_cfg):
    """
    Resolve a colormap from config.

    Accepts either:
      - A string  -> returned as-is (Matplotlib named colormap, e.g. "YlOrRd")
      - A list    -> interpolated into a smooth LinearSegmentedColormap.
                     Any number of hex colour strings is supported.

    Usage in config.py:
      dhw_cmap   = "plasma"                              # built-in
      dhw_cmap   = ["#FFFFFF", "#FFD700", "#8B0000"]     # 3-stop custom
      error_cmap = ["#053061", "#F7F7F7", "#67001F"]     # diverging custom
    """
    import matplotlib.colors as mcolors
    if isinstance(cmap_cfg, str):
        return cmap_cfg
    colours = [mcolors.to_rgb(c) for c in cmap_cfg]
    return mcolors.LinearSegmentedColormap.from_list("custom", colours)


def _valid_chain_indices(ds):
    """
    Return indices of chains that have at least one finite DHW value.
    Chain 0 is commonly all-NaN when the first init date falls before
    sufficient actual SST history is available for the sliding window.
    """
    dhw = ds["predicted_dhw"].values   # (chain, lead, lat, lon)
    valid = [c for c in range(dhw.shape[0]) if np.isfinite(dhw[c]).any()]
    if len(valid) < dhw.shape[0]:
        skipped = [c for c in range(dhw.shape[0]) if c not in valid]
        logger.warning(f"  Skipping all-NaN chains: {skipped}")
    return valid


def _resolve_chain_indices(init_times, n_chains, valid_chains=None):
    """
    Return the list of chain indices to plot.

    If cfg.dhw_plots.plot_init_dates is non-empty, find the chain whose
    init_time is closest to each requested date (nearest-date matching).
    Duplicate matches and all-NaN chains are skipped.

    If plot_init_dates is empty, fall back to n_sample_chains evenly-spaced
    indices drawn only from valid (non-NaN) chains.
    """
    conf = cfg.dhw_plots

    # Restrict to chains that actually have data
    pool = valid_chains if valid_chains is not None else list(range(n_chains))

    if conf.plot_init_dates:
        requested = [np.datetime64(d, "D") for d in conf.plot_init_dates]
        init_days = init_times.astype("datetime64[D]")
        indices = []
        for req in requested:
            # Only consider valid chains when finding nearest date
            diffs = np.abs((init_days - req).astype(np.int64))
            # Set invalid chains to a very large distance so they are never chosen
            diffs_filtered = diffs.copy().astype(float)
            for c in range(n_chains):
                if c not in pool:
                    diffs_filtered[c] = np.inf
            best = int(np.argmin(diffs_filtered))
            actual_date = str(init_days[best])
            if best not in indices:
                indices.append(best)
                logger.info(f"  Date {req} -> chain {best} (init {actual_date})")
            else:
                logger.warning(f"  Date {req} -> chain {best} already selected, skipping")
        return indices

    # Default: evenly-spaced sample from valid chains only
    pool_arr = np.array(pool)
    n_select = min(conf.n_sample_chains, len(pool_arr))
    idx = np.round(np.linspace(0, len(pool_arr) - 1, n_select)).astype(int)
    return list(pool_arr[idx])


# =============================================================================
# CONFIGURATION SUMMARY PAGE
# =============================================================================

def _plot_config_summary(pdf):
    """Generate a multi-page configuration summary at the start of the PDF."""
    from dataclasses import fields
    from datetime import datetime

    fig, ax = plt.subplots(figsize=(11, 16))
    ax.axis("off")

    lines = []
    lines.append("DHW ANALYSIS - CONFIGURATION SUMMARY")
    lines.append(f"Project: {cfg.project_name}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 72)
    lines.append("")

    # Iterate over all config dataclasses relevant to the DHW pipeline
    config_sections = [
        ("Data Preparation", cfg.data_preparation),
        ("Model Training",   cfg.model_training),
        ("Model Inference",  cfg.model_inference),
        ("Concat Chains",    cfg.concat_chains),
        ("Calc DHW",         cfg.calc_dhw),
        ("DHW Plots",        cfg.dhw_plots),
    ]

    for section_name, section_cfg in config_sections:
        lines.append(f"-- {section_name} --")
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
            elif isinstance(val, (dict, list)):
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


# =============================================================================
# OCEAN MASK
# =============================================================================

def compute_ocean_mask(ds):
    """
    Boolean (lat, lon) array: True = ocean.

    regionmask and shapefiles expect monotone -180/+180 lon.
    We sort the wrapped lon, build the mask, then unsort back to the
    original column order so mask[:,j] matches data[:,j].
    """
    lat = ds["lat"].values
    lon = ds["lon"].values
    lon_w = _wrap_lon(lon)

    sort_idx     = np.argsort(lon_w)
    unsort_idx   = np.argsort(sort_idx)
    lon_w_sorted = lon_w[sort_idx]

    if HAS_REGIONMASK:
        try:
            logger.info("  Ocean mask via regionmask")
            try:
                land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
            except AttributeError:
                try:
                    land = regionmask.defined_regions.natural_earth.land_110
                except AttributeError:
                    land = regionmask.defined_regions.natural_earth_v4_1_0.land_110

            land_mask    = land.mask(lon_w_sorted, lat)
            ocean_sorted = np.isnan(land_mask.values)
            ocean        = ocean_sorted[:, unsort_idx]
            logger.info(f"  Ocean pixels: {ocean.sum():,} / {ocean.size:,}")
            return ocean
        except Exception as e:
            logger.warning(f"  regionmask failed: {e}")

    if HAS_CARTOPY:
        try:
            from matplotlib.path import Path as MplPath
            from cartopy.io.shapereader import natural_earth
            import cartopy.io.shapereader as shpreader

            logger.info("  Ocean mask via cartopy rasterisation")
            shpfile = natural_earth(resolution="110m", category="physical", name="land")
            reader  = shpreader.Reader(shpfile)
            lon_g, lat_g = np.meshgrid(lon_w, lat)
            points = np.column_stack([lon_g.ravel(), lat_g.ravel()])
            land = np.zeros(len(points), dtype=bool)
            for geom in reader.geometries():
                polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
                for poly in polys:
                    land |= MplPath(np.array(poly.exterior.coords)).contains_points(points)
            ocean = ~land.reshape(len(lat), len(lon))
            logger.info(f"  Ocean pixels: {ocean.sum():,} / {ocean.size:,}")
            return ocean
        except Exception as e:
            logger.warning(f"  Cartopy mask failed: {e}")

    logger.info("  Ocean mask via variance fallback")
    std   = np.nanstd(ds["actual_dhw"].values, axis=(0, 1))
    ocean = std > 0.01
    logger.info(f"  Ocean pixels: {ocean.sum():,} / {ocean.size:,}")
    return ocean


def _mask_land(data, ocean_mask):
    """Return copy of data with land pixels set to NaN."""
    out = data.copy().astype(float)
    out[~ocean_mask] = np.nan
    return out


def _subset_ds_to_region(ds, lat_bounds, lon_bounds):
    """
    Return a copy of ds spatially subsetted to a lat/lon bounding box.

    Parameters
    ----------
    ds         : xr.Dataset with lat and lon coordinates
    lat_bounds : (lat_min, lat_max) in degrees
    lon_bounds : (lon_min, lon_max) in 0-360 degrees

    Returns
    -------
    xr.Dataset - same variables, reduced spatial extent.
    All (chain, lead, lat, lon) and (lat, lon) variables are sliced.
    """
    lat = ds["lat"].values
    lon = ds["lon"].values

    lat_mask = (lat >= lat_bounds[0]) & (lat <= lat_bounds[1])
    lon_mask = (lon >= lon_bounds[0]) & (lon <= lon_bounds[1])

    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]

    if len(lat_idx) == 0 or len(lon_idx) == 0:
        raise ValueError(
            f"No grid points found in region lat={lat_bounds}, lon={lon_bounds}. "
            f"Grid lat: {lat.min():.1f}–{lat.max():.1f}, "
            f"lon: {lon.min():.1f}–{lon.max():.1f}"
        )

    return ds.isel(lat=lat_idx, lon=lon_idx)


# =============================================================================
# METRICS
# =============================================================================

def _rmse(p, a): return float(np.sqrt(np.nanmean((p - a) ** 2)))
def _mae(p, a):  return float(np.nanmean(np.abs(p - a)))
def _bias(p, a): return float(np.nanmean(p - a))
def _r2(p, a):
    ss_res = np.nansum((a - p) ** 2)
    ss_tot = np.nansum((a - np.nanmean(a)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

def _acc(pred, actual):
    """
    Anomaly Correlation Coefficient across chains and space.

    Uses the time-mean of actual DHW as the climatology (simple centering),
    which is appropriate when no external climatology is available.
    """
    clim = np.nanmean(actual, axis=0, keepdims=True)   # (1, lat, lon)
    f_anom = pred   - clim
    a_anom = actual - clim
    num  = float(np.nansum(f_anom * a_anom))
    den  = float(np.sqrt(np.nansum(f_anom**2) * np.nansum(a_anom**2)))
    return float(num / den) if den > 0 else 0.0


def _compute_persistence_dhw(ds):
    """
    DHW persistence baseline: hold the DHW value at lead step 1 constant
    across all lead steps.

    This is the natural DHW persistence - if you expect no further heat
    stress beyond what has already accumulated, the DHW stays flat.

    Returns
    -------
    persist : np.ndarray, same shape as predicted_dhw (chain, lead, lat, lon)
    """
    actual = ds["actual_dhw"].values    # (chain, lead, lat, lon)
    # Use lead index 0 (lead step 1) as the anchor
    anchor = actual[:, 0:1, :, :]      # (chain, 1, lat, lon)
    n_leads = actual.shape[1]
    return np.tile(anchor, (1, n_leads, 1, 1))


def _compute_metrics_by_lead(ds, persistence):
    """
    Compute per-lead-step metrics for both the model and persistence baseline.

    Returns a list of dicts with keys:
      lead_step, rmse, mae, bias, acc, r2,
      persist_rmse, persist_mae, persist_acc, skill_score
    """
    pred   = ds["predicted_dhw"].values   # (chain, lead, lat, lon)
    actual = ds["actual_dhw"].values
    leads  = ds["lead_step"].values

    rows = []
    for i, ls in enumerate(leads):
        p  = pred[:,   i]    # (chain, lat, lon)
        a  = actual[:, i]
        pe = persistence[:, i]

        p_rmse  = _rmse(p,  a)
        pe_rmse = _rmse(pe, a)
        skill   = (1.0 - p_rmse / pe_rmse) * 100.0 if pe_rmse > 0 else 0.0

        rows.append({
            "lead_step":    int(ls),
            "rmse":         p_rmse,
            "mae":          _mae(p,  a),
            "bias":         _bias(p, a),
            "acc":          _acc(p,  a),
            "r2":           _r2(p,   a),
            "persist_rmse": pe_rmse,
            "persist_mae":  _mae(pe, a),
            "persist_acc":  _acc(pe, a),
            "skill_score":  skill,
        })
    return rows


# =============================================================================
# PAGE 1 - TITLE
# =============================================================================

def _page_title(pdf, splits_info):
    conf = cfg.dhw_plots
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis("off")
    lines = [
        "DEGREE HEATING WEEKS (DHW) - COMBINED ANALYSIS",
        "=" * 68, "",
        "METHODOLOGY  (NOAA Coral Reef Watch v3.1)",
        "  SSTA      = SST - DC              (DC = daily climatology)",
        "  HotSpot   = max(SST - MMM, 0)     (MMM = Maximum Monthly Mean)",
        f"  DHW_i     = (1/7) x sum(HotSpot_j) for j=i-83..i, "
        f"HotSpot_j >= {cfg.calc_dhw.hotspot_threshold}C",
        "",
        "SLIDING WINDOW CONSTRUCTION",
        "  Lead L:  (84-L) actual + L predicted SST days",
        "  Lead 1:  83 actual +  1 predicted",
        "  Lead 30: 54 actual + 30 predicted",
        "  Lead 84:  0 actual + 84 predicted",
        "  Lead 90: previous 84 predicted days (days 7-90)",
        "",
        "BLEACHING ALERT THRESHOLDS",
    ]
    for lo, hi, label, _ in conf.bleaching_thresholds:
        hi_str = f"{hi}" if hi < 999 else ">="
        lines.append(f"  DHW {lo:>2}-{hi_str:<3}: {label}")
    lines += ["", "DATASETS"]
    for name, n_chains, n_leads, nlat, nlon in splits_info:
        lines.append(f"  {name:12s}  {n_chains} chains x {n_leads} leads x {nlat} lat x {nlon} lon")
    ax.text(0.05, 0.97, "\n".join(lines), transform=ax.transAxes,
            fontsize=11, verticalalignment="top", fontfamily="monospace", linespacing=1.4)
    ax.set_title("DHW Analysis - Summary", fontsize=16, fontweight="bold", pad=15)
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


# =============================================================================
# PAGE 2 - PER-LEAD METRICS TABLE
# =============================================================================

def _page_metrics_table(rows, split, pdf):
    """Tabular summary including persistence baseline columns."""
    fig, ax = plt.subplots(figsize=(18, max(4, len(rows) * 0.4 + 2)))
    ax.axis("off")
    headers = [
        "Lead\nStep", "SFNO\nRMSE", "Persist\nRMSE", "Skill\n%",
        "SFNO\nMAE", "Persist\nMAE", "SFNO\nBias",
        "SFNO\nACC", "Persist\nACC", "SFNO\nR²",
    ]
    cells = [[
        str(r["lead_step"]),
        f"{r['rmse']:.4f}",    f"{r['persist_rmse']:.4f}",
        f"{r['skill_score']:.1f}%",
        f"{r['mae']:.4f}",     f"{r['persist_mae']:.4f}",
        f"{r['bias']:.4f}",
        f"{r['acc']:.4f}",     f"{r['persist_acc']:.4f}",
        f"{r['r2']:.4f}",
    ] for r in rows]
    table = ax.table(cellText=cells, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.4)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#D6E4F0")
    ax.set_title(f"DHW Metrics by Lead Step - {split}", fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


# =============================================================================
# PAGE 3 - METRICS VS LEAD STEP
# =============================================================================

def _page_metrics_vs_lead(rows, split, pdf):
    """
    2x3 grid matching results_analysis.py style:
      RMSE | MAE | ACC
      Bias | R²  | RMSE Skill vs Persistence
    Both model and persistence lines shown on RMSE, MAE, ACC panels.
    """
    leads = [r["lead_step"] for r in rows]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"DHW Forecast Skill vs Lead Step - {split}",
                 fontsize=16, fontweight="bold")

    # Colour palette: SFNO = solid darker colour, Persistence = dashed lighter contrast colour
    C_SFNO    = {"rmse": "#1B4F8A", "mae": "#B84C00", "acc": "#5B2182"}   # dark blue / burnt orange / dark purple
    C_PERSIST = {"rmse": "#E63946", "mae": "#2CA02C", "acc": "#FF7F0E"}   # red / green / orange

    # ── RMSE ──
    ax = axes[0, 0]
    ax.plot(leads, [r["rmse"]         for r in rows], "o-",  ms=4, lw=2.5,
            color=C_SFNO["rmse"],    label="SFNO")
    ax.plot(leads, [r["persist_rmse"] for r in rows], "s--", ms=4, lw=2,
            color=C_PERSIST["rmse"], label="Persistence")
    ax.set(xlabel="Lead step", ylabel="RMSE (C-weeks)", title="RMSE")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ── MAE ──
    ax = axes[0, 1]
    ax.plot(leads, [r["mae"]         for r in rows], "o-",  ms=4, lw=2.5,
            color=C_SFNO["mae"],    label="SFNO")
    ax.plot(leads, [r["persist_mae"] for r in rows], "s--", ms=4, lw=2,
            color=C_PERSIST["mae"], label="Persistence")
    ax.set(xlabel="Lead step", ylabel="MAE (C-weeks)", title="MAE")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ── ACC ──
    ax = axes[0, 2]
    ax.plot(leads, [r["acc"]         for r in rows], "o-",  ms=4, lw=2.5,
            color=C_SFNO["acc"],    label="SFNO")
    ax.plot(leads, [r["persist_acc"] for r in rows], "s--", ms=4, lw=2,
            color=C_PERSIST["acc"], label="Persistence")
    ax.set(xlabel="Lead step", ylabel="ACC", title="ACC")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ── Bias ──
    ax = axes[1, 0]
    ax.plot(leads, [r["bias"] for r in rows], "o-", ms=3, lw=2, color="crimson")
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set(xlabel="Lead step", ylabel="Bias (C-weeks)", title="Mean Bias")
    ax.grid(alpha=0.3)

    # ── R² ──
    ax = axes[1, 1]
    ax.plot(leads, [r["r2"] for r in rows], "o-", ms=3, lw=2, color="green")
    ax.set(xlabel="Lead step", ylabel="R²", title="R²")
    ax.grid(alpha=0.3)

    # ── RMSE Skill vs Persistence ──
    ax = axes[1, 2]
    skill = [r["skill_score"] for r in rows]
    colours = ["steelblue" if s >= 0 else "firebrick" for s in skill]
    ax.bar(leads, skill, color=colours, edgecolor="black", alpha=0.8, width=0.8)
    ax.axhline(0, color="red", ls="--", lw=1)
    ax.set(xlabel="Lead step", ylabel="Skill (%)",
           title="RMSE Skill vs Persistence")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


# =============================================================================
# PAGE 4 - SPATIAL RMSE & BIAS MAPS
# =============================================================================

def _page_spatial_maps(ds, ocean_mask, split, pdf):
    conf = cfg.dhw_plots
    pred, actual  = ds["predicted_dhw"].values, ds["actual_dhw"].values
    all_leads     = ds["lead_step"].values
    lat, lon      = ds["lat"].values, ds["lon"].values
    valid_leads   = [ls for ls in conf.spatial_lead_steps if ls <= len(all_leads)]

    spatial = {}
    for ls in valid_leads:
        idx_arr = np.where(all_leads == ls)[0]
        if not len(idx_arr): continue
        i = idx_arr[0]
        p, a = pred[:, i], actual[:, i]
        spatial[int(ls)] = {
            "rmse": np.sqrt(np.nanmean((p - a) ** 2, axis=0)),
            "bias": np.nanmean(p - a, axis=0),
        }
    if not spatial: return

    xlim, ylim = _data_map_extent(lon, lat)

    def _draw_panel(field_key, cmap, title_prefix, label,
                    vmin=None, vmax=None, symmetric=False, mask=None):
        n, ncols = len(valid_leads), 3
        nrows = (n + ncols - 1) // ncols
        fig = plt.figure(figsize=(5.5 * ncols, 3.5 * nrows + 1.5))
        fig.suptitle(f"{title_prefix} - {split}", fontsize=16, fontweight="bold", y=0.98)

        vals = [_mask_land(spatial[ls][field_key], mask) if mask is not None
                else spatial[ls][field_key] for ls in valid_leads]
        flat = np.concatenate([v.ravel() for v in vals])
        flat = flat[np.isfinite(flat)]
        if symmetric:
            ab = float(np.percentile(np.abs(flat), 98)) if len(flat) else 1.0
            vmin, vmax = -ab, ab
        else:
            if vmin is None: vmin = float(np.nanmin(flat))
            if vmax is None: vmax = float(np.nanmax(flat))

        im = None
        for k, ls in enumerate(valid_leads):
            ax = _create_map_ax(fig, nrows, ncols, k + 1, xlim=xlim, ylim=ylim)
            d = _mask_land(spatial[ls][field_key], mask) if mask is not None else spatial[ls][field_key]
            im = _map_pcolormesh(ax, lon, lat, d, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
            ax.set_title(f"Step {ls}")

        for k in range(n, nrows * ncols):
            fig.add_subplot(nrows, ncols, k + 1).set_visible(False)

        fig.subplots_adjust(left=0.03, right=0.97, top=0.94, bottom=0.12, wspace=0.08, hspace=0.15)
        if im is not None: _add_shared_colorbar(fig, im, label=label)
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    _draw_panel("rmse", _make_cmap(conf.rmse_cmap),  "Spatial DHW RMSE (all grid)",   "RMSE (C-weeks)")
    _draw_panel("bias", _make_cmap(conf.error_cmap), "Spatial DHW Bias (all grid)",   "Bias (C-weeks)", symmetric=True)
    _draw_panel("rmse", _make_cmap(conf.rmse_cmap),  "Spatial DHW RMSE (ocean only)", "RMSE (C-weeks)", mask=ocean_mask)
    _draw_panel("bias", _make_cmap(conf.error_cmap), "Spatial DHW Bias (ocean only)", "Bias (C-weeks)", symmetric=True, mask=ocean_mask)


# =============================================================================
# PAGE 5 - PER-CHAIN SPATIAL MAPS
# =============================================================================

def _page_sample_chains(ds, ocean_mask, split, pdf, valid_chains=None):
    conf = cfg.dhw_plots
    pred, actual   = ds["predicted_dhw"].values, ds["actual_dhw"].values
    init_times     = ds["init_time"].values
    lead_steps     = ds["lead_step"].values
    lat, lon       = ds["lat"].values, ds["lon"].values
    n_chains       = pred.shape[0]
    indices        = _resolve_chain_indices(init_times, n_chains, valid_chains=valid_chains)
    show           = [0] + ([len(lead_steps) // 2] if len(lead_steps) > 2 else []) + [len(lead_steps) - 1]
    xlim, ylim     = _data_map_extent(lon, lat)

    for ci in indices:
        fig = plt.figure(figsize=(18, 4 * len(show) + 1.5))
        fig.suptitle(f"DHW Chain init: {str(init_times[ci])[:10]} - {split} (C-weeks)",
                     fontsize=16, fontweight="bold", y=0.98)

        for row, li in enumerate(show):
            ls   = lead_steps[li]
            p    = _mask_land(pred[ci, li].copy(),   ocean_mask)
            a    = _mask_land(actual[ci, li].copy(), ocean_mask)
            diff = p - a
            fd   = diff[np.isfinite(diff)]
            eabs = min(float(np.percentile(np.abs(fd), 98)) if len(fd) else conf.dhw_diff_vmax,
                       conf.dhw_diff_vmax)

            ax = _create_map_ax(fig, len(show), 3, row * 3 + 1, xlim=xlim, ylim=ylim)
            im_dhw = _map_pcolormesh(ax, lon, lat, a, cmap=_make_cmap(conf.dhw_cmap),
                                     vmin=0, vmax=conf.dhw_vmax, shading="auto")
            ax.set_title(f"Actual DHW - Step {ls}")

            ax = _create_map_ax(fig, len(show), 3, row * 3 + 2, xlim=xlim, ylim=ylim)
            _map_pcolormesh(ax, lon, lat, p, cmap=_make_cmap(conf.dhw_cmap),
                            vmin=0, vmax=conf.dhw_vmax, shading="auto")
            ax.set_title(f"Predicted DHW - Step {ls}")

            ax = _create_map_ax(fig, len(show), 3, row * 3 + 3, xlim=xlim, ylim=ylim)
            im_err = _map_pcolormesh(ax, lon, lat, diff, cmap=_make_cmap(conf.error_cmap),
                                     vmin=-eabs, vmax=eabs, shading="auto")
            ax.set_title(f"Error (Pred - Actual) - Step {ls}")

            if row == len(show) - 1:
                fig.colorbar(im_dhw,
                             cax=fig.add_axes([0.05, 0.06, 0.58, 0.018]),
                             orientation="horizontal", label="DHW (C-weeks)")
                fig.colorbar(im_err,
                             cax=fig.add_axes([0.68, 0.06, 0.27, 0.018]),
                             orientation="horizontal", label="Error (C-weeks)")

        fig.subplots_adjust(left=0.03, right=0.97, top=0.94, bottom=0.12, wspace=0.08, hspace=0.15)
        pdf.savefig(fig, dpi=150)
        plt.close(fig)


# =============================================================================
# PAGE 6 - GLOBAL-MEAN DHW TIME SERIES
# =============================================================================

def _page_global_mean_chains(ds, split, pdf, valid_chains=None):
    pred, actual = ds["predicted_dhw"].values, ds["actual_dhw"].values
    init_times   = ds["init_time"].values
    lead_steps   = ds["lead_step"].values
    indices      = _resolve_chain_indices(init_times, pred.shape[0], valid_chains=valid_chains)
    ncols        = 3
    nrows        = (len(indices) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.5 * nrows + 1))
    fig.suptitle(f"Global-Mean DHW - {split} (C-weeks)", fontsize=16, fontweight="bold")
    axes = np.atleast_2d(axes)

    for i, ci in enumerate(indices):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        ax.plot(lead_steps, np.nanmean(actual[ci], axis=(1, 2)), "k-",  lw=2, label="Actual")
        ax.plot(lead_steps, np.nanmean(pred[ci],   axis=(1, 2)), "r--", lw=2, label="Predicted")
        ax.set_title(f"Init: {str(init_times[ci])[:10]}", fontsize=10)
        if c == 0: ax.set_ylabel("DHW (C-weeks)")
        if r == nrows - 1: ax.set_xlabel("Lead step")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    for i in range(len(indices), nrows * ncols):
        r, c = divmod(i, ncols); axes[r, c].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.0, w_pad=2.0)
    pdf.savefig(fig, dpi=150); plt.close(fig)


# =============================================================================
# PAGE 7 - TROPICAL-MEAN DHW TIME SERIES
# =============================================================================

def _page_tropical_mean_chains(ds, split, pdf, valid_chains=None):
    conf = cfg.dhw_plots
    pred, actual = ds["predicted_dhw"].values, ds["actual_dhw"].values
    init_times   = ds["init_time"].values
    lead_steps   = ds["lead_step"].values
    lat          = ds["lat"].values
    trop         = np.where(np.abs(lat) <= conf.tropic_lat)[0]
    if not len(trop): return

    pred_t, actual_t = pred[:, :, trop, :], actual[:, :, trop, :]
    indices = _resolve_chain_indices(init_times, pred.shape[0], valid_chains=valid_chains)
    ncols   = 3
    nrows   = (len(indices) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.5 * nrows + 1))
    fig.suptitle(f"Tropical-Mean DHW (|lat|<={conf.tropic_lat:.0f}) - {split} (C-weeks)",
                 fontsize=16, fontweight="bold")
    axes = np.atleast_2d(axes)

    for i, ci in enumerate(indices):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        ax.plot(lead_steps, np.nanmean(actual_t[ci], axis=(1, 2)), "k-",  lw=2, label="Actual")
        ax.plot(lead_steps, np.nanmean(pred_t[ci],   axis=(1, 2)), "r--", lw=2, label="Predicted")
        ax.set_title(f"Init: {str(init_times[ci])[:10]}", fontsize=10)
        if c == 0: ax.set_ylabel("DHW (C-weeks)")
        if r == nrows - 1: ax.set_xlabel("Lead step")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    for i in range(len(indices), nrows * ncols):
        r, c = divmod(i, ncols); axes[r, c].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.0, w_pad=2.0)
    pdf.savefig(fig, dpi=150); plt.close(fig)


# =============================================================================
# PAGE 8 - PEAK DHW MAP
# =============================================================================

def _page_max_dhw_map(ds, ocean_mask, split, pdf):
    conf = cfg.dhw_plots
    pred, actual = ds["predicted_dhw"].values, ds["actual_dhw"].values
    lat, lon     = ds["lat"].values, ds["lon"].values

    pred_max   = _mask_land(np.nanmax(pred,   axis=(0, 1)), ocean_mask)
    actual_max = _mask_land(np.nanmax(actual, axis=(0, 1)), ocean_mask)
    diff_max   = _mask_land(pred_max - actual_max, ocean_mask)
    fd         = diff_max[np.isfinite(diff_max)]
    eabs       = float(np.nanpercentile(np.abs(fd), 98)) if len(fd) else conf.dhw_diff_vmax

    xlim, ylim = _data_map_extent(lon, lat)
    fig = plt.figure(figsize=(18, 6.5))
    fig.suptitle(f"Peak DHW (max over all chains & leads) - {split} (C-weeks)",
                 fontsize=15, fontweight="bold", y=0.98)

    ax = _create_map_ax(fig, 1, 3, 1, xlim=xlim, ylim=ylim)
    im_a = _map_pcolormesh(ax, lon, lat, actual_max, cmap=_make_cmap(conf.dhw_cmap),
                           vmin=0, vmax=conf.dhw_vmax, shading="auto")
    ax.set_title("Actual - Max DHW")

    ax = _create_map_ax(fig, 1, 3, 2, xlim=xlim, ylim=ylim)
    _map_pcolormesh(ax, lon, lat, pred_max, cmap=_make_cmap(conf.dhw_cmap),
                    vmin=0, vmax=conf.dhw_vmax, shading="auto")
    ax.set_title("Predicted - Max DHW")

    ax = _create_map_ax(fig, 1, 3, 3, xlim=xlim, ylim=ylim)
    im_e = _map_pcolormesh(ax, lon, lat, diff_max, cmap=_make_cmap(conf.error_cmap),
                           vmin=-eabs, vmax=eabs, shading="auto")
    ax.set_title("Max DHW Error (Pred - Actual)")

    fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.18, wspace=0.08)
    fig.colorbar(im_a, cax=fig.add_axes([0.05, 0.08, 0.58, 0.025]),
                 orientation="horizontal", label="DHW (C-weeks)")
    fig.colorbar(im_e, cax=fig.add_axes([0.68, 0.08, 0.27, 0.025]),
                 orientation="horizontal", label="Error (C-weeks)")
    # ── Reef legend ──────────────────────────────────────────────────────────
    # Build legend from proxy patches (same colours as the boxes on the map).
    import matplotlib.patches as mpatches
    conf = cfg.dhw_plots
    handles = [
        mpatches.Patch(facecolor=r["colour"], edgecolor=r["colour"],
                       alpha=0.85, label=r["name"])
        for r in conf.reef_boxes
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(4, len(handles)),
        bbox_to_anchor=(0.5, -0.02),
        fontsize=8,
        framealpha=0.9,
        title="Major Coral Reef Systems",
        title_fontsize=9,
    )
    fig.subplots_adjust(bottom=0.22)   # make room for legend below colorbars
    # ─────────────────────────────────────────────────────────────────────────

    pdf.savefig(fig, dpi=150); plt.close(fig)


# =============================================================================
# PAGE 9 - BLEACHING ALERT DISTRIBUTION
# =============================================================================

def _page_bleaching_distribution(ds, ocean_mask, split, pdf):
    conf = cfg.dhw_plots
    pred, actual = ds["predicted_dhw"].values, ds["actual_dhw"].values
    pf = pred[:, :, ocean_mask].ravel();   pf = pf[np.isfinite(pf)]
    af = actual[:, :, ocean_mask].ravel(); af = af[np.isfinite(af)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Bleaching Alert Level Distribution - {split}", fontsize=15, fontweight="bold")

    for ax, data, label in zip(axes, [af, pf], ["Actual", "Predicted"]):
        total  = len(data)
        cats   = [t[2] for t in conf.bleaching_thresholds]
        colors = [t[3] for t in conf.bleaching_thresholds]
        vals   = [((data >= lo) & (data < hi)).sum() / total * 100 if total else 0.0
                  for lo, hi, _, _ in conf.bleaching_thresholds]
        bars = ax.bar(cats, vals, color=colors, edgecolor="black", linewidth=1.0)
        ax.set_ylabel("Percentage (%)"); ax.set_title(label); ax.set_ylim(0, 110)
        ax.tick_params(axis="x", rotation=15); ax.grid(axis="y", alpha=0.3, linestyle="--")
        for bar, val in zip(bars, vals):
            if val > 0.3:
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1.5,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, dpi=150); plt.close(fig)


# =============================================================================
# PAGE 10 - AREA UNDER STRESS
# =============================================================================

def _page_area_under_stress(ds, ocean_mask, split, pdf):
    conf = cfg.dhw_plots
    pred, actual = ds["predicted_dhw"].values, ds["actual_dhw"].values
    lead_steps   = ds["lead_step"].values
    lat          = ds["lat"].values
    trop         = np.where(np.abs(lat) <= conf.tropic_lat)[0]
    trop_ocean   = ocean_mask[trop, :]

    pt = pred[:, :, trop, :][:, :, trop_ocean]
    at = actual[:, :, trop, :][:, :, trop_ocean]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f"Tropical Ocean Area Under Bleaching Stress - {split}",
                 fontsize=15, fontweight="bold")

    for thr, ax, title in [(4, axes[0], "Bleaching Warning (DHW >= 4 C-weeks)"),
                           (8, axes[1], "Alert Level 1+ (DHW >= 8 C-weeks)")]:
        ax.plot(lead_steps, np.nanmean(at >= thr, axis=(0, 2)) * 100, "k-",  lw=2, label="Actual")
        ax.plot(lead_steps, np.nanmean(pt >= thr, axis=(0, 2)) * 100, "r--", lw=2, label="Predicted")
        ax.set_title(title); ax.set_ylabel("Tropical Ocean Area (%)")
        ax.legend(); ax.grid(alpha=0.3)

    axes[1].set_xlabel("Lead step")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, dpi=150); plt.close(fig)


# =============================================================================
# CSV
# =============================================================================

def _write_csv(all_rows, csv_path):
    if not all_rows: return
    keys = list(dict.fromkeys(k for r in all_rows for k in r))
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(all_rows)
    logger.info(f"CSV saved -> {csv_path}")


# =============================================================================
# ANALYSE ONE SPLIT
# =============================================================================

def analyse_split(ds, split, pdf):
    logger.info(f"  {split}: chains={ds.sizes['chain']}, leads={ds.sizes['lead_step']}, "
                f"grid={ds.sizes['lat']}x{ds.sizes['lon']}")
    logger.info(f"  Lon range: {float(ds['lon'].min()):.3f} to {float(ds['lon'].max()):.3f}")

    # ── Lead-step sanity check ────────────────────────────────────────────────
    # The lead_step coordinate should be integer indices [1, 2, ..., N].
    # If values are tiny decimals (e.g. -0.04 to 0.04) the coordinate was
    # written with wrong values (normalised/scaled).  Detect and replace so
    # all plots use the correct x-axis.
    lead_vals = ds["lead_step"].values
    n_leads   = len(lead_vals)
    if lead_vals.max() < 2.0:
        logger.warning(
            f"  lead_step max={lead_vals.max():.4f} - looks wrong. "
            f"Replacing with arange(1, {n_leads + 1})."
        )
        ds = ds.assign_coords(lead_step=np.arange(1, n_leads + 1))
    # ─────────────────────────────────────────────────────────────────────────

    # ── Init-date availability check ─────────────────────────────────────────
    # Log the actual init dates in this split so the user can set
    # cfg.dhw_plots.plot_init_dates to dates that exist in the data.
    init_days = ds["init_time"].values.astype("datetime64[D]")
    logger.info(f"  Available init dates ({split}):")
    for d in init_days:
        logger.info(f"    {d}")
    # ─────────────────────────────────────────────────────────────────────────

    ocean_mask = compute_ocean_mask(ds)

    # Identify chains with actual data - chain 0 is often all-NaN
    valid_chains = _valid_chain_indices(ds)
    logger.info(f"  Valid chains (non-NaN): {valid_chains}")

    # Persistence baseline and per-lead metrics (includes skill vs persistence)
    logger.info("  Computing persistence DHW baseline")
    persistence = _compute_persistence_dhw(ds)
    logger.info("  Computing per-lead metrics")
    rows = _compute_metrics_by_lead(ds, persistence)

    logger.info("  Per-lead metrics table")
    _page_metrics_table(rows, split, pdf)
    logger.info("  Metrics vs lead step chart")
    _page_metrics_vs_lead(rows, split, pdf)
    logger.info("  Spatial RMSE / bias maps")
    _page_spatial_maps(ds, ocean_mask, split, pdf)
    logger.info("  Per-chain spatial sample maps")
    _page_sample_chains(ds, ocean_mask, split, pdf, valid_chains=valid_chains)
    logger.info("  Global-mean DHW time series")
    _page_global_mean_chains(ds, split, pdf, valid_chains=valid_chains)
    logger.info("  Tropical-mean DHW time series")
    _page_tropical_mean_chains(ds, split, pdf, valid_chains=valid_chains)
    logger.info("  Peak DHW spatial map")
    _page_max_dhw_map(ds, ocean_mask, split, pdf)
    logger.info("  Bleaching alert distribution")
    _page_bleaching_distribution(ds, ocean_mask, split, pdf)
    logger.info("  Area under stress vs lead step")
    _page_area_under_stress(ds, ocean_mask, split, pdf)
    return rows


# =============================================================================
# GBR SECTION
# =============================================================================

def analyse_gbr(ds, split, pdf):
    """
    Produce the same set of diagnostic pages as analyse_split but restricted
    to the Great Barrier Reef bounding box.

    The GBR bounds and section title are read from config so they can be
    adjusted without touching this file:
      cfg.dhw_plots.gbr_lat_bounds  - default (-25, -10) degrees
      cfg.dhw_plots.gbr_lon_bounds  - default (142, 157) degrees in 0-360

    All existing page functions are reused - the only difference is that ds
    is spatially subsetted before being passed to them, and the map xlim/ylim
    are zoomed to the GBR extent.
    """
    conf = cfg.dhw_plots

    lat_bounds = conf.gbr_lat_bounds
    lon_bounds = conf.gbr_lon_bounds

    logger.info(f"  GBR subset: lat {lat_bounds}, lon {lon_bounds}")

    # ── Subset to GBR ─────────────────────────────────────────────────────────
    try:
        ds_gbr = _subset_ds_to_region(ds, lat_bounds, lon_bounds)
    except ValueError as e:
        logger.warning(f"  GBR subset failed: {e} - skipping GBR section")
        return

    n_lat = ds_gbr.sizes["lat"]
    n_lon = ds_gbr.sizes["lon"]
    logger.info(f"  GBR grid: {n_lat} lat x {n_lon} lon")

    if n_lat < 2 or n_lon < 2:
        logger.warning("  GBR region too small (< 2 grid points) - skipping")
        return

    # ── Lead-step sanity check (same as analyse_split) ────────────────────────
    lead_vals = ds_gbr["lead_step"].values
    n_leads   = len(lead_vals)
    if lead_vals.max() < 2.0:
        ds_gbr = ds_gbr.assign_coords(lead_step=np.arange(1, n_leads + 1))

    # ── Title page for GBR section ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")
    ax.text(0.5, 0.5,
            f"GREAT BARRIER REEF - {split.upper()}\n"
            f"Lat {lat_bounds[0]} deg to {lat_bounds[1]} deg  |  "
            f"Lon {lon_bounds[0]} deg to {lon_bounds[1]} deg  (0-360)\n"
            f"Grid: {n_lat} lat x {n_lon} lon points",
            transform=ax.transAxes, fontsize=14, fontfamily="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#E8F4FD",
                      edgecolor="#2471A3", linewidth=2))
    ax.set_title(f"Great Barrier Reef Analysis - {split}",
                 fontsize=18, fontweight="bold", pad=15)
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

    # ── Ocean mask for GBR subset ─────────────────────────────────────────────
    ocean_mask_gbr = compute_ocean_mask(ds_gbr)

    # ── Valid chains ──────────────────────────────────────────────────────────
    valid_chains = _valid_chain_indices(ds_gbr)
    logger.info(f"  GBR valid chains: {valid_chains}")

    # ── Metrics (GBR only) ────────────────────────────────────────────────────
    logger.info("  GBR: persistence baseline")
    persistence_gbr = _compute_persistence_dhw(ds_gbr)
    logger.info("  GBR: per-lead metrics")
    rows_gbr = _compute_metrics_by_lead(ds_gbr, persistence_gbr)

    gbr_label = f"{split} - GBR"

    logger.info("  GBR: metrics table")
    _page_metrics_table(rows_gbr, gbr_label, pdf)
    logger.info("  GBR: metrics vs lead step")
    _page_metrics_vs_lead(rows_gbr, gbr_label, pdf)
    logger.info("  GBR: spatial RMSE / bias maps")
    _page_spatial_maps(ds_gbr, ocean_mask_gbr, gbr_label, pdf)
    logger.info("  GBR: per-chain spatial maps")
    _page_sample_chains(ds_gbr, ocean_mask_gbr, gbr_label, pdf,
                        valid_chains=valid_chains)
    logger.info("  GBR: global-mean DHW time series (GBR-mean)")
    _page_global_mean_chains(ds_gbr, gbr_label, pdf, valid_chains=valid_chains)
    logger.info("  GBR: peak DHW map")
    _page_max_dhw_map(ds_gbr, ocean_mask_gbr, gbr_label, pdf)
    logger.info("  GBR: bleaching alert distribution")
    _page_bleaching_distribution(ds_gbr, ocean_mask_gbr, gbr_label, pdf)
    logger.info("  GBR: area under stress")
    _page_area_under_stress(ds_gbr, ocean_mask_gbr, gbr_label, pdf)

    return rows_gbr


# =============================================================================
# MAIN
# =============================================================================

def run():
    conf = cfg.dhw_plots

    print()
    logger.info("=" * 60)
    logger.info("DHW PLOTS - START")
    logger.info("=" * 60)

    dhw_dir    = conf.dhw_dir
    output_pdf = os.path.join(dhw_dir, conf.output_pdf)
    output_csv = os.path.join(dhw_dir, conf.output_csv)

    Path(dhw_dir).mkdir(parents=True, exist_ok=True)
    datasets, splits_info = {}, []

    for split in ["test", "val"]:
        nc_path = os.path.join(dhw_dir, f"dhw_products_{split}.nc")
        if not os.path.exists(nc_path):
            logger.warning(f"Not found, skipping: {nc_path}"); continue
        logger.info(f"Loading {nc_path}")
        ds = xr.open_dataset(nc_path)
        datasets[split] = ds
        splits_info.append((split.capitalize(), ds.sizes["chain"], ds.sizes["lead_step"],
                            ds.sizes["lat"], ds.sizes["lon"]))

    if not datasets:
        logger.error("No DHW files found. Run DHW_calculation.py first."); return

    logger.info(f"Writing combined PDF -> {output_pdf}")
    all_metric_rows = []

    with PdfPages(output_pdf) as pdf:
        _plot_config_summary(pdf)      # NEW - configuration summary, always first
        _page_title(pdf, splits_info)
        for split, ds in datasets.items():
            logger.info(f"\nProcessing {split.upper()}")
            rows = analyse_split(ds, split.capitalize(), pdf)
            for r in rows:
                all_metric_rows.append({"split": split.capitalize(), **r})

            # ── GBR section ───────────────────────────────────────────────────
            logger.info(f"\nProcessing {split.upper()} - GBR")
            rows_gbr = analyse_gbr(ds, split.capitalize(), pdf)
            if rows_gbr:
                for r in rows_gbr:
                    all_metric_rows.append({"split": f"{split.capitalize()}_GBR", **r})
            # ─────────────────────────────────────────────────────────────────

            ds.close()

    logger.info(f"PDF saved -> {output_pdf}")
    _write_csv(all_metric_rows, output_csv)
    logger.info("=" * 60)
    logger.info("DHW PLOTS - COMPLETE")
    logger.info("=" * 60)
    print()


if __name__ == "__main__":
    run()