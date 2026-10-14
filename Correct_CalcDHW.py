"""
Calc_DHW02.py — Phase 5.2: DHW calculation based on NOAA CRW Methodology
Reference: https://coralreefwatch.noaa.gov/product/5km/methodology.php

SLIDING WINDOW CONSTRUCTION (NOAA CRW standard)
------------------------------------------------
Every DHW value is a sum over exactly 84 daily SST fields ENDING ON THE DAY
THE VALUE IS REPORTED FOR.  For lead step L of a chain that day is
valid_time[chain, L].

  Lead L <= 84:  (84-L) days actual SST  +  L days predicted SST
  Lead  1: 83 actual + 1 predicted   → DHW nearly identical to actual
  Lead 30: 54 actual + 30 predicted  → increasing weight on predictions
  Lead 84:  0 actual + 84 predicted  → fully predicted DHW
  Lead 90:  0 actual + predicted days 7-90 (the most recent 84 only)

HOW A CHAIN IS PLACED ON THE OBSERVED RECORD
--------------------------------------------
Each chain is located from its own valid_time.  Lead step L is verified
against the observed field carrying the same date.  The observed history
in the forecast window ends on the LAST OBSERVED DAY, defined as the
observed field immediately before the lead-1 valid date.

init_time is NOT used for alignment.  In the inference output init_time is
the same date as the lead-1 valid date (it is the first day being
forecast, not the last observed day).  The script prints which convention
it finds.

The observed record is treated as a sequence of daily fields, exactly as
the model steps through it.  If a calendar day is absent from the record
(and therefore from the forecast chains), a window is still the 84 most
recent fields, so a window spanning the absent day covers 85 calendar
days.  Absent dates and the chains they touch are printed.

FIXES IN THIS VERSION
---------------------
  1. Last observed day.  The earlier code took init_time as the last
     observed day.  Because init_time is the lead-1 valid date, the
     observed part of every forecast window was one day late: it included
     the OBSERVED SST of the lead-1 valid date (which the forecast also
     supplies, so that day was counted twice) and left out the earliest
     day.  The observed part now ends on the day before the lead-1 valid
     date.  actual_dhw / actual_hs use the 84 fields ending on the valid
     date.

  2. Window length beyond lead 84.  The predicted part of the window is
     now the most recent 84 forecast days.  Previously leads 85-90 summed
     85-90 days of HotSpot, inflating predicted DHW.

  3. First chain.  A value is only produced when every observed field it
     needs exists; otherwise it is left NaN.  The first chain of a split
     has no observed field before its lead-1 valid date, so its forecast
     and its verification are both defined from lead 84 only and it has
     no persistence value.  (Previously its forecast started at lead 83
     and its verification at lead 84.)

  4. Persistence.  persist_dhw now holds the OBSERVED SST of the last
     observed day constant (previously it tiled the model's lead-1
     prediction).  A new variable last_obs_dhw stores the observed DHW on
     the last observed day, which is the anchor for the flat-line DHW
     persistence baseline used in dhw_plots.py.

  5. MMM interpolation wraps in longitude, so a model column lying past the
     last MMM cell centre (lon = 180 on the T42 grid) is no longer lost
     as "land".

OUTPUT VARIABLES
----------------
  (chain, lead_step, lat, lon)
    predicted_dhw : blended window using predicted SST for the lead days
    actual_dhw    : 84 observed fields ending on the valid date
                    (ground truth)
    persist_dhw   : blended window using the last observed SST held constant
    predicted_hs / actual_hs / persist_hs :
                    HotSpot (SST - MMM) on the final day of each window,
                    i.e. the valid date.  HS is required alongside DHW to
                    classify pixels into NOAA CRW's Bleaching Alert Area
                    categories (Table 1.1), since every category above
                    "Bleaching Watch" is gated on HS >= 1.

  (chain, lat, lon)
    last_obs_dhw / last_obs_hs :
                    observed DHW and HotSpot on the last observed day (the
                    day before the lead-1 valid date).  Anchor for the
                    flat-line DHW persistence baseline.

WHEN VALUES ARE NaN
-------------------
  A chain with no observed history before its lead-1 valid date (normally
  only the first chain of a split) gets:
    predicted_dhw : NaN for leads 1-83, defined for leads >= 84
    actual_dhw    : NaN for leads 1-83, defined for leads >= 84
    persist_dhw, last_obs_dhw : NaN at all leads
  dhw_plots.py scores only chains that are defined at every lead step, so
  the verification sample does not change with lead time.

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


ONE_DAY = np.timedelta64(1, "D")


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

def _pad_mmm_cyclic(mmm):
    """
    Add one wrapped column on each side of the MMM in longitude.

    The NOAA grid stores cell centres (-179.975 ... 179.975), so a target
    longitude of exactly 180 lies just outside the source range and linear
    interpolation would return NaN for that whole column.  Padding with the
    opposite edge makes the interpolation periodic.  If the source already
    spans a full 360 degrees, or its lon is not ascending, it is returned
    unchanged.
    """
    lon = mmm.lon.values
    if lon.size < 2 or not np.all(np.diff(lon) > 0):
        return mmm
    if (lon[-1] - lon[0]) >= 360.0:
        return mmm

    left  = mmm.isel(lon=[-1]).assign_coords(lon=[lon[-1] - 360.0])
    right = mmm.isel(lon=[0]).assign_coords(lon=[lon[0] + 360.0])
    return xr.concat([left, mmm, right], dim="lon")


def interpolate_mmm_to_grid(mmm, ds_ref):
    """
    Bilinearly interpolate the NOAA 0.05 deg MMM onto the coarser model
    grid (64 x 128).

    The MMM file uses -180/+180 lon; the model grid uses 0-360.  We convert
    target lons to -180/+180 before interpolating, then restore the original
    column order afterwards.  The MMM is wrapped in longitude first so that
    a target column at the dateline is interpolated rather than dropped.
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

    mmm_cyclic = _pad_mmm_cyclic(mmm)
    mmm_interp = mmm_cyclic.interp(lat=target_lat, lon=target_lon_sorted,
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
# STEP 4: Observed SST record and chain placement
# ============================================================================

def build_observed_timeline(ds_actual_cont):
    """
    Observed SST as a date-sorted sequence of daily fields.

    The DHW windows below are taken by position in this sequence ("the 84
    fields ending at field i"), which is how the model itself steps through
    the data.  If a calendar day is absent from the record the absent dates
    are printed; a window spanning one then covers more calendar days than
    it has fields.

    Returns
    -------
    obs_days : np.ndarray (n_fields,) datetime64[D], strictly increasing
    sst      : np.ndarray (n_fields, lat, lon)
    """
    window = cfg.calc_dhw.window_days

    sst  = ds_actual_cont["sst_actual"].transpose("time", "lat", "lon").values
    days = ds_actual_cont["time"].values.astype("datetime64[D]")

    order = np.argsort(days, kind="stable")
    days, sst = days[order], sst[order]

    steps = np.diff(days).astype(np.int64)
    if np.any(steps == 0):
        raise ValueError(
            "Observed SST timeline contains duplicate dates. "
            "Re-run ConcatChainsUpdated.py (it removes duplicates)."
        )

    print(f"  Observed SST: {days[0]} to {days[-1]} ({len(days)} daily fields)")

    gaps = np.where(steps > 1)[0]
    if gaps.size:
        n_absent = int((steps[gaps] - 1).sum())
        print(f"  NOTE: {n_absent} calendar day(s) absent from the observed record:")
        for g in gaps:
            first, last = days[g] + ONE_DAY, days[g + 1] - ONE_DAY
            if first == last:
                print(f"          {first}")
            else:
                print(f"          {first} to {last} ({int(steps[g]) - 1} days)")
        print(f"        Windows are the {window} most recent daily fields, so a "
              f"window spanning an absent")
        print(f"        day covers more than {window} calendar days.")

    return days, sst


def _fields_available(n_fields, start, stop):
    """True if the half-open index range [start, stop) lies inside the record."""
    return 0 <= start < stop <= n_fields


def _locate_chain(valid_days_c, obs_days, chain_number):
    """
    Index, in the observed record, of the LAST OBSERVED DAY of one chain:
    the field immediately before the chain's lead-1 valid date.

    The chain is placed from its own valid_time, so lead step L is verified
    against the observed field with the same date.  The returned index is
    -1 when the lead-1 valid date is the first field of the record (no
    observed history at all).

    Raises ValueError if a valid date is not in the observed record, or if
    the chain's lead steps do not fall on consecutive observed fields.
    """
    idx = np.searchsorted(obs_days, valid_days_c)
    in_range = idx < obs_days.size
    found = np.zeros(idx.shape, dtype=bool)
    found[in_range] = obs_days[idx[in_range]] == valid_days_c[in_range]
    if not found.all():
        bad = valid_days_c[~found]
        raise ValueError(
            f"Chain {chain_number}: {int((~found).sum())} valid date(s) are not "
            f"in the observed SST record (first: {bad[0]}). The observed "
            f"record and the forecast chains must come from the same "
            f"ConcatChainsUpdated.py run."
        )

    if not np.array_equal(idx, idx[0] + np.arange(idx.size)):
        raise ValueError(
            f"Chain {chain_number}: lead steps do not fall on consecutive "
            f"fields of the observed record, so they cannot be matched to "
            f"it one-to-one."
        )

    return int(idx[0]) - 1


# ============================================================================
# STEP 5: DHW and HotSpot for one 84-day window
# ============================================================================

def _dhw_and_final_hs(window_sst, mmm_np, ocean_mask, threshold):
    """
    DHW and final-day HotSpot for one window of daily SST.

      HotSpot(t) = window(t) - MMM
      DHW        = sum of HotSpot(t) over days where HotSpot(t) >= threshold,
                   divided by 7
      HS_final   = HotSpot on the last day of the window (not thresholded)
    """
    hotspot       = window_sst - mmm_np[np.newaxis, :, :]
    hotspot_above = np.where(hotspot >= threshold, hotspot, 0.0)

    dhw = np.nansum(hotspot_above, axis=0) / 7.0
    dhw[~ocean_mask] = np.nan

    hs = hotspot[-1].copy()
    hs[~ocean_mask] = np.nan

    return dhw.astype(np.float32), hs.astype(np.float32)


def compute_dhw_blended(actual_sst, forecast_sst,
                        last_obs_idx, mmm_np, ocean_mask):
    """
    Compute DHW and the final-day HotSpot for one forecast chain, using
    the NOAA CRW blended window.

    For lead step s (1-based), the window is the 84 daily fields ending on
    the valid date of that lead:

      n_pred   = min(s, 84)     most recent forecast days, ending at lead s
      n_actual = 84 - n_pred    observed days ending on the last observed
                                day (the day before the lead-1 valid date)

      s <= 84: forecast days 1..s, preceded by (84 - s) observed days
      s >  84: forecast days (s-83)..s only — the earliest forecast days
               have dropped out of the window, exactly as old observed
               days do

    A lead step is left NaN when the observed days it needs are not all
    present in the record.

    Parameters
    ----------
    actual_sst   : np.ndarray (n_fields, lat, lon)
        Observed SST record, one field per day in date order (real degC).
    forecast_sst : np.ndarray (n_leads, lat, lon)
        SST for lead steps 1..n_leads of one chain (real degC).
    last_obs_idx : int
        Index in actual_sst of the last observed day (the field before the
        lead-1 valid date).  -1 if the chain has no observed history.
    mmm_np       : np.ndarray (lat, lon)
        Interpolated MMM climatology.
    ocean_mask   : np.ndarray (lat, lon) bool

    Returns
    -------
    dhw      : np.ndarray (n_leads, lat, lon) float32
    hs_final : np.ndarray (n_leads, lat, lon) float32
        Final-day HotSpot (SST - MMM, not thresholded) for each lead step.
    """
    conf      = cfg.calc_dhw
    window    = conf.window_days        # 84
    threshold = conf.hotspot_threshold  # 1.0

    n_leads, nlat, nlon = forecast_sst.shape
    dhw      = np.full((n_leads, nlat, nlon), np.nan, dtype=np.float32)
    hs_final = np.full((n_leads, nlat, nlon), np.nan, dtype=np.float32)

    for L in range(n_leads):
        step     = L + 1
        n_pred   = min(step, window)    # forecast days in the window
        n_actual = window - n_pred      # observed days in the window

        # ── Forecast days: the most recent n_pred, ending at this lead ───────
        pred_slice = forecast_sst[step - n_pred : step]

        # ── Observed days: the n_actual days ending on the last observed day ─
        if n_actual > 0:
            a_start = last_obs_idx - n_actual + 1
            a_stop  = last_obs_idx + 1
            if not _fields_available(actual_sst.shape[0], a_start, a_stop):
                continue   # not enough observed history — leave NaN
            window_sst = np.concatenate(
                [actual_sst[a_start:a_stop], pred_slice], axis=0)
        else:
            window_sst = pred_slice

        if window_sst.shape[0] != window:
            raise RuntimeError(
                f"Lead step {step}: window has {window_sst.shape[0]} days, "
                f"expected {window}.")

        dhw[L], hs_final[L] = _dhw_and_final_hs(
            window_sst, mmm_np, ocean_mask, threshold)

    return dhw, hs_final


def compute_dhw_actual(actual_sst, last_obs_idx, n_leads,
                       mmm_np, ocean_mask):
    """
    Ground-truth DHW and HotSpot for one chain, from observed SST only.

    For lead step s the window is the 84 observed fields ending ON the
    valid date, i.e. record indices (last_obs_idx + s - 83) ..
    (last_obs_idx + s).  This is the same final day as the blended forecast
    window, so forecast and verification refer to the same date.

    Also returns the observed DHW / HotSpot on the last observed day
    itself (window ending at last_obs_idx), used as the anchor for DHW
    persistence.

    Returns
    -------
    dhw, hs                   : np.ndarray (n_leads, lat, lon) float32
    last_obs_dhw, last_obs_hs : np.ndarray (lat, lon) float32
    """
    conf      = cfg.calc_dhw
    window    = conf.window_days
    threshold = conf.hotspot_threshold

    n_fields   = actual_sst.shape[0]
    nlat, nlon = mmm_np.shape
    dhw = np.full((n_leads, nlat, nlon), np.nan, dtype=np.float32)
    hs  = np.full((n_leads, nlat, nlon), np.nan, dtype=np.float32)

    for L in range(n_leads):
        valid_idx = last_obs_idx + L + 1      # record index of the valid date
        w_start   = valid_idx - window + 1
        w_stop    = valid_idx + 1             # exclusive → last day = valid date
        if not _fields_available(n_fields, w_start, w_stop):
            continue
        dhw[L], hs[L] = _dhw_and_final_hs(
            actual_sst[w_start:w_stop], mmm_np, ocean_mask, threshold)

    last_obs_dhw = np.full((nlat, nlon), np.nan, dtype=np.float32)
    last_obs_hs  = np.full((nlat, nlon), np.nan, dtype=np.float32)
    w_start, w_stop = last_obs_idx - window + 1, last_obs_idx + 1
    if _fields_available(n_fields, w_start, w_stop):
        last_obs_dhw, last_obs_hs = _dhw_and_final_hs(
            actual_sst[w_start:w_stop], mmm_np, ocean_mask, threshold)

    return dhw, hs, last_obs_dhw, last_obs_hs


# ============================================================================
# STEP 6: Compute DHW and HotSpot for All Chains
# ============================================================================

def _describe_availability(name, field):
    """Short text describing which lead steps of one chain are defined."""
    have = np.isfinite(field).any(axis=(1, 2))
    if have.all():
        return None
    if not have.any():
        return f"{name}: none"
    steps = np.where(have)[0] + 1
    return f"{name}: leads {steps[0]}-{steps[-1]}"


def compute_dhw_all_chains(ds_pred, ds_actual_cont, mmm_interp, ocean_mask):
    """
    Compute the DHW and HotSpot products for every forecast chain.

    "Last observed day" below means the observed field immediately before
    a chain's lead-1 valid date.

    predicted_dhw / predicted_hs
        Blended window ending on the valid date: observed SST up to the
        last observed day, then the model forecast.

    actual_dhw / actual_hs
        84 observed days ending on the valid date.  Ground truth.

    persist_dhw / persist_hs
        Blended window: observed SST up to the last observed day, then
        that day's OBSERVED SST held constant.  SST persistence baseline —
        DHW still evolves as old heat stress leaves the window.

    last_obs_dhw / last_obs_hs
        Observed DHW / HotSpot on the last observed day.  Holding
        last_obs_dhw constant across lead steps gives the flat-line DHW
        persistence baseline used by dhw_plots.py.

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
    xr.Dataset with predicted_dhw, actual_dhw, persist_dhw, predicted_hs,
    actual_hs, persist_hs on (chain, lead_step, lat, lon) and last_obs_dhw,
    last_obs_hs on (chain, lat, lon).
    """
    conf = cfg.calc_dhw

    n_chains = ds_pred.sizes["chain"]
    n_leads  = ds_pred.sizes["lead_step"]
    nlat     = ds_pred.sizes["lat"]
    nlon     = ds_pred.sizes["lon"]
    lat      = ds_pred.lat.values
    lon      = ds_pred.lon.values

    # Observed SST as a date-sorted sequence of daily fields
    obs_days, actual_daily = build_observed_timeline(ds_actual_cont)
    n_fields = actual_daily.shape[0]
    window   = conf.window_days

    pred_all   = ds_pred["sst_predicted"].transpose(
        "chain", "lead_step", "lat", "lon").values
    init_days  = ds_pred["init_time"].values.astype("datetime64[D]")
    valid_days = ds_pred["valid_time"].transpose(
        "chain", "lead_step").values.astype("datetime64[D]")

    predicted_dhw = np.full((n_chains, n_leads, nlat, nlon), np.nan,
                            dtype=np.float32)
    actual_dhw    = np.full_like(predicted_dhw, np.nan)
    persist_dhw   = np.full_like(predicted_dhw, np.nan)

    predicted_hs  = np.full_like(predicted_dhw, np.nan)
    actual_hs     = np.full_like(predicted_dhw, np.nan)
    persist_hs    = np.full_like(predicted_dhw, np.nan)

    last_obs_dhw = np.full((n_chains, nlat, nlon), np.nan, dtype=np.float32)
    last_obs_hs  = np.full_like(last_obs_dhw, np.nan)

    # How init_time relates to the chain's own valid dates (reported below)
    n_init_is_first_valid = 0
    n_init_is_day_before  = 0

    mmm_np = mmm_interp.values   # (lat, lon)

    print(f"\n  Processing {n_chains} chains x {n_leads} lead steps ...")

    for c in range(n_chains):

        # Last observed day = field before this chain's lead-1 valid date,
        # located from the chain's own valid dates (init_time is not used).
        last_obs_idx = _locate_chain(valid_days[c], obs_days, c)

        if init_days[c] == valid_days[c, 0]:
            n_init_is_first_valid += 1
        elif init_days[c] == valid_days[c, 0] - ONE_DAY:
            n_init_is_day_before += 1

        # ── predicted_dhw / predicted_hs: observed history + model forecast ──
        predicted_dhw[c], predicted_hs[c] = compute_dhw_blended(
            actual_daily, pred_all[c], last_obs_idx, mmm_np, ocean_mask)

        # ── actual_dhw / actual_hs: ground truth, window ends on valid date ──
        (actual_dhw[c], actual_hs[c],
         last_obs_dhw[c], last_obs_hs[c]) = compute_dhw_actual(
            actual_daily, last_obs_idx, n_leads, mmm_np, ocean_mask)

        # ── persist_dhw / persist_hs: last OBSERVED SST held constant ────────
        # Needs that observation itself; left NaN if the chain has no
        # observed history (first chain of a split).
        if _fields_available(n_fields, last_obs_idx, last_obs_idx + 1):
            persist_sst = np.broadcast_to(
                actual_daily[last_obs_idx], (n_leads, nlat, nlon))
            persist_dhw[c], persist_hs[c] = compute_dhw_blended(
                actual_daily, persist_sst, last_obs_idx, mmm_np, ocean_mask)

        # ── Progress / availability report ───────────────────────────────────
        notes = [n for n in (
            _describe_availability("predicted",   predicted_dhw[c]),
            _describe_availability("actual",      actual_dhw[c]),
            _describe_availability("persistence", persist_dhw[c]),
        ) if n]
        if not np.isfinite(last_obs_dhw[c]).any():
            notes.append("last-observed-day DHW: none")

        # Does any window of this chain span an absent calendar day?
        lo = max(last_obs_idx - window + 1, 0)
        hi = min(last_obs_idx + n_leads, n_fields - 1)
        spans_absent_day = (
            int((obs_days[hi] - obs_days[lo]).astype(np.int64)) != hi - lo)

        if (notes or spans_absent_day
                or (c + 1) % 5 == 0 or c == n_chains - 1):
            print(f"    Chain {c + 1:>3d}/{n_chains} — "
                  f"lead 1 valid: {valid_days[c, 0]}")
            if notes:
                print("      incomplete observed history → "
                      + "; ".join(notes))
            if spans_absent_day:
                print("      some windows span an absent calendar day")

    # ── Report how init_time relates to the valid dates ──────────────────────
    if n_init_is_first_valid == n_chains:
        print("\n  init_time equals the lead-1 valid date in every chain.")
        print("  Last observed day used = the observed field before that date.")
    elif n_init_is_day_before == n_chains:
        print("\n  init_time is the day before the lead-1 valid date "
              "(the last observed day).")
    else:
        print(f"\n  NOTE: init_time equals the lead-1 valid date in "
              f"{n_init_is_first_valid} of {n_chains} chains and the day "
              f"before it in {n_init_is_day_before}.")
        print("  Alignment uses valid_time only, so this does not affect "
              "the results.")

    # ── Pack outputs into an xarray Dataset ──────────────────────────────────
    lead_steps  = np.arange(1, n_leads + 1)
    init_times  = ds_pred["init_time"].values
    valid_times = ds_pred["valid_time"].transpose("chain", "lead_step").values

    thr     = conf.hotspot_threshold

    dims4 = ["chain", "lead_step", "lat", "lon"]
    dims3 = ["chain", "lat", "lon"]

    ds_out = xr.Dataset(
        {
            "predicted_dhw": (dims4, predicted_dhw),
            "actual_dhw":    (dims4, actual_dhw),
            "persist_dhw":   (dims4, persist_dhw),
            "predicted_hs":  (dims4, predicted_hs),
            "actual_hs":     (dims4, actual_hs),
            "persist_hs":    (dims4, persist_hs),
            "last_obs_dhw":  (dims3, last_obs_dhw),
            "last_obs_hs":   (dims3, last_obs_hs),
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

    ds_out.attrs = {
        "window_alignment": "every DHW/HS value uses the "
                            f"{window} daily fields ending on its valid "
                            "date; observed history ends on the day before "
                            "the lead-1 valid date",
        "window_days":       window,
        "hotspot_threshold": thr,
    }

    ds_out["predicted_dhw"].attrs = {
        "long_name": "Predicted DHW — blended sliding window",
        "units":     "degC-weeks",
        "method":    f"{window} days ending on the valid date: observed SST "
                     f"to the last observed day, then predicted SST "
                     f"(most recent {window} forecast days beyond lead "
                     f"{window}); HotSpot >= {thr} degC",
    }
    ds_out["actual_dhw"].attrs = {
        "long_name": "Actual DHW — ground truth (observed SST only)",
        "units":     "degC-weeks",
        "method":    f"{window} observed days ending on the valid date, "
                     f"HotSpot >= {thr} degC",
    }
    ds_out["persist_dhw"].attrs = {
        "long_name": "Persistence DHW — last observed SST held constant",
        "units":     "degC-weeks",
        "method":    f"{window} days ending on the valid date: observed SST "
                     f"to the last observed day, then that day's SST repeated; "
                     f"HotSpot >= {thr} degC",
    }
    ds_out["predicted_hs"].attrs = {
        "long_name": "Predicted HotSpot (SST - MMM) on the valid date",
        "units":     "degC",
        "method":    "Instantaneous HotSpot on the last day of the "
                     "blended predicted window, used for Bleaching Alert "
                     "Area classification alongside predicted_dhw.",
    }
    ds_out["actual_hs"].attrs = {
        "long_name": "Actual HotSpot (SST - MMM) on the valid date",
        "units":     "degC",
        "method":    "Instantaneous HotSpot on the last day of the "
                     "actual SST window, used for Bleaching Alert Area "
                     "classification alongside actual_dhw.",
    }
    ds_out["persist_hs"].attrs = {
        "long_name": "Persistence HotSpot (SST - MMM) on the valid date",
        "units":     "degC",
        "method":    "Instantaneous HotSpot on the last day of the "
                     "persistence window (equals the HotSpot on the last "
                     "observed day), used for Bleaching Alert Area "
                     "classification alongside persist_dhw.",
    }
    ds_out["last_obs_dhw"].attrs = {
        "long_name": "Observed DHW on the last observed day "
                     "(day before the lead-1 valid date)",
        "units":     "degC-weeks",
        "method":    f"{window} observed days ending on the last observed "
                     f"day, HotSpot >= {thr} degC. Held constant across lead "
                     f"steps this is the flat-line DHW persistence baseline.",
    }
    ds_out["last_obs_hs"].attrs = {
        "long_name": "Observed HotSpot (SST - MMM) on the last observed day "
                     "(day before the lead-1 valid date)",
        "units":     "degC",
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
    """
    Print DHW statistics (mean / max / DHW-only bins) for all three DHW
    variables.

    The bins are plain DHW ranges.  They are NOT Bleaching Alert Area
    categories, which also depend on HotSpot — see STEP 7 for those.
    """
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
        print(f"    DHW < 4:        {np.sum(finite < 4):>10,} "
              f"({100 * np.mean(finite < 4):.1f}%)")
        print(f"    4 <= DHW < 8:   {np.sum((finite >= 4) & (finite < 8)):>10,} "
              f"({100 * np.mean((finite >= 4) & (finite < 8)):.1f}%)")
        print(f"    DHW >= 8:       {np.sum(finite >= 8):>10,} "
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

        # ── Compute DHW and HS with the blended sliding window ────────────────
        print("\n" + "=" * 60)
        print("STEP 6: Compute DHW and HotSpot — blended sliding window")
        print("=" * 60)
        ds_dhw = compute_dhw_all_chains(ds_pred, ds_actual,
                                        mmm_model, ocean_mask)

        # ── DHW summary (DHW-only bins, retained for continuity) ──────────────
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