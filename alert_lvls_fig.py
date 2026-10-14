"""
make_alert_levels_figure.py - poster figure: share of reef-zone ocean in each NOAA
Coral Reef Watch Bleaching Alert Area level, observed vs forecast, at ONE lead time.

Categories (NOAA CRW, Table 1.1; uses HotSpot AND DHW):
  Bleaching Warning : HotSpot >= 1 and 0 < DHW < 4
  Alert Level 1     : HotSpot >= 1 and 4 <= DHW < 8
  Alert Level 2     : HotSpot >= 1 and 8 <= DHW < 12
  Alert Level 3-5   : HotSpot >= 1 and DHW >= 12
("No Stress" and "Bleaching Watch" are left out.)

Sample: the same chains dhw_plots.py scores (defined at every lead step), ocean pixels
within the latitude band, area-weighted by cos(latitude) so the bars are shares of
ocean AREA rather than grid cells.

Usage (from the DHW-project folder, on Gadi):
  python make_alert_levels_figure.py dhw_output_64x128/dhw_products_test.nc
  python make_alert_levels_figure.py FILE --step 90 --lat 35
Writes alert_levels_day<step>.png (300 dpi) and .pdf.
"""
import argparse
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("file")
ap.add_argument("--step", type=int, default=60, help="lead step (day) to show")
ap.add_argument("--lat", type=float, default=35.0, help="use |latitude| <= this (reef zone)")
args = ap.parse_args()

ds = xr.open_dataset(args.file)
dims4 = ("chain", "lead_step", "lat", "lon")
pred_all = ds["predicted_dhw"].transpose(*dims4).values
act_all = ds["actual_dhw"].transpose(*dims4).values
anchor = ds["last_obs_dhw"].transpose("chain", "lat", "lon").values if "last_obs_dhw" in ds else None

# Same verification sample as dhw_plots.py: chains defined at every lead step
usable = np.isfinite(pred_all) & np.isfinite(act_all)
if anchor is not None:
    usable &= np.isfinite(anchor)[:, None]
complete = usable.any(axis=(2, 3)).all(axis=1)
chains = np.where(complete)[0]

i = int(np.where(ds["lead_step"].values == args.step)[0][0])
lat = ds["lat"].values
band = np.abs(lat) <= args.lat
w_lat = np.cos(np.deg2rad(lat))[band]                       # area weight per row

def get(var):
    return ds[var].transpose(*dims4).values[chains, i][:, band, :]

obs_dhw, obs_hs = get("actual_dhw"), get("actual_hs")
fc_dhw, fc_hs = get("predicted_dhw"), get("predicted_hs")
valid = np.isfinite(obs_dhw) & np.isfinite(obs_hs) & np.isfinite(fc_dhw) & np.isfinite(fc_hs)
W = np.broadcast_to(w_lat[None, :, None], obs_dhw.shape)
total = W[valid].sum()

CATS = [("Bleaching\nWarning", 0, 4, True), ("Alert\nLevel 1", 4, 8, False),
        ("Alert\nLevel 2", 8, 12, False), ("Alert\nLevel 3–5", 12, np.inf, False)]

def shares(dhw, hs):
    out = []
    for _, lo, hi, warning in CATS:
        lower = (dhw > lo) if warning else (dhw >= lo)
        m = valid & (hs >= 1) & lower & (dhw < hi)
        out.append(100 * W[m].sum() / total)
    return np.array(out)

obs, fc = shares(obs_dhw, obs_hs), shares(fc_dhw, fc_hs)

# ── Figure (styled like the poster's other charts) ────────────────────────────
C_OBS, C_FC = "#2b2b2b", "#e0302b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15})
fig, ax = plt.subplots(figsize=(11, 6.0), facecolor="white")
x = np.arange(len(CATS)); bw = 0.36
b1 = ax.bar(x - bw / 2 - 0.01, obs, bw, color=C_OBS, label="Observed", zorder=3)
b2 = ax.bar(x + bw / 2 + 0.01, fc, bw, color=C_FC, label="Forecast", zorder=3)

ymax = max(obs.max(), fc.max()) * 1.32 if max(obs.max(), fc.max()) > 0 else 1
for bars, vals in ((b1, obs), (b2, fc)):
    for r, v in zip(bars, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + ymax * 0.015, f"{v:.2f}%",
                ha="center", va="bottom", fontsize=14, color="#1a1a1a")
for k in range(len(CATS)):
    if obs[k] > 0:
        ax.text(x[k], max(obs[k], fc[k]) + ymax * 0.10, f"forecast flags {100 * fc[k] / obs[k]:.0f}%\nof observed",
                ha="center", va="bottom", fontsize=12.5, color="#444444", linespacing=1.2)

ax.set_xticks(x); ax.set_xticklabels([c[0] for c in CATS], fontsize=15)
ax.set_ylim(0, ymax * 1.18)
ax.set_ylabel(f"Share of reef-zone ocean (%)", color="#444444")
ax.grid(axis="y", color="#e4e3df", lw=1, zorder=0)
ax.tick_params(colors="#444444", length=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#bdbdbd")
ax.legend(frameon=False, fontsize=15, loc="upper right", ncol=2)
ax.set_title(f"Day {args.step} forecast, {len(chains)} forecasts, ocean within {args.lat:.0f}° of the equator",
             fontsize=14, color="#444444", loc="left", pad=12)
fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.14)

stem = f"alert_levels_day{args.step}"
fig.savefig(stem + ".png", dpi=300, facecolor="white"); fig.savefig(stem + ".pdf", facecolor="white")
print(f"day {args.step}: {len(chains)} chains, |lat| <= {args.lat}")
for (name, *_), o, f in zip(CATS, obs, fc):
    print(f"  {name.replace(chr(10), ' '):<20s} observed {o:6.3f}%  forecast {f:6.3f}%")
print(f"saved {stem}.png and {stem}.pdf")