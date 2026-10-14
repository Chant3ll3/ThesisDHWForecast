import numpy as np
import pandas as pd

def compute_mmm_climatology(sst, times):
    df = pd.DataFrame({"time": times})
    df["month"] = df["time"].dt.month

    monthly_means = []
    for m in range(1, 13):
        monthly_means.append(sst[df["month"] == m].mean(axis=0))

    monthly_means = np.stack(monthly_means, axis=0)
    mmm = monthly_means.max(axis=0)
    return mmm


def compute_hotspot(sst, mmm):
    hotspot = sst - mmm
    hotspot = np.maximum(hotspot, 0.0)
    return hotspot


def compute_dhw(hotspot):
    T = hotspot.shape[0]
    dhw = np.zeros_like(hotspot)

    for t in range(T):
        start = max(0, t - 83)
        window = hotspot[start:t+1]

        window = np.where(window >= 1.0, window, 0.0)
        dhw[t] = window.sum(axis=0) / 7.0

    return dhw

# Write a function that classifies bleeching...
def classify_bleaching...
