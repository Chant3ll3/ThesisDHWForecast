"""
sst_dataset.py — PyTorch Dataset for autoregressive SST forecasting.

Loads the processed NetCDF, extracts sliding windows of
``input_history`` consecutive time steps, and serves
(input, target, day_of_year) tuples for the FNO training loop.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr
from config import cfg


def _compute_fractional_day_of_year(times):
    """
    Convert datetime64 array to fractional day-of-year.

    Works for daily, weekly, and monthly time steps.
    Returns an array of floats in [0, ~365].
    """
    # Convert to pandas Timestamp for easy day-of-year extraction
    import pandas as pd
    ts = pd.DatetimeIndex(times)
    return ts.dayofyear.values.astype(np.float32)


class SSTForecastDataset(Dataset):
    """
    Sliding-window dataset over the time axis of processed SST.

    Each sample returns:
      - x           : (input_history, lat, lon)   — past SST fields
      - y           : (max_rollout, lat, lon)      — future SST targets
      - day_of_year : float — day of year of the most recent input frame
    """

    def __init__(self, sst_data: np.ndarray, times: np.ndarray,
                 max_rollout: int):
        """
        Parameters
        ----------
        sst_data    : np.ndarray, shape (T, lat, lon)
        times       : np.ndarray of datetime64, shape (T,)
        max_rollout : int — number of future steps per sample
        """
        conf = cfg.model_training
        self.history = conf.input_history
        self.max_rollout = max_rollout
        self.data = torch.tensor(sst_data, dtype=torch.float32)

        # Precompute day-of-year for all time steps
        self.day_of_year = torch.tensor(
            _compute_fractional_day_of_year(times), dtype=torch.float32
        )

        self.n_samples = len(self.data) - self.history - max_rollout + 1
        assert self.n_samples > 0, (
            f"Not enough time steps ({len(self.data)}) for "
            f"history={self.history}, rollout={max_rollout}"
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        t0 = idx + self.history - 1  # index of the most recent input frame

        # Input: history frames (most recent first)
        x = self.data[t0 - self.history + 1 : t0 + 1]    # (history, H, W)

        # Targets: next max_rollout frames
        y = self.data[t0 + 1 : t0 + 1 + self.max_rollout]  # (rollout, H, W)

        # Day of year of the most recent input frame
        doy = self.day_of_year[t0]

        return x, y, doy


def load_split_data():
    """
    Load processed SST and split into train / val / test numpy arrays
    based on year ranges in config.

    Returns
    -------
    train_data, val_data, test_data : np.ndarray each (T, lat, lon)
    train_times, val_times, test_times : np.ndarray of datetime64
    lat, lon : np.ndarray coordinate vectors
    """
    conf = cfg.model_training

    ds = xr.open_dataset(conf.input_data_path)
    sst = ds[cfg.data_preparation.variable]

    lat = sst.lat.values
    lon = sst.lon.values

    def _select_years(da, yr_start, yr_end):
        subset = da.sel(time=slice(f"{yr_start}-01-01", f"{yr_end}-12-31"))
        return subset.values, subset.time.values

    train, train_t = _select_years(sst, *conf.train_years)
    val, val_t = _select_years(sst, *conf.val_years)
    test, test_t = _select_years(sst, *conf.test_years)

    return train, val, test, train_t, val_t, test_t, lat, lon
