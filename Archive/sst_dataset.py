"""
sst_dataset.py — PyTorch Dataset for autoregressive SST forecasting.

Loads the processed NetCDF, extracts sliding windows of
``input_history`` consecutive days, and serves (input, target) pairs
for the SFNO training loop.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr
from config import cfg


class SSTForecastDataset(Dataset):
    """
    Sliding-window dataset over the time axis of processed SST.

    Each sample returns:
      - x : (input_history, lat, lon)   — past SST fields
      - y : (max_rollout, lat, lon)     — future SST targets

    The number of target steps (``max_rollout``) is configurable so
    the training loop can vary the rollout length per epoch.
    """

    def __init__(self, sst_data: np.ndarray, max_rollout: int):
        """
        Parameters
        ----------
        sst_data : np.ndarray, shape (T, lat, lon)
            Contiguous block of processed SST (e.g. train years).
        max_rollout : int
            Number of future steps stored per sample.
        """
        conf = cfg.model_training
        self.history = conf.input_history
        self.max_rollout = max_rollout
        self.data = torch.tensor(sst_data, dtype=torch.float32)

        # Valid starting indices: need history behind + rollout ahead
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

        return x, y


def load_split_data():
    """
    Load processed SST and split into train / val / test numpy arrays
    based on year ranges in config.

    Returns
    -------
    train_data, val_data, test_data : np.ndarray each (T, lat, lon)
    lat, lon : np.ndarray coordinate vectors
    """
    conf = cfg.model_training

    ds = xr.open_dataset(conf.input_data_path)
    sst = ds[cfg.data_preparation.variable]

    lat = sst.lat.values
    lon = sst.lon.values

    def _select_years(da, yr_start, yr_end):
        return da.sel(time=slice(f"{yr_start}-01-01", f"{yr_end}-12-31")).values

    train = _select_years(sst, *conf.train_years)
    val = _select_years(sst, *conf.val_years)
    test = _select_years(sst, *conf.test_years)

    return train, val, test, lat, lon
