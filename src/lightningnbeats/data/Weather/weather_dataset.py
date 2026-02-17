import os
import urllib.request

import numpy as np
import pandas as pd

from ..benchmark_dataset import BenchmarkDataset

# THUML Time-Series-Library dataset on Hugging Face (CC-BY-4.0).
# 21 meteorological indicators, ~52,696 timesteps (10-minute intervals).
_DOWNLOAD_URL = (
    "https://huggingface.co/datasets/thuml/Time-Series-Library/"
    "resolve/main/weather/weather.csv?download=true"
)
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "lightningnbeats", "Weather")
_CACHE_FILE = os.path.join(_CACHE_DIR, "weather.csv")


class WeatherDataset(BenchmarkDataset):
    """Weather benchmark dataset (21 meteorological indicators, 10-min intervals).

    Downloads the standard weather.csv from THUML/Hugging Face on first use
    and caches it at ~/.cache/lightningnbeats/Weather/weather.csv.

    Parameters
    ----------
    horizon : int
        Forecast horizon (commonly 96, 192, 336, or 720).
    train_ratio : float
        Fraction of data used for training (default 0.8). The DataModule
        will internally split this into train+val.
    """

    supports_owa = False

    def __init__(self, horizon, train_ratio=0.8):
        self.horizon = horizon
        self.forecast_length = horizon
        self.frequency = 144  # 10-min intervals, daily seasonality = 6*24 = 144

        df = self._load_or_download()

        # Drop the date column — keep only numeric indicator columns
        if "date" in df.columns:
            df = df.drop(columns=["date"])
        # Drop the OT target column if present
        if "OT" in df.columns:
            df = df.drop(columns=["OT"])

        n_total = len(df)
        n_train = int(n_total * train_ratio)

        # train_data: everything before the split point
        # test_data: the horizon-length window immediately after train
        self.train_data = df.iloc[:n_train].reset_index(drop=True)
        self.test_data = df.iloc[n_train:n_train + horizon].reset_index(drop=True)

    @property
    def name(self):
        return f"Weather-{self.horizon}"

    @staticmethod
    def _load_or_download():
        """Load cached CSV or download from Hugging Face."""
        if os.path.exists(_CACHE_FILE):
            return pd.read_csv(_CACHE_FILE)

        os.makedirs(_CACHE_DIR, exist_ok=True)
        print(f"Downloading Weather dataset to {_CACHE_FILE} ...")
        urllib.request.urlretrieve(_DOWNLOAD_URL, _CACHE_FILE)
        print("Download complete.")
        return pd.read_csv(_CACHE_FILE)
