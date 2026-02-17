import os
import urllib.request

import numpy as np
import pandas as pd

from ..benchmark_dataset import BenchmarkDataset

# THUML Time-Series-Library dataset on Hugging Face (CC-BY-4.0).
# 862 PeMS road sensors, hourly occupancy rates, ~17,544 timesteps.
_DOWNLOAD_URL = (
    "https://huggingface.co/datasets/thuml/Time-Series-Library/"
    "resolve/main/traffic/traffic.csv?download=true"
)
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "lightningnbeats", "Traffic")
_CACHE_FILE = os.path.join(_CACHE_DIR, "traffic.csv")


class TrafficDataset(BenchmarkDataset):
    """PeMS Traffic benchmark dataset (862 sensors, hourly occupancy).

    Downloads the standard traffic.csv from THUML/Hugging Face on first use
    and caches it at ~/.cache/lightningnbeats/Traffic/traffic.csv.

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
        self.frequency = 24  # hourly data, daily seasonality

        df = self._load_or_download()

        # Drop the date column — keep only numeric sensor columns
        if "date" in df.columns:
            df = df.drop(columns=["date"])
        # Drop the OT target column if present (aggregate; we use individual sensors)
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
        return f"Traffic-{self.horizon}"

    @staticmethod
    def _load_or_download():
        """Load cached CSV or download from Hugging Face."""
        if os.path.exists(_CACHE_FILE):
            return pd.read_csv(_CACHE_FILE)

        os.makedirs(_CACHE_DIR, exist_ok=True)
        print(f"Downloading PeMS Traffic dataset to {_CACHE_FILE} ...")
        urllib.request.urlretrieve(_DOWNLOAD_URL, _CACHE_FILE)
        print("Download complete.")
        return pd.read_csv(_CACHE_FILE)
