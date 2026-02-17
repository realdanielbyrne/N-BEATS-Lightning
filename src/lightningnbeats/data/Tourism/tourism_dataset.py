import os

import numpy as np
import pandas as pd

from ..benchmark_dataset import BenchmarkDataset


class TourismDataset(BenchmarkDataset):
    """Tourism competition dataset — very small series for rapid benchmarking.

    Bundled CSV data files (no download needed):
      - Yearly: tourism1/tourism_data.csv — 43 rows x 518 series (Y1-Y518)
      - Monthly: tourism2/tourism2_revision2.csv cols m1-m366 — 309 rows x 366 series
      - Quarterly: same file, cols q1-q427 — 309 rows x 427 series

    Parameters
    ----------
    period : str
        One of "Yearly", "Monthly", "Quarterly".
    """

    PERIODS = {
        "Yearly":    {"forecast_length": 4,  "frequency": 1},
        "Monthly":   {"forecast_length": 24, "frequency": 12},
        "Quarterly": {"forecast_length": 8,  "frequency": 4},
    }

    supports_owa = False

    def __init__(self, period):
        if period not in self.PERIODS:
            raise ValueError(
                f"Unknown Tourism period '{period}'. "
                f"Choose from: {list(self.PERIODS.keys())}"
            )

        cfg = self.PERIODS[period]
        self.period = period
        self.forecast_length = cfg["forecast_length"]
        self.frequency = cfg["frequency"]

        data_dir = os.path.join(os.path.dirname(__file__), "..")
        df = self._load_period(period, data_dir)

        # Pad short series with zeros (matches examples/utils.py fill_columnar_ts_gaps)
        min_length = self.forecast_length + 1  # need at least 1 train obs after split
        df = self._pad_short_series(df, min_length)

        # Split: train = all but last forecast_length rows, test = last forecast_length rows
        self.train_data = df.iloc[:-self.forecast_length].reset_index(drop=True)
        self.test_data = df.iloc[-self.forecast_length:].reset_index(drop=True)

    @property
    def name(self):
        return f"Tourism-{self.period}"

    @staticmethod
    def _load_period(period, data_dir):
        """Load and filter the CSV for the given period."""
        if period == "Yearly":
            path = os.path.join(data_dir, "tourism1", "tourism_data.csv")
            df = pd.read_csv(path)
            # All columns are Y-prefixed series
            df = df[[c for c in df.columns if c.startswith("Y")]]
        elif period == "Monthly":
            path = os.path.join(data_dir, "tourism2", "tourism2_revision2.csv")
            df = pd.read_csv(path)
            df = df[[c for c in df.columns if c.startswith("m")]]
        elif period == "Quarterly":
            path = os.path.join(data_dir, "tourism2", "tourism2_revision2.csv")
            df = pd.read_csv(path)
            df = df[[c for c in df.columns if c.startswith("q")]]

        # Drop rows that are entirely NaN
        df = df.dropna(how="all").reset_index(drop=True)
        return df

    @staticmethod
    def _pad_short_series(df, min_length):
        """Prepend zeros to columns shorter than min_length."""
        for col in df.columns:
            valid_count = df[col].dropna().shape[0]
            if valid_count < min_length:
                deficit = min_length - valid_count
                # Build padded series: zeros + existing non-NaN values
                non_na = df[col].dropna().values.tolist()
                padded = [0.0] * deficit + non_na
                # Extend with NaN to match df length
                if len(padded) < len(df):
                    padded = [np.nan] * (len(df) - len(padded)) + padded
                df[col] = padded[:len(df)]
        return df
