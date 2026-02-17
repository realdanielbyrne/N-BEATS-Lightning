from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BenchmarkDataset(ABC):
    """Abstract base class for benchmark datasets.

    All benchmark datasets must provide columnar-format DataFrames
    (rows=timesteps, cols=series) for train and test data, plus metadata
    needed by the experiment runner.
    """

    train_data: pd.DataFrame
    test_data: pd.DataFrame
    forecast_length: int
    frequency: int
    name: str

    supports_owa: bool = False

    def compute_owa(self, smape, mase):
        """Compute OWA metric. Only meaningful for datasets with Naive2 baselines."""
        return float("nan")

    def get_training_series(self):
        """Extract per-column training arrays (NaN removed).

        Returns a list of 1-D numpy arrays, one per series (column).
        """
        series_list = []
        for col in self.train_data.columns:
            vals = self.train_data[col].dropna().values.astype(np.float64)
            series_list.append(vals)
        return series_list
