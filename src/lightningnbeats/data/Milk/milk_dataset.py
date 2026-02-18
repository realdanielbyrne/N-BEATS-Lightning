import os

import numpy as np
import pandas as pd

from ..benchmark_dataset import BenchmarkDataset


class MilkDataset(BenchmarkDataset):
    """Milk production dataset (168 monthly observations, 1962-1975).

    Single univariate series wrapped as a one-column DataFrame so the
    experiment runner can use the same columnar code path for all datasets.
    """

    supports_owa = False

    def __init__(self):
        csv_path = os.path.join(os.path.dirname(__file__), "..", "milk.csv")
        df = pd.read_csv(csv_path)
        values = df["milk_production_pounds"].values.astype(float)

        # Single-column DataFrame (columnar format)
        full = pd.DataFrame({"milk": values})

        self.forecast_length = 6
        self.frequency = 12
        self.train_data = full.iloc[: -self.forecast_length].reset_index(drop=True)
        self.test_data = full.iloc[-self.forecast_length :].reset_index(drop=True)

    @property
    def name(self):
        return "Milk"
