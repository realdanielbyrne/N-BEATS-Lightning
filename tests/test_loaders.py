"""Tests for data loaders — RowCollection setup, Columnar iloc, validation."""
import pytest
import numpy as np
import pandas as pd

from lightningnbeats.loaders import (
    RowCollectionTimeSeriesDataModule,
    ColumnarCollectionTimeSeriesTestDataModule,
)


BACKCAST_LENGTH = 8
FORECAST_LENGTH = 4


def _make_row_dataframe(n_series=10, n_timesteps=20):
    """Create a DataFrame where rows=series, cols=time steps."""
    data = np.random.rand(n_series, n_timesteps)
    return pd.DataFrame(data)


def _make_columnar_dataframes(n_timesteps=30, n_series=5, forecast_length=4):
    """Create train/test DataFrames where cols=series, rows=time steps."""
    train_data = pd.DataFrame(
        np.random.rand(n_timesteps, n_series),
        columns=[f"ts_{i}" for i in range(n_series)])
    test_data = pd.DataFrame(
        np.random.rand(forecast_length, n_series),
        columns=[f"ts_{i}" for i in range(n_series)])
    return train_data, test_data


# --- RowCollectionTimeSeriesDataModule tests ---

class TestRowCollectionSetup:
    """Verify RowCollectionTimeSeriesDataModule.setup() works correctly."""

    def test_setup_does_not_crash(self):
        df = _make_row_dataframe(n_series=10, n_timesteps=20)
        dm = RowCollectionTimeSeriesDataModule(
            data=df, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, batch_size=4)
        dm.setup()

    def test_setup_splits_time_dimension(self):
        n_timesteps = 20
        df = _make_row_dataframe(n_series=10, n_timesteps=n_timesteps)
        dm = RowCollectionTimeSeriesDataModule(
            data=df, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, batch_size=4)
        dm.setup()
        expected_train_cols = n_timesteps - FORECAST_LENGTH
        expected_val_cols = BACKCAST_LENGTH + FORECAST_LENGTH
        assert dm.train_data.shape[1] == expected_train_cols
        assert dm.val_data.shape[1] == expected_val_cols

    def test_setup_preserves_all_series(self):
        n_series = 15
        df = _make_row_dataframe(n_series=n_series, n_timesteps=20)
        dm = RowCollectionTimeSeriesDataModule(
            data=df, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, batch_size=4)
        dm.setup()
        assert dm.train_data.shape[0] == n_series
        assert dm.val_data.shape[0] == n_series

    def test_setup_returns_numpy_arrays(self):
        df = _make_row_dataframe(n_series=10, n_timesteps=20)
        dm = RowCollectionTimeSeriesDataModule(
            data=df, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, batch_size=4)
        dm.setup()
        assert isinstance(dm.train_data, np.ndarray)
        assert isinstance(dm.val_data, np.ndarray)

    def test_train_dataloader(self):
        df = _make_row_dataframe(n_series=10, n_timesteps=20)
        dm = RowCollectionTimeSeriesDataModule(
            data=df, backcast_length=BACKCAST_LENGTH,
            forecast_length=FORECAST_LENGTH, batch_size=4)
        dm.setup()
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        x, y = batch
        assert x.shape[1] == BACKCAST_LENGTH
        assert y.shape[1] == FORECAST_LENGTH


# --- ColumnarCollectionTimeSeriesTestDataModule tests ---

class TestColumnarTestModule:
    """Verify ColumnarCollectionTimeSeriesTestDataModule iloc and validation."""

    def test_iloc_slicing_works(self):
        train_data, test_data = _make_columnar_dataframes(
            n_timesteps=30, n_series=5, forecast_length=FORECAST_LENGTH)
        dm = ColumnarCollectionTimeSeriesTestDataModule(
            train_data=train_data, test_data=test_data,
            backcast_length=BACKCAST_LENGTH, forecast_length=FORECAST_LENGTH)
        expected_rows = BACKCAST_LENGTH + FORECAST_LENGTH
        assert len(dm.test_data) == expected_rows

    def test_backcast_exceeds_train_length_raises(self):
        train_data, test_data = _make_columnar_dataframes(
            n_timesteps=5, n_series=3, forecast_length=FORECAST_LENGTH)
        with pytest.raises(ValueError, match="cannot exceed training data length"):
            ColumnarCollectionTimeSeriesTestDataModule(
                train_data=train_data, test_data=test_data,
                backcast_length=10, forecast_length=FORECAST_LENGTH)

    def test_non_default_index(self):
        train_data, test_data = _make_columnar_dataframes(
            n_timesteps=30, n_series=5, forecast_length=FORECAST_LENGTH)
        train_data.index = range(100, 130)
        test_data.index = range(200, 204)
        dm = ColumnarCollectionTimeSeriesTestDataModule(
            train_data=train_data, test_data=test_data,
            backcast_length=BACKCAST_LENGTH, forecast_length=FORECAST_LENGTH)
        expected_rows = BACKCAST_LENGTH + FORECAST_LENGTH
        assert len(dm.test_data) == expected_rows
        assert dm.test_data.index[0] == 0

