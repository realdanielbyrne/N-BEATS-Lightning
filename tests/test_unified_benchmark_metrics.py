"""Tests for the per-series metric helpers in run_unified_benchmark.py.

Regression coverage for the rolling-window test-set layout introduced by
commit 41888b4 ("Add horizon-adaptive pooling and val split"), which changed
Traffic/Weather from a single-horizon test block (exactly one prediction row
per series) to a full rolling-window test set (thousands of rows per series).

``compute_m4_mase`` and ``compute_normalized_mae_mse`` both assumed
"prediction row i corresponds to series i". Under the rolling-window layout
that assumption breaks: MASE raised IndexError, and normalized MAE/MSE
silently fell back to the last column's statistics for every row past the
series count.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

_EXPERIMENTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "experiments")
)
if _EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, _EXPERIMENTS_DIR)

import run_unified_benchmark as rub  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures: two series, deliberately different scales and naive denominators
# ---------------------------------------------------------------------------

FREQUENCY = 2


@pytest.fixture
def train_data_df():
    """Columnar training frame: rows = timesteps, cols = series."""
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "b": [10.0, 30.0, 50.0, 70.0, 90.0, 110.0, 130.0, 150.0],
        }
    )


@pytest.fixture
def train_series_list(train_data_df):
    return [
        train_data_df[c].values.astype(np.float64) for c in train_data_df.columns
    ]


def _naive_mae(series, m):
    return float(np.mean(np.abs(series[m:] - series[:-m])))


# ---------------------------------------------------------------------------
# compute_m4_mase
# ---------------------------------------------------------------------------


class TestComputeM4Mase:
    def test_one_row_per_series_unchanged(self, train_series_list):
        """M4/Tourism convention: row i IS series i. Behaviour must not change."""
        preds = np.array([[1.0, 1.0], [10.0, 10.0]])
        targets = np.array([[2.0, 2.0], [12.0, 12.0]])

        got = rub.compute_m4_mase(preds, targets, train_series_list, FREQUENCY)

        expected = np.mean(
            [
                1.0 / _naive_mae(train_series_list[0], FREQUENCY),
                2.0 / _naive_mae(train_series_list[1], FREQUENCY),
            ]
        )
        assert got == pytest.approx(expected)

    def test_rolling_window_rows_do_not_raise(self, train_series_list):
        """More prediction rows than series must not IndexError.

        This is the Traffic-96 failure: 2,943,730 rows vs 862 series.
        """
        # 2 series x 3 windows each = 6 rows, but only 2 training series
        preds = np.zeros((6, 2))
        targets = np.ones((6, 2))
        series_idx = np.array([0, 0, 0, 1, 1, 1])

        got = rub.compute_m4_mase(
            preds, targets, train_series_list, FREQUENCY, series_idx=series_idx
        )
        assert np.isfinite(got)

    def test_rolling_window_uses_correct_denominator(self, train_series_list):
        """Each row must be scaled by ITS OWN series' naive denominator."""
        # series 'a' naive MAE = 2.0; series 'b' naive MAE = 40.0
        preds = np.zeros((4, 2))
        targets = np.ones((4, 2))  # forecast MAE = 1.0 for every row
        series_idx = np.array([0, 0, 1, 1])

        got = rub.compute_m4_mase(
            preds, targets, train_series_list, FREQUENCY, series_idx=series_idx
        )

        na = _naive_mae(train_series_list[0], FREQUENCY)
        nb = _naive_mae(train_series_list[1], FREQUENCY)
        expected = np.mean([1.0 / na, 1.0 / na, 1.0 / nb, 1.0 / nb])
        assert got == pytest.approx(expected)

    def test_series_idx_none_matches_identity_mapping(self, train_series_list):
        """series_idx=None must equal an explicit identity mapping."""
        preds = np.array([[1.0, 1.0], [10.0, 10.0]])
        targets = np.array([[2.0, 2.0], [12.0, 12.0]])

        implicit = rub.compute_m4_mase(preds, targets, train_series_list, FREQUENCY)
        explicit = rub.compute_m4_mase(
            preds, targets, train_series_list, FREQUENCY, series_idx=np.array([0, 1])
        )
        assert implicit == pytest.approx(explicit)

    def test_degenerate_series_skipped(self):
        """A constant series has zero naive MAE and must be skipped, not inf."""
        flat = [np.ones(8, dtype=np.float64), np.arange(8, dtype=np.float64)]
        preds = np.zeros((2, 2))
        targets = np.ones((2, 2))

        got = rub.compute_m4_mase(preds, targets, flat, FREQUENCY)
        # only series 1 contributes: naive MAE = 2.0, forecast MAE = 1.0
        assert got == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# compute_normalized_mae_mse
# ---------------------------------------------------------------------------


class TestComputeNormalizedMaeMse:
    def test_one_row_per_series_unchanged(self, train_data_df):
        preds = np.array([[1.0, 1.0], [10.0, 10.0]])
        targets = np.array([[2.0, 2.0], [12.0, 12.0]])

        mae, mse = rub.compute_normalized_mae_mse(preds, targets, train_data_df)

        sa = float(np.std(train_data_df["a"].values))
        sb = float(np.std(train_data_df["b"].values))
        assert mae == pytest.approx(np.mean([1.0 / sa, 2.0 / sb]))
        assert mse == pytest.approx(np.mean([(1.0 / sa) ** 2, (2.0 / sb) ** 2]))

    def test_rolling_window_uses_correct_series_stats(self, train_data_df):
        """Rows past the series count must NOT fall back to the last column."""
        preds = np.zeros((4, 2))
        targets = np.ones((4, 2))
        series_idx = np.array([0, 0, 1, 1])

        mae, _ = rub.compute_normalized_mae_mse(
            preds, targets, train_data_df, series_idx=series_idx
        )

        sa = float(np.std(train_data_df["a"].values))
        sb = float(np.std(train_data_df["b"].values))
        expected = np.mean([1.0 / sa, 1.0 / sa, 1.0 / sb, 1.0 / sb])
        assert mae == pytest.approx(expected)

    def test_last_column_fallback_is_gone(self, train_data_df):
        """The old code normalized every overflow row by cols[-1] ('b')."""
        preds = np.zeros((4, 2))
        targets = np.ones((4, 2))
        series_idx = np.array([0, 0, 1, 1])

        mae, _ = rub.compute_normalized_mae_mse(
            preds, targets, train_data_df, series_idx=series_idx
        )

        sb = float(np.std(train_data_df["b"].values))
        all_b = 1.0 / sb  # what the buggy fallback would have produced
        assert mae != pytest.approx(all_b)


# ---------------------------------------------------------------------------
# build_series_index_map
# ---------------------------------------------------------------------------


class TestBuildSeriesIndexMap:
    def test_maps_col_indices_to_positions(self, train_data_df):
        col_indices = [("a", 0), ("a", 1), ("b", 0), ("b", 1)]
        got = rub.build_series_index_map(col_indices, train_data_df)
        assert list(got) == [0, 0, 1, 1]

    def test_length_matches_col_indices(self, train_data_df):
        col_indices = [("b", i) for i in range(7)]
        got = rub.build_series_index_map(col_indices, train_data_df)
        assert len(got) == 7
        assert set(got) == {1}
