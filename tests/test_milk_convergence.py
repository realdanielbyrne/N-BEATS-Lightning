"""Tests for experiments/run_milk_convergence.py — utility functions, callbacks,
CSV helpers, single-run worker, and summary statistics."""

import csv
import importlib.util
import json
import math
import os
import sys
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Import the experiment module by file path (it's not a package)
# ---------------------------------------------------------------------------
_SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "experiments", "run_milk_convergence.py"
)
_spec = importlib.util.spec_from_file_location("run_milk_convergence", _SCRIPT_PATH)
mc = importlib.util.module_from_spec(_spec)
mc.__file__ = os.path.abspath(_SCRIPT_PATH)
_spec.loader.exec_module(mc)


# ===================================================================
# Constants & Configuration
# ===================================================================

class TestConstants:
    """Verify configuration constants are consistent."""

    def test_backcast_is_4x_forecast(self):
        assert mc.BACKCAST_LENGTH == 4 * mc.FORECAST_LENGTH

    def test_two_configs_defined(self):
        assert len(mc.MILK_CONVERGENCE_CONFIGS) == 2
        assert "Milk6_baseline" in mc.MILK_CONVERGENCE_CONFIGS
        assert "Milk6_activeG" in mc.MILK_CONVERGENCE_CONFIGS

    def test_baseline_active_g_false(self):
        assert mc.MILK_CONVERGENCE_CONFIGS["Milk6_baseline"]["active_g"] is False

    def test_active_g_config_true(self):
        assert mc.MILK_CONVERGENCE_CONFIGS["Milk6_activeG"]["active_g"] is True

    def test_csv_columns_non_empty(self):
        assert len(mc.MILK_CONVERGENCE_CSV_COLUMNS) > 0

    def test_csv_columns_contain_required_fields(self):
        required = [
            "config_name", "run", "seed", "active_g",
            "best_val_loss", "epochs_trained", "diverged", "healthy",
            "val_loss_curve",
        ]
        for field in required:
            assert field in mc.MILK_CONVERGENCE_CSV_COLUMNS, f"Missing: {field}"


# ===================================================================
# Utility Functions
# ===================================================================

class TestLoadMilkData:
    """Verify milk dataset loading."""

    def test_returns_numpy_array(self):
        data = mc.load_milk_data()
        assert isinstance(data, np.ndarray)

    def test_dtype_is_float32(self):
        data = mc.load_milk_data()
        assert data.dtype == np.float32

    def test_shape_is_1d(self):
        data = mc.load_milk_data()
        assert data.ndim == 1

    def test_has_168_observations(self):
        data = mc.load_milk_data()
        assert len(data) == 168

    def test_values_are_positive(self):
        data = mc.load_milk_data()
        assert np.all(data > 0)


class TestCountParameters:
    """Verify parameter counting."""

    def test_known_linear_layer(self):
        model = torch.nn.Linear(10, 5, bias=True)
        # 10*5 + 5 = 55
        assert mc.count_parameters(model) == 55

    def test_frozen_params_excluded(self):
        model = torch.nn.Linear(10, 5, bias=True)
        for p in model.parameters():
            p.requires_grad = False
        assert mc.count_parameters(model) == 0


# ===================================================================
# CSV Helpers
# ===================================================================

class TestCSVHelpers:
    """Verify CSV init, append, and existence check."""

    def _make_row(self, config_name="test_cfg", run_idx=0):
        """Create a minimal valid result row dict."""
        row = {col: "" for col in mc.MILK_CONVERGENCE_CSV_COLUMNS}
        row.update({
            "experiment": "test",
            "config_name": config_name,
            "run": run_idx,
            "seed": 42,
            "active_g": False,
            "activation": "ReLU",
            "best_val_loss": "10.0",
            "final_val_loss": "10.0",
            "final_train_loss": "10.0",
            "best_epoch": 5,
            "epochs_trained": 10,
            "stopping_reason": "EARLY_STOPPED",
            "loss_ratio": "1.0",
            "diverged": False,
            "healthy": True,
            "n_params": 1000,
            "training_time_seconds": "5.0",
            "val_loss_curve": "[]",
        })
        return row

    def test_init_csv_creates_file(self, tmp_path):
        csv_path = str(tmp_path / "test.csv")
        mc.init_csv(csv_path)
        assert os.path.exists(csv_path)

    def test_init_csv_writes_header(self, tmp_path):
        csv_path = str(tmp_path / "test.csv")
        mc.init_csv(csv_path)
        with open(csv_path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == mc.MILK_CONVERGENCE_CSV_COLUMNS

    def test_init_csv_idempotent(self, tmp_path):
        csv_path = str(tmp_path / "test.csv")
        mc.init_csv(csv_path)
        mc.init_csv(csv_path)  # second call should not duplicate header
        with open(csv_path) as f:
            lines = f.readlines()
        assert len(lines) == 1  # header only

    def test_append_result_adds_row(self, tmp_path):
        csv_path = str(tmp_path / "test.csv")
        mc.init_csv(csv_path)
        mc.append_result(csv_path, self._make_row())
        with open(csv_path) as f:
            lines = f.readlines()
        assert len(lines) == 2  # header + 1 row

    def test_result_exists_false_when_missing(self, tmp_path):
        csv_path = str(tmp_path / "test.csv")
        assert mc.result_exists(csv_path, "cfg", 0) is False

    def test_result_exists_false_on_empty_csv(self, tmp_path):
        csv_path = str(tmp_path / "test.csv")
        mc.init_csv(csv_path)
        assert mc.result_exists(csv_path, "cfg", 0) is False

    def test_result_exists_true_after_append(self, tmp_path):
        csv_path = str(tmp_path / "test.csv")
        mc.init_csv(csv_path)
        mc.append_result(csv_path, self._make_row("my_cfg", 7))
        assert mc.result_exists(csv_path, "my_cfg", 7) is True

    def test_result_exists_false_for_different_run(self, tmp_path):
        csv_path = str(tmp_path / "test.csv")
        mc.init_csv(csv_path)
        mc.append_result(csv_path, self._make_row("my_cfg", 7))
        assert mc.result_exists(csv_path, "my_cfg", 8) is False


# ===================================================================
# ConvergenceTracker Callback
# ===================================================================

class TestConvergenceTracker:
    """Verify ConvergenceTracker records per-epoch losses."""

    def _make_trainer_mock(self, val_loss=None, train_loss=None):
        trainer = MagicMock()
        metrics = {}
        if val_loss is not None:
            metrics["val_loss"] = torch.tensor(val_loss)
        if train_loss is not None:
            metrics["train_loss"] = torch.tensor(train_loss)
        trainer.callback_metrics = metrics
        return trainer

    def test_initial_state_empty(self):
        tracker = mc.ConvergenceTracker()
        assert tracker.val_losses == []
        assert tracker.train_losses == []

    def test_records_val_loss(self):
        tracker = mc.ConvergenceTracker()
        trainer = self._make_trainer_mock(val_loss=5.0)
        tracker.on_validation_epoch_end(trainer, None)
        assert len(tracker.val_losses) == 1
        assert tracker.val_losses[0] == pytest.approx(5.0)

    def test_records_train_loss(self):
        tracker = mc.ConvergenceTracker()
        trainer = self._make_trainer_mock(train_loss=3.0)
        tracker.on_train_epoch_end(trainer, None)
        assert len(tracker.train_losses) == 1
        assert tracker.train_losses[0] == pytest.approx(3.0)

    def test_ignores_missing_val_loss(self):
        tracker = mc.ConvergenceTracker()
        trainer = self._make_trainer_mock()  # no val_loss
        tracker.on_validation_epoch_end(trainer, None)
        assert tracker.val_losses == []

    def test_accumulates_multiple_epochs(self):
        tracker = mc.ConvergenceTracker()
        for val in [10.0, 8.0, 6.0]:
            trainer = self._make_trainer_mock(val_loss=val)
            tracker.on_validation_epoch_end(trainer, None)
        assert len(tracker.val_losses) == 3
        assert tracker.val_losses == [pytest.approx(v) for v in [10.0, 8.0, 6.0]]


# ===================================================================
# DivergenceDetector Callback
# ===================================================================

class TestDivergenceDetector:
    """Verify DivergenceDetector stops training on divergence."""

    def _make_trainer_mock(self, val_loss):
        trainer = MagicMock()
        trainer.callback_metrics = {"val_loss": torch.tensor(val_loss)}
        trainer.should_stop = False
        return trainer

    def test_initial_state(self):
        dd = mc.DivergenceDetector()
        assert dd.diverged is False
        assert dd.best_val_loss == float("inf")

    def test_tracks_best_val_loss(self):
        dd = mc.DivergenceDetector()
        trainer = self._make_trainer_mock(10.0)
        dd.on_validation_epoch_end(trainer, None)
        assert dd.best_val_loss == pytest.approx(10.0)

    def test_nan_triggers_immediate_divergence(self):
        dd = mc.DivergenceDetector()
        trainer = self._make_trainer_mock(float("nan"))
        dd.on_validation_epoch_end(trainer, None)
        assert dd.diverged is True
        assert trainer.should_stop is True

    def test_inf_triggers_immediate_divergence(self):
        dd = mc.DivergenceDetector()
        trainer = self._make_trainer_mock(float("inf"))
        dd.on_validation_epoch_end(trainer, None)
        assert dd.diverged is True
        assert trainer.should_stop is True

    def test_no_divergence_within_threshold(self):
        dd = mc.DivergenceDetector(relative_threshold=3.0, consecutive_epochs=3)
        # Set best to 10.0
        trainer = self._make_trainer_mock(10.0)
        dd.on_validation_epoch_end(trainer, None)
        # Loss at 2.5x best — below 3.0 threshold
        trainer = self._make_trainer_mock(25.0)
        dd.on_validation_epoch_end(trainer, None)
        assert dd.diverged is False

    def test_divergence_after_consecutive_bad_epochs(self):
        dd = mc.DivergenceDetector(relative_threshold=2.0, consecutive_epochs=2)
        # Set best to 5.0
        trainer = self._make_trainer_mock(5.0)
        dd.on_validation_epoch_end(trainer, None)
        # Two consecutive epochs above 2x threshold
        for _ in range(2):
            trainer = self._make_trainer_mock(15.0)  # 3x best
            dd.on_validation_epoch_end(trainer, None)
        assert dd.diverged is True
        assert trainer.should_stop is True

    def test_bad_count_resets_on_good_epoch(self):
        dd = mc.DivergenceDetector(relative_threshold=2.0, consecutive_epochs=3)
        # Set best to 5.0
        trainer = self._make_trainer_mock(5.0)
        dd.on_validation_epoch_end(trainer, None)
        # One bad epoch
        trainer = self._make_trainer_mock(15.0)
        dd.on_validation_epoch_end(trainer, None)
        assert dd.bad_epoch_count == 1
        # Good epoch resets count
        trainer = self._make_trainer_mock(7.0)  # above best but below threshold
        dd.on_validation_epoch_end(trainer, None)
        assert dd.bad_epoch_count == 0
        assert dd.diverged is False

    def test_ignores_none_val_loss(self):
        dd = mc.DivergenceDetector()
        trainer = MagicMock()
        trainer.callback_metrics = {}
        dd.on_validation_epoch_end(trainer, None)
        assert dd.diverged is False


# ===================================================================
# Single-Run Worker (Integration Test — short training)
# ===================================================================

class TestSingleRunWorker:
    """Integration test: run_single_milk_convergence_experiment with 3 epochs."""

    @pytest.fixture(scope="class")
    def result(self):
        """Run a single short experiment and cache the result for all tests."""
        return mc.run_single_milk_convergence_experiment(
            config_name="Milk6_baseline",
            active_g=False,
            activation="ReLU",
            run_idx=0,
            seed=42,
            max_epochs=3,
            n_threads=1,
        )

    def test_result_is_dict(self, result):
        assert isinstance(result, dict)

    def test_result_has_all_csv_columns(self, result):
        for col in mc.MILK_CONVERGENCE_CSV_COLUMNS:
            assert col in result, f"Missing key: {col}"

    def test_experiment_name(self, result):
        assert result["experiment"] == "milk_convergence"

    def test_config_name(self, result):
        assert result["config_name"] == "Milk6_baseline"

    def test_run_index(self, result):
        assert result["run"] == 0

    def test_seed_value(self, result):
        assert result["seed"] == 42

    def test_active_g_value(self, result):
        assert result["active_g"] is False

    def test_epochs_trained_positive(self, result):
        assert int(result["epochs_trained"]) > 0

    def test_best_val_loss_is_numeric(self, result):
        val = float(result["best_val_loss"])
        assert math.isfinite(val) or math.isnan(val)

    def test_n_params_positive(self, result):
        assert int(result["n_params"]) > 0

    def test_training_time_positive(self, result):
        assert float(result["training_time_seconds"]) > 0

    def test_val_loss_curve_is_json_list(self, result):
        curve = json.loads(result["val_loss_curve"])
        assert isinstance(curve, list)
        assert len(curve) > 0

    def test_stopping_reason_valid(self, result):
        assert result["stopping_reason"] in (
            "DIVERGED", "EARLY_STOPPED", "MAX_EPOCHS"
        )

    def test_diverged_is_bool(self, result):
        assert isinstance(result["diverged"], bool)

    def test_healthy_is_bool(self, result):
        assert isinstance(result["healthy"], bool)

    def test_model_hyperparams_in_result(self, result):
        assert result["forecast_length"] == mc.FORECAST_LENGTH
        assert result["backcast_length"] == mc.BACKCAST_LENGTH
        assert result["n_stacks"] == mc.N_STACKS


# ===================================================================
# Summary Statistics (print_summary_statistics)
# ===================================================================

class TestPrintSummaryStatistics:
    """Verify print_summary_statistics handles various CSV states."""

    def _write_csv(self, path, rows):
        """Write a CSV with header + rows."""
        mc.init_csv(path)
        for row in rows:
            mc.append_result(path, row)

    def _make_healthy_row(self, config_name, run_idx, best_val=10.0, epochs=50):
        row = {col: "" for col in mc.MILK_CONVERGENCE_CSV_COLUMNS}
        row.update({
            "experiment": "milk_convergence",
            "config_name": config_name,
            "forecast_length": 6,
            "backcast_length": 24,
            "n_stacks": 6,
            "n_blocks_per_stack": 1,
            "share_weights": True,
            "run": run_idx,
            "seed": 42 + run_idx,
            "active_g": config_name == "Milk6_activeG",
            "activation": "ReLU",
            "best_val_loss": f"{best_val:.8f}",
            "final_val_loss": f"{best_val + 1:.8f}",
            "final_train_loss": f"{best_val - 1:.8f}",
            "best_epoch": epochs - 10,
            "epochs_trained": epochs,
            "stopping_reason": "EARLY_STOPPED",
            "loss_ratio": "1.1",
            "diverged": False,
            "healthy": True,
            "n_params": 4896768,
            "training_time_seconds": "30.0",
            "val_loss_curve": "[]",
        })
        return row

    def test_no_crash_on_missing_file(self, tmp_path, capsys):
        mc.print_summary_statistics(str(tmp_path / "nonexistent.csv"))
        captured = capsys.readouterr()
        assert "No results CSV found" in captured.out

    def test_no_crash_on_empty_csv(self, tmp_path, capsys):
        csv_path = str(tmp_path / "empty.csv")
        mc.init_csv(csv_path)
        mc.print_summary_statistics(csv_path)
        captured = capsys.readouterr()
        assert "empty" in captured.out.lower()

    def test_prints_summary_for_valid_data(self, tmp_path, capsys):
        csv_path = str(tmp_path / "results.csv")
        rows = [
            self._make_healthy_row("Milk6_baseline", i, best_val=12.0 + i * 0.5)
            for i in range(5)
        ]
        self._write_csv(csv_path, rows)
        mc.print_summary_statistics(csv_path)
        captured = capsys.readouterr()
        assert "SUMMARY STATISTICS" in captured.out
        assert "Milk6_baseline" in captured.out
        assert "COMPARISON TABLE" in captured.out

    def test_prints_convergence_rate(self, tmp_path, capsys):
        csv_path = str(tmp_path / "results.csv")
        rows = [
            self._make_healthy_row("Milk6_activeG", i, best_val=8.0 + i * 0.1)
            for i in range(3)
        ]
        self._write_csv(csv_path, rows)
        mc.print_summary_statistics(csv_path)
        captured = capsys.readouterr()
        assert "3/3" in captured.out

