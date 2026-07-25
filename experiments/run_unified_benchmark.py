"""
Unified Benchmark for N-BEATS Lightning — Paper Experiments

Tests three hypotheses:
  H1 (Basis invariance): N-BEATS performance is driven by architecture, not
      choice of basis function.
  H2 (active_g stabilization): active_g='forecast' stabilizes training and
      produces competitive results vs active_g=False.
  H3 (AE efficiency): Autoencoder-backbone blocks achieve comparable
      performance with 5-6x fewer parameters and faster training.

Two-pass design:
  Pass 1 ("baseline"):      active_g=False,      sum_losses=False, activation="ReLU"
  Pass 2 ("activeG_fcast"):  active_g="forecast", sum_losses=False, activation="ReLU"

Datasets: M4 (6 periods), Tourism (3 periods), Milk (1 period).

Usage:
    python experiments/run_unified_benchmark.py --dataset m4 --max-epochs 100
    python experiments/run_unified_benchmark.py --dataset m4 --periods Yearly --max-epochs 100
    python experiments/run_unified_benchmark.py --dataset tourism --max-epochs 100
    python experiments/run_unified_benchmark.py --dataset milk --max-epochs 500
    python experiments/run_unified_benchmark.py --dataset all
    python experiments/run_unified_benchmark.py --dataset m4 --max-epochs 100 --wandb
"""

import argparse
import csv
import gc
import json
import math
import multiprocessing as mp
import os
import queue
import shutil
import signal
import sys
import tempfile
import time
import warnings
from contextlib import contextmanager

# fcntl is Unix-only; use msvcrt on Windows for cross-platform file locking
if sys.platform == "win32":
    import msvcrt

    @contextmanager
    def _exclusive_lock(file_obj):
        file_obj.seek(0)
        while True:
            try:
                msvcrt.locking(file_obj.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                time.sleep(0.05)
        try:
            yield
        finally:
            file_obj.seek(0)
            msvcrt.locking(file_obj.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    @contextmanager
    def _exclusive_lock(file_obj):
        fcntl.flock(file_obj, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(file_obj, fcntl.LOCK_UN)


import numpy as np
import torch

# Intel XPU: import IPEX before lightning so its tuned XPU kernels override
# the native PyTorch XPU kernels. This emits a one-time C++ kernel-override
# warning at startup; it indicates IPEX kernels are active (faster on Arc).
try:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        import intel_extension_for_pytorch as _ipex  # noqa: F401
except ImportError:
    pass

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch import loggers as pl_loggers

# Allow running from project root or experiments/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lightningnbeats.models import NBeatsNet
from lightningnbeats.loaders import (
    ColumnarCollectionTimeSeriesDataModule,
    ColumnarCollectionTimeSeriesTestDataModule,
    TimeSeriesDataModule,
)
from lightningnbeats.data import (
    M4Dataset,
    TourismDataset,
    MilkDataset,
    TrafficDataset,
    WeatherDataset,
)

torch.set_float32_matmul_precision("medium")


# ---------------------------------------------------------------------------
# Signal Handling — Graceful Shutdown
# ---------------------------------------------------------------------------

_shutdown_requested = False
_shutdown_event = None  # multiprocessing.Event — set when running in parallel mode


def _signal_handler(signum, frame):
    global _shutdown_requested
    if _shutdown_requested:
        # Second signal — force exit
        print("\n[SIGNAL] Force exit.")
        os._exit(1)
    _shutdown_requested = True
    if _shutdown_event is not None:
        _shutdown_event.set()
    print(
        "\n[SIGNAL] Shutdown requested. Finishing current run... "
        "(signal again to force)"
    )


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ---------------------------------------------------------------------------
# Training Hyperparameters
# ---------------------------------------------------------------------------

LEARNING_RATE = 1e-3
THETAS_DIM = 5
LATENT_DIM = 4
BASIS_DIM = 128
LOSS = "SMAPELoss"
SHARE_WEIGHTS = True
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
PATIENCE = EARLY_STOPPING_PATIENCE  # alias for imports

BASE_SEED = 42
N_RUNS_DEFAULT = 10

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CLAIMS_DIR = os.path.join(RESULTS_DIR, ".claims")
# A claim older than this is assumed to belong to a crashed worker and may be
# re-taken. It MUST exceed the longest legitimate single-run wall time: the
# result_exists guards all run before training, never before append_result, so a
# prematurely stolen claim makes two workers train the same job and BOTH write a
# row — silently inflating the per-cell run count. Long-horizon sliding-window
# runs (Traffic/Weather at L=480) take 3-6h, well past the previous 4h value.
STALE_CLAIM_SECONDS = 86400  # 24 hours

# ---------------------------------------------------------------------------
# Dataset Periods
# ---------------------------------------------------------------------------

M4_PERIODS = {
    "Yearly": {"frequency": 1, "horizon": 6},
    "Quarterly": {"frequency": 4, "horizon": 8},
    "Monthly": {"frequency": 12, "horizon": 18},
    "Weekly": {"frequency": 1, "horizon": 13},
    "Daily": {"frequency": 1, "horizon": 14},
    "Hourly": {"frequency": 24, "horizon": 48},
}

TOURISM_PERIODS = {
    "Tourism-Yearly": {"frequency": 1, "horizon": 4},
    "Tourism-Monthly": {"frequency": 12, "horizon": 24},
    "Tourism-Quarterly": {"frequency": 4, "horizon": 8},
}

MILK_PERIODS = {
    "Milk": {"frequency": 12, "horizon": 6},
}

TRAFFIC_PERIODS = {
    "Traffic-96": {"frequency": 24, "horizon": 96},
    "Traffic-192": {"frequency": 24, "horizon": 192},
}

WEATHER_PERIODS = {
    "Weather-96": {"frequency": 144, "horizon": 96},
    "Weather-192": {"frequency": 144, "horizon": 192},
}

DATASET_PERIODS = {
    "m4": M4_PERIODS,
    "tourism": TOURISM_PERIODS,
    "milk": MILK_PERIODS,
    "traffic": TRAFFIC_PERIODS,
    "weather": WEATHER_PERIODS,
}

FORECAST_MULTIPLIERS = {
    "m4": 5,
    "tourism": 2,
    "milk": 4,
    "traffic": 2,
    "weather": 2,
}

# ---------------------------------------------------------------------------
# Batch Sizes
# ---------------------------------------------------------------------------

BATCH_SIZES = {
    ("m4", "Yearly"): 16384,
    ("m4", "Quarterly"): 16384,
    ("m4", "Monthly"): 16384,
    ("m4", "Weekly"): 8192,
    ("m4", "Daily"): 16384,
    ("m4", "Hourly"): 4096,
    ("tourism", "Tourism-Yearly"): 8192,
    ("tourism", "Tourism-Monthly"): 32768,
    ("tourism", "Tourism-Quarterly"): 65536,
    ("milk", "Milk"): 128,
    ("traffic", "Traffic-96"): 65536,
    ("traffic", "Traffic-192"): 65536,
    ("weather", "Weather-96"): 65536,
    ("weather", "Weather-192"): 65536,
}
DEFAULT_BATCH_SIZE = 65536

# ---------------------------------------------------------------------------
# Milk Overrides
# ---------------------------------------------------------------------------

MILK_N_STACKS = 6
MILK_MAX_EPOCHS = 500
MILK_PATIENCE = 20
MILK_N_RUNS = 100

# ---------------------------------------------------------------------------
# CSV Schema
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "experiment",
    "config_name",
    "category",
    "stack_types",
    "period",
    "frequency",
    "forecast_length",
    "backcast_length",
    "n_stacks",
    "n_blocks_per_stack",
    "share_weights",
    "run",
    "seed",
    "smape",
    "mase",
    "mae",
    "mse",
    "owa",
    "norm_mae",
    "norm_mse",
    "n_params",
    "t_width",
    "training_time_seconds",
    "epochs_trained",
    "active_g",
    "sum_losses",
    "activation",
    "stopping_reason",
    "best_val_loss",
    "final_val_loss",
    "final_train_loss",
    "best_epoch",
    "loss_ratio",
    "diverged",
    "val_loss_curve",
]

# ---------------------------------------------------------------------------
# 16 Unified Configs
# ---------------------------------------------------------------------------

UNIFIED_CONFIGS = {
    # --- Paper Baselines (3) ---
    "NBEATS-G": {
        "category": "paper_baseline",
        "stack_types": ["Generic"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "NBEATS-I": {
        "category": "paper_baseline",
        "stack_types": ["Trend", "Seasonality"],
        "n_blocks_per_stack": 3,
        "share_weights": True,
    },
    "NBEATS-I+G": {
        "category": "paper_baseline",
        "stack_types": ["Trend", "Seasonality"] + ["Generic"] * 28,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    # --- Novel Homogeneous — AE Backbone (3) ---
    "GenericAE": {
        "category": "novel_ae",
        "stack_types": ["GenericAE"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "BottleneckGenericAE": {
        "category": "novel_ae",
        "stack_types": ["BottleneckGenericAE"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "AutoEncoder": {
        "category": "novel_ae",
        "stack_types": ["AutoEncoder"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    # --- Novel Homogeneous — Basis Alternatives (3) ---
    "BottleneckGeneric": {
        "category": "novel_basis",
        "stack_types": ["BottleneckGeneric"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "Coif2WaveletV3": {
        "category": "novel_basis",
        "stack_types": ["Coif2WaveletV3"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "DB4WaveletV3": {
        "category": "novel_basis",
        "stack_types": ["DB4WaveletV3"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    # --- Novel Mixed Stacks (7) ---
    "Trend+Coif2WaveletV3": {
        "category": "novel_mixed",
        "stack_types": ["Trend", "Coif2WaveletV3"] * 15,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "Trend+DB3WaveletV3": {
        "category": "novel_mixed",
        "stack_types": ["Trend", "DB3WaveletV3"] * 15,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "Trend+HaarWaveletV3": {
        "category": "novel_mixed",
        "stack_types": ["Trend", "HaarWaveletV3"] * 15,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "Generic+DB3WaveletV3": {
        "category": "novel_mixed",
        "stack_types": ["Generic", "DB3WaveletV3"] * 15,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "NBEATS-I+GenericAE": {
        "category": "novel_mixed",
        "stack_types": ["Trend", "Seasonality"] + ["GenericAE"] * 28,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "NBEATS-I+BottleneckGeneric": {
        "category": "novel_mixed",
        "stack_types": ["Trend", "Seasonality"] + ["BottleneckGeneric"] * 28,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "NBEATS-I-AE": {
        "category": "novel_mixed",
        "stack_types": ["TrendAE", "SeasonalityAE"],
        "n_blocks_per_stack": 3,
        "share_weights": True,
    },
}


# ---------------------------------------------------------------------------
# Milk Config Scaler
# ---------------------------------------------------------------------------


def scale_config_for_milk(cfg):
    """Adapt a 30-stack config to 6 stacks for Milk dataset.

    - Homogeneous 30-stack -> 6-stack
    - NBEATS-I (2-stack interpretable) -> keep as-is
    - Mixed I+G pattern (2 + 28) -> 2 + 4
    - Alternating mixed (A, B) * 15 -> (A, B) * 3
    """
    stack_types = cfg["stack_types"]
    n_blocks = cfg["n_blocks_per_stack"]
    unique = list(dict.fromkeys(stack_types))
    n_stacks = len(stack_types)

    # 2-stack interpretable configs (NBEATS-I, NBEATS-I-AE) — keep as-is
    if n_stacks == 2:
        return stack_types, n_blocks

    # I+G pattern: starts with Trend, Seasonality, then 28x same block
    if (
        n_stacks == 30
        and stack_types[0] == "Trend"
        and stack_types[1] == "Seasonality"
        and len(set(stack_types[2:])) == 1
    ):
        # 2 interpretable + 4 generic-backend
        return ["Trend", "Seasonality"] + [stack_types[2]] * 4, n_blocks

    # I+G variant: Trend, Seasonality then 28 of same block (already covered above)
    # But also handle TrendAE-style if any exist
    if (
        n_stacks == 30
        and "Trend" in stack_types[0]
        and "Seasonality" in stack_types[1]
        and len(set(stack_types[2:])) == 1
    ):
        return [stack_types[0], stack_types[1]] + [stack_types[2]] * 4, n_blocks

    # Alternating mixed: [A, B] * 15 -> [A, B] * 3
    if n_stacks == 30 and len(unique) == 2:
        pattern = stack_types[:2]
        is_alternating = all(stack_types[i] == pattern[i % 2] for i in range(n_stacks))
        if is_alternating:
            return pattern * 3, n_blocks

    # Homogeneous 30-stack -> 6-stack
    if len(unique) == 1:
        return [unique[0]] * MILK_N_STACKS, n_blocks

    # Fallback: truncate to 6
    return stack_types[:MILK_N_STACKS], n_blocks


# ---------------------------------------------------------------------------
# Logger Helpers
# ---------------------------------------------------------------------------


def build_loggers(
    log_dir,
    log_name,
    wandb_enabled,
    wandb_project,
    wandb_group,
    wandb_config,
    tb_enabled=False,
):
    """Build list of loggers (optionally TensorBoard + W&B).

    TensorBoard is disabled by default for benchmark runs to avoid
    accumulating large event files on disk.  Enable with ``tb_enabled=True``
    or the ``--tensorboard`` CLI flag.
    """
    loggers = []
    if tb_enabled:
        loggers.append(pl_loggers.TensorBoardLogger(save_dir=log_dir, name=log_name))
    if wandb_enabled:
        loggers.append(
            pl_loggers.WandbLogger(
                project=wandb_project,
                group=wandb_group,
                name=log_name,
                config=wandb_config,
                save_dir=log_dir,
                reinit=True,
            )
        )
    return loggers if loggers else False


def finish_wandb(wandb_enabled):
    """Cleanly close the current wandb run if enabled."""
    if wandb_enabled:
        import wandb

        wandb.finish(quiet=True)


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def set_seed(seed):
    pl.seed_everything(seed, workers=True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_smape(y_pred, y_true):
    """sMAPE matching the M4 competition protocol."""
    eps = 1e-8
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps
    return np.mean(numerator / denominator) * 100.0


def build_series_index_map(col_indices, train_data_df):
    """Map each test prediction row to its source series index.

    ``ColumnarTimeSeriesDataset.col_indices`` is a list of ``(column, start_idx)``
    in dataloader order (``shuffle=False`` for test), so entry ``i`` identifies the
    series behind prediction row ``i``. This converts those column names into
    positional indices into ``train_data_df.columns`` — the same ordering used by
    ``BenchmarkDataset.get_training_series()``.

    Returns
    -------
    np.ndarray of int64, shape (len(col_indices),)
    """
    pos = {col: i for i, col in enumerate(train_data_df.columns)}
    return np.fromiter(
        (pos[col] for col, _ in col_indices),
        dtype=np.int64,
        count=len(col_indices),
    )


def _row_error_means(y_pred, y_true, chunk=200_000):
    """Row-wise mean |error| and mean error², computed in chunks to bound memory.

    The rolling-window test sets (Traffic/Weather) produce millions of rows, so a
    whole-array temporary would cost gigabytes.
    """
    n = y_pred.shape[0]
    row_mae = np.empty(n, dtype=np.float64)
    row_mse = np.empty(n, dtype=np.float64)
    for s in range(0, n, chunk):
        e = y_true[s : s + chunk].astype(np.float64) - y_pred[s : s + chunk]
        row_mae[s : s + chunk] = np.mean(np.abs(e), axis=1)
        row_mse[s : s + chunk] = np.mean(e * e, axis=1)
    return row_mae, row_mse


def compute_m4_mase(y_pred, y_true, train_series_list, frequency, series_idx=None):
    """M4-faithful MASE using full training history for naive seasonal denominator.

    Parameters
    ----------
    y_pred, y_true : np.ndarray, shape (n_rows, forecast_length)
        Predictions and ground truth, one row per test window.
    train_series_list : list of 1-D np.ndarray
        Full training history per series, in ``train_data.columns`` order
        (i.e. ``BenchmarkDataset.get_training_series()``).
    frequency : int
        Seasonal period used for the naive denominator.
    series_idx : array-like of int, optional
        ``series_idx[i]`` gives the index into ``train_series_list`` for prediction
        row ``i``. **Required whenever the test set yields more than one window per
        series** — the rolling-window layout Traffic and Weather have used since
        commit 41888b4. ``None`` keeps the M4/Tourism convention that row ``i`` *is*
        series ``i``, which holds only when there is exactly one window per series.
    """
    n_rows = y_pred.shape[0]
    m = max(1, frequency)

    if series_idx is None:
        if n_rows > len(train_series_list):
            raise ValueError(
                f"compute_m4_mase received {n_rows} prediction rows but only "
                f"{len(train_series_list)} training series. This test set yields "
                "more than one window per series; pass series_idx (see "
                "build_series_index_map) to map rows onto series."
            )
        idx = np.arange(n_rows, dtype=np.int64)
    else:
        idx = np.asarray(series_idx, dtype=np.int64)
        if idx.shape[0] != n_rows:
            raise ValueError(
                f"series_idx has {idx.shape[0]} entries but y_pred has {n_rows} rows."
            )
        if idx.size and (idx.max() >= len(train_series_list) or idx.min() < 0):
            raise ValueError(
                f"series_idx out of range for {len(train_series_list)} training series."
            )

    # Naive seasonal denominator, computed ONCE per series rather than per row.
    # NaN marks a series that the original implementation skipped via `continue`.
    denom = np.full(len(train_series_list), np.nan, dtype=np.float64)
    for s in np.unique(idx):
        train_s = train_series_list[s]
        if len(train_s) <= m:
            continue
        naive_mae = float(np.mean(np.abs(train_s[m:] - train_s[:-m])))
        if naive_mae < 1e-10:
            continue
        denom[s] = naive_mae

    forecast_mae, _ = _row_error_means(y_pred, y_true)
    row_denom = denom[idx]
    keep = np.isfinite(row_denom)
    if not keep.any():
        return float("nan")
    return float(np.mean(forecast_mae[keep] / row_denom[keep]))


def compute_mae(y_pred, y_true):
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_mse(y_pred, y_true):
    return float(np.mean((y_true - y_pred) ** 2))


def compute_normalized_mae_mse(preds, targets, train_data_df, series_idx=None):
    """Compute Z-score normalized MAE and MSE for comparison with published benchmarks.

    Normalizes per-series using per-column std from the training data, matching the
    methodology of N-HiTS, PatchTST, and related long-horizon papers. (The per-column
    mean cancels in the error difference, so only sigma is needed.)

    Parameters
    ----------
    preds : np.ndarray, shape (n_rows, forecast_length)
        Model predictions, one row per test window.
    targets : np.ndarray, shape (n_rows, forecast_length)
        Ground-truth values.
    train_data_df : pd.DataFrame, shape (n_timesteps, n_series)
        Columnar training data used to compute per-series normalization statistics.
    series_idx : array-like of int, optional
        ``series_idx[i]`` gives the column position in ``train_data_df`` for
        prediction row ``i``. Required when the test set yields more than one window
        per series (Traffic/Weather rolling-window layout). ``None`` keeps the
        one-row-per-series convention used by M4 and Tourism.

    Returns
    -------
    norm_mae, norm_mse : float
        Normalized MAE and MSE.
    """
    eps = 1e-8
    cols = train_data_df.columns
    n_rows = preds.shape[0]

    if series_idx is None:
        if n_rows > len(cols):
            raise ValueError(
                f"compute_normalized_mae_mse received {n_rows} prediction rows but "
                f"only {len(cols)} series. This test set yields more than one window "
                "per series; pass series_idx (see build_series_index_map)."
            )
        idx = np.arange(n_rows, dtype=np.int64)
    else:
        idx = np.asarray(series_idx, dtype=np.int64)
        if idx.shape[0] != n_rows:
            raise ValueError(
                f"series_idx has {idx.shape[0]} entries but preds has {n_rows} rows."
            )
        if idx.size and (idx.max() >= len(cols) or idx.min() < 0):
            raise ValueError(f"series_idx out of range for {len(cols)} series.")

    # Per-series sigma, computed once. NaN marks an all-empty column, which the
    # original implementation skipped via `continue`.
    sigma = np.full(len(cols), np.nan, dtype=np.float64)
    for s in np.unique(idx):
        series_vals = train_data_df[cols[s]].dropna().values
        if len(series_vals) == 0:
            continue
        sd = float(np.std(series_vals))
        sigma[s] = 1.0 if sd < eps else sd

    row_mae, row_mse = _row_error_means(preds, targets)
    row_sigma = sigma[idx]
    keep = np.isfinite(row_sigma)
    if not keep.any():
        return float("nan"), float("nan")

    norm_mae = float(np.mean(row_mae[keep] / row_sigma[keep]))
    norm_mse = float(np.mean(row_mse[keep] / row_sigma[keep] ** 2))
    return norm_mae, norm_mse


def resolve_accelerator(accelerator_override):
    """Resolve accelerator and device from override string."""
    if accelerator_override == "auto":
        if torch.cuda.is_available():
            accelerator = "cuda"
            device = torch.device("cuda")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            accelerator = "xpu"
            device = torch.device("xpu")
        elif torch.backends.mps.is_available():
            accelerator = "mps"
            device = torch.device("mps")
        else:
            accelerator = "cpu"
            device = torch.device("cpu")
    else:
        accelerator = accelerator_override
        device = torch.device(accelerator)
    return accelerator, device


def run_inference(model, test_dm, device):
    """Run inference, collecting (predictions, targets) as numpy arrays."""
    model.eval()
    model.to(device)

    all_preds = []
    all_targets = []

    test_dm.setup("test")

    with torch.no_grad():
        for batch in test_dm.test_dataloader():
            x, y = batch
            x = x.to(device)
            _, forecast = model(x)
            all_preds.append(forecast.cpu().numpy())
            all_targets.append(y.numpy())

    return np.concatenate(all_preds, axis=0), np.concatenate(all_targets, axis=0)


def load_dataset(dataset_name, period, train_ratio=None, include_target=None):
    """Factory function to load the appropriate benchmark dataset.

    ``train_ratio`` and ``include_target`` are only used by the Traffic and
    Weather benchmark datasets; all other dataset types ignore them.
    """
    if dataset_name == "m4":
        return M4Dataset(period, "All")
    elif dataset_name == "tourism":
        period_name = period.replace("Tourism-", "")
        return TourismDataset(period_name)
    elif dataset_name == "milk":
        return MilkDataset()
    elif dataset_name == "traffic":
        horizon = TRAFFIC_PERIODS[period]["horizon"]
        dataset_kwargs = {}
        if train_ratio is not None:
            dataset_kwargs["train_ratio"] = train_ratio
        if include_target is not None:
            dataset_kwargs["include_target"] = include_target
        return TrafficDataset(horizon, **dataset_kwargs)
    elif dataset_name == "weather":
        horizon = WEATHER_PERIODS[period]["horizon"]
        dataset_kwargs = {}
        if train_ratio is not None:
            dataset_kwargs["train_ratio"] = train_ratio
        if include_target is not None:
            dataset_kwargs["include_target"] = include_target
        return WeatherDataset(horizon, **dataset_kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_batch_size(dataset_name, period, override=None):
    """Look up per-dataset/period batch size, with optional override.

    Priority: explicit override > tuned per-dataset/period > DEFAULT_BATCH_SIZE.
    The override comes from YAML training.batch_size or CLI --batch-size.
    """
    if override is not None:
        return override
    tuned = BATCH_SIZES.get((dataset_name, period))
    if tuned is not None:
        return tuned
    return DEFAULT_BATCH_SIZE


# ---------------------------------------------------------------------------
# DivergenceDetector Callback
# ---------------------------------------------------------------------------


class DivergenceDetector(pl.Callback):
    """Stop training when val_loss exceeds best by relative_threshold for consecutive epochs."""

    def __init__(self, relative_threshold=3.0, consecutive_epochs=3):
        super().__init__()
        self.relative_threshold = relative_threshold
        self.consecutive_epochs = consecutive_epochs
        self.best_val_loss = float("inf")
        self.bad_epoch_count = 0
        self.diverged = False

    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip sanity check — random weights + AMP often produce NaN val_loss
        # before any training has occurred.
        if trainer.sanity_checking:
            return

        v = trainer.callback_metrics.get("val_loss")
        if v is None:
            return

        val_loss = float(v)

        if not math.isfinite(val_loss):
            self.diverged = True
            trainer.should_stop = True
            return

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.bad_epoch_count = 0
            return

        if (
            self.best_val_loss > 0
            and val_loss > self.best_val_loss * self.relative_threshold
        ):
            self.bad_epoch_count += 1
        else:
            self.bad_epoch_count = 0

        if self.bad_epoch_count >= self.consecutive_epochs:
            self.diverged = True
            trainer.should_stop = True


class ConvergenceTracker(pl.Callback):
    """Records per-epoch losses for convergence analysis."""

    def __init__(self):
        super().__init__()
        self.val_losses = []
        self.train_losses = []

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        v = trainer.callback_metrics.get("val_loss")
        if v is not None:
            self.val_losses.append(float(v))

    def on_train_epoch_end(self, trainer, pl_module):
        v = trainer.callback_metrics.get("train_loss")
        if v is not None:
            self.train_losses.append(float(v))


class LearningRateLogger(pl.Callback):
    """Logs the learning rate at the start of each training epoch."""

    def on_train_epoch_start(self, trainer, pl_module):
        lr = trainer.optimizers[0].param_groups[0]["lr"]
        pl_module.log("lr_epoch", lr, prog_bar=True)
        print(f"  Epoch {trainer.current_epoch + 1}: lr = {lr:.6f}")


# ---------------------------------------------------------------------------
# CSV Helpers
# ---------------------------------------------------------------------------


def init_csv(path, columns=None):
    """Create CSV with header if it doesn't exist, or migrate header if schema changed."""
    columns = columns or CSV_COLUMNS
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
        return

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        try:
            existing_header = next(reader)
        except StopIteration:
            existing_header = []
    if existing_header == columns:
        return

    # Schema mismatch — migrate the file
    print(
        f"  [MIGRATE] {os.path.basename(path)}: "
        f"header {len(existing_header)} cols -> {len(columns)} cols"
    )
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        old_header = next(reader)
        raw_rows = list(reader)

    migrated = []
    for raw in raw_rows:
        row_dict = {}
        for i, col_name in enumerate(old_header):
            if col_name in columns and i < len(raw):
                row_dict[col_name] = raw[i]
        if len(raw) > len(old_header):
            missing_cols = [c for c in columns if c not in old_header]
            extra_values = raw[len(old_header) :]
            for j, col_name in enumerate(missing_cols):
                if j < len(extra_values):
                    row_dict[col_name] = extra_values[j]
        for col in columns:
            if col not in row_dict:
                row_dict[col] = ""
        migrated.append(row_dict)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(migrated)
    print(f"  [MIGRATE] {os.path.basename(path)}: migrated {len(migrated)} rows")


def append_result(path, row_dict, columns=None):
    """Append a single result row to CSV (process-safe via file locking)."""
    columns = columns or CSV_COLUMNS
    lock_path = path + ".lock"
    with open(lock_path, "w") as lock_f:
        with _exclusive_lock(lock_f):
            with open(path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writerow(row_dict)


def result_exists(path, experiment, config_name, period, run):
    """Check if a result row already exists in the CSV."""
    if not os.path.exists(path):
        return False
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (
                row["experiment"] == experiment
                and row["config_name"] == config_name
                and row["period"] == period
                and row["run"] == str(run)
            ):
                return True
    return False


def _claim_key(config_name, dataset_name, horizon_label, run_idx):
    """Deterministic filename for a job claim file."""
    return f"{config_name}__{dataset_name}__{horizon_label}__run{run_idx}.claim"


def build_claim_config_name(experiment_name, config_name):
    """Make claim identity pass-aware for two-pass/search YAML jobs."""
    if not experiment_name:
        return config_name
    return f"{experiment_name}__{config_name}"


def claim_job(config_name, dataset_name, horizon_label, run_idx, worker_id=""):
    """Atomically claim a job. Returns True if this process won the claim."""
    os.makedirs(CLAIMS_DIR, exist_ok=True)
    claim_path = os.path.join(
        CLAIMS_DIR,
        _claim_key(config_name, dataset_name, horizon_label, run_idx),
    )
    try:
        file_descriptor = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(file_descriptor, "w") as claim_file:
            claim_file.write(
                json.dumps(
                    {
                        "worker_id": worker_id,
                        "pid": os.getpid(),
                        "claimed_at": time.time(),
                        "config_name": config_name,
                        "dataset": dataset_name,
                        "horizon": horizon_label,
                        "run": run_idx,
                    }
                )
            )
        return True
    except FileExistsError:
        try:
            claim_age_seconds = time.time() - os.path.getmtime(claim_path)
            if claim_age_seconds > STALE_CLAIM_SECONDS:
                print(
                    f"  [STALE] Reclaiming stale job: "
                    f"{config_name}/{dataset_name}-{horizon_label}/run{run_idx}"
                )
                os.unlink(claim_path)
                return claim_job(
                    config_name,
                    dataset_name,
                    horizon_label,
                    run_idx,
                    worker_id,
                )
        except OSError:
            pass
        return False


def release_claim(config_name, dataset_name, horizon_label, run_idx):
    """Remove a claim file after the job is complete or aborted."""
    claim_path = os.path.join(
        CLAIMS_DIR,
        _claim_key(config_name, dataset_name, horizon_label, run_idx),
    )
    try:
        os.unlink(claim_path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Single Run Function
# ---------------------------------------------------------------------------


def run_single_experiment(
    experiment_name,
    config_name,
    category,
    stack_types,
    period,
    run_idx,
    dataset,
    train_series_list,
    csv_path,
    n_blocks_per_stack=1,
    share_weights=True,
    active_g=False,
    sum_losses=False,
    activation="ReLU",
    max_epochs=100,
    patience=EARLY_STOPPING_PATIENCE,
    batch_size=65536,
    accelerator_override="auto",
    forecast_multiplier=5,
    num_workers=0,
    wandb_enabled=False,
    wandb_project="nbeats-lightning",
    save_predictions=True,
    predictions_dir=None,
    gpu_id=None,
    dataset_name=None,
    worker_id="",
    basis_dim=BASIS_DIM,
    basis_offset=0,
    stack_basis_offsets=None,
    forecast_basis_dim=None,
    extra_row=None,
    csv_columns=None,
    tb_enabled=False,
    thetas_dim_override=None,
    latent_dim_override=None,
    latent_gate_fn="sigmoid",
    trend_thetas_dim=None,
    lr_scheduler_config=None,
    optimizer_name="Adam",
    learning_rate=None,  # None → uses module-level LEARNING_RATE constant
    loss_override=None,  # None → uses module-level LOSS constant
    normalize=False,
    val_ratio=None,
    datamodule_type="columnar",  # "columnar" (temporal split) or "univariate" (80/20 random split)
    seed=None,  # None → BASE_SEED + run_idx
    wavelet_type="db3",  # Wavelet family for TrendWaveletAE/TrendWaveletAELG blocks
    backcast_wavelet_type=None,  # Override wavelet family for backcast path only
    forecast_wavelet_type=None,  # Override wavelet family for forecast path only
    skip_distance=0,  # Re-inject original input every N stacks (0 = disabled)
    skip_alpha=0.0,  # Mixing weight for skip injection (float or "learnable")
    generic_dim=5,  # Learned generic branch rank for TrendWaveletGeneric* blocks
    t_width=256,  # Hidden layer width for Trend/TrendWavelet-family blocks
    kl_weight=0.1,  # KL divergence loss weight for VAE-family blocks
    sampling_style="sliding",  # "sliding" | "nhits_paper" | "nbeats_paper"
    steps_per_epoch=None,  # required when sampling_style in ("nhits_paper","nbeats_paper")
    sampling_weights="uniform",  # "uniform" | "by_series"
    lh_multiplier=None,  # required when sampling_style="nbeats_paper"; Lh = lh_multiplier * H
    acknowledge_epoch_semantics=False,  # opt-in for max_epochs under paper sampling
    max_steps=None,  # paper-faithful total gradient-step budget; overrides max_epochs when set
    val_check_interval=None,  # int (steps) for nbeats_paper; None = every epoch
    min_delta=0.0,  # minimum improvement to reset EarlyStopping patience
    min_epochs=0,  # minimum epochs before EarlyStopping can fire
):
    """Run a single training + evaluation experiment and save results to CSV."""

    prefix = f"[GPU {gpu_id}] " if gpu_id is not None else ""

    # Check resumability
    if result_exists(csv_path, experiment_name, config_name, period, run_idx):
        print(
            f"  {prefix}[SKIP] {config_name} / {period} / run {run_idx} -- already exists"
        )
        return

    effective_dataset_name = dataset_name or getattr(dataset, "name", "unknown")
    claim_config_name = build_claim_config_name(experiment_name, config_name)
    if not claim_job(
        claim_config_name,
        effective_dataset_name,
        period,
        run_idx,
        worker_id,
    ):
        print(
            f"  {prefix}[CLAIMED] {config_name} / {period} / run {run_idx} "
            f"(another worker)"
        )
        return

    try:
        # ── Sampling-style validation + epochs-vs-steps guard ───────────
        # Paper sampling changes epoch semantics (1 epoch = steps_per_epoch
        # gradient updates). Validate up front before any expensive work.
        if sampling_style not in ("sliding", "nhits_paper", "nbeats_paper"):
            raise ValueError(
                f"sampling_style must be one of {{'sliding','nhits_paper','nbeats_paper'}}, "
                f"got {sampling_style!r}"
            )
        if sampling_weights not in ("uniform", "by_series"):
            raise ValueError(
                f"sampling_weights must be one of {{'uniform','by_series'}}, "
                f"got {sampling_weights!r}"
            )
        if sampling_style in ("nhits_paper", "nbeats_paper"):
            if (
                steps_per_epoch is None
                or not isinstance(steps_per_epoch, int)
                or isinstance(steps_per_epoch, bool)
                or steps_per_epoch <= 0
            ):
                raise ValueError(
                    f"sampling_style={sampling_style!r} requires protocol.steps_per_epoch "
                    f"to be a positive int, got {steps_per_epoch!r}"
                )
            if datamodule_type != "columnar":
                raise ValueError(
                    f"sampling_style={sampling_style!r} is only supported with "
                    "datamodule='columnar'; got datamodule_type="
                    f"{datamodule_type!r}."
                )
            if max_steps is None:
                if not acknowledge_epoch_semantics:
                    raise ValueError(
                        f"sampling_style={sampling_style!r} changes epoch semantics: one "
                        "epoch = steps_per_epoch gradient updates. Specify "
                        "training.max_steps for an unambiguous budget, or "
                        "set protocol.acknowledge_epoch_semantics: true to "
                        "keep using training.max_epochs (total steps will "
                        "be max_epochs * steps_per_epoch)."
                    )
                warnings.warn(
                    f"sampling_style={sampling_style!r} with max_epochs={max_epochs} "
                    f"and no max_steps: total training = "
                    f"{max_epochs * steps_per_epoch} gradient steps "
                    f"(steps_per_epoch={steps_per_epoch}).",
                    UserWarning,
                    stacklevel=2,
                )

        if sampling_style == "nbeats_paper":
            if (
                lh_multiplier is None
                or isinstance(lh_multiplier, bool)
                or not isinstance(lh_multiplier, (int, float))
                or lh_multiplier <= 0
            ):
                raise ValueError(
                    "sampling_style='nbeats_paper' requires protocol.lh_multiplier "
                    f"to be a positive number, got {lh_multiplier!r}"
                )

        seed = seed if seed is not None else (BASE_SEED + run_idx)
        set_seed(seed)

        forecast_length = dataset.forecast_length
        frequency = dataset.frequency
        backcast_length = forecast_length * forecast_multiplier

        n_stacks = len(stack_types)

        accelerator, device = resolve_accelerator(accelerator_override)

        # Mixed precision on CUDA — prefer bf16 (same exponent range as fp32,
        # avoids the fp16 overflow that kills 30-stack N-BEATS on M4).
        if accelerator == "cuda" and torch.cuda.is_bf16_supported():
            precision = "bf16-mixed"
        elif accelerator == "cuda":
            precision = "32-true"  # fp16 overflows; fall back to fp32
        else:
            precision = "32-true"

        # Create model
        effective_thetas_dim = (
            thetas_dim_override if thetas_dim_override is not None else THETAS_DIM
        )
        effective_latent_dim = (
            latent_dim_override if latent_dim_override is not None else LATENT_DIM
        )
        model = NBeatsNet(
            backcast_length=backcast_length,
            forecast_length=forecast_length,
            stack_types=stack_types,
            n_blocks_per_stack=n_blocks_per_stack,
            share_weights=share_weights,
            thetas_dim=effective_thetas_dim,
            loss=loss_override if loss_override is not None else LOSS,
            active_g=active_g,
            sum_losses=sum_losses,
            activation=activation,
            latent_dim=effective_latent_dim,
            latent_gate_fn=latent_gate_fn,
            basis_dim=basis_dim,
            basis_offset=basis_offset,
            stack_basis_offsets=stack_basis_offsets,
            forecast_basis_dim=forecast_basis_dim,
            trend_thetas_dim=trend_thetas_dim,
            generic_dim=generic_dim,
            wavelet_type=wavelet_type,
            backcast_wavelet_type=backcast_wavelet_type,
            forecast_wavelet_type=forecast_wavelet_type,
            learning_rate=(
                learning_rate if learning_rate is not None else LEARNING_RATE
            ),
            optimizer_name=optimizer_name,
            no_val=False,
            lr_scheduler_config=lr_scheduler_config,
            skip_distance=skip_distance,
            skip_alpha=skip_alpha,
            t_width=t_width,
            kl_weight=kl_weight,
        )

        n_params = count_parameters(model)

        # Data modules
        train_data = dataset.train_data
        test_data = dataset.test_data
        ext_val_data = getattr(dataset, "val_data", None)

        pin_memory = num_workers > 0
        if datamodule_type == "univariate":
            # Single-series mode: flat numpy array, 80/20 random split
            # Matches standalone convergence scripts (e.g. run_milk_convergence_10stack.py)
            flat_data = train_data.values.flatten().astype(np.float32)
            dm = TimeSeriesDataModule(
                data=flat_data,
                batch_size=batch_size,
                backcast_length=backcast_length,
                forecast_length=forecast_length,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            dm.setup()
        else:
            dm = ColumnarCollectionTimeSeriesDataModule(
                train_data,
                backcast_length=backcast_length,
                forecast_length=forecast_length,
                batch_size=batch_size,
                no_val=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                normalize=normalize,
                val_data=ext_val_data,
                val_ratio=val_ratio if ext_val_data is None else None,
                sampling_style=sampling_style,
                steps_per_epoch=steps_per_epoch,
                sampling_weights=sampling_weights,
                lh_multiplier=lh_multiplier,
            )
            dm.setup()

        test_dm = ColumnarCollectionTimeSeriesTestDataModule(
            train_data,
            test_data,
            backcast_length=backcast_length,
            forecast_length=forecast_length,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            col_means=dm.col_means if hasattr(dm, "col_means") else None,
            col_stds=dm.col_stds if hasattr(dm, "col_stds") else None,
        )

        # Trainer
        log_dir = os.path.join(RESULTS_DIR, "lightning_logs")
        log_name = f"unified/{experiment_name}/{config_name}/{period}/run{run_idx}"

        # Use a temp directory for checkpoints — they are only needed to reload
        # the best-epoch weights for evaluation, then deleted immediately.
        ckpt_tmp_dir = tempfile.mkdtemp(prefix="nbeats_ckpt_")
        chk_callback = ModelCheckpoint(
            dirpath=ckpt_tmp_dir,
            filename="best-checkpoint",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
        )

        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=patience,
            mode="min",
            verbose=False,
            min_delta=min_delta,
        )

        wandb_group = f"unified/{period}"
        exp_loggers = build_loggers(
            log_dir=log_dir,
            log_name=log_name,
            wandb_enabled=wandb_enabled,
            wandb_project=wandb_project,
            wandb_group=wandb_group,
            wandb_config={
                "dataset": dataset.name,
                "period": period,
                "config_name": config_name,
                "category": category,
                "stack_types": stack_types,
                "n_stacks": n_stacks,
                "n_blocks_per_stack": n_blocks_per_stack,
                "share_weights": share_weights,
                "backcast_length": backcast_length,
                "forecast_length": forecast_length,
                "batch_size": batch_size,
                "max_epochs": max_epochs,
                "seed": seed,
                "run_idx": run_idx,
                "active_g": active_g,
                "sum_losses": sum_losses,
                "activation": activation,
                "n_params": n_params,
            },
            tb_enabled=tb_enabled,
        )

        divergence_detector = DivergenceDetector(
            relative_threshold=3.0,
            consecutive_epochs=3,
        )
        convergence_tracker = ConvergenceTracker()
        lr_logger = LearningRateLogger()

        # Lightning 2.6.x _choose_strategy hardcodes device="cpu" for any
        # accelerator that isn't CUDA/MPS, breaking custom XPU accelerators.
        # Passing an explicit SingleDeviceStrategy bypasses that code path.
        if accelerator == "xpu":
            from lightning.pytorch.strategies import SingleDeviceStrategy
            _xpu_strategy = SingleDeviceStrategy(device=device)
        else:
            _xpu_strategy = "auto"

        trainer = pl.Trainer(
            accelerator=accelerator,
            strategy=_xpu_strategy,
            devices=1,
            max_epochs=max_epochs,
            min_epochs=min_epochs if min_epochs > 0 else None,
            max_steps=max_steps if max_steps is not None else -1,
            val_check_interval=(
                val_check_interval if val_check_interval is not None else 1.0
            ),
            precision=precision,
            gradient_clip_val=1.0,
            callbacks=[
                chk_callback,
                early_stop_callback,
                divergence_detector,
                convergence_tracker,
                lr_logger,
            ],
            logger=exp_loggers,
            enable_progress_bar=True,
            deterministic=False,
            log_every_n_steps=1,
        )

        # Train
        stack_summary = (
            f"{n_stacks}x{stack_types[0]}"
            if len(set(stack_types)) == 1
            else f"{n_stacks} mixed"
        )
        print(
            f"  {prefix}[RUN]  {config_name} / {period} / run {run_idx} "
            f"(seed={seed}, {stack_summary}, blocks/stack={n_blocks_per_stack}, "
            f"share_w={share_weights}, params={n_params:,})"
        )
        t0 = time.time()
        trainer.fit(model, datamodule=dm)
        training_time = time.time() - t0

        # Load best checkpoint, then delete temp checkpoint directory
        best_path = trainer.checkpoint_callback.best_model_path
        if best_path:
            model = NBeatsNet.load_from_checkpoint(best_path, weights_only=False)
        shutil.rmtree(ckpt_tmp_dir, ignore_errors=True)
        epochs_trained = trainer.current_epoch

        # Classify stopping reason
        if divergence_detector.diverged:
            stopping_reason = "DIVERGED"
        elif (
            hasattr(early_stop_callback, "stopped_epoch")
            and early_stop_callback.stopped_epoch > 0
        ):
            stopping_reason = "EARLY_STOPPED"
        else:
            stopping_reason = "MAX_EPOCHS"

        # Convergence statistics
        best_val_loss = (
            float(trainer.checkpoint_callback.best_model_score)
            if trainer.checkpoint_callback.best_model_score is not None
            else float("nan")
        )
        final_val_loss = (
            convergence_tracker.val_losses[-1]
            if convergence_tracker.val_losses
            else float("nan")
        )
        final_train_loss = (
            convergence_tracker.train_losses[-1]
            if convergence_tracker.train_losses
            else float("nan")
        )
        best_epoch = (
            int(np.argmin(convergence_tracker.val_losses))
            if convergence_tracker.val_losses
            else 0
        )
        loss_ratio = (
            final_val_loss / best_val_loss
            if best_val_loss > 0 and math.isfinite(best_val_loss)
            else float("nan")
        )
        diverged = divergence_detector.diverged or not math.isfinite(best_val_loss)

        # Inference
        preds, targets = run_inference(model, test_dm, device)

        # Map each prediction row onto its source series. M4/Tourism emit exactly
        # one window per series, but the Traffic/Weather rolling-window test sets
        # emit thousands per series, so the per-series metrics below need an
        # explicit row -> series mapping rather than assuming row i == series i.
        series_idx = None
        test_ds = getattr(test_dm, "test_dataset", None)
        if test_ds is not None and hasattr(test_ds, "col_indices"):
            series_idx = build_series_index_map(test_ds.col_indices, train_data)

        # Metrics
        smape = compute_smape(preds, targets)
        mase = compute_m4_mase(
            preds, targets, train_series_list, frequency, series_idx=series_idx
        )
        mae = compute_mae(preds, targets)
        mse = compute_mse(preds, targets)
        owa = dataset.compute_owa(smape, mase)
        # Normalized MAE/MSE: per-series Z-score using training-data statistics,
        # matching the evaluation protocol in N-HiTS, PatchTST, etc. for
        # Traffic and Weather datasets.  For M4, sMAPE/OWA remain primary;
        # norm_mae/norm_mse are still recorded for completeness.
        norm_mae, norm_mse = compute_normalized_mae_mse(
            preds, targets, train_data, series_idx=series_idx
        )

        # Update diverged flag — only for genuinely non-finite metrics.
        # The smape >= 200 threshold was removed: a high sMAPE from a model that
        # early-stopped is "poor" but not "diverged".  Actual divergence is already
        # caught by DivergenceDetector (NaN val_loss or 3× spike for 3 epochs).
        diverged = diverged or not math.isfinite(smape)

        print(
            f"  {prefix}       sMAPE={smape:.4f}  MASE={mase:.4f}  OWA={owa:.4f}  "
            f"nMAE={norm_mae:.4f}  nMSE={norm_mse:.4f}  "
            f"time={training_time:.1f}s  epochs={epochs_trained}  [{stopping_reason}]"
        )

        # Save predictions
        if save_predictions and predictions_dir is not None:
            os.makedirs(predictions_dir, exist_ok=True)
            npz_name = f"{experiment_name}_{config_name}_{period}_run{run_idx}.npz"
            np.savez(
                os.path.join(predictions_dir, npz_name),
                preds=preds,
                targets=targets,
            )

        # Save result
        unique_types = list(dict.fromkeys(stack_types))
        row = {
            "experiment": experiment_name,
            "config_name": config_name,
            "category": category,
            "stack_types": str(unique_types),
            "period": period,
            "frequency": frequency,
            "forecast_length": forecast_length,
            "backcast_length": backcast_length,
            "n_stacks": n_stacks,
            "n_blocks_per_stack": n_blocks_per_stack,
            "share_weights": share_weights,
            "run": run_idx,
            "seed": seed,
            "smape": f"{smape:.6f}",
            "mase": f"{mase:.6f}",
            "mae": f"{mae:.6f}",
            "mse": f"{mse:.6f}",
            "owa": f"{owa:.6f}",
            "norm_mae": f"{norm_mae:.6f}" if math.isfinite(norm_mae) else "nan",
            "norm_mse": f"{norm_mse:.6f}" if math.isfinite(norm_mse) else "nan",
            "n_params": n_params,
            "t_width": model.t_width,
            "training_time_seconds": f"{training_time:.2f}",
            "epochs_trained": epochs_trained,
            "active_g": active_g,
            "sum_losses": sum_losses,
            "activation": activation,
            "stopping_reason": stopping_reason,
            "best_val_loss": f"{best_val_loss:.8f}",
            "final_val_loss": f"{final_val_loss:.8f}",
            "final_train_loss": f"{final_train_loss:.8f}",
            "best_epoch": best_epoch,
            "loss_ratio": f"{loss_ratio:.6f}" if math.isfinite(loss_ratio) else "nan",
            "diverged": diverged,
            "val_loss_curve": json.dumps(
                [f"{value:.8f}" for value in convergence_tracker.val_losses]
            ),
        }
        if extra_row:
            row.update(extra_row)
        append_result(csv_path, row, columns=csv_columns)
        finish_wandb(wandb_enabled)

        # Cleanup
        del model, trainer, dm, test_dm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
    finally:
        release_claim(claim_config_name, effective_dataset_name, period, run_idx)


# ---------------------------------------------------------------------------
# Multi-GPU Support
# ---------------------------------------------------------------------------


def resolve_n_gpus(args):
    """Determine number of GPUs to use for parallel execution."""
    if args.accelerator in ("auto", "cuda") and torch.cuda.is_available():
        available = torch.cuda.device_count()
        if args.n_gpus is not None:
            return min(args.n_gpus, available)
        return available
    if (
        args.accelerator in ("auto", "xpu")
        and hasattr(torch, "xpu")
        and torch.xpu.is_available()
    ):
        available = torch.xpu.device_count()
        if args.n_gpus is not None:
            return min(args.n_gpus, available)
        return available
    return 0


def _build_job_list(periods, n_runs, dataset_name, batch_size_override):
    """Build flat list of job dicts for all (period, pass, config, run_idx) combos."""
    passes = [
        ("baseline", False),
        ("activeG_fcast", "forecast"),
    ]
    jobs = []
    for period in periods:
        batch_size = get_batch_size(dataset_name, period, batch_size_override)
        for pass_name, active_g in passes:
            for config_name, cfg in UNIFIED_CONFIGS.items():
                for run_idx in range(n_runs):
                    jobs.append(
                        {
                            "period": period,
                            "pass_name": pass_name,
                            "active_g": active_g,
                            "config_name": config_name,
                            "cfg": cfg,
                            "run_idx": run_idx,
                            "batch_size": batch_size,
                        }
                    )
    return jobs


def _gpu_worker(gpu_id, job_queue, shutdown_event, worker_args):
    """Worker process: pins to GPU gpu_id, pulls jobs from shared queue."""
    # Pin to specific device before any GPU context is created
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        os.environ["ONEAPI_DEVICE_SELECTOR"] = f"level_zero:{gpu_id}"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Ignore signals in worker — main process handles shutdown via event
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    torch.set_float32_matmul_precision("medium")

    prefix = f"[GPU {gpu_id}]"
    print(f"{prefix} Worker started (device={gpu_id}).")

    # Cache datasets per period to avoid redundant loading
    dataset_cache = {}
    series_cache = {}

    while not shutdown_event.is_set():
        try:
            job = job_queue.get_nowait()
        except queue.Empty:
            break

        dataset_name = worker_args["dataset_name"]
        is_milk = worker_args["is_milk"]
        period = job["period"]

        # Load dataset (cache by period)
        if period not in dataset_cache:
            dataset_cache[period] = load_dataset(dataset_name, period)
            series_cache[period] = dataset_cache[period].get_training_series()
        dataset = dataset_cache[period]
        train_series_list = series_cache[period]

        # Adapt config for Milk
        cfg = job["cfg"]
        if is_milk:
            stack_types, n_blocks = scale_config_for_milk(cfg)
        else:
            stack_types = cfg["stack_types"]
            n_blocks = cfg["n_blocks_per_stack"]

        run_single_experiment(
            experiment_name=job["pass_name"],
            config_name=job["config_name"],
            category=cfg["category"],
            stack_types=stack_types,
            period=period,
            run_idx=job["run_idx"],
            dataset=dataset,
            train_series_list=train_series_list,
            csv_path=worker_args["csv_path"],
            n_blocks_per_stack=n_blocks,
            share_weights=cfg["share_weights"],
            active_g=job["active_g"],
            sum_losses=False,
            activation="ReLU",
            max_epochs=worker_args["max_epochs"],
            patience=worker_args["patience"],
            batch_size=job["batch_size"],
            accelerator_override=(
                "xpu"
                if hasattr(torch, "xpu") and torch.xpu.is_available()
                else "cuda"
            ),
            forecast_multiplier=worker_args["forecast_multiplier"],
            num_workers=worker_args["num_workers"],
            wandb_enabled=worker_args["wandb_enabled"],
            wandb_project=worker_args["wandb_project"],
            save_predictions=worker_args["save_predictions"],
            predictions_dir=worker_args["predictions_dir"],
            gpu_id=gpu_id,
            dataset_name=dataset_name,
            worker_id=worker_args.get("worker_id") or f"gpu{gpu_id}",
            tb_enabled=worker_args.get("tb_enabled", False),
        )

    print(f"{prefix} Worker finished.")


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------


def _resolve_benchmark_params(args, dataset_override=None):
    """Resolve common benchmark parameters from CLI args.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    dataset_override : str or None
        If provided, use this dataset name instead of ``args.dataset``.
        Used by the ``--dataset all`` path to iterate over datasets while
        keeping the rest of the CLI args intact.
    """
    dataset_name = dataset_override or args.dataset
    is_milk = dataset_name == "milk"

    # Resolve periods
    all_periods = DATASET_PERIODS[dataset_name]
    if args.periods:
        periods = [p for p in args.periods if p in all_periods]
        if not periods:
            print(
                f"[ERROR] No valid periods for dataset '{dataset_name}'. "
                f"Available: {list(all_periods.keys())}"
            )
            return None
    else:
        periods = list(all_periods.keys())

    # Resolve run count and training params
    if is_milk:
        n_runs = args.n_runs if args.n_runs is not None else MILK_N_RUNS
        max_epochs = (
            args.max_epochs if args.max_epochs != MAX_EPOCHS else MILK_MAX_EPOCHS
        )
        patience = MILK_PATIENCE
    else:
        n_runs = args.n_runs if args.n_runs is not None else N_RUNS_DEFAULT
        max_epochs = args.max_epochs
        patience = EARLY_STOPPING_PATIENCE

    forecast_multiplier = FORECAST_MULTIPLIERS[dataset_name]

    # Output paths
    results_dir = os.path.join(RESULTS_DIR, dataset_name)
    csv_path = os.path.join(results_dir, "unified_benchmark_results.csv")
    predictions_dir = os.path.join(results_dir, "predictions")
    init_csv(csv_path)

    return {
        "dataset_name": dataset_name,
        "is_milk": is_milk,
        "periods": periods,
        "n_runs": n_runs,
        "max_epochs": max_epochs,
        "patience": patience,
        "forecast_multiplier": forecast_multiplier,
        "batch_size_override": args.batch_size,
        "csv_path": csv_path,
        "predictions_dir": predictions_dir,
    }


def _run_sequential(args, params):
    """Run benchmark sequentially on a single device (original behavior)."""
    dataset_name = params["dataset_name"]
    is_milk = params["is_milk"]
    periods = params["periods"]
    n_runs = params["n_runs"]
    max_epochs = params["max_epochs"]
    patience = params["patience"]
    forecast_multiplier = params["forecast_multiplier"]
    batch_size_override = params["batch_size_override"]
    csv_path = params["csv_path"]
    predictions_dir = params["predictions_dir"]

    passes = [
        ("baseline", False),
        ("activeG_fcast", "forecast"),
    ]

    for period in periods:
        if _shutdown_requested:
            print("[SHUTDOWN] Exiting before next period.")
            return

        print(f"\n{'='*60}")
        print(f"Period: {period}")
        print(f"{'='*60}")

        dataset = load_dataset(dataset_name, period)
        train_series_list = dataset.get_training_series()
        batch_size = get_batch_size(dataset_name, period, batch_size_override)

        for pass_name, active_g in passes:
            if _shutdown_requested:
                print("[SHUTDOWN] Exiting before next pass.")
                return

            print(f"\n  --- Pass: {pass_name} (active_g={active_g}) ---")

            for config_name, cfg in UNIFIED_CONFIGS.items():
                if _shutdown_requested:
                    print("[SHUTDOWN] Exiting before next config.")
                    return

                if is_milk:
                    stack_types, n_blocks = scale_config_for_milk(cfg)
                else:
                    stack_types = cfg["stack_types"]
                    n_blocks = cfg["n_blocks_per_stack"]

                category = cfg["category"]

                for run_idx in range(n_runs):
                    if _shutdown_requested:
                        print("[SHUTDOWN] Exiting before next run.")
                        return

                    run_single_experiment(
                        experiment_name=pass_name,
                        config_name=config_name,
                        category=category,
                        stack_types=stack_types,
                        period=period,
                        run_idx=run_idx,
                        dataset=dataset,
                        train_series_list=train_series_list,
                        csv_path=csv_path,
                        n_blocks_per_stack=n_blocks,
                        share_weights=cfg["share_weights"],
                        active_g=active_g,
                        sum_losses=False,
                        activation="ReLU",
                        max_epochs=max_epochs,
                        patience=patience,
                        batch_size=batch_size,
                        accelerator_override=args.accelerator,
                        forecast_multiplier=forecast_multiplier,
                        num_workers=args.num_workers,
                        wandb_enabled=args.wandb,
                        wandb_project=args.wandb_project,
                        save_predictions=args.save_predictions,
                        predictions_dir=predictions_dir,
                        dataset_name=dataset_name,
                        worker_id=(
                            f"gpu{args.gpu_id}"
                            if getattr(args, "gpu_id", None) is not None
                            else ""
                        ),
                        tb_enabled=getattr(args, "tensorboard", False),
                    )


def _run_parallel(args, params, n_gpus):
    """Run benchmark in parallel across multiple GPUs."""
    global _shutdown_event

    dataset_name = params["dataset_name"]
    is_milk = params["is_milk"]
    periods = params["periods"]
    n_runs = params["n_runs"]
    csv_path = params["csv_path"]

    # Build flat job list
    jobs = _build_job_list(
        periods,
        n_runs,
        dataset_name,
        params["batch_size_override"],
    )

    # Pre-filter completed jobs
    pending_jobs = [
        job
        for job in jobs
        if not result_exists(
            csv_path,
            job["pass_name"],
            job["config_name"],
            job["period"],
            job["run_idx"],
        )
    ]

    n_complete = len(jobs) - len(pending_jobs)
    print(
        f"  Jobs: {len(jobs)} total, {len(pending_jobs)} pending, "
        f"{n_complete} already complete"
    )

    if not pending_jobs:
        print("  All jobs already complete!")
        return

    # Set up multiprocessing with spawn context (clean CUDA state per worker)
    ctx = mp.get_context("spawn")
    job_queue = ctx.Queue()
    for job in pending_jobs:
        job_queue.put(job)

    _shutdown_event = ctx.Event()

    worker_args = {
        "dataset_name": dataset_name,
        "is_milk": is_milk,
        "csv_path": csv_path,
        "max_epochs": params["max_epochs"],
        "patience": params["patience"],
        "forecast_multiplier": params["forecast_multiplier"],
        "num_workers": args.num_workers,
        "wandb_enabled": args.wandb,
        "wandb_project": args.wandb_project,
        "save_predictions": args.save_predictions,
        "predictions_dir": params["predictions_dir"],
        "tb_enabled": getattr(args, "tensorboard", False),
    }

    # Spawn workers
    workers = []
    for gid in range(n_gpus):
        p = ctx.Process(
            target=_gpu_worker,
            args=(gid, job_queue, _shutdown_event, worker_args),
        )
        p.start()
        workers.append(p)

    print(f"  Spawned {n_gpus} GPU worker processes.")

    # Wait for all workers to finish
    for p in workers:
        p.join()

    # Clean up any workers still alive (shouldn't happen normally)
    for p in workers:
        if p.is_alive():
            print(f"  [WARN] Terminating worker PID {p.pid}")
            p.terminate()
            p.join(timeout=10)

    _shutdown_event = None


def _run_dataset_benchmark(args, params, n_gpus):
    """Print header, dispatch to sequential or parallel, and print footer
    for a single dataset.  Extracted so ``run_unified_benchmark`` can call
    it in a loop for ``--dataset all`` without duplicating logic.
    """
    dataset_name = params["dataset_name"]
    periods = params["periods"]
    n_runs = params["n_runs"]
    max_epochs = params["max_epochs"]
    patience = params["patience"]

    total_configs = len(UNIFIED_CONFIGS)
    total_passes = 2  # baseline + activeG_fcast

    print(f"\n{'='*70}")
    print(f"Unified Benchmark — {dataset_name.upper()}")
    print(f"  Periods: {periods}")
    print(
        f"  Configs: {total_configs}  |  Passes: {total_passes}  |  Runs/config: {n_runs}"
    )
    print(f"  Max epochs: {max_epochs}  |  Patience: {patience}")
    print(f"  Total runs per period: {total_configs * total_passes * n_runs}")
    if n_gpus >= 2:
        print(f"  GPUs: {n_gpus} (parallel execution)")
    else:
        print(f"  Mode: sequential (GPUs available: {max(n_gpus, 0)})")
    print(f"{'='*70}")

    if n_gpus >= 2:
        _run_parallel(args, params, n_gpus)
    else:
        _run_sequential(args, params)

    print(f"\n{'='*70}")
    print(f"Unified Benchmark COMPLETE — {dataset_name.upper()}")
    print(f"Results: {params['csv_path']}")
    print(f"{'='*70}")


def run_unified_benchmark(args):
    """Main orchestrator: resolve params and dispatch to sequential or parallel.

    When ``args.dataset == "all"``, iterates through every dataset defined in
    ``DATASET_PERIODS`` (M4 → Tourism → Milk), resolving dataset-specific
    defaults for each and running them sequentially.
    """
    n_gpus = resolve_n_gpus(args)

    # --dataset all: run each dataset in turn
    if args.dataset == "all":
        all_dataset_names = list(DATASET_PERIODS.keys())

        print(f"\n{'#'*70}")
        print(f"Unified Benchmark — ALL DATASETS")
        print(f"  Datasets: {[d.upper() for d in all_dataset_names]}")
        if n_gpus >= 2:
            print(f"  GPUs: {n_gpus} (parallel execution per dataset)")
        else:
            print(f"  Mode: sequential")
        print(f"{'#'*70}")

        for ds_idx, dataset_name in enumerate(all_dataset_names, 1):
            if _shutdown_requested:
                print("[SHUTDOWN] Exiting before next dataset.")
                return

            print(
                f"\n  >>> Dataset {ds_idx}/{len(all_dataset_names)}: "
                f"{dataset_name.upper()} <<<"
            )

            params = _resolve_benchmark_params(args, dataset_override=dataset_name)
            if params is None:
                print(f"  Skipping {dataset_name} (no matching periods).")
                continue

            _run_dataset_benchmark(args, params, n_gpus)

        print(f"\n{'#'*70}")
        print(f"Unified Benchmark COMPLETE — ALL DATASETS")
        print(f"{'#'*70}")
        return

    # Single-dataset path (original behaviour)
    params = _resolve_benchmark_params(args)
    if params is None:
        return

    _run_dataset_benchmark(args, params, n_gpus)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Unified Benchmark for N-BEATS Lightning paper experiments"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["m4", "tourism", "milk", "traffic", "weather", "all"],
        help="Dataset to benchmark (use 'all' to run all datasets sequentially)",
    )
    parser.add_argument(
        "--periods",
        nargs="+",
        default=None,
        help="Filter to specific periods (default: all for dataset)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=MAX_EPOCHS,
        help=f"Maximum training epochs (default: {MAX_EPOCHS}; Milk auto-uses {MILK_MAX_EPOCHS})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override default batch size"
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=None,
        help=f"Runs per config (default: {N_RUNS_DEFAULT}; Milk auto-uses {MILK_N_RUNS})",
    )
    parser.add_argument(
        "--n-gpus",
        type=int,
        default=None,
        help="Number of GPUs for parallel execution (default: auto-detect)",
    )
    parser.add_argument(
        "--accelerator",
        default="auto",
        choices=["auto", "cuda", "xpu", "mps", "cpu"],
        help="Accelerator (default: auto)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="DataLoader workers (default: 0)"
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-project", default="nbeats-lightning", help="W&B project name"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        default=True,
        help="Save NPZ predictions (default: True)",
    )
    parser.add_argument(
        "--no-save-predictions",
        action="store_false",
        dest="save_predictions",
        help="Disable NPZ prediction saving",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging (disabled by default to save disk space)",
    )

    args = parser.parse_args()
    run_unified_benchmark(args)


if __name__ == "__main__":
    main()
