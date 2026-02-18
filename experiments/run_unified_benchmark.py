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
    python experiments/run_unified_benchmark.py --dataset m4 --max-epochs 100 --wandb
"""

import argparse
import csv
import fcntl
import gc
import json
import math
import multiprocessing as mp
import os
import queue
import signal
import sys
import time

import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch import loggers as pl_loggers

# Allow running from project root or experiments/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lightningnbeats.models import NBeatsNet
from lightningnbeats.loaders import (
    ColumnarCollectionTimeSeriesDataModule,
    ColumnarCollectionTimeSeriesTestDataModule,
)
from lightningnbeats.data import M4Dataset, TourismDataset, MilkDataset

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
    print("\n[SIGNAL] Shutdown requested. Finishing current run... "
          "(signal again to force)")


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

BASE_SEED = 42
N_RUNS_DEFAULT = 10

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ---------------------------------------------------------------------------
# Dataset Periods
# ---------------------------------------------------------------------------

M4_PERIODS = {
    "Yearly":    {"frequency": 1,  "horizon": 6},
    "Quarterly": {"frequency": 4,  "horizon": 8},
    "Monthly":   {"frequency": 12, "horizon": 18},
    "Weekly":    {"frequency": 1,  "horizon": 13},
    "Daily":     {"frequency": 1,  "horizon": 14},
    "Hourly":    {"frequency": 24, "horizon": 48},
}

TOURISM_PERIODS = {
    "Tourism-Yearly":    {"frequency": 1,  "horizon": 4},
    "Tourism-Monthly":   {"frequency": 12, "horizon": 24},
    "Tourism-Quarterly": {"frequency": 4,  "horizon": 8},
}

MILK_PERIODS = {
    "Milk": {"frequency": 12, "horizon": 6},
}

DATASET_PERIODS = {
    "m4": M4_PERIODS,
    "tourism": TOURISM_PERIODS,
    "milk": MILK_PERIODS,
}

FORECAST_MULTIPLIERS = {
    "m4": 5,
    "tourism": 2,
    "milk": 4,
}

# ---------------------------------------------------------------------------
# Batch Sizes
# ---------------------------------------------------------------------------

BATCH_SIZES = {
    ("m4", "Yearly"): 32768,
    ("tourism", "Tourism-Yearly"): 8192,
    ("tourism", "Tourism-Monthly"): 32768,
    ("tourism", "Tourism-Quarterly"): 65536,
    ("milk", "Milk"): 128,
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
    "experiment", "config_name", "category", "stack_types",
    "period", "frequency", "forecast_length", "backcast_length",
    "n_stacks", "n_blocks_per_stack", "share_weights",
    "run", "seed",
    "smape", "mase", "mae", "mse", "owa",
    "n_params", "training_time_seconds", "epochs_trained",
    "active_g", "sum_losses", "activation", "stopping_reason",
    "best_val_loss", "final_val_loss", "final_train_loss",
    "best_epoch", "loss_ratio", "diverged",
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
    if (n_stacks == 30
            and stack_types[0] == "Trend"
            and stack_types[1] == "Seasonality"
            and len(set(stack_types[2:])) == 1):
        # 2 interpretable + 4 generic-backend
        return ["Trend", "Seasonality"] + [stack_types[2]] * 4, n_blocks

    # I+G variant: Trend, Seasonality then 28 of same block (already covered above)
    # But also handle TrendAE-style if any exist
    if (n_stacks == 30
            and "Trend" in stack_types[0]
            and "Seasonality" in stack_types[1]
            and len(set(stack_types[2:])) == 1):
        return [stack_types[0], stack_types[1]] + [stack_types[2]] * 4, n_blocks

    # Alternating mixed: [A, B] * 15 -> [A, B] * 3
    if n_stacks == 30 and len(unique) == 2:
        pattern = stack_types[:2]
        is_alternating = all(
            stack_types[i] == pattern[i % 2] for i in range(n_stacks)
        )
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

def build_loggers(log_dir, log_name, wandb_enabled, wandb_project, wandb_group, wandb_config):
    """Build list of loggers (TensorBoard + optionally W&B)."""
    loggers = [pl_loggers.TensorBoardLogger(save_dir=log_dir, name=log_name)]
    if wandb_enabled:
        loggers.append(pl_loggers.WandbLogger(
            project=wandb_project,
            group=wandb_group,
            name=log_name,
            config=wandb_config,
            save_dir=log_dir,
            reinit=True,
        ))
    return loggers


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


def compute_m4_mase(y_pred, y_true, train_series_list, frequency):
    """M4-faithful MASE using full training history for naive seasonal denominator."""
    n_series = y_pred.shape[0]
    mase_values = []

    for i in range(n_series):
        train_i = train_series_list[i]
        pred_i = y_pred[i]
        true_i = y_true[i]

        forecast_mae = np.mean(np.abs(true_i - pred_i))

        m = max(1, frequency)
        if len(train_i) <= m:
            continue

        naive_errors = np.abs(train_i[m:] - train_i[:-m])
        naive_mae = np.mean(naive_errors)

        if naive_mae < 1e-10:
            continue

        mase_values.append(forecast_mae / naive_mae)

    if len(mase_values) == 0:
        return float("nan")
    return float(np.mean(mase_values))


def compute_mae(y_pred, y_true):
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_mse(y_pred, y_true):
    return float(np.mean((y_true - y_pred) ** 2))


def resolve_accelerator(accelerator_override):
    """Resolve accelerator and device from override string."""
    if accelerator_override == "auto":
        if torch.cuda.is_available():
            accelerator = "cuda"
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            accelerator = "mps"
            device = torch.device("mps")
        else:
            accelerator = "cpu"
            device = torch.device("cpu")
    else:
        accelerator = accelerator_override
        device = torch.device(accelerator if accelerator != "cuda" else "cuda")
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


def load_dataset(dataset_name, period):
    """Factory function to load the appropriate benchmark dataset."""
    if dataset_name == "m4":
        return M4Dataset(period, "All")
    elif dataset_name == "tourism":
        period_name = period.replace("Tourism-", "")
        return TourismDataset(period_name)
    elif dataset_name == "milk":
        return MilkDataset()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_batch_size(dataset_name, period, override=None):
    """Look up per-dataset/period batch size, with optional override.

    Tuned per-dataset/period values in BATCH_SIZES always take precedence
    over the CLI override to prevent divergence (e.g. M4 Yearly needs 32768).
    The override is only applied to dataset/period combos that are NOT in
    BATCH_SIZES.
    """
    tuned = BATCH_SIZES.get((dataset_name, period))
    if tuned is not None:
        return tuned
    if override is not None:
        return override
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

        if self.best_val_loss > 0 and val_loss > self.best_val_loss * self.relative_threshold:
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
    print(f"  [MIGRATE] {os.path.basename(path)}: "
          f"header {len(existing_header)} cols -> {len(columns)} cols")
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
            extra_values = raw[len(old_header):]
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
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            with open(path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writerow(row_dict)
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)


def result_exists(path, experiment, config_name, period, run):
    """Check if a result row already exists in the CSV."""
    if not os.path.exists(path):
        return False
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row["experiment"] == experiment
                    and row["config_name"] == config_name
                    and row["period"] == period
                    and row["run"] == str(run)):
                return True
    return False


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
):
    """Run a single training + evaluation experiment and save results to CSV."""

    prefix = f"[GPU {gpu_id}] " if gpu_id is not None else ""

    # Check resumability
    if result_exists(csv_path, experiment_name, config_name, period, run_idx):
        print(f"  {prefix}[SKIP] {config_name} / {period} / run {run_idx} -- already exists")
        return

    seed = BASE_SEED + run_idx
    set_seed(seed)

    forecast_length = dataset.forecast_length
    frequency = dataset.frequency
    backcast_length = forecast_length * forecast_multiplier

    n_stacks = len(stack_types)

    accelerator, device = resolve_accelerator(accelerator_override)

    # Mixed precision on CUDA
    precision = "16-mixed" if accelerator == "cuda" else "32-true"

    # Create model
    model = NBeatsNet(
        backcast_length=backcast_length,
        forecast_length=forecast_length,
        stack_types=stack_types,
        n_blocks_per_stack=n_blocks_per_stack,
        share_weights=share_weights,
        thetas_dim=THETAS_DIM,
        loss=LOSS,
        active_g=active_g,
        sum_losses=sum_losses,
        activation=activation,
        latent_dim=LATENT_DIM,
        basis_dim=BASIS_DIM,
        learning_rate=LEARNING_RATE,
        no_val=False,
    )

    n_params = count_parameters(model)

    # Data modules
    train_data = dataset.train_data
    test_data = dataset.test_data

    pin_memory = num_workers > 0
    dm = ColumnarCollectionTimeSeriesDataModule(
        train_data,
        backcast_length=backcast_length,
        forecast_length=forecast_length,
        batch_size=batch_size,
        no_val=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_dm = ColumnarCollectionTimeSeriesTestDataModule(
        train_data,
        test_data,
        backcast_length=backcast_length,
        forecast_length=forecast_length,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Trainer
    log_dir = os.path.join(RESULTS_DIR, "lightning_logs")
    log_name = f"unified/{experiment_name}/{config_name}/{period}/run{run_idx}"

    chk_callback = ModelCheckpoint(
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
    )

    wandb_group = f"unified/{period}"
    exp_loggers = build_loggers(
        log_dir=log_dir, log_name=log_name,
        wandb_enabled=wandb_enabled, wandb_project=wandb_project,
        wandb_group=wandb_group,
        wandb_config={
            "dataset": dataset.name, "period": period,
            "config_name": config_name, "category": category,
            "stack_types": stack_types,
            "n_stacks": n_stacks, "n_blocks_per_stack": n_blocks_per_stack,
            "share_weights": share_weights, "backcast_length": backcast_length,
            "forecast_length": forecast_length, "batch_size": batch_size,
            "max_epochs": max_epochs, "seed": seed,
            "run_idx": run_idx, "active_g": active_g,
            "sum_losses": sum_losses, "activation": activation,
            "n_params": n_params,
        },
    )

    divergence_detector = DivergenceDetector(relative_threshold=3.0, consecutive_epochs=3)
    convergence_tracker = ConvergenceTracker()

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=max_epochs,
        precision=precision,
        callbacks=[chk_callback, early_stop_callback, divergence_detector, convergence_tracker],
        logger=exp_loggers,
        enable_progress_bar=True,
        deterministic=False,
        log_every_n_steps=10,
    )

    # Train
    stack_summary = (f"{n_stacks}x{stack_types[0]}" if len(set(stack_types)) == 1
                     else f"{n_stacks} mixed")
    print(f"  {prefix}[RUN]  {config_name} / {period} / run {run_idx} "
          f"(seed={seed}, {stack_summary}, blocks/stack={n_blocks_per_stack}, "
          f"share_w={share_weights}, params={n_params:,})")
    t0 = time.time()
    trainer.fit(model, datamodule=dm)
    training_time = time.time() - t0

    # Load best checkpoint
    best_path = trainer.checkpoint_callback.best_model_path
    if best_path:
        model = NBeatsNet.load_from_checkpoint(best_path, weights_only=False)
    epochs_trained = trainer.current_epoch

    # Classify stopping reason
    if divergence_detector.diverged:
        stopping_reason = "DIVERGED"
    elif hasattr(early_stop_callback, "stopped_epoch") and early_stop_callback.stopped_epoch > 0:
        stopping_reason = "EARLY_STOPPED"
    else:
        stopping_reason = "MAX_EPOCHS"

    # Convergence statistics
    best_val_loss = (float(trainer.checkpoint_callback.best_model_score)
                     if trainer.checkpoint_callback.best_model_score is not None
                     else float("nan"))
    final_val_loss = (convergence_tracker.val_losses[-1]
                      if convergence_tracker.val_losses else float("nan"))
    final_train_loss = (convergence_tracker.train_losses[-1]
                        if convergence_tracker.train_losses else float("nan"))
    best_epoch = (int(np.argmin(convergence_tracker.val_losses))
                  if convergence_tracker.val_losses else 0)
    loss_ratio = (final_val_loss / best_val_loss
                  if best_val_loss > 0 and math.isfinite(best_val_loss)
                  else float("nan"))
    diverged = (divergence_detector.diverged
                or not math.isfinite(best_val_loss))

    # Inference
    preds, targets = run_inference(model, test_dm, device)

    # Metrics
    smape = compute_smape(preds, targets)
    mase = compute_m4_mase(preds, targets, train_series_list, frequency)
    mae = compute_mae(preds, targets)
    mse = compute_mse(preds, targets)
    owa = dataset.compute_owa(smape, mase)

    # Update diverged flag — only for genuinely non-finite metrics.
    # The smape >= 200 threshold was removed: a high sMAPE from a model that
    # early-stopped is "poor" but not "diverged".  Actual divergence is already
    # caught by DivergenceDetector (NaN val_loss or 3× spike for 3 epochs).
    diverged = diverged or not math.isfinite(smape)

    print(f"  {prefix}       sMAPE={smape:.4f}  MASE={mase:.4f}  MAE={mae:.4f}  "
          f"MSE={mse:.4f}  OWA={owa:.4f}  "
          f"time={training_time:.1f}s  epochs={epochs_trained}  [{stopping_reason}]")

    # Save predictions
    if save_predictions and predictions_dir is not None:
        os.makedirs(predictions_dir, exist_ok=True)
        npz_name = f"{experiment_name}_{config_name}_{period}_run{run_idx}.npz"
        np.savez(
            os.path.join(predictions_dir, npz_name),
            preds=preds, targets=targets,
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
        "n_params": n_params,
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
        "val_loss_curve": json.dumps([f"{v:.8f}" for v in convergence_tracker.val_losses]),
    }
    append_result(csv_path, row)
    finish_wandb(wandb_enabled)

    # Cleanup
    del model, trainer, dm, test_dm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Multi-GPU Support
# ---------------------------------------------------------------------------

def resolve_n_gpus(args):
    """Determine number of GPUs to use for parallel execution."""
    if args.accelerator not in ("auto", "cuda"):
        return 0
    if not torch.cuda.is_available():
        return 0
    available = torch.cuda.device_count()
    if args.n_gpus is not None:
        return min(args.n_gpus, available)
    return available


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
                    jobs.append({
                        "period": period,
                        "pass_name": pass_name,
                        "active_g": active_g,
                        "config_name": config_name,
                        "cfg": cfg,
                        "run_idx": run_idx,
                        "batch_size": batch_size,
                    })
    return jobs


def _gpu_worker(gpu_id, job_queue, shutdown_event, worker_args):
    """Worker process: pins to GPU gpu_id, pulls jobs from shared queue."""
    # Pin to specific GPU before any CUDA context is created
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Ignore signals in worker — main process handles shutdown via event
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    torch.set_float32_matmul_precision("medium")

    prefix = f"[GPU {gpu_id}]"
    print(f"{prefix} Worker started (CUDA_VISIBLE_DEVICES={gpu_id}).")

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
            accelerator_override="cuda",
            forecast_multiplier=worker_args["forecast_multiplier"],
            num_workers=worker_args["num_workers"],
            wandb_enabled=worker_args["wandb_enabled"],
            wandb_project=worker_args["wandb_project"],
            save_predictions=worker_args["save_predictions"],
            predictions_dir=worker_args["predictions_dir"],
            gpu_id=gpu_id,
        )

    print(f"{prefix} Worker finished.")


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def _resolve_benchmark_params(args):
    """Resolve common benchmark parameters from CLI args."""
    dataset_name = args.dataset
    is_milk = dataset_name == "milk"

    # Resolve periods
    all_periods = DATASET_PERIODS[dataset_name]
    if args.periods:
        periods = [p for p in args.periods if p in all_periods]
        if not periods:
            print(f"[ERROR] No valid periods for dataset '{dataset_name}'. "
                  f"Available: {list(all_periods.keys())}")
            return None
    else:
        periods = list(all_periods.keys())

    # Resolve run count and training params
    if is_milk:
        n_runs = args.n_runs if args.n_runs is not None else MILK_N_RUNS
        max_epochs = args.max_epochs if args.max_epochs != MAX_EPOCHS else MILK_MAX_EPOCHS
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
        periods, n_runs, dataset_name, params["batch_size_override"],
    )

    # Pre-filter completed jobs
    pending_jobs = [
        job for job in jobs
        if not result_exists(csv_path, job["pass_name"], job["config_name"],
                             job["period"], job["run_idx"])
    ]

    n_complete = len(jobs) - len(pending_jobs)
    print(f"  Jobs: {len(jobs)} total, {len(pending_jobs)} pending, "
          f"{n_complete} already complete")

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


def run_unified_benchmark(args):
    """Main orchestrator: resolve params and dispatch to sequential or parallel."""
    params = _resolve_benchmark_params(args)
    if params is None:
        return

    n_gpus = resolve_n_gpus(args)

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
    print(f"  Configs: {total_configs}  |  Passes: {total_passes}  |  Runs/config: {n_runs}")
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified Benchmark for N-BEATS Lightning paper experiments"
    )
    parser.add_argument(
        "--dataset", required=True, choices=["m4", "tourism", "milk"],
        help="Dataset to benchmark"
    )
    parser.add_argument(
        "--periods", nargs="+", default=None,
        help="Filter to specific periods (default: all for dataset)"
    )
    parser.add_argument(
        "--max-epochs", type=int, default=MAX_EPOCHS,
        help=f"Maximum training epochs (default: {MAX_EPOCHS}; Milk auto-uses {MILK_MAX_EPOCHS})"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override default batch size"
    )
    parser.add_argument(
        "--n-runs", type=int, default=None,
        help=f"Runs per config (default: {N_RUNS_DEFAULT}; Milk auto-uses {MILK_N_RUNS})"
    )
    parser.add_argument(
        "--n-gpus", type=int, default=None,
        help="Number of GPUs for parallel execution (default: auto-detect)"
    )
    parser.add_argument(
        "--accelerator", default="auto", choices=["auto", "cuda", "mps", "cpu"],
        help="Accelerator (default: auto)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader workers (default: 0)"
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-project", default="nbeats-lightning",
        help="W&B project name"
    )
    parser.add_argument(
        "--save-predictions", action="store_true", default=True,
        help="Save NPZ predictions (default: True)"
    )
    parser.add_argument(
        "--no-save-predictions", action="store_false", dest="save_predictions",
        help="Disable NPZ prediction saving"
    )

    args = parser.parse_args()
    run_unified_benchmark(args)


if __name__ == "__main__":
    main()
