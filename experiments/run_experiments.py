"""
Benchmark Experiment Script for N-BEATS Lightning

Supports multiple datasets (M4, Traffic) with a unified experiment framework.
Use --dataset to select the dataset and --periods to select sub-configs.

Part 1: Block-type benchmark — Paper baselines (N-BEATS-G, N-BEATS-I, N-BEATS-I+G)
        plus novel block types at the same 30-stack scale for fair comparison.
Part 2: Ablation studies on 30-stack Generic — active_g, sum_losses, activations.
Part 3: Multi-horizon ensemble — Train G, I, I+G at backcast lengths 2H-7H and
        take median forecast across all models (paper's ensemble strategy).
Part 4: Wavelet V2 benchmark — Numerically stabilized wavelet blocks with spectral
        normalization, LayerNorm, Xavier init, and output clamping.
Part 5: Wavelet V3 benchmark — Orthonormal DWT basis via impulse-response synthesis
        + SVD orthogonalization (condition number = 1.0).
Part 6: Convergence study — Multi-dataset (M4-Yearly + Traffic-96), random seeds,
        10-stack Generic with 2x2 factorial (active_g x sum_losses),
        50 runs per config to disentangle data distribution vs initialization effects.
Part 7: Novel mixed stack benchmark + ensemble — I+G compositional pattern with novel
        G-position blocks (GenericAE, BottleneckGeneric, AutoEncoder) plus AE
        interpretable front-end. Benchmark + multi-horizon ensemble for each config.

Usage:
    python experiments/run_experiments.py --dataset m4 --part 1 --periods Yearly --max-epochs 100
    python experiments/run_experiments.py --dataset traffic --part 1 --periods Traffic-96 --max-epochs 100
    python experiments/run_experiments.py --dataset m4 --part all
    python experiments/run_experiments.py --dataset m4 --part 2 --periods Yearly Monthly --max-epochs 100
    python experiments/run_experiments.py --dataset m4 --part 3 --periods Yearly --max-epochs 100
    python experiments/run_experiments.py --dataset m4 --part 4 --periods Yearly --max-epochs 100
    python experiments/run_experiments.py --dataset m4 --part 5 --periods Yearly --max-epochs 100
    python experiments/run_experiments.py --part 6 --max-epochs 100
    python experiments/run_experiments.py --dataset m4 --part 7 --periods Yearly --max-epochs 100
"""

import argparse
import csv
import gc
import os
import random
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
from lightningnbeats.data import M4Dataset, TrafficDataset, WeatherDataset

torch.set_float32_matmul_precision("medium")

# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------

M4_PERIODS = {
    "Yearly":    {"frequency": 1,  "horizon": 6},
    "Quarterly": {"frequency": 4,  "horizon": 8},
    "Monthly":   {"frequency": 12, "horizon": 18},
    "Weekly":    {"frequency": 1,  "horizon": 13},
    "Daily":     {"frequency": 1,  "horizon": 14},
    "Hourly":    {"frequency": 24, "horizon": 48},
}

TRAFFIC_HORIZONS = {
    "Traffic-96":  {"frequency": 24, "horizon": 96},
}

WEATHER_HORIZONS = {
    "Weather-96":  {"frequency": 144, "horizon": 96},
}

# Dataset-specific hyperparameter defaults
DATASET_DEFAULTS = {
    "m4":      {"loss": "SMAPELoss", "forecast_multiplier": 5},
    "traffic": {"loss": "SMAPELoss", "forecast_multiplier": 2},
    "weather": {"loss": "SMAPELoss", "forecast_multiplier": 2},
}

# Fixed training hyperparameters — matches original paper where applicable
BATCH_SIZE = 1024                # Paper: 1024
FORECAST_MULTIPLIER = 5          # Paper uses 2-7 for ensemble; 5H is a reasonable single point
FORECAST_MULTIPLIERS = [2, 3, 4, 5, 6, 7]  # Paper: ensemble across backcast lengths 2H-7H
TOTAL_STACKS = 30                # Paper: 30 for Generic architecture
THETAS_DIM = 5                   # Polynomial degree for Trend; bottleneck dim for BottleneckGeneric
LATENT_DIM = 4                   # For AE backbone blocks
BASIS_DIM = 128                  # For Wavelet blocks
LOSS = "SMAPELoss"               # Paper primary metric and training loss
LEARNING_RATE = 1e-3             # Paper: 1e-3
EARLY_STOPPING_PATIENCE = 10    # Paper uses early stopping on validation loss

N_RUNS = 5
BASE_SEED = 42

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ---------------------------------------------------------------------------
# Block Benchmark Configs (Part 1)
# ---------------------------------------------------------------------------

BLOCK_CONFIGS = {
    # ===== Paper Baselines (1:1 reproduction) =====
    # N-BEATS-G: 30 stacks x 1 Generic block, shared weights (paper Section 3)
    "NBEATS-G": {
        "stack_types": ["Generic"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    # N-BEATS-I: Trend + Seasonality, 3 blocks/stack, shared weights (paper Section 3)
    "NBEATS-I": {
        "stack_types": ["Trend", "Seasonality"],
        "n_blocks_per_stack": 3,
        "share_weights": True,
    },
    # N-BEATS-I+G: Interpretable + Generic combined (paper Section 3 — top M4 performer)
    "NBEATS-I+G": {
        "stack_types": ["Trend", "Seasonality"] + ["Generic"] * 28,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    # ===== Novel Single-Type Blocks (30 stacks — fair comparison with G) =====
    "BottleneckGeneric": {
        "stack_types": ["BottleneckGeneric"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "AutoEncoder": {
        "stack_types": ["AutoEncoder"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "GenericAE": {
        "stack_types": ["GenericAE"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "BottleneckGenericAE": {
        "stack_types": ["BottleneckGenericAE"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "GenericAEBackcast": {
        "stack_types": ["GenericAEBackcast"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },

    # ===== Novel Interpretable (fair comparison with I) =====
    "NBEATS-I-AE": {
        "stack_types": ["TrendAE", "SeasonalityAE"],
        "n_blocks_per_stack": 3,
        "share_weights": True,
    },
    # ===== Mixed Stacks (30 total — novel compositions) =====
    "Trend+HaarWavelet": {
        "stack_types": ["Trend", "HaarWavelet"] * 15,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "Trend+DB3Wavelet": {
        "stack_types": ["Trend", "DB3Wavelet"] * 15,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "Generic+DB3Wavelet": {
        "stack_types": ["Generic", "DB3Wavelet"] * 15,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
}

# ---------------------------------------------------------------------------
# Mixed Stack Benchmark Configs (Part 7) — I+G pattern with novel G blocks
# ---------------------------------------------------------------------------

MIXED_STACK_CONFIGS = {
    "NBEATS-I+GenericAE": {
        "stack_types": ["Trend", "Seasonality"] + ["GenericAE"] * 28,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "NBEATS-I-AE+GenericAE": {
        "stack_types": ["TrendAE", "SeasonalityAE"] + ["GenericAE"] * 28,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "NBEATS-I+BottleneckGeneric": {
        "stack_types": ["Trend", "Seasonality"] + ["BottleneckGeneric"] * 28,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "NBEATS-I+AutoEncoder": {
        "stack_types": ["Trend", "Seasonality"] + ["AutoEncoder"] * 28,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
}

# ---------------------------------------------------------------------------
# Wavelet V2 Benchmark Configs (Part 4) — Numerically stabilized wavelets
# ---------------------------------------------------------------------------

WAVELET_V2_CONFIGS = {
    # ===== Homogeneous 30-stack (direct comparison with Part 1 wavelet results) =====
    "HaarWaveletV2": {
        "stack_types": ["HaarWaveletV2"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "DB3WaveletV2": {
        "stack_types": ["DB3WaveletV2"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "DB3AltWaveletV2": {
        "stack_types": ["DB3AltWaveletV2"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "Coif2WaveletV2": {
        "stack_types": ["Coif2WaveletV2"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "Symlet3WaveletV2": {
        "stack_types": ["Symlet3WaveletV2"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    # ===== Mixed Stacks (30 total — direct comparison with Part 1 mixed results) =====
    "Trend+HaarWaveletV2": {
        "stack_types": ["Trend", "HaarWaveletV2"] * 15,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "Trend+DB3WaveletV2": {
        "stack_types": ["Trend", "DB3WaveletV2"] * 15,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "Generic+DB3WaveletV2": {
        "stack_types": ["Generic", "DB3WaveletV2"] * 15,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
}

# ---------------------------------------------------------------------------
# Wavelet V3 Benchmark Configs (Part 5) — Orthonormal DWT basis
# ---------------------------------------------------------------------------

WAVELET_V3_CONFIGS = {
    # ===== Homogeneous 30-stack (direct comparison with V1 and V2 results) =====
    "HaarWaveletV3": {
        "stack_types": ["HaarWaveletV3"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "DB3WaveletV3": {
        "stack_types": ["DB3WaveletV3"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "Coif2WaveletV3": {
        "stack_types": ["Coif2WaveletV3"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "Symlet3WaveletV3": {
        "stack_types": ["Symlet3WaveletV3"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "DB2WaveletV3": {
        "stack_types": ["DB2WaveletV3"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "DB4WaveletV3": {
        "stack_types": ["DB4WaveletV3"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    # ===== Mixed Stacks (30 total — mirror V2 Part 4 + novel combos) =====
    "Trend+HaarWaveletV3": {
        "stack_types": ["Trend", "HaarWaveletV3"] * 15,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "Trend+DB3WaveletV3": {
        "stack_types": ["Trend", "DB3WaveletV3"] * 15,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "Generic+DB3WaveletV3": {
        "stack_types": ["Generic", "DB3WaveletV3"] * 15,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "Trend+Coif2WaveletV3": {
        "stack_types": ["Trend", "Coif2WaveletV3"] * 15,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "Seasonality+DB3WaveletV3": {
        "stack_types": ["Seasonality", "DB3WaveletV3"] * 15,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "NBEATS-I+WaveletV3": {
        "stack_types": ["Trend", "Seasonality"] + ["DB3WaveletV3"] * 28,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
}

# V3 Ablation Configs — on 30-stack DB3WaveletV3
V3_ABLATION_CONFIGS = {
    "DB3WaveletV3_baseline":      {"active_g": False, "sum_losses": False, "activation": "ReLU"},
    "DB3WaveletV3_activeG":       {"active_g": True,  "sum_losses": False, "activation": "ReLU"},
    "DB3WaveletV3_sumLosses":     {"active_g": False, "sum_losses": True,  "activation": "ReLU"},
    "DB3WaveletV3_activeG+sumL":  {"active_g": True,  "sum_losses": True,  "activation": "ReLU"},
}

# ---------------------------------------------------------------------------
# Ablation Configs (Part 2) — all on 30-stack Generic
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = {
    "Generic_baseline":      {"active_g": False, "sum_losses": False, "activation": "ReLU"},
    "Generic_activeG":       {"active_g": True,  "sum_losses": False, "activation": "ReLU"},
    "Generic_sumLosses":     {"active_g": False, "sum_losses": True,  "activation": "ReLU"},
    "Generic_activeG+sumL":  {"active_g": True,  "sum_losses": True,  "activation": "ReLU"},
    "Generic_GELU":          {"active_g": False, "sum_losses": False, "activation": "GELU"},
    "Generic_ELU":           {"active_g": False, "sum_losses": False, "activation": "ELU"},
    "Generic_LeakyReLU":     {"active_g": False, "sum_losses": False, "activation": "LeakyReLU"},
    "Generic_SELU":          {"active_g": False, "sum_losses": False, "activation": "SELU"},
}

# ---------------------------------------------------------------------------
# Convergence Study Configs (Part 6) — active_g effect on 10-stack Generic
# ---------------------------------------------------------------------------

CONVERGENCE_STUDY_CONFIGS = {
    "Generic10_baseline":      {"active_g": False, "sum_losses": False, "activation": "ReLU"},
    "Generic10_activeG":       {"active_g": True,  "sum_losses": False, "activation": "ReLU"},
    "Generic10_sumLosses":     {"active_g": False, "sum_losses": True,  "activation": "ReLU"},
    "Generic10_activeG+sumL":  {"active_g": True,  "sum_losses": True,  "activation": "ReLU"},
}

CONVERGENCE_STUDY_STACKS = 10
CONVERGENCE_STUDY_N_RUNS = 50

# Convergence study iterates over multiple datasets internally (Part 6 only)
CONVERGENCE_STUDY_DATASETS = {
    "m4":      ["Yearly"],
    "weather": ["Weather-96"],
}

# ---------------------------------------------------------------------------
# Ensemble Configs (Part 3) — paper's key architectures at multiple horizons
# ---------------------------------------------------------------------------

ENSEMBLE_CONFIGS = {
    "NBEATS-G":   BLOCK_CONFIGS["NBEATS-G"],
    "NBEATS-I":   BLOCK_CONFIGS["NBEATS-I"],
    "NBEATS-I+G": BLOCK_CONFIGS["NBEATS-I+G"],
}

# ---------------------------------------------------------------------------
# Mixed Stack Ensemble Configs (Part 7)
# ---------------------------------------------------------------------------

MIXED_STACK_ENSEMBLE_CONFIGS = {
    "NBEATS-I+GenericAE":         MIXED_STACK_CONFIGS["NBEATS-I+GenericAE"],
    "NBEATS-I-AE+GenericAE":      MIXED_STACK_CONFIGS["NBEATS-I-AE+GenericAE"],
    "NBEATS-I+BottleneckGeneric": MIXED_STACK_CONFIGS["NBEATS-I+BottleneckGeneric"],
    "NBEATS-I+AutoEncoder":       MIXED_STACK_CONFIGS["NBEATS-I+AutoEncoder"],
}

# ---------------------------------------------------------------------------
# CSV Column Schemas
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "experiment", "config_name", "stack_types", "period", "frequency",
    "forecast_length", "backcast_length", "n_stacks", "n_blocks_per_stack",
    "share_weights", "run", "seed",
    "smape", "mase", "mae", "mse", "owa", "n_params",
    "training_time_seconds", "epochs_trained",
    "active_g", "sum_losses", "activation",
]

ENSEMBLE_INDIVIDUAL_COLUMNS = [
    "experiment", "config_name", "stack_types", "period", "frequency",
    "forecast_length", "backcast_length", "forecast_multiplier",
    "n_stacks", "n_blocks_per_stack", "share_weights", "run", "seed",
    "smape", "mase", "mae", "mse", "owa", "n_params",
    "training_time_seconds", "epochs_trained",
    "active_g", "sum_losses", "activation",
]

ENSEMBLE_SUMMARY_COLUMNS = [
    "config_name", "period", "frequency", "forecast_length",
    "n_models", "multipliers", "n_runs_per_multiplier",
    "ensemble_smape", "ensemble_mase", "ensemble_mae", "ensemble_mse",
    "ensemble_owa",
]


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
    """M4-faithful MASE using full training history for naive seasonal denominator.

    For each series i:
        MASE_i = MAE(forecast_i) / MAE(naive_seasonal_on_train_i)

    Returns the mean MASE across all valid series.
    """
    n_series = y_pred.shape[0]
    mase_values = []

    for i in range(n_series):
        train_i = train_series_list[i]
        pred_i = y_pred[i]
        true_i = y_true[i]

        # Forecast MAE
        forecast_mae = np.mean(np.abs(true_i - pred_i))

        # Naive seasonal forecast denominator from training data
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
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_mse(y_pred, y_true):
    """Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


def load_dataset(dataset_name, period):
    """Factory function to load the appropriate benchmark dataset."""
    if dataset_name == "m4":
        return M4Dataset(period, "All")
    elif dataset_name == "traffic":
        horizon = TRAFFIC_HORIZONS[period]["horizon"]
        return TrafficDataset(horizon=horizon)
    elif dataset_name == "weather":
        horizon = WEATHER_HORIZONS[period]["horizon"]
        return WeatherDataset(horizon=horizon)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_forecast_multiplier(dataset_name):
    """Return the dataset-specific forecast multiplier for single-model runs."""
    return DATASET_DEFAULTS[dataset_name]["forecast_multiplier"]


def resolve_accelerator(accelerator_override):
    """Resolve accelerator and device from override string.

    'auto' uses CUDA > MPS > CPU detection; any other value forces that accelerator.
    """
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
    """Run inference manually, collecting (predictions, targets) as numpy arrays.

    The test dataloader yields (x, y) tuples. We run model(x) and collect
    the forecast outputs along with the ground truth y.
    """
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


# ---------------------------------------------------------------------------
# CSV Helpers (incremental save with resumability)
# ---------------------------------------------------------------------------

def init_csv(path, columns=None):
    """Create CSV with header if it doesn't exist."""
    columns = columns or CSV_COLUMNS
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()


def append_result(path, row_dict, columns=None):
    """Append a single result row to CSV."""
    columns = columns or CSV_COLUMNS
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
            if (row["experiment"] == experiment
                    and row["config_name"] == config_name
                    and row["period"] == period
                    and row["run"] == str(run)):
                return True
    return False


def ensemble_summary_exists(path, config_name, period):
    """Check if an ensemble summary row already exists in the CSV."""
    if not os.path.exists(path):
        return False
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row["config_name"] == config_name
                    and row["period"] == period):
                return True
    return False


# ---------------------------------------------------------------------------
# Single Run Function
# ---------------------------------------------------------------------------

def run_single_experiment(
    experiment_name,
    config_name,
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
    batch_size=1024,
    accelerator_override="auto",
    forecast_multiplier=None,
    seed_override=None,
    num_workers=0,
):
    """Run a single training + evaluation experiment and save results to CSV."""

    # Check resumability
    if result_exists(csv_path, experiment_name, config_name, period, run_idx):
        print(f"  [SKIP] {config_name} / {period} / run {run_idx} — already exists")
        return

    seed = seed_override if seed_override is not None else BASE_SEED + run_idx
    set_seed(seed)

    forecast_length = dataset.forecast_length
    frequency = dataset.frequency
    if forecast_multiplier is None:
        forecast_multiplier = FORECAST_MULTIPLIER
    backcast_length = forecast_length * forecast_multiplier

    n_stacks = len(stack_types)

    accelerator, device = resolve_accelerator(accelerator_override)

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

    # Trainer — paper uses early stopping on validation loss
    log_dir = os.path.join(RESULTS_DIR, "lightning_logs")
    log_name = f"{experiment_name}/{config_name}/{period}/run{run_idx}"

    chk_callback = ModelCheckpoint(
        filename="best-checkpoint",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        mode="min",
        verbose=False,
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir, name=log_name)

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=max_epochs,
        callbacks=[chk_callback, early_stop_callback],
        logger=[tb_logger],
        enable_progress_bar=True,
        deterministic=False,
        log_every_n_steps=50,
    )

    # Train
    stack_summary = (f"{n_stacks}x{stack_types[0]}" if len(set(stack_types)) == 1
                     else f"{n_stacks} mixed")
    print(f"  [RUN]  {config_name} / {period} / run {run_idx} "
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

    # Inference
    preds, targets = run_inference(model, test_dm, device)

    # Metrics
    smape = compute_smape(preds, targets)
    mase = compute_m4_mase(preds, targets, train_series_list, frequency)
    mae = compute_mae(preds, targets)
    mse = compute_mse(preds, targets)
    owa = dataset.compute_owa(smape, mase)

    print(f"         sMAPE={smape:.4f}  MASE={mase:.4f}  MAE={mae:.4f}  "
          f"MSE={mse:.4f}  OWA={owa:.4f}  "
          f"time={training_time:.1f}s  epochs={epochs_trained}")

    # Save result — record unique block types for readability
    unique_types = list(dict.fromkeys(stack_types))  # preserves order, deduplicates
    row = {
        "experiment": experiment_name,
        "config_name": config_name,
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
    }
    append_result(csv_path, row)

    # Cleanup
    del model, trainer, dm, test_dm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Runner Functions
# ---------------------------------------------------------------------------

def run_block_benchmark(dataset_name, periods, max_epochs, batch_size, accelerator_override, num_workers=0):
    """Part 1: Block-type benchmark across periods."""
    results_dir = os.path.join(RESULTS_DIR, dataset_name)
    csv_path = os.path.join(results_dir, "block_benchmark_results.csv")
    init_csv(csv_path)
    fm = get_forecast_multiplier(dataset_name)

    for period in periods:
        print(f"\n{'='*60}")
        print(f"Block Benchmark — {period}")
        print(f"{'='*60}")

        dataset = load_dataset(dataset_name, period)
        train_series_list = dataset.get_training_series()

        for config_name, cfg in BLOCK_CONFIGS.items():
            for run_idx in range(N_RUNS):
                run_single_experiment(
                    experiment_name="block_benchmark",
                    config_name=config_name,
                    stack_types=cfg["stack_types"],
                    period=period,
                    run_idx=run_idx,
                    dataset=dataset,
                    train_series_list=train_series_list,
                    csv_path=csv_path,
                    n_blocks_per_stack=cfg["n_blocks_per_stack"],
                    share_weights=cfg["share_weights"],
                    active_g=False,
                    sum_losses=False,
                    activation="ReLU",
                    max_epochs=max_epochs,
                    batch_size=batch_size,
                    accelerator_override=accelerator_override,
                    forecast_multiplier=fm,
                    num_workers=num_workers,
                )


def run_wavelet_v2_benchmark(dataset_name, periods, max_epochs, batch_size, accelerator_override, num_workers=0):
    """Part 4: Numerically stabilized wavelet V2 benchmark."""
    results_dir = os.path.join(RESULTS_DIR, dataset_name)
    csv_path = os.path.join(results_dir, "wavelet_v2_benchmark_results.csv")
    init_csv(csv_path)
    fm = get_forecast_multiplier(dataset_name)

    for period in periods:
        print(f"\n{'='*60}")
        print(f"Wavelet V2 Benchmark — {period}")
        print(f"{'='*60}")

        dataset = load_dataset(dataset_name, period)
        train_series_list = dataset.get_training_series()

        for config_name, cfg in WAVELET_V2_CONFIGS.items():
            for run_idx in range(N_RUNS):
                run_single_experiment(
                    experiment_name="wavelet_v2_benchmark",
                    config_name=config_name,
                    stack_types=cfg["stack_types"],
                    period=period,
                    run_idx=run_idx,
                    dataset=dataset,
                    train_series_list=train_series_list,
                    csv_path=csv_path,
                    n_blocks_per_stack=cfg["n_blocks_per_stack"],
                    share_weights=cfg["share_weights"],
                    active_g=False,
                    sum_losses=False,
                    activation="ReLU",
                    max_epochs=max_epochs,
                    batch_size=batch_size,
                    accelerator_override=accelerator_override,
                    forecast_multiplier=fm,
                    num_workers=num_workers,
                )


def run_wavelet_v3_benchmark(dataset_name, periods, max_epochs, batch_size, accelerator_override, num_workers=0):
    """Part 5: Orthonormal DWT wavelet V3 benchmark."""
    results_dir = os.path.join(RESULTS_DIR, dataset_name)
    csv_path = os.path.join(results_dir, "wavelet_v3_benchmark_results.csv")
    init_csv(csv_path)
    fm = get_forecast_multiplier(dataset_name)

    for period in periods:
        print(f"\n{'='*60}")
        print(f"Wavelet V3 Benchmark — {period}")
        print(f"{'='*60}")

        dataset = load_dataset(dataset_name, period)
        train_series_list = dataset.get_training_series()

        # Standalone V3 block configs
        for config_name, cfg in WAVELET_V3_CONFIGS.items():
            for run_idx in range(N_RUNS):
                run_single_experiment(
                    experiment_name="wavelet_v3_benchmark",
                    config_name=config_name,
                    stack_types=cfg["stack_types"],
                    period=period,
                    run_idx=run_idx,
                    dataset=dataset,
                    train_series_list=train_series_list,
                    csv_path=csv_path,
                    n_blocks_per_stack=cfg["n_blocks_per_stack"],
                    share_weights=cfg["share_weights"],
                    active_g=False,
                    sum_losses=False,
                    activation="ReLU",
                    max_epochs=max_epochs,
                    batch_size=batch_size,
                    accelerator_override=accelerator_override,
                    forecast_multiplier=fm,
                    num_workers=num_workers,
                )

        # V3 ablation configs (on 30-stack DB3WaveletV3)
        v3_ablation_stack = ["DB3WaveletV3"] * TOTAL_STACKS
        for config_name, ablation in V3_ABLATION_CONFIGS.items():
            for run_idx in range(N_RUNS):
                run_single_experiment(
                    experiment_name="wavelet_v3_benchmark",
                    config_name=config_name,
                    stack_types=v3_ablation_stack,
                    period=period,
                    run_idx=run_idx,
                    dataset=dataset,
                    train_series_list=train_series_list,
                    csv_path=csv_path,
                    n_blocks_per_stack=1,
                    share_weights=True,
                    active_g=ablation["active_g"],
                    sum_losses=ablation["sum_losses"],
                    activation=ablation["activation"],
                    max_epochs=max_epochs,
                    batch_size=batch_size,
                    accelerator_override=accelerator_override,
                    forecast_multiplier=fm,
                    num_workers=num_workers,
                )


def run_convergence_study(max_epochs, batch_size, accelerator_override,
                          config_filter=None, num_workers=0):
    """Part 6: Convergence study — multi-dataset, random seeds.

    Tests active_g/sum_losses stabilization across datasets to
    disentangle data distribution effects from initialization effects.
    Uses random seeds (tracked in CSV) instead of fixed seeds.

    Args:
        config_filter: Optional config name to run only that config (for parallel execution).
    """
    stack_types = ["Generic"] * CONVERGENCE_STUDY_STACKS

    # Filter configs if requested
    configs = CONVERGENCE_STUDY_CONFIGS
    if config_filter:
        if config_filter not in configs:
            print(f"Unknown convergence config: {config_filter}")
            print(f"Available: {list(configs.keys())}")
            return
        configs = {config_filter: configs[config_filter]}

    for dataset_name, periods in CONVERGENCE_STUDY_DATASETS.items():
        results_dir = os.path.join(RESULTS_DIR, dataset_name)
        csv_path = os.path.join(results_dir, "convergence_study_results.csv")
        init_csv(csv_path)
        fm = get_forecast_multiplier(dataset_name)

        for period in periods:
            print(f"\n{'='*60}")
            print(f"Convergence Study — {dataset_name} / {period}")
            print(f"{'='*60}")

            dataset = load_dataset(dataset_name, period)
            train_series_list = dataset.get_training_series()

            for config_name, cfg in configs.items():
                for run_idx in range(CONVERGENCE_STUDY_N_RUNS):
                    # Random seed — tracked in CSV for reproducibility
                    seed = random.randint(0, 2**31 - 1)

                    run_single_experiment(
                        experiment_name="convergence_study",
                        config_name=config_name,
                        stack_types=stack_types,
                        period=period,
                        run_idx=run_idx,
                        dataset=dataset,
                        train_series_list=train_series_list,
                        csv_path=csv_path,
                        n_blocks_per_stack=1,
                        share_weights=True,
                        active_g=cfg["active_g"],
                        sum_losses=cfg["sum_losses"],
                        activation=cfg["activation"],
                        max_epochs=max_epochs,
                        batch_size=batch_size,
                        accelerator_override=accelerator_override,
                        forecast_multiplier=fm,
                        seed_override=seed,
                        num_workers=num_workers,
                    )


def run_ablation_studies(dataset_name, periods, max_epochs, batch_size, accelerator_override, num_workers=0):
    """Part 2: Ablation studies on 30-stack Generic."""
    results_dir = os.path.join(RESULTS_DIR, dataset_name)
    csv_path = os.path.join(results_dir, "ablation_results.csv")
    init_csv(csv_path)
    fm = get_forecast_multiplier(dataset_name)

    # Ablation baseline: 30-stack Generic with shared weights (paper config)
    ablation_stack_types = ["Generic"] * TOTAL_STACKS

    for period in periods:
        print(f"\n{'='*60}")
        print(f"Ablation Studies — {period}")
        print(f"{'='*60}")

        dataset = load_dataset(dataset_name, period)
        train_series_list = dataset.get_training_series()

        for config_name, ablation in ABLATION_CONFIGS.items():
            for run_idx in range(N_RUNS):
                run_single_experiment(
                    experiment_name="ablation",
                    config_name=config_name,
                    stack_types=ablation_stack_types,
                    period=period,
                    run_idx=run_idx,
                    dataset=dataset,
                    train_series_list=train_series_list,
                    csv_path=csv_path,
                    n_blocks_per_stack=1,
                    share_weights=True,
                    active_g=ablation["active_g"],
                    sum_losses=ablation["sum_losses"],
                    activation=ablation["activation"],
                    max_epochs=max_epochs,
                    batch_size=batch_size,
                    accelerator_override=accelerator_override,
                    forecast_multiplier=fm,
                    num_workers=num_workers,
                )


def run_ensemble_experiment(dataset_name, periods, max_epochs, batch_size, accelerator_override, num_workers=0):
    """Part 3: Multi-horizon ensemble following the original N-BEATS paper.

    For each architecture (G, I, I+G) and each period:
      - Train models at 6 backcast multipliers (2H-7H) x N_RUNS seeds
      - Take element-wise median of all forecasts (paper's ensemble strategy)
      - Report per-model and ensemble sMAPE, MASE, OWA
    """
    results_dir = os.path.join(RESULTS_DIR, dataset_name)
    individual_csv = os.path.join(results_dir, "ensemble_individual_results.csv")
    summary_csv = os.path.join(results_dir, "ensemble_summary_results.csv")
    preds_dir = os.path.join(results_dir, "ensemble_predictions")
    os.makedirs(preds_dir, exist_ok=True)

    init_csv(individual_csv, ENSEMBLE_INDIVIDUAL_COLUMNS)
    init_csv(summary_csv, ENSEMBLE_SUMMARY_COLUMNS)

    for period in periods:
        print(f"\n{'='*60}")
        print(f"Ensemble Experiment — {period}")
        print(f"{'='*60}")

        dataset = load_dataset(dataset_name, period)
        train_series_list = dataset.get_training_series()
        forecast_length = dataset.forecast_length
        frequency = dataset.frequency

        for config_name, cfg in ENSEMBLE_CONFIGS.items():
            # Skip if ensemble summary already computed
            if ensemble_summary_exists(summary_csv, config_name, period):
                print(f"  [SKIP] {config_name} / {period} ensemble — already exists")
                continue

            stack_types = cfg["stack_types"]
            n_stacks = len(stack_types)
            n_blocks_per_stack = cfg["n_blocks_per_stack"]
            share_weights = cfg["share_weights"]

            all_predictions = []
            targets = None

            for multiplier in FORECAST_MULTIPLIERS:
                backcast_length = forecast_length * multiplier

                for run_idx in range(N_RUNS):
                    seed = BASE_SEED + run_idx
                    pred_file = os.path.join(
                        preds_dir,
                        f"{config_name}_{period}_m{multiplier}_run{run_idx}.npz",
                    )

                    if os.path.exists(pred_file):
                        # Load cached predictions
                        data = np.load(pred_file)
                        preds = data["preds"]
                        targets = data["targets"]
                        all_predictions.append(preds)
                        print(f"  [LOAD] {config_name} / {period} / "
                              f"m={multiplier} / run {run_idx}")
                        continue

                    set_seed(seed)

                    accelerator, device = resolve_accelerator(accelerator_override)

                    model = NBeatsNet(
                        backcast_length=backcast_length,
                        forecast_length=forecast_length,
                        stack_types=stack_types,
                        n_blocks_per_stack=n_blocks_per_stack,
                        share_weights=share_weights,
                        thetas_dim=THETAS_DIM,
                        loss=LOSS,
                        active_g=False,
                        sum_losses=False,
                        activation="ReLU",
                        latent_dim=LATENT_DIM,
                        basis_dim=BASIS_DIM,
                        learning_rate=LEARNING_RATE,
                        no_val=False,
                    )

                    n_params = count_parameters(model)
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
                        train_data, test_data,
                        backcast_length=backcast_length,
                        forecast_length=forecast_length,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                    )

                    log_dir = os.path.join(RESULTS_DIR, "lightning_logs")
                    log_name = (f"ensemble/{config_name}/{period}/"
                                f"m{multiplier}/run{run_idx}")

                    chk_cb = ModelCheckpoint(
                        filename="best-checkpoint",
                        save_top_k=1, monitor="val_loss", mode="min",
                    )
                    es_cb = EarlyStopping(
                        monitor="val_loss",
                        patience=EARLY_STOPPING_PATIENCE,
                        mode="min", verbose=False,
                    )
                    tb_logger = pl_loggers.TensorBoardLogger(
                        save_dir=log_dir, name=log_name,
                    )

                    trainer = pl.Trainer(
                        accelerator=accelerator, devices=1,
                        max_epochs=max_epochs,
                        callbacks=[chk_cb, es_cb],
                        logger=[tb_logger],
                        enable_progress_bar=True,
                        deterministic=False,
                        log_every_n_steps=50,
                    )

                    stack_summary = (
                        f"{n_stacks}x{stack_types[0]}"
                        if len(set(stack_types)) == 1
                        else f"{n_stacks} mixed"
                    )
                    print(f"  [RUN]  {config_name} / {period} / "
                          f"m={multiplier} / run {run_idx} "
                          f"(seed={seed}, {stack_summary}, "
                          f"params={n_params:,})")

                    t0 = time.time()
                    trainer.fit(model, datamodule=dm)
                    training_time = time.time() - t0

                    best_path = trainer.checkpoint_callback.best_model_path
                    if best_path:
                        model = NBeatsNet.load_from_checkpoint(
                            best_path, weights_only=False,
                        )
                    epochs_trained = trainer.current_epoch

                    preds, tgts = run_inference(model, test_dm, device)
                    targets = tgts  # same for all multipliers

                    # Cache predictions
                    np.savez_compressed(pred_file, preds=preds, targets=targets)
                    all_predictions.append(preds)

                    # Per-model metrics
                    smape = compute_smape(preds, targets)
                    mase = compute_m4_mase(
                        preds, targets, train_series_list, frequency,
                    )
                    mae = compute_mae(preds, targets)
                    mse = compute_mse(preds, targets)
                    owa = dataset.compute_owa(smape, mase)

                    print(f"         sMAPE={smape:.4f}  MASE={mase:.4f}  "
                          f"OWA={owa:.4f}  time={training_time:.1f}s  "
                          f"epochs={epochs_trained}")

                    unique_types = list(dict.fromkeys(stack_types))
                    row = {
                        "experiment": "ensemble",
                        "config_name": config_name,
                        "stack_types": str(unique_types),
                        "period": period,
                        "frequency": frequency,
                        "forecast_length": forecast_length,
                        "backcast_length": backcast_length,
                        "forecast_multiplier": multiplier,
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
                        "active_g": False,
                        "sum_losses": False,
                        "activation": "ReLU",
                    }
                    append_result(
                        individual_csv, row, ENSEMBLE_INDIVIDUAL_COLUMNS,
                    )

                    del model, trainer, dm, test_dm
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # --- Ensemble aggregation: median across all models ---
            n_models = len(all_predictions)
            if n_models == 0:
                print(f"  [WARN] No predictions for {config_name}/{period}")
                continue

            ensemble_preds = np.median(np.stack(all_predictions), axis=0)
            ens_smape = compute_smape(ensemble_preds, targets)
            ens_mase = compute_m4_mase(
                ensemble_preds, targets, train_series_list, frequency,
            )
            ens_mae = compute_mae(ensemble_preds, targets)
            ens_mse = compute_mse(ensemble_preds, targets)
            ens_owa = dataset.compute_owa(ens_smape, ens_mase)

            print(f"  [ENS]  {config_name} / {period} — "
                  f"{n_models} models → median ensemble")
            print(f"         sMAPE={ens_smape:.4f}  MASE={ens_mase:.4f}  "
                  f"OWA={ens_owa:.4f}")

            summary_row = {
                "config_name": config_name,
                "period": period,
                "frequency": frequency,
                "forecast_length": forecast_length,
                "n_models": n_models,
                "multipliers": str(FORECAST_MULTIPLIERS),
                "n_runs_per_multiplier": N_RUNS,
                "ensemble_smape": f"{ens_smape:.6f}",
                "ensemble_mase": f"{ens_mase:.6f}",
                "ensemble_mae": f"{ens_mae:.6f}",
                "ensemble_mse": f"{ens_mse:.6f}",
                "ensemble_owa": f"{ens_owa:.6f}",
            }
            append_result(summary_csv, summary_row, ENSEMBLE_SUMMARY_COLUMNS)


def run_mixed_stack_benchmark(dataset_name, periods, max_epochs, batch_size, accelerator_override, num_workers=0):
    """Part 7a: Mixed stack benchmark — I+G pattern with novel G blocks."""
    results_dir = os.path.join(RESULTS_DIR, dataset_name)
    csv_path = os.path.join(results_dir, "block_benchmark_results.csv")
    init_csv(csv_path)
    fm = get_forecast_multiplier(dataset_name)

    for period in periods:
        print(f"\n{'='*60}")
        print(f"Mixed Stack Benchmark — {period}")
        print(f"{'='*60}")

        dataset = load_dataset(dataset_name, period)
        train_series_list = dataset.get_training_series()

        for config_name, cfg in MIXED_STACK_CONFIGS.items():
            for run_idx in range(N_RUNS):
                run_single_experiment(
                    experiment_name="mixed_stack_benchmark",
                    config_name=config_name,
                    stack_types=cfg["stack_types"],
                    period=period,
                    run_idx=run_idx,
                    dataset=dataset,
                    train_series_list=train_series_list,
                    csv_path=csv_path,
                    n_blocks_per_stack=cfg["n_blocks_per_stack"],
                    share_weights=cfg["share_weights"],
                    active_g=False,
                    sum_losses=False,
                    activation="ReLU",
                    max_epochs=max_epochs,
                    batch_size=batch_size,
                    accelerator_override=accelerator_override,
                    forecast_multiplier=fm,
                    num_workers=num_workers,
                )


def run_mixed_stack_ensemble(dataset_name, periods, max_epochs, batch_size, accelerator_override, num_workers=0):
    """Part 7b: Multi-horizon ensemble for mixed stack configs.

    Same strategy as Part 3: train at 6 backcast multipliers (2H-7H) x N_RUNS
    seeds, take element-wise median forecast across all models.
    """
    results_dir = os.path.join(RESULTS_DIR, dataset_name)
    individual_csv = os.path.join(results_dir, "ensemble_individual_results.csv")
    summary_csv = os.path.join(results_dir, "ensemble_summary_results.csv")
    preds_dir = os.path.join(results_dir, "ensemble_predictions")
    os.makedirs(preds_dir, exist_ok=True)

    init_csv(individual_csv, ENSEMBLE_INDIVIDUAL_COLUMNS)
    init_csv(summary_csv, ENSEMBLE_SUMMARY_COLUMNS)

    for period in periods:
        print(f"\n{'='*60}")
        print(f"Mixed Stack Ensemble — {period}")
        print(f"{'='*60}")

        dataset = load_dataset(dataset_name, period)
        train_series_list = dataset.get_training_series()
        forecast_length = dataset.forecast_length
        frequency = dataset.frequency

        for config_name, cfg in MIXED_STACK_ENSEMBLE_CONFIGS.items():
            # Skip if ensemble summary already computed
            if ensemble_summary_exists(summary_csv, config_name, period):
                print(f"  [SKIP] {config_name} / {period} ensemble — already exists")
                continue

            stack_types = cfg["stack_types"]
            n_stacks = len(stack_types)
            n_blocks_per_stack = cfg["n_blocks_per_stack"]
            share_weights = cfg["share_weights"]

            all_predictions = []
            targets = None

            for multiplier in FORECAST_MULTIPLIERS:
                backcast_length = forecast_length * multiplier

                for run_idx in range(N_RUNS):
                    seed = BASE_SEED + run_idx
                    pred_file = os.path.join(
                        preds_dir,
                        f"{config_name}_{period}_m{multiplier}_run{run_idx}.npz",
                    )

                    if os.path.exists(pred_file):
                        # Load cached predictions
                        data = np.load(pred_file)
                        preds = data["preds"]
                        targets = data["targets"]
                        all_predictions.append(preds)
                        print(f"  [LOAD] {config_name} / {period} / "
                              f"m={multiplier} / run {run_idx}")
                        continue

                    set_seed(seed)

                    accelerator, device = resolve_accelerator(accelerator_override)

                    model = NBeatsNet(
                        backcast_length=backcast_length,
                        forecast_length=forecast_length,
                        stack_types=stack_types,
                        n_blocks_per_stack=n_blocks_per_stack,
                        share_weights=share_weights,
                        thetas_dim=THETAS_DIM,
                        loss=LOSS,
                        active_g=False,
                        sum_losses=False,
                        activation="ReLU",
                        latent_dim=LATENT_DIM,
                        basis_dim=BASIS_DIM,
                        learning_rate=LEARNING_RATE,
                        no_val=False,
                    )

                    n_params = count_parameters(model)
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
                        train_data, test_data,
                        backcast_length=backcast_length,
                        forecast_length=forecast_length,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                    )

                    log_dir = os.path.join(RESULTS_DIR, "lightning_logs")
                    log_name = (f"mixed_ensemble/{config_name}/{period}/"
                                f"m{multiplier}/run{run_idx}")

                    chk_cb = ModelCheckpoint(
                        filename="best-checkpoint",
                        save_top_k=1, monitor="val_loss", mode="min",
                    )
                    es_cb = EarlyStopping(
                        monitor="val_loss",
                        patience=EARLY_STOPPING_PATIENCE,
                        mode="min", verbose=False,
                    )
                    tb_logger = pl_loggers.TensorBoardLogger(
                        save_dir=log_dir, name=log_name,
                    )

                    trainer = pl.Trainer(
                        accelerator=accelerator, devices=1,
                        max_epochs=max_epochs,
                        callbacks=[chk_cb, es_cb],
                        logger=[tb_logger],
                        enable_progress_bar=True,
                        deterministic=False,
                        log_every_n_steps=50,
                    )

                    stack_summary = (
                        f"{n_stacks}x{stack_types[0]}"
                        if len(set(stack_types)) == 1
                        else f"{n_stacks} mixed"
                    )
                    print(f"  [RUN]  {config_name} / {period} / "
                          f"m={multiplier} / run {run_idx} "
                          f"(seed={seed}, {stack_summary}, "
                          f"params={n_params:,})")

                    t0 = time.time()
                    trainer.fit(model, datamodule=dm)
                    training_time = time.time() - t0

                    best_path = trainer.checkpoint_callback.best_model_path
                    if best_path:
                        model = NBeatsNet.load_from_checkpoint(
                            best_path, weights_only=False,
                        )
                    epochs_trained = trainer.current_epoch

                    preds, tgts = run_inference(model, test_dm, device)
                    targets = tgts  # same for all multipliers

                    # Cache predictions
                    np.savez_compressed(pred_file, preds=preds, targets=targets)
                    all_predictions.append(preds)

                    # Per-model metrics
                    smape = compute_smape(preds, targets)
                    mase = compute_m4_mase(
                        preds, targets, train_series_list, frequency,
                    )
                    mae = compute_mae(preds, targets)
                    mse = compute_mse(preds, targets)
                    owa = dataset.compute_owa(smape, mase)

                    print(f"         sMAPE={smape:.4f}  MASE={mase:.4f}  "
                          f"OWA={owa:.4f}  time={training_time:.1f}s  "
                          f"epochs={epochs_trained}")

                    unique_types = list(dict.fromkeys(stack_types))
                    row = {
                        "experiment": "mixed_stack_ensemble",
                        "config_name": config_name,
                        "stack_types": str(unique_types),
                        "period": period,
                        "frequency": frequency,
                        "forecast_length": forecast_length,
                        "backcast_length": backcast_length,
                        "forecast_multiplier": multiplier,
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
                        "active_g": False,
                        "sum_losses": False,
                        "activation": "ReLU",
                    }
                    append_result(
                        individual_csv, row, ENSEMBLE_INDIVIDUAL_COLUMNS,
                    )

                    del model, trainer, dm, test_dm
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # --- Ensemble aggregation: median across all models ---
            n_models = len(all_predictions)
            if n_models == 0:
                print(f"  [WARN] No predictions for {config_name}/{period}")
                continue

            ensemble_preds = np.median(np.stack(all_predictions), axis=0)
            ens_smape = compute_smape(ensemble_preds, targets)
            ens_mase = compute_m4_mase(
                ensemble_preds, targets, train_series_list, frequency,
            )
            ens_mae = compute_mae(ensemble_preds, targets)
            ens_mse = compute_mse(ensemble_preds, targets)
            ens_owa = dataset.compute_owa(ens_smape, ens_mase)

            print(f"  [ENS]  {config_name} / {period} — "
                  f"{n_models} models → median ensemble")
            print(f"         sMAPE={ens_smape:.4f}  MASE={ens_mase:.4f}  "
                  f"OWA={ens_owa:.4f}")

            summary_row = {
                "config_name": config_name,
                "period": period,
                "frequency": frequency,
                "forecast_length": forecast_length,
                "n_models": n_models,
                "multipliers": str(FORECAST_MULTIPLIERS),
                "n_runs_per_multiplier": N_RUNS,
                "ensemble_smape": f"{ens_smape:.6f}",
                "ensemble_mase": f"{ens_mase:.6f}",
                "ensemble_mae": f"{ens_mae:.6f}",
                "ensemble_mse": f"{ens_mse:.6f}",
                "ensemble_owa": f"{ens_owa:.6f}",
            }
            append_result(summary_csv, summary_row, ENSEMBLE_SUMMARY_COLUMNS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark — N-BEATS paper comparison + novel extensions"
    )
    parser.add_argument(
        "--dataset", choices=["m4", "traffic", "weather"], default="m4",
        help="Dataset to benchmark (default: m4). Part 6 ignores this flag.",
    )
    parser.add_argument(
        "--part", choices=["1", "2", "3", "4", "5", "6", "7", "all"], default="all",
        help=("Which experiments to run: 1=block benchmark, 2=ablation, "
              "3=multi-horizon ensemble, 4=wavelet V2 benchmark, "
              "5=wavelet V3 benchmark, 6=convergence study (multi-dataset), "
              "7=mixed stack benchmark+ensemble, "
              "all=1-3 (use 4/5/6/7 explicitly)"),
    )
    parser.add_argument(
        "--periods", nargs="+", default=None,
        help=("Periods/horizons to benchmark. Defaults: all for the chosen dataset. "
              "M4: Yearly Quarterly Monthly Weekly Daily Hourly. "
              "Traffic: Traffic-96."),
    )
    parser.add_argument(
        "--max-epochs", type=int, default=100,
        help="Maximum training epochs per run (default: 100, early stopping may end sooner)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1024,
        help="Training batch size (default: 1024, paper: 1024)",
    )
    parser.add_argument(
        "--accelerator", choices=["auto", "cpu", "cuda", "mps"], default="auto",
        help="Accelerator to use (default: auto = CUDA > MPS > CPU detection)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="Number of DataLoader worker processes (default: 4). Set 0 for single-threaded loading.",
    )
    parser.add_argument(
        "--convergence-config", default=None,
        choices=list(CONVERGENCE_STUDY_CONFIGS.keys()),
        help="Run only this convergence config (Part 6). Enables parallel execution.",
    )

    args = parser.parse_args()

    # Resolve periods based on dataset
    if args.dataset == "m4":
        all_periods = list(M4_PERIODS.keys())
        valid_periods = M4_PERIODS
    elif args.dataset == "traffic":
        all_periods = list(TRAFFIC_HORIZONS.keys())
        valid_periods = TRAFFIC_HORIZONS
    elif args.dataset == "weather":
        all_periods = list(WEATHER_HORIZONS.keys())
        valid_periods = WEATHER_HORIZONS

    periods = args.periods or all_periods

    # Validate periods against the chosen dataset
    for p in periods:
        if p not in valid_periods:
            parser.error(
                f"Unknown period '{p}' for dataset '{args.dataset}'. "
                f"Choose from: {list(valid_periods.keys())}"
            )

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Summary
    resolved_accel, _ = resolve_accelerator(args.accelerator)
    device_name = resolved_accel.upper()

    num_workers = args.num_workers

    print(f"Dataset: {args.dataset}")
    print(f"Device: {device_name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Periods: {periods}")
    print(f"Max epochs: {args.max_epochs}")

    if args.part in ("1", "all"):
        n_block_runs = len(BLOCK_CONFIGS) * len(periods) * N_RUNS
        print(f"Part 1 — Block benchmark: {n_block_runs} runs "
              f"({len(BLOCK_CONFIGS)} configs x {len(periods)} periods x {N_RUNS} runs)")
    if args.part in ("2", "all"):
        n_ablation_runs = len(ABLATION_CONFIGS) * len(periods) * N_RUNS
        print(f"Part 2 — Ablation studies: {n_ablation_runs} runs "
              f"({len(ABLATION_CONFIGS)} configs x {len(periods)} periods x {N_RUNS} runs)")
    if args.part in ("3", "all"):
        n_ensemble_runs = (len(ENSEMBLE_CONFIGS) * len(FORECAST_MULTIPLIERS)
                           * len(periods) * N_RUNS)
        print(f"Part 3 — Multi-horizon ensemble: {n_ensemble_runs} runs "
              f"({len(ENSEMBLE_CONFIGS)} configs x {len(FORECAST_MULTIPLIERS)} multipliers "
              f"x {len(periods)} periods x {N_RUNS} runs)")
    if args.part == "4":
        n_wavelet_v2_runs = len(WAVELET_V2_CONFIGS) * len(periods) * N_RUNS
        print(f"Part 4 — Wavelet V2 benchmark: {n_wavelet_v2_runs} runs "
              f"({len(WAVELET_V2_CONFIGS)} configs x {len(periods)} periods x {N_RUNS} runs)")
    if args.part == "5":
        n_wavelet_v3_runs = (len(WAVELET_V3_CONFIGS) + len(V3_ABLATION_CONFIGS)) * len(periods) * N_RUNS
        print(f"Part 5 — Wavelet V3 benchmark: {n_wavelet_v3_runs} runs "
              f"({len(WAVELET_V3_CONFIGS)}+{len(V3_ABLATION_CONFIGS)} configs x {len(periods)} periods x {N_RUNS} runs)")
    if args.part == "6":
        n_conv_runs = sum(
            len(pds) for pds in CONVERGENCE_STUDY_DATASETS.values()
        ) * len(CONVERGENCE_STUDY_CONFIGS) * CONVERGENCE_STUDY_N_RUNS
        print(f"Part 6 — Convergence study (multi-dataset, random seeds): {n_conv_runs} runs "
              f"({len(CONVERGENCE_STUDY_CONFIGS)} configs x "
              f"{sum(len(p) for p in CONVERGENCE_STUDY_DATASETS.values())} dataset-periods x "
              f"{CONVERGENCE_STUDY_N_RUNS} runs)")
    if args.part == "7":
        n_mixed_bench_runs = len(MIXED_STACK_CONFIGS) * len(periods) * N_RUNS
        n_mixed_ens_runs = (len(MIXED_STACK_ENSEMBLE_CONFIGS) * len(FORECAST_MULTIPLIERS)
                            * len(periods) * N_RUNS)
        print(f"Part 7 — Mixed stack benchmark: {n_mixed_bench_runs} runs "
              f"({len(MIXED_STACK_CONFIGS)} configs x {len(periods)} periods x {N_RUNS} runs)")
        print(f"Part 7 — Mixed stack ensemble: {n_mixed_ens_runs} runs "
              f"({len(MIXED_STACK_ENSEMBLE_CONFIGS)} configs x {len(FORECAST_MULTIPLIERS)} multipliers "
              f"x {len(periods)} periods x {N_RUNS} runs)")

    if args.part in ("1", "all"):
        run_block_benchmark(args.dataset, periods, args.max_epochs, args.batch_size, args.accelerator, num_workers)

    if args.part in ("2", "all"):
        run_ablation_studies(args.dataset, periods, args.max_epochs, args.batch_size, args.accelerator, num_workers)

    if args.part in ("3", "all"):
        run_ensemble_experiment(args.dataset, periods, args.max_epochs, args.batch_size, args.accelerator, num_workers)

    if args.part == "4":
        run_wavelet_v2_benchmark(args.dataset, periods, args.max_epochs, args.batch_size, args.accelerator, num_workers)

    if args.part == "5":
        run_wavelet_v3_benchmark(args.dataset, periods, args.max_epochs, args.batch_size, args.accelerator, num_workers)

    if args.part == "6":
        run_convergence_study(args.max_epochs, args.batch_size, args.accelerator,
                              config_filter=args.convergence_config, num_workers=num_workers)

    if args.part == "7":
        run_mixed_stack_benchmark(args.dataset, periods, args.max_epochs, args.batch_size, args.accelerator, num_workers)
        run_mixed_stack_ensemble(args.dataset, periods, args.max_epochs, args.batch_size, args.accelerator, num_workers)

    results_dir = os.path.join(RESULTS_DIR, args.dataset)
    print("\nDone. Results saved to:")
    if args.part in ("1", "all"):
        print(f"  {os.path.join(results_dir, 'block_benchmark_results.csv')}")
    if args.part in ("2", "all"):
        print(f"  {os.path.join(results_dir, 'ablation_results.csv')}")
    if args.part in ("3", "all"):
        print(f"  {os.path.join(results_dir, 'ensemble_individual_results.csv')}")
        print(f"  {os.path.join(results_dir, 'ensemble_summary_results.csv')}")
    if args.part == "4":
        print(f"  {os.path.join(results_dir, 'wavelet_v2_benchmark_results.csv')}")
    if args.part == "5":
        print(f"  {os.path.join(results_dir, 'wavelet_v3_benchmark_results.csv')}")
    if args.part == "6":
        for ds_name in CONVERGENCE_STUDY_DATASETS:
            print(f"  {os.path.join(RESULTS_DIR, ds_name, 'convergence_study_results.csv')}")
    if args.part == "7":
        print(f"  {os.path.join(results_dir, 'block_benchmark_results.csv')}")
        print(f"  {os.path.join(results_dir, 'ensemble_individual_results.csv')}")
        print(f"  {os.path.join(results_dir, 'ensemble_summary_results.csv')}")


if __name__ == "__main__":
    main()
