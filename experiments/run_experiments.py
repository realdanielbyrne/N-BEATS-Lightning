"""
M4 Benchmark Experiment Script for N-BEATS Lightning

Runs systematic benchmarks of all block-type extensions against paper baselines
on the M4 dataset. Produces CSV results suitable for generating tables and charts.

Part 1: Block-type benchmark (14 configurations)
Part 2: Ablation studies on Generic (7 configurations)

Usage:
    python experiments/run_experiments.py --part 1 --periods Yearly --max-epochs 50
    python experiments/run_experiments.py --part all
    python experiments/run_experiments.py --part 2 --periods Yearly Monthly --max-epochs 100
"""

import argparse
import csv
import gc
import os
import sys
import time

import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import loggers as pl_loggers

# Allow running from project root or experiments/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lightningnbeats.models import NBeatsNet
from lightningnbeats.loaders import (
    ColumnarCollectionTimeSeriesDataModule,
    ColumnarCollectionTimeSeriesTestDataModule,
)
from lightningnbeats.data import M4Dataset

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

# Fixed training hyperparameters
BATCH_SIZE = 2048
FORECAST_MULTIPLIER = 5
TOTAL_STACKS = 8
THETAS_DIM = 5
LATENT_DIM = 4
BASIS_DIM = 128
LOSS = "SMAPELoss"
LEARNING_RATE = 1e-4

N_RUNS = 3
BASE_SEED = 42

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ---------------------------------------------------------------------------
# Block Benchmark Configs (Part 1)
# ---------------------------------------------------------------------------

BLOCK_CONFIGS = {
    # Paper baselines
    "Generic":              ["Generic"],
    "Trend+Seasonality":    ["Trend", "Seasonality"],
    # Novel block types
    "BottleneckGeneric":    ["BottleneckGeneric"],
    "AutoEncoder":          ["AutoEncoder"],
    "GenericAE":            ["GenericAE"],
    "BottleneckGenericAE":  ["BottleneckGenericAE"],
    "GenericAEBackcast":    ["GenericAEBackcast"],
    # Wavelets (representative subset)
    "HaarWavelet":          ["HaarWavelet"],
    "DB3Wavelet":           ["DB3Wavelet"],
    "DB3AltWavelet":        ["DB3AltWavelet"],
    "Coif2Wavelet":         ["Coif2Wavelet"],
    "Symlet3Wavelet":       ["Symlet3Wavelet"],
    # Mixed stacks
    "Trend+HaarWavelet":    ["Trend", "HaarWavelet"],
    "Generic+DB3Wavelet":   ["Generic", "DB3Wavelet"],
}

# ---------------------------------------------------------------------------
# Ablation Configs (Part 2) — all on Generic
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = {
    "Generic_baseline":  {"active_g": False, "sum_losses": False, "activation": "ReLU"},
    "Generic_activeG":   {"active_g": True,  "sum_losses": False, "activation": "ReLU"},
    "Generic_sumLosses": {"active_g": False, "sum_losses": True,  "activation": "ReLU"},
    "Generic_GELU":      {"active_g": False, "sum_losses": False, "activation": "GELU"},
    "Generic_ELU":       {"active_g": False, "sum_losses": False, "activation": "ELU"},
    "Generic_LeakyReLU": {"active_g": False, "sum_losses": False, "activation": "LeakyReLU"},
    "Generic_SELU":      {"active_g": False, "sum_losses": False, "activation": "SELU"},
}

# ---------------------------------------------------------------------------
# CSV Column Schema
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "experiment", "config_name", "stack_types", "period", "frequency",
    "forecast_length", "backcast_length", "run", "seed",
    "smape", "mase", "n_params",
    "training_time_seconds", "epochs_trained",
    "active_g", "sum_losses", "activation",
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


def get_training_series(m4_dataset):
    """Extract per-column training arrays (NaN removed) from M4Dataset.

    m4_dataset.train_data is a DataFrame where columns are series and rows
    are time steps (NaN-padded at top for shorter series).
    """
    series_list = []
    train_df = m4_dataset.train_data
    for col in train_df.columns:
        vals = train_df[col].dropna().values.astype(np.float64)
        series_list.append(vals)
    return series_list


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

def init_csv(path):
    """Create CSV with header if it doesn't exist."""
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def append_result(path, row_dict):
    """Append a single result row to CSV."""
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
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


# ---------------------------------------------------------------------------
# Single Run Function
# ---------------------------------------------------------------------------

def run_single_experiment(
    experiment_name,
    config_name,
    stack_types_base,
    period,
    run_idx,
    m4,
    train_series_list,
    csv_path,
    active_g=False,
    sum_losses=False,
    activation="ReLU",
    max_epochs=50,
):
    """Run a single training + evaluation experiment and save results to CSV."""

    # Check resumability
    if result_exists(csv_path, experiment_name, config_name, period, run_idx):
        print(f"  [SKIP] {config_name} / {period} / run {run_idx} — already exists")
        return

    seed = BASE_SEED + run_idx
    set_seed(seed)

    forecast_length = m4.forecast_length
    frequency = m4.frequency
    backcast_length = forecast_length * FORECAST_MULTIPLIER

    # Expand stack types to fill TOTAL_STACKS
    n_repeats = TOTAL_STACKS // len(stack_types_base)
    stack_types = stack_types_base * n_repeats

    # Detect accelerator
    if torch.cuda.is_available():
        accelerator = "cuda"
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        accelerator = "mps"
        device = torch.device("mps")
    else:
        accelerator = "cpu"
        device = torch.device("cpu")

    # Create model
    model = NBeatsNet(
        backcast_length=backcast_length,
        forecast_length=forecast_length,
        stack_types=stack_types,
        n_blocks_per_stack=1,
        share_weights=False,
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
    train_data = m4.train_data
    test_data = m4.test_data

    dm = ColumnarCollectionTimeSeriesDataModule(
        train_data,
        backcast_length=backcast_length,
        forecast_length=forecast_length,
        batch_size=BATCH_SIZE,
        no_val=False,
    )

    test_dm = ColumnarCollectionTimeSeriesTestDataModule(
        train_data,
        test_data,
        backcast_length=backcast_length,
        forecast_length=forecast_length,
        batch_size=BATCH_SIZE,
    )

    # Trainer
    log_dir = os.path.join(RESULTS_DIR, "lightning_logs")
    log_name = f"{experiment_name}/{config_name}/{period}/run{run_idx}"

    chk_callback = ModelCheckpoint(
        filename="best-checkpoint",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir, name=log_name)

    trainer = pl.Trainer(
        accelerator=accelerator,
        max_epochs=max_epochs,
        callbacks=[chk_callback],
        logger=[tb_logger],
        enable_progress_bar=True,
        deterministic=False,
    )

    # Train
    print(f"  [RUN]  {config_name} / {period} / run {run_idx} "
          f"(seed={seed}, stacks={stack_types}, params={n_params:,})")
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

    print(f"         sMAPE={smape:.4f}  MASE={mase:.4f}  "
          f"time={training_time:.1f}s  epochs={epochs_trained}")

    # Save result
    row = {
        "experiment": experiment_name,
        "config_name": config_name,
        "stack_types": str(stack_types_base),
        "period": period,
        "frequency": frequency,
        "forecast_length": forecast_length,
        "backcast_length": backcast_length,
        "run": run_idx,
        "seed": seed,
        "smape": f"{smape:.6f}",
        "mase": f"{mase:.6f}",
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

def run_block_benchmark(periods, max_epochs):
    """Part 1: Block-type benchmark across M4 periods."""
    csv_path = os.path.join(RESULTS_DIR, "block_benchmark_results.csv")
    init_csv(csv_path)

    for period in periods:
        print(f"\n{'='*60}")
        print(f"Block Benchmark — {period}")
        print(f"{'='*60}")

        m4 = M4Dataset(period, "All")
        train_series_list = get_training_series(m4)

        for config_name, stack_types_base in BLOCK_CONFIGS.items():
            for run_idx in range(N_RUNS):
                run_single_experiment(
                    experiment_name="block_benchmark",
                    config_name=config_name,
                    stack_types_base=stack_types_base,
                    period=period,
                    run_idx=run_idx,
                    m4=m4,
                    train_series_list=train_series_list,
                    csv_path=csv_path,
                    active_g=False,
                    sum_losses=False,
                    activation="ReLU",
                    max_epochs=max_epochs,
                )


def run_ablation_studies(periods, max_epochs):
    """Part 2: Ablation studies on Generic block across M4 periods."""
    csv_path = os.path.join(RESULTS_DIR, "ablation_results.csv")
    init_csv(csv_path)

    for period in periods:
        print(f"\n{'='*60}")
        print(f"Ablation Studies — {period}")
        print(f"{'='*60}")

        m4 = M4Dataset(period, "All")
        train_series_list = get_training_series(m4)

        for config_name, ablation in ABLATION_CONFIGS.items():
            for run_idx in range(N_RUNS):
                run_single_experiment(
                    experiment_name="ablation",
                    config_name=config_name,
                    stack_types_base=["Generic"],
                    period=period,
                    run_idx=run_idx,
                    m4=m4,
                    train_series_list=train_series_list,
                    csv_path=csv_path,
                    active_g=ablation["active_g"],
                    sum_losses=ablation["sum_losses"],
                    activation=ablation["activation"],
                    max_epochs=max_epochs,
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="M4 Benchmark Experiments for N-BEATS Lightning"
    )
    parser.add_argument(
        "--part", choices=["1", "2", "all"], default="all",
        help="Which experiments to run: 1=block benchmark, 2=ablation, all=both",
    )
    parser.add_argument(
        "--periods", nargs="+",
        default=["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"],
        help="M4 periods to benchmark (default: all 6)",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=50,
        help="Maximum training epochs per run (default: 50)",
    )

    args = parser.parse_args()

    # Validate periods
    for p in args.periods:
        if p not in M4_PERIODS:
            parser.error(f"Unknown period '{p}'. Choose from: {list(M4_PERIODS.keys())}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Summary
    n_block_runs = len(BLOCK_CONFIGS) * len(args.periods) * N_RUNS
    n_ablation_runs = len(ABLATION_CONFIGS) * len(args.periods) * N_RUNS

    device_name = "CUDA" if torch.cuda.is_available() else (
        "MPS" if torch.backends.mps.is_available() else "CPU"
    )

    print(f"Device: {device_name}")
    print(f"Periods: {args.periods}")
    print(f"Max epochs: {args.max_epochs}")

    if args.part in ("1", "all"):
        print(f"Part 1 — Block benchmark: {n_block_runs} runs "
              f"({len(BLOCK_CONFIGS)} configs x {len(args.periods)} periods x {N_RUNS} runs)")
    if args.part in ("2", "all"):
        print(f"Part 2 — Ablation studies: {n_ablation_runs} runs "
              f"({len(ABLATION_CONFIGS)} configs x {len(args.periods)} periods x {N_RUNS} runs)")

    if args.part in ("1", "all"):
        run_block_benchmark(args.periods, args.max_epochs)

    if args.part in ("2", "all"):
        run_ablation_studies(args.periods, args.max_epochs)

    print("\nDone. Results saved to:")
    if args.part in ("1", "all"):
        print(f"  {os.path.join(RESULTS_DIR, 'block_benchmark_results.csv')}")
    if args.part in ("2", "all"):
        print(f"  {os.path.join(RESULTS_DIR, 'ablation_results.csv')}")


if __name__ == "__main__":
    main()
