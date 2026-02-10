"""
M4 Benchmark Experiment Script for N-BEATS Lightning

1:1 comparison with the original N-BEATS paper (Oreshkin et al., 2019) using
paper-faithful hyperparameters (30 stacks, batch=1024, lr=1e-3, early stopping),
alongside novel extensions from this codebase.

Part 1: Block-type benchmark — Paper baselines (N-BEATS-G, N-BEATS-I) plus
        novel block types at the same 30-stack scale for fair comparison.
Part 2: Ablation studies on 30-stack Generic — active_g, sum_losses, activations.

Usage:
    python experiments/run_experiments.py --part 1 --periods Yearly --max-epochs 100
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
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
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

# Naïve2 baseline values from the M4 competition (Makridakis et al., 2020).
# Used to compute OWA = 0.5 * (sMAPE/sMAPE_Naive2 + MASE/MASE_Naive2).
NAIVE2_SMAPE = {
    "Yearly": 16.342, "Quarterly": 11.012, "Monthly": 14.427,
    "Weekly": 9.161,  "Daily": 3.045,      "Hourly": 18.383,
}
NAIVE2_MASE = {
    "Yearly": 3.974, "Quarterly": 1.371, "Monthly": 1.063,
    "Weekly": 2.777, "Daily": 3.278,     "Hourly": 2.395,
}

# Fixed training hyperparameters — matches original paper where applicable
BATCH_SIZE = 1024                # Paper: 1024
FORECAST_MULTIPLIER = 5          # Paper uses 2-7 for ensemble; 5H is a reasonable single point
TOTAL_STACKS = 30                # Paper: 30 for Generic architecture
THETAS_DIM = 5                   # Polynomial degree for Trend; bottleneck dim for BottleneckGeneric
LATENT_DIM = 4                   # For AE backbone blocks
BASIS_DIM = 128                  # For Wavelet blocks
LOSS = "SMAPELoss"               # Paper primary metric and training loss
LEARNING_RATE = 1e-3             # Paper: 1e-3
EARLY_STOPPING_PATIENCE = 10    # Paper uses early stopping on validation loss

N_RUNS = 3
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
    # Wavelets (representative from each family)
    "HaarWavelet": {
        "stack_types": ["HaarWavelet"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "DB3Wavelet": {
        "stack_types": ["DB3Wavelet"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "DB3AltWavelet": {
        "stack_types": ["DB3AltWavelet"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "Coif2Wavelet": {
        "stack_types": ["Coif2Wavelet"] * 30,
        "n_blocks_per_stack": 1,
        "share_weights": True,
    },
    "Symlet3Wavelet": {
        "stack_types": ["Symlet3Wavelet"] * 30,
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
# CSV Column Schema
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "experiment", "config_name", "stack_types", "period", "frequency",
    "forecast_length", "backcast_length", "n_stacks", "n_blocks_per_stack",
    "share_weights", "run", "seed",
    "smape", "mase", "owa", "n_params",
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


def compute_owa(smape, mase, period):
    """OWA (Overall Weighted Average) as defined in the M4 competition.

    OWA = 0.5 * (sMAPE / sMAPE_Naive2 + MASE / MASE_Naive2)

    A seasonally-adjusted naïve forecast obtains OWA = 1.0. Lower is better.
    """
    return 0.5 * (smape / NAIVE2_SMAPE[period] + mase / NAIVE2_MASE[period])


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
    stack_types,
    period,
    run_idx,
    m4,
    train_series_list,
    csv_path,
    n_blocks_per_stack=1,
    share_weights=True,
    active_g=False,
    sum_losses=False,
    activation="ReLU",
    max_epochs=100,
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

    n_stacks = len(stack_types)

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
    owa = compute_owa(smape, mase, period)

    print(f"         sMAPE={smape:.4f}  MASE={mase:.4f}  OWA={owa:.4f}  "
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

        for config_name, cfg in BLOCK_CONFIGS.items():
            for run_idx in range(N_RUNS):
                run_single_experiment(
                    experiment_name="block_benchmark",
                    config_name=config_name,
                    stack_types=cfg["stack_types"],
                    period=period,
                    run_idx=run_idx,
                    m4=m4,
                    train_series_list=train_series_list,
                    csv_path=csv_path,
                    n_blocks_per_stack=cfg["n_blocks_per_stack"],
                    share_weights=cfg["share_weights"],
                    active_g=False,
                    sum_losses=False,
                    activation="ReLU",
                    max_epochs=max_epochs,
                )


def run_ablation_studies(periods, max_epochs):
    """Part 2: Ablation studies on 30-stack Generic across M4 periods."""
    csv_path = os.path.join(RESULTS_DIR, "ablation_results.csv")
    init_csv(csv_path)

    # Ablation baseline: 30-stack Generic with shared weights (paper config)
    ablation_stack_types = ["Generic"] * TOTAL_STACKS

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
                    stack_types=ablation_stack_types,
                    period=period,
                    run_idx=run_idx,
                    m4=m4,
                    train_series_list=train_series_list,
                    csv_path=csv_path,
                    n_blocks_per_stack=1,
                    share_weights=True,
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
        description="M4 Benchmark — 1:1 N-BEATS paper comparison + novel extensions"
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
        "--max-epochs", type=int, default=100,
        help="Maximum training epochs per run (default: 100, early stopping may end sooner)",
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
