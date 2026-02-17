"""
Milk Production Convergence Study — active_g effect on small-scale N-BEATS

Evaluates whether `active_g` stabilizes convergence on the milk production dataset
(156 monthly observations, 1962-1974) using a 6-stack Generic architecture with
100 independent random-seed runs per configuration.

Modeled after Part 6 of run_experiments.py (Section 5.6 of paper.md), adapted for
the single-series TimeSeriesDataModule workflow from simple_example.ipynb.

Experimental Conditions:
  - Baseline: active_g=False, activation=ReLU
  - Active-G: active_g=True, activation=ReLU
  - 100 runs per config (200 total), each with a unique random seed
  - 6 stacks of Generic blocks, 1 block/stack, shared weights
  - backcast_length=24, forecast_length=6, batch_size=64
  - SMAPELoss, Adam optimizer, max 500 epochs, early stopping patience=20
  - "Healthy" run threshold: val_loss < 20.0, no NaN

Results saved to experiments/results/milk_convergence/

Usage:
    python experiments/run_milk_convergence.py
    python experiments/run_milk_convergence.py --max-epochs 500 --n-runs 100
    python experiments/run_milk_convergence.py --config Milk6_activeG --max-workers 8
"""

import argparse
import csv
import gc
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch import loggers as pl_loggers

# Allow running from project root or experiments/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lightningnbeats.models import NBeatsNet
from lightningnbeats.loaders import TimeSeriesDataModule

torch.set_float32_matmul_precision("medium")

# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------

# Model hyperparameters (from simple_example.ipynb)
FORECAST_LENGTH = 6
BACKCAST_LENGTH = 4 * FORECAST_LENGTH  # 24
BATCH_SIZE = 64
N_STACKS = 6
N_BLOCKS_PER_STACK = 1
SHARE_WEIGHTS = True
THETAS_DIM = 5
LOSS = "SMAPELoss"
LEARNING_RATE = 1e-3
LATENT_DIM = 4

# Training protocol
MAX_EPOCHS = 500
EARLY_STOPPING_PATIENCE = 20

# Convergence study parameters
N_RUNS = 100
HEALTHY_VAL_LOSS_THRESHOLD = 20.0  # sMAPE threshold for "healthy" run

# Experimental conditions
MILK_CONVERGENCE_CONFIGS = {
    "Milk6_baseline": {"active_g": False, "activation": "ReLU"},
    "Milk6_activeG":  {"active_g": True,  "activation": "ReLU"},
}

# Paths
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "milk_convergence")
MILK_CSV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "src", "lightningnbeats", "data", "milk.csv"
)

# CSV schema — matches Part 6 convergence study with milk-specific adaptations
MILK_CONVERGENCE_CSV_COLUMNS = [
    "experiment", "config_name",
    "forecast_length", "backcast_length",
    "n_stacks", "n_blocks_per_stack", "share_weights",
    "run", "seed", "active_g", "activation",
    "best_val_loss", "final_val_loss", "final_train_loss",
    "best_epoch", "epochs_trained", "stopping_reason",
    "loss_ratio", "diverged", "healthy",
    "n_params", "training_time_seconds",
    "val_loss_curve",
]


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def set_seed(seed):
    pl.seed_everything(seed, workers=True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_milk_data():
    """Load milk production dataset as a flat numpy array."""
    milk = pd.read_csv(MILK_CSV_PATH, index_col=0)
    return milk.values.flatten().astype(np.float32)


# ---------------------------------------------------------------------------
# CSV Helpers (incremental save with resumability)
# ---------------------------------------------------------------------------

def init_csv(path):
    """Create CSV with header if it doesn't exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MILK_CONVERGENCE_CSV_COLUMNS)
            writer.writeheader()


def append_result(path, row_dict):
    """Append a single result row to CSV."""
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MILK_CONVERGENCE_CSV_COLUMNS)
        writer.writerow(row_dict)


def result_exists(csv_path, config_name, run_idx):
    """Check if a result row already exists in the CSV."""
    if not os.path.exists(csv_path):
        return False
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("config_name") == config_name
                    and row.get("run") == str(run_idx)):
                return True
    return False


# ---------------------------------------------------------------------------
# Lightning Callbacks
# ---------------------------------------------------------------------------

class ConvergenceTracker(pl.Callback):
    """Records per-epoch losses for convergence analysis."""

    def __init__(self):
        super().__init__()
        self.val_losses = []
        self.train_losses = []

    def on_validation_epoch_end(self, trainer, pl_module):
        v = trainer.callback_metrics.get("val_loss")
        if v is not None:
            self.val_losses.append(float(v))

    def on_train_epoch_end(self, trainer, pl_module):
        v = trainer.callback_metrics.get("train_loss")
        if v is not None:
            self.train_losses.append(float(v))


class DivergenceDetector(pl.Callback):
    """Stop training when val_loss exceeds best by relative_threshold
    for consecutive_epochs in a row."""

    def __init__(self, relative_threshold=3.0, consecutive_epochs=3):
        super().__init__()
        self.relative_threshold = relative_threshold
        self.consecutive_epochs = consecutive_epochs
        self.best_val_loss = float("inf")
        self.bad_epoch_count = 0
        self.diverged = False

    def on_validation_epoch_end(self, trainer, pl_module):
        v = trainer.callback_metrics.get("val_loss")
        if v is None:
            return

        val_loss = float(v)

        # NaN / Inf → immediate stop
        if not math.isfinite(val_loss):
            self.diverged = True
            trainer.should_stop = True
            return

        # Track best
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.bad_epoch_count = 0
            return

        # Check if loss exceeds threshold
        if (self.best_val_loss > 0
                and val_loss > self.best_val_loss * self.relative_threshold):
            self.bad_epoch_count += 1
        else:
            self.bad_epoch_count = 0

        if self.bad_epoch_count >= self.consecutive_epochs:
            self.diverged = True
            trainer.should_stop = True


# ---------------------------------------------------------------------------
# Single-Run Worker (top-level for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------

def run_single_milk_convergence_experiment(
    config_name, active_g, activation, run_idx, seed,
    max_epochs, n_threads,
    wandb_enabled=False, wandb_project="nbeats-lightning",
):
    """Train one N-BEATS model on the milk dataset and return convergence metrics.

    Designed as a top-level function so ProcessPoolExecutor can pickle it.
    """
    torch.set_num_threads(n_threads)
    torch.set_float32_matmul_precision("medium")

    set_seed(seed)

    # Load data inside worker to avoid pickling large arrays
    milk_data = load_milk_data()

    stack_types = ["Generic"] * N_STACKS

    model = NBeatsNet(
        backcast_length=BACKCAST_LENGTH,
        forecast_length=FORECAST_LENGTH,
        stack_types=stack_types,
        n_blocks_per_stack=N_BLOCKS_PER_STACK,
        share_weights=SHARE_WEIGHTS,
        thetas_dim=THETAS_DIM,
        loss=LOSS,
        active_g=active_g,
        sum_losses=False,
        activation=activation,
        latent_dim=LATENT_DIM,
        learning_rate=LEARNING_RATE,
        no_val=False,
    )

    n_params = count_parameters(model)

    dm = TimeSeriesDataModule(
        data=milk_data,
        batch_size=BATCH_SIZE,
        backcast_length=BACKCAST_LENGTH,
        forecast_length=FORECAST_LENGTH,
        num_workers=0,
        pin_memory=False,
    )

    # Callbacks
    tracker = ConvergenceTracker()
    divergence_detector = DivergenceDetector(
        relative_threshold=3.0, consecutive_epochs=3
    )
    checkpoint_callback = ModelCheckpoint(
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

    log_dir = os.path.join(RESULTS_DIR, "lightning_logs")
    log_name = f"milk_convergence/{config_name}/run{run_idx}"

    exp_loggers = [pl_loggers.TensorBoardLogger(save_dir=log_dir, name=log_name)]
    if wandb_enabled:
        exp_loggers.append(pl_loggers.WandbLogger(
            project=wandb_project,
            group="milk_convergence",
            name=log_name,
            config={
                "dataset": "milk", "config_name": config_name,
                "n_stacks": N_STACKS, "backcast_length": BACKCAST_LENGTH,
                "forecast_length": FORECAST_LENGTH, "batch_size": BATCH_SIZE,
                "max_epochs": max_epochs, "seed": seed,
                "run_idx": run_idx, "active_g": active_g,
                "activation": activation, "n_params": n_params,
            },
            save_dir=log_dir,
            reinit=True,
        ))

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=max_epochs,
        callbacks=[
            checkpoint_callback, early_stop_callback,
            tracker, divergence_detector,
        ],
        logger=exp_loggers,
        enable_progress_bar=False,
        deterministic=False,
        log_every_n_steps=50,
    )

    t0 = time.time()
    trainer.fit(model, datamodule=dm)
    training_time = time.time() - t0

    epochs_trained = trainer.current_epoch

    # Extract convergence details
    best_val_loss = (
        float(checkpoint_callback.best_model_score)
        if checkpoint_callback.best_model_score is not None
        else float("nan")
    )
    final_val_loss = (
        tracker.val_losses[-1] if tracker.val_losses else float("nan")
    )
    final_train_loss = (
        tracker.train_losses[-1] if tracker.train_losses else float("nan")
    )
    best_epoch = (
        int(np.argmin(tracker.val_losses)) if tracker.val_losses else 0
    )

    # Stopping reason
    if divergence_detector.diverged:
        stopping_reason = "DIVERGED"
    elif (hasattr(early_stop_callback, "stopped_epoch")
          and early_stop_callback.stopped_epoch > 0):
        stopping_reason = "EARLY_STOPPED"
    else:
        stopping_reason = "MAX_EPOCHS"

    # Loss ratio and health flags
    loss_ratio = (
        final_val_loss / best_val_loss
        if best_val_loss > 0 and math.isfinite(best_val_loss)
        else float("nan")
    )
    diverged = (
        divergence_detector.diverged
        or not math.isfinite(final_val_loss)
    )
    healthy = (
        not diverged
        and math.isfinite(best_val_loss)
        and best_val_loss < HEALTHY_VAL_LOSS_THRESHOLD
    )

    result = {
        "experiment": "milk_convergence",
        "config_name": config_name,
        "forecast_length": FORECAST_LENGTH,
        "backcast_length": BACKCAST_LENGTH,
        "n_stacks": N_STACKS,
        "n_blocks_per_stack": N_BLOCKS_PER_STACK,
        "share_weights": SHARE_WEIGHTS,
        "run": run_idx,
        "seed": seed,
        "active_g": active_g,
        "activation": activation,
        "best_val_loss": f"{best_val_loss:.8f}",
        "final_val_loss": f"{final_val_loss:.8f}",
        "final_train_loss": f"{final_train_loss:.8f}",
        "best_epoch": best_epoch,
        "epochs_trained": epochs_trained,
        "stopping_reason": stopping_reason,
        "loss_ratio": (
            f"{loss_ratio:.6f}" if math.isfinite(loss_ratio) else "nan"
        ),
        "diverged": diverged,
        "healthy": healthy,
        "n_params": n_params,
        "training_time_seconds": f"{training_time:.2f}",
        "val_loss_curve": json.dumps(
            [f"{v:.8f}" for v in tracker.val_losses]
        ),
    }

    if wandb_enabled:
        import wandb
        wandb.finish(quiet=True)

    # Cleanup
    del model, trainer, dm
    gc.collect()

    return result


# ---------------------------------------------------------------------------
# Parallel Runner
# ---------------------------------------------------------------------------

def run_milk_convergence_study(
    max_epochs=MAX_EPOCHS,
    n_runs=N_RUNS,
    config_filter=None,
    max_workers=5,
    n_threads_override=None,
    wandb_enabled=False,
    wandb_project="nbeats-lightning",
):
    """Run the full milk convergence study with parallel execution.

    Args:
        max_epochs:    Maximum training epochs per run.
        n_runs:        Number of independent runs per configuration.
        config_filter: Optional config name to run only that config.
        max_workers:   Number of parallel worker processes.
    """
    configs = MILK_CONVERGENCE_CONFIGS
    if config_filter:
        if config_filter not in configs:
            print(f"Unknown config: {config_filter}")
            print(f"Available: {list(configs.keys())}")
            return
        configs = {config_filter: configs[config_filter]}

    # Compute threads per worker
    n_cpus = os.cpu_count() or 1
    n_threads = n_threads_override if n_threads_override is not None else max(1, n_cpus // max_workers)

    csv_path = os.path.join(RESULTS_DIR, "milk_convergence_results.csv")
    init_csv(csv_path)

    # Build flat job list with resumability check
    jobs = []
    for config_name, cfg in configs.items():
        for run_idx in range(n_runs):
            if result_exists(csv_path, config_name, run_idx):
                continue
            seed = random.randint(0, 2**31 - 1)
            jobs.append({
                "config_name": config_name,
                "active_g": cfg["active_g"],
                "activation": cfg["activation"],
                "run_idx": run_idx,
                "seed": seed,
                "max_epochs": max_epochs,
                "n_threads": n_threads,
                "wandb_enabled": wandb_enabled,
                "wandb_project": wandb_project,
            })

    total_jobs = len(jobs)
    if total_jobs == 0:
        print("All milk convergence study runs already complete.")
    else:
        print(f"\nMilk Convergence Study — {total_jobs} jobs, "
              f"{max_workers} workers, {n_threads} threads/worker")
        print(f"Results: {csv_path}\n")

        completed = 0
        diverged_count = 0

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_job = {}
            for job in jobs:
                future = executor.submit(
                    run_single_milk_convergence_experiment, **job
                )
                future_to_job[future] = job

            for future in as_completed(future_to_job):
                job = future_to_job[future]
                completed += 1

                try:
                    result = future.result()
                    append_result(csv_path, result)

                    diverged_tag = " [DIVERGED]" if result["diverged"] else ""
                    healthy_tag = " [HEALTHY]" if result["healthy"] else ""
                    if result["diverged"]:
                        diverged_count += 1

                    print(
                        f"  [{completed}/{total_jobs}] "
                        f"{result['config_name']} / run {result['run']} — "
                        f"best_val={result['best_val_loss']}  "
                        f"epochs={result['epochs_trained']}  "
                        f"time={result['training_time_seconds']}s"
                        f"{diverged_tag}{healthy_tag}"
                    )

                except Exception as e:
                    print(
                        f"  [{completed}/{total_jobs}] FAILED: "
                        f"{job['config_name']} / run {job['run_idx']} — {e}"
                    )

        print(f"\nTraining complete: {completed} runs, "
              f"{diverged_count} diverged")

    # Print summary statistics
    print_summary_statistics(csv_path)


# ---------------------------------------------------------------------------
# Summary Statistics
# ---------------------------------------------------------------------------

def print_summary_statistics(csv_path):
    """Load results CSV and print summary tables comparing configurations."""
    if not os.path.exists(csv_path):
        print("No results CSV found.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("Results CSV is empty.")
        return

    # Parse numeric columns
    for col in ["best_val_loss", "final_val_loss", "final_train_loss",
                "training_time_seconds"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["diverged"] = df["diverged"].astype(str).str.lower().isin(
        ["true", "1"]
    )
    df["healthy"] = df["healthy"].astype(str).str.lower().isin(
        ["true", "1"]
    )

    print("\n" + "=" * 75)
    print("MILK CONVERGENCE STUDY — SUMMARY STATISTICS")
    print("=" * 75)

    for config_name in sorted(df["config_name"].unique()):
        cfg_df = df[df["config_name"] == config_name]
        total_runs = len(cfg_df)
        healthy_df = cfg_df[cfg_df["healthy"]]
        n_healthy = len(healthy_df)
        n_diverged = cfg_df["diverged"].sum()
        convergence_rate = n_healthy / total_runs if total_runs > 0 else 0.0
        divergence_rate = n_diverged / total_runs if total_runs > 0 else 0.0

        print(f"\n--- {config_name} ({total_runs} runs) ---")
        print(f"  Convergence rate:  {n_healthy}/{total_runs} "
              f"({convergence_rate:.1%})")
        print(f"  Divergence rate:   {int(n_diverged)}/{total_runs} "
              f"({divergence_rate:.1%})")

        if n_healthy > 0:
            val_losses = healthy_df["best_val_loss"]
            epochs = healthy_df["epochs_trained"].astype(float)
            times = healthy_df["training_time_seconds"]

            mean_val = val_losses.mean()
            std_val = val_losses.std()
            cv_val = (std_val / mean_val * 100) if mean_val > 0 else float("nan")
            median_val = val_losses.median()

            mean_epochs = epochs.mean()
            std_epochs = epochs.std()
            median_epochs = epochs.median()

            mean_time = times.mean()
            std_time = times.std()

            print(f"\n  Best Val Loss (healthy runs):")
            print(f"    Mean ± Std:   {mean_val:.4f} ± {std_val:.4f}")
            print(f"    CV%:          {cv_val:.2f}%")
            print(f"    Median:       {median_val:.4f}")
            print(f"    Min:          {val_losses.min():.4f}")
            print(f"    Max:          {val_losses.max():.4f}")

            print(f"\n  Epochs to Convergence (healthy runs):")
            print(f"    Mean ± Std:   {mean_epochs:.1f} ± {std_epochs:.1f}")
            print(f"    Median:       {median_epochs:.1f}")
            print(f"    Min:          {epochs.min():.0f}")
            print(f"    Max:          {epochs.max():.0f}")

            print(f"\n  Training Time (healthy runs):")
            print(f"    Mean ± Std:   {mean_time:.1f}s ± {std_time:.1f}s")

            # Stopping reason distribution
            reasons = healthy_df["stopping_reason"].value_counts()
            print(f"\n  Stopping Reasons (healthy):")
            for reason, count in reasons.items():
                print(f"    {reason}: {count}")

    # Comparison table
    print("\n" + "=" * 75)
    print("COMPARISON TABLE")
    print("=" * 75)
    header = (f"{'Config':<20} {'Conv%':>6} {'ValLoss':>12} "
              f"{'CV%':>7} {'Epochs':>14} {'Time(s)':>10}")
    print(header)
    print("-" * 75)

    for config_name in sorted(df["config_name"].unique()):
        cfg_df = df[df["config_name"] == config_name]
        total_runs = len(cfg_df)
        healthy_df = cfg_df[cfg_df["healthy"]]
        n_healthy = len(healthy_df)
        conv_pct = f"{n_healthy}/{total_runs}"

        if n_healthy > 0:
            val_mean = healthy_df["best_val_loss"].mean()
            val_std = healthy_df["best_val_loss"].std()
            cv_pct = (val_std / val_mean * 100) if val_mean > 0 else float("nan")
            ep_mean = healthy_df["epochs_trained"].astype(float).mean()
            ep_std = healthy_df["epochs_trained"].astype(float).std()
            t_mean = healthy_df["training_time_seconds"].mean()

            val_str = f"{val_mean:.2f}±{val_std:.2f}"
            cv_str = f"{cv_pct:.2f}%"
            ep_str = f"{ep_mean:.1f}±{ep_std:.1f}"
            t_str = f"{t_mean:.1f}"
        else:
            val_str = "N/A"
            cv_str = "N/A"
            ep_str = "N/A"
            t_str = "N/A"

        print(f"{config_name:<20} {conv_pct:>6} {val_str:>12} "
              f"{cv_str:>7} {ep_str:>14} {t_str:>10}")

    print()
    print(f"Results saved to: {csv_path}")
    print("=" * 75)


# ---------------------------------------------------------------------------
# Main — CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Milk Production Convergence Study — "
            "active_g effect on small-scale N-BEATS"
        ),
    )
    parser.add_argument(
        "--max-epochs", type=int, default=MAX_EPOCHS,
        help=f"Maximum training epochs per run (default: {MAX_EPOCHS})",
    )
    parser.add_argument(
        "--n-runs", type=int, default=N_RUNS,
        help=f"Number of independent runs per config (default: {N_RUNS})",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        choices=list(MILK_CONVERGENCE_CONFIGS.keys()),
        help="Run only the specified configuration (default: all)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=5,
        help="Number of parallel worker processes (default: 5)",
    )
    parser.add_argument(
        "--n-threads", type=int, default=None,
        help="PyTorch threads per worker (default: auto = n_cpus // max_workers).",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Skip training, only print summary statistics from existing CSV",
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="Enable Weights & Biases logging alongside TensorBoard",
    )
    parser.add_argument(
        "--wandb-project", default="nbeats-lightning",
        help="W&B project name (default: nbeats-lightning)",
    )

    args = parser.parse_args()

    if args.summary_only:
        csv_path = os.path.join(RESULTS_DIR, "milk_convergence_results.csv")
        print_summary_statistics(csv_path)
        return

    run_milk_convergence_study(
        max_epochs=args.max_epochs,
        n_runs=args.n_runs,
        config_filter=args.config,
        max_workers=args.max_workers,
        n_threads_override=args.n_threads,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
    )


if __name__ == "__main__":
    main()

