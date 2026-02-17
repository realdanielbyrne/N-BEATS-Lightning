"""
Milk Production Unbalanced active_g Convergence Study

Investigates the effect of applying active_g to only one output path (forecast
or backcast) rather than both.  Extends the balanced active_g study in
run_milk_convergence.py which already provides Baseline (active_g=False) and
Active-G (active_g=True) data in the same CSV.

Experimental Conditions (new — added to existing CSV):
  - Forecast-Only: active_g='forecast', activation=ReLU
  - Backcast-Only: active_g='backcast', activation=ReLU
  - 100 runs per config (200 new runs), each with a unique random seed

Existing conditions (reused from run_milk_convergence.py — not re-run):
  - Baseline:  active_g=False,  activation=ReLU   (100 runs)
  - Active-G:  active_g=True,   activation=ReLU   (100 runs)

All four conditions share:
  - 6 stacks of Generic blocks, 1 block/stack, shared weights
  - backcast_length=24, forecast_length=6, batch_size=64
  - SMAPELoss, Adam optimizer, max 500 epochs, early stopping patience=20
  - "Healthy" run threshold: val_loss < 20.0, no NaN

Results appended to experiments/results/milk_convergence/milk_convergence_results.csv

Usage:
    python experiments/run_milk_unbalanced_activeg.py
    python experiments/run_milk_unbalanced_activeg.py --max-epochs 500 --n-runs 100
    python experiments/run_milk_unbalanced_activeg.py --config Milk6_activeG_forecastOnly
    python experiments/run_milk_unbalanced_activeg.py --summary-only
"""

import argparse
import os
import sys

# Allow running from project root or experiments/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Reuse all infrastructure from the balanced experiment
from run_milk_convergence import (
    MILK_CONVERGENCE_CSV_COLUMNS,
    MAX_EPOCHS,
    N_RUNS,
    RESULTS_DIR,
    init_csv,
    append_result,
    result_exists,
    run_single_milk_convergence_experiment,
    print_summary_statistics,
)

import random
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Configuration — only the two NEW unbalanced conditions
# ---------------------------------------------------------------------------

UNBALANCED_CONFIGS = {
    "Milk6_activeG_forecastOnly": {"active_g": "forecast", "activation": "ReLU"},
    "Milk6_activeG_backcastOnly": {"active_g": "backcast", "activation": "ReLU"},
}


# ---------------------------------------------------------------------------
# Parallel Runner (mirrors run_milk_convergence.run_milk_convergence_study)
# ---------------------------------------------------------------------------

def run_unbalanced_activeg_study(
    max_epochs=MAX_EPOCHS,
    n_runs=N_RUNS,
    config_filter=None,
    max_workers=5,
    n_threads_override=None,
    wandb_enabled=False,
    wandb_project="nbeats-lightning",
):
    """Run the unbalanced active_g study, appending results to the shared CSV."""
    configs = UNBALANCED_CONFIGS
    if config_filter:
        if config_filter not in configs:
            print(f"Unknown config: {config_filter}")
            print(f"Available: {list(configs.keys())}")
            return
        configs = {config_filter: configs[config_filter]}

    n_cpus = os.cpu_count() or 1
    n_threads = (
        n_threads_override if n_threads_override is not None
        else max(1, n_cpus // max_workers)
    )

    csv_path = os.path.join(RESULTS_DIR, "milk_convergence_results.csv")
    init_csv(csv_path)

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
        print("All unbalanced active_g runs already complete.")
    else:
        print(
            f"\nUnbalanced active_g Study — {total_jobs} jobs, "
            f"{max_workers} workers, {n_threads} threads/worker"
        )
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

    # Print combined summary (all configs in the CSV)
    print_summary_statistics(csv_path)



# ---------------------------------------------------------------------------
# Main — CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Milk Production Unbalanced active_g Study — "
            "forecast-only vs backcast-only activation"
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
        choices=list(UNBALANCED_CONFIGS.keys()),
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

    run_unbalanced_activeg_study(
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