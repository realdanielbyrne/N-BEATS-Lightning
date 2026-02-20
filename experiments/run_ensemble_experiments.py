"""
Architecture-Diversity Ensemble Experiment for N-BEATS Lightning

Tests the hypothesis that mixing architecturally diverse stack types
(paper baselines, AE-backbone variants, wavelet blocks) provides additive
ensemble benefit beyond horizon diversity alone.

2D factorial design per ensemble:
  - horizon_only:  single best config, all 6 multipliers (2H-7H), N runs
  - arch_only:     all member configs, fixed 5H multiplier, N runs
  - combined:      all member configs x all 6 multipliers x N runs

Aggregation: element-wise median (primary) and mean (secondary).

Reuses existing 5H predictions from experiments/results/m4/predictions/
whenever a matching (pass, config, period, run) tuple is found.

Usage:
    python experiments/run_ensemble_experiments.py --dataset m4 --periods Yearly
    python experiments/run_ensemble_experiments.py --dataset m4 --periods Yearly --ensemble-name Paper-Horizon
    python experiments/run_ensemble_experiments.py --dataset m4 --periods Yearly --n-runs 1
    python experiments/run_ensemble_experiments.py --dataset m4 --periods Yearly Quarterly --n-gpus 2
"""

import argparse
import csv
import fcntl
import gc
import math
import multiprocessing as mp
import os
import queue
import signal
import sys
import time

import numpy as np
import torch

# Allow running from project root or experiments/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from run_unified_benchmark import (
    run_single_experiment,
    load_dataset,
    get_batch_size,
    compute_smape,
    compute_m4_mase,
    compute_mae,
    compute_mse,
    init_csv,
    append_result,
    result_exists,
    set_seed,
    resolve_accelerator,
    resolve_n_gpus,
    UNIFIED_CONFIGS,
    BATCH_SIZES,
    FORECAST_MULTIPLIERS,
    DATASET_PERIODS,
    BASE_SEED,
    N_RUNS_DEFAULT,
    RESULTS_DIR,
    CSV_COLUMNS,
    LEARNING_RATE,
    LOSS,
    EARLY_STOPPING_PATIENCE,
    DEFAULT_BATCH_SIZE,
)

torch.set_float32_matmul_precision("medium")


# ---------------------------------------------------------------------------
# Signal Handling — Graceful Shutdown
# ---------------------------------------------------------------------------

_shutdown_requested = False
_shutdown_event = None


def _signal_handler(signum, frame):
    global _shutdown_requested
    if _shutdown_requested:
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
# Ensemble Constants
# ---------------------------------------------------------------------------

ENSEMBLE_MULTIPLIERS = [2, 3, 4, 5, 6, 7]
N_RUNS_ENSEMBLE = 3

ENSEMBLE_PROPOSALS = {
    "Paper-Horizon": {
        "members": [("NBEATS-I+G", "baseline")],
        "description": "Paper replication: single arch, multi-horizon",
    },
    "Paper-Diverse": {
        "members": [
            ("NBEATS-G", "baseline"),
            ("NBEATS-I", "baseline"),
            ("NBEATS-I+G", "baseline"),
        ],
        "description": "Paper baselines with architecture diversity",
    },
    "Basis-Diverse": {
        "members": [
            ("NBEATS-G", "baseline"),
            ("NBEATS-I", "baseline"),
            ("Trend+Coif2WaveletV3", "baseline"),
            ("GenericAE", "baseline"),
        ],
        "description": "Maximum basis-function diversity",
    },
    "Efficient-Diverse": {
        "members": [
            ("NBEATS-I", "baseline"),
            ("GenericAE", "baseline"),
            ("Trend+Coif2WaveletV3", "baseline"),
        ],
        "description": "Parameter-efficient diverse ensemble",
    },
    "Full-Spectrum": {
        "members": [
            ("NBEATS-G", "baseline"),
            ("NBEATS-I", "baseline"),
            ("Trend+Coif2WaveletV3", "baseline"),
            ("Coif2WaveletV3", "activeG_fcast"),
            ("GenericAE", "baseline"),
        ],
        "description": "Cross-pass + architecture + backbone diversity",
    },
}

# Pass name → active_g value mapping
PASS_ACTIVE_G = {
    "baseline": False,
    "activeG_fcast": "forecast",
}

# CSV schemas
ENSEMBLE_INDIVIDUAL_COLUMNS = CSV_COLUMNS + ["forecast_multiplier"]

ENSEMBLE_SUMMARY_COLUMNS = [
    "ensemble_name", "ensemble_type", "period", "aggregation",
    "member_configs", "n_members", "n_models",
    "ensemble_smape", "ensemble_mase", "ensemble_mae", "ensemble_mse",
    "ensemble_owa",
    "best_single_smape", "best_single_mase", "best_single_owa",
    "improvement_vs_single_pct", "improvement_vs_paper_horizon_pct",
]


# ---------------------------------------------------------------------------
# Prediction Reuse
# ---------------------------------------------------------------------------

def find_existing_5h_prediction(config_name, pass_name, period, run_idx,
                                predictions_dir):
    """Check for an existing 5H prediction NPZ in the main predictions dir.

    The unified benchmark saves predictions as:
        {experiment_name}_{config_name}_{period}_run{run_idx}.npz
    where experiment_name matches pass_name (e.g., "baseline", "activeG_fcast").
    """
    npz_name = f"{pass_name}_{config_name}_{period}_run{run_idx}.npz"
    path = os.path.join(predictions_dir, npz_name)
    if os.path.exists(path):
        return path
    return None


# ---------------------------------------------------------------------------
# Training Plan
# ---------------------------------------------------------------------------

def build_training_plan(ensemble_proposals, periods, n_runs, multipliers,
                        main_predictions_dir):
    """Build the full set of jobs needed across all ensembles.

    Returns
    -------
    jobs_to_train : list[dict]
        Unique (config, pass, period, multiplier, run) tuples that need training.
    jobs_to_reuse : dict
        Mapping from (config, pass, period, multiplier, run) → existing NPZ path.
    """
    # Collect all unique jobs across all ensembles
    all_jobs = set()
    for ens_name, ens in ensemble_proposals.items():
        for config_name, pass_name in ens["members"]:
            for period in periods:
                for mult in multipliers:
                    for run_idx in range(n_runs):
                        all_jobs.add((config_name, pass_name, period, mult, run_idx))

    jobs_to_train = []
    jobs_to_reuse = {}

    for config_name, pass_name, period, mult, run_idx in sorted(all_jobs):
        key = (config_name, pass_name, period, mult, run_idx)
        if mult == 5:
            existing = find_existing_5h_prediction(
                config_name, pass_name, period, run_idx, main_predictions_dir)
            if existing is not None:
                jobs_to_reuse[key] = existing
                continue
        jobs_to_train.append({
            "config_name": config_name,
            "pass_name": pass_name,
            "period": period,
            "multiplier": mult,
            "run_idx": run_idx,
        })

    return jobs_to_train, jobs_to_reuse


def get_ensemble_npz_name(pass_name, config_name, period, mult, run_idx):
    """Return the filename for an ensemble prediction NPZ."""
    return f"{pass_name}_{config_name}_{period}_m{mult}_run{run_idx}.npz"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train_single_ensemble_job(job, dataset, train_series_list, csv_path,
                               ensemble_preds_dir, args):
    """Train a single model and save predictions to ensemble predictions dir."""
    config_name = job["config_name"]
    pass_name = job["pass_name"]
    period = job["period"]
    mult = job["multiplier"]
    run_idx = job["run_idx"]

    cfg = UNIFIED_CONFIGS[config_name]
    active_g = PASS_ACTIVE_G[pass_name]
    batch_size = get_batch_size("m4", period, args.batch_size)

    npz_name = get_ensemble_npz_name(pass_name, config_name, period, mult, run_idx)
    npz_path = os.path.join(ensemble_preds_dir, npz_name)

    # Skip if prediction already exists in ensemble dir
    if os.path.exists(npz_path):
        print(f"  [SKIP] {npz_name} — already exists in ensemble predictions")
        return

    # Use a distinct experiment name that includes the multiplier
    experiment_name = f"ens_{pass_name}_m{mult}"

    run_single_experiment(
        experiment_name=experiment_name,
        config_name=config_name,
        category=cfg["category"],
        stack_types=cfg["stack_types"],
        period=period,
        run_idx=run_idx,
        dataset=dataset,
        train_series_list=train_series_list,
        csv_path=csv_path,
        n_blocks_per_stack=cfg["n_blocks_per_stack"],
        share_weights=cfg["share_weights"],
        active_g=active_g,
        sum_losses=False,
        activation="ReLU",
        max_epochs=args.max_epochs,
        patience=EARLY_STOPPING_PATIENCE,
        batch_size=batch_size,
        accelerator_override=args.accelerator,
        forecast_multiplier=mult,
        num_workers=args.num_workers,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        save_predictions=True,
        predictions_dir=ensemble_preds_dir,
    )

    # The unified benchmark saves as {experiment_name}_{config}_{period}_run{run}.npz
    # We need it as {pass}_{config}_{period}_m{mult}_run{run}.npz
    # Rename the file to our naming convention
    unified_name = f"{experiment_name}_{config_name}_{period}_run{run_idx}.npz"
    unified_path = os.path.join(ensemble_preds_dir, unified_name)
    if os.path.exists(unified_path) and not os.path.exists(npz_path):
        os.rename(unified_path, npz_path)


def copy_reused_predictions(jobs_to_reuse, ensemble_preds_dir):
    """Copy existing 5H predictions into the ensemble predictions directory."""
    os.makedirs(ensemble_preds_dir, exist_ok=True)
    copied = 0
    for (config_name, pass_name, period, mult, run_idx), src_path in jobs_to_reuse.items():
        npz_name = get_ensemble_npz_name(pass_name, config_name, period, mult, run_idx)
        dst_path = os.path.join(ensemble_preds_dir, npz_name)
        if not os.path.exists(dst_path):
            # Copy the file (symlinks could break if source is deleted)
            data = np.load(src_path)
            np.savez(dst_path, preds=data["preds"], targets=data["targets"])
            data.close()
            copied += 1
    if copied:
        print(f"  Copied {copied} existing 5H predictions to ensemble dir")


def train_ensemble_models_sequential(jobs_to_train, csv_path, ensemble_preds_dir,
                                     args):
    """Train all ensemble models sequentially."""
    global _shutdown_requested

    dataset_cache = {}
    series_cache = {}

    total = len(jobs_to_train)
    for i, job in enumerate(jobs_to_train):
        if _shutdown_requested:
            print("[SHUTDOWN] Stopping training loop.")
            break

        period = job["period"]
        if period not in dataset_cache:
            dataset_cache[period] = load_dataset("m4", period)
            series_cache[period] = dataset_cache[period].get_training_series()

        print(f"\n  [{i+1}/{total}] {job['pass_name']} / {job['config_name']} / "
              f"{period} / m{job['multiplier']} / run{job['run_idx']}")

        _train_single_ensemble_job(
            job, dataset_cache[period], series_cache[period],
            csv_path, ensemble_preds_dir, args)


def _ensemble_gpu_worker(gpu_id, job_queue, shutdown_event, worker_args):
    """GPU worker process for parallel ensemble training."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    torch.set_float32_matmul_precision("medium")

    prefix = f"[GPU {gpu_id}]"
    print(f"{prefix} Ensemble worker started (CUDA_VISIBLE_DEVICES={gpu_id}).")

    dataset_cache = {}
    series_cache = {}

    while not shutdown_event.is_set():
        try:
            job = job_queue.get_nowait()
        except queue.Empty:
            break

        period = job["period"]
        if period not in dataset_cache:
            dataset_cache[period] = load_dataset("m4", period)
            series_cache[period] = dataset_cache[period].get_training_series()

        config_name = job["config_name"]
        pass_name = job["pass_name"]
        mult = job["multiplier"]
        run_idx = job["run_idx"]
        cfg = UNIFIED_CONFIGS[config_name]
        active_g = PASS_ACTIVE_G[pass_name]
        batch_size = get_batch_size("m4", period, worker_args["batch_size"])

        ensemble_preds_dir = worker_args["ensemble_preds_dir"]
        npz_name = get_ensemble_npz_name(pass_name, config_name, period, mult, run_idx)
        npz_path = os.path.join(ensemble_preds_dir, npz_name)

        if os.path.exists(npz_path):
            print(f"  {prefix} [SKIP] {npz_name} — already exists")
            continue

        experiment_name = f"ens_{pass_name}_m{mult}"

        run_single_experiment(
            experiment_name=experiment_name,
            config_name=config_name,
            category=cfg["category"],
            stack_types=cfg["stack_types"],
            period=period,
            run_idx=run_idx,
            dataset=dataset_cache[period],
            train_series_list=series_cache[period],
            csv_path=worker_args["csv_path"],
            n_blocks_per_stack=cfg["n_blocks_per_stack"],
            share_weights=cfg["share_weights"],
            active_g=active_g,
            sum_losses=False,
            activation="ReLU",
            max_epochs=worker_args["max_epochs"],
            patience=EARLY_STOPPING_PATIENCE,
            batch_size=batch_size,
            accelerator_override="cuda",
            forecast_multiplier=mult,
            num_workers=worker_args["num_workers"],
            wandb_enabled=worker_args["wandb_enabled"],
            wandb_project=worker_args["wandb_project"],
            save_predictions=True,
            predictions_dir=ensemble_preds_dir,
            gpu_id=gpu_id,
        )

        # Rename to ensemble naming convention
        unified_name = f"{experiment_name}_{config_name}_{period}_run{run_idx}.npz"
        unified_path = os.path.join(ensemble_preds_dir, unified_name)
        if os.path.exists(unified_path) and not os.path.exists(npz_path):
            os.rename(unified_path, npz_path)

    print(f"{prefix} Ensemble worker finished.")


def train_ensemble_models_parallel(jobs_to_train, csv_path, ensemble_preds_dir,
                                   args, n_gpus):
    """Train ensemble models in parallel across GPUs."""
    global _shutdown_event

    # Filter already-completed jobs
    filtered = []
    for job in jobs_to_train:
        npz_name = get_ensemble_npz_name(
            job["pass_name"], job["config_name"], job["period"],
            job["multiplier"], job["run_idx"])
        npz_path = os.path.join(ensemble_preds_dir, npz_name)
        if not os.path.exists(npz_path):
            filtered.append(job)

    if not filtered:
        print("  All ensemble models already trained.")
        return

    print(f"  Training {len(filtered)} models across {n_gpus} GPUs...")

    ctx = mp.get_context("spawn")
    job_queue = ctx.Queue()
    for job in filtered:
        job_queue.put(job)

    shutdown_event = ctx.Event()
    _shutdown_event = shutdown_event

    worker_args = {
        "csv_path": csv_path,
        "ensemble_preds_dir": ensemble_preds_dir,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "wandb_enabled": args.wandb,
        "wandb_project": args.wandb_project,
    }

    workers = []
    for gpu_id in range(n_gpus):
        p = ctx.Process(
            target=_ensemble_gpu_worker,
            args=(gpu_id, job_queue, shutdown_event, worker_args))
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    _shutdown_event = None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def load_ensemble_predictions(members, condition, period, multipliers, n_runs,
                              ensemble_preds_dir):
    """Load prediction arrays for an ensemble condition.

    Parameters
    ----------
    members : list of (config_name, pass_name) tuples
    condition : str — "horizon_only", "arch_only", or "combined"
    period : str
    multipliers : list[int]
    n_runs : int
    ensemble_preds_dir : str

    Returns
    -------
    preds_list : list[np.ndarray]  — each shape (n_series, forecast_length)
    targets : np.ndarray — shape (n_series, forecast_length) from first valid file
    n_loaded : int
    n_diverged : int
    """
    if condition == "horizon_only":
        # Single best config (first member), all multipliers
        selected_members = [members[0]]
        selected_mults = multipliers
    elif condition == "arch_only":
        # All members, fixed 5H
        selected_members = members
        selected_mults = [5]
    elif condition == "combined":
        # All members x all multipliers
        selected_members = members
        selected_mults = multipliers
    else:
        raise ValueError(f"Unknown condition: {condition}")

    preds_list = []
    targets = None
    n_diverged = 0

    for config_name, pass_name in selected_members:
        for mult in selected_mults:
            for run_idx in range(n_runs):
                npz_name = get_ensemble_npz_name(
                    pass_name, config_name, period, mult, run_idx)
                npz_path = os.path.join(ensemble_preds_dir, npz_name)

                if not os.path.exists(npz_path):
                    continue

                data = np.load(npz_path)
                p = data["preds"]
                t = data["targets"]
                data.close()

                # Filter diverged (non-finite predictions)
                if not np.all(np.isfinite(p)):
                    n_diverged += 1
                    continue

                preds_list.append(p)
                if targets is None:
                    targets = t

    return preds_list, targets, len(preds_list), n_diverged


def aggregate_predictions(preds_list, method="median"):
    """Aggregate a list of prediction arrays via element-wise median or mean."""
    if not preds_list:
        return None
    stacked = np.stack(preds_list, axis=0)  # (n_models, n_series, forecast_length)
    if method == "median":
        return np.median(stacked, axis=0)
    elif method == "mean":
        return np.mean(stacked, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def compute_ensemble_metrics(preds, targets, dataset, train_series_list):
    """Compute all metrics for ensemble predictions."""
    smape = compute_smape(preds, targets)
    mase = compute_m4_mase(preds, targets, train_series_list, dataset.frequency)
    mae = compute_mae(preds, targets)
    mse = compute_mse(preds, targets)
    owa = dataset.compute_owa(smape, mase)
    return {
        "smape": smape,
        "mase": mase,
        "mae": mae,
        "mse": mse,
        "owa": owa,
    }


def find_best_single_model_metrics(members, period, n_runs, ensemble_preds_dir,
                                   dataset, train_series_list):
    """Find the best single-model OWA among ensemble members (5H only)."""
    best_owa = float("inf")
    best_smape = float("nan")
    best_mase = float("nan")

    for config_name, pass_name in members:
        for run_idx in range(n_runs):
            npz_name = get_ensemble_npz_name(pass_name, config_name, period, 5, run_idx)
            npz_path = os.path.join(ensemble_preds_dir, npz_name)
            if not os.path.exists(npz_path):
                continue

            data = np.load(npz_path)
            p, t = data["preds"], data["targets"]
            data.close()

            if not np.all(np.isfinite(p)):
                continue

            smape = compute_smape(p, t)
            mase = compute_m4_mase(p, t, train_series_list, dataset.frequency)
            owa = dataset.compute_owa(smape, mase)

            if math.isfinite(owa) and owa < best_owa:
                best_owa = owa
                best_smape = smape
                best_mase = mase

    return best_smape, best_mase, best_owa


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def init_summary_csv(path):
    """Create ensemble summary CSV with header if it doesn't exist."""
    init_csv(path, ENSEMBLE_SUMMARY_COLUMNS)


def append_summary(path, row_dict):
    """Append a summary row (process-safe)."""
    append_result(path, row_dict, ENSEMBLE_SUMMARY_COLUMNS)


def summary_exists(path, ensemble_name, ensemble_type, period, aggregation):
    """Check if a summary row already exists."""
    if not os.path.exists(path):
        return False
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("ensemble_name") == ensemble_name
                    and row.get("ensemble_type") == ensemble_type
                    and row.get("period") == period
                    and row.get("aggregation") == aggregation):
                return True
    return False


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def run_ensemble_experiment(args):
    """Main entry point: build plan, train models, aggregate, save results."""
    global _shutdown_requested

    dataset_name = args.dataset

    # Resolve periods
    all_periods = DATASET_PERIODS[dataset_name]
    if args.periods:
        periods = [p for p in args.periods if p in all_periods]
        if not periods:
            print(f"[ERROR] No valid periods for dataset '{dataset_name}'. "
                  f"Available: {list(all_periods.keys())}")
            return
    else:
        periods = list(all_periods.keys())

    # Resolve ensemble proposals
    if args.ensemble_name:
        if args.ensemble_name not in ENSEMBLE_PROPOSALS:
            print(f"[ERROR] Unknown ensemble: {args.ensemble_name}. "
                  f"Available: {list(ENSEMBLE_PROPOSALS.keys())}")
            return
        proposals = {args.ensemble_name: ENSEMBLE_PROPOSALS[args.ensemble_name]}
    else:
        proposals = ENSEMBLE_PROPOSALS

    n_runs = args.n_runs
    multipliers = ENSEMBLE_MULTIPLIERS

    # Directories
    dataset_results_dir = os.path.join(RESULTS_DIR, dataset_name)
    main_predictions_dir = os.path.join(dataset_results_dir, "predictions")
    ensemble_preds_dir = os.path.join(dataset_results_dir, "ensemble_arch_predictions")
    os.makedirs(ensemble_preds_dir, exist_ok=True)

    csv_path = os.path.join(dataset_results_dir, "ensemble_arch_individual_results.csv")
    summary_path = os.path.join(dataset_results_dir, "ensemble_arch_summary_results.csv")

    init_csv(csv_path, ENSEMBLE_INDIVIDUAL_COLUMNS)
    init_summary_csv(summary_path)

    # Build training plan
    print("=" * 72)
    print("ARCHITECTURE-DIVERSITY ENSEMBLE EXPERIMENT")
    print("=" * 72)
    print(f"  Dataset:    {dataset_name}")
    print(f"  Periods:    {periods}")
    print(f"  Ensembles:  {list(proposals.keys())}")
    print(f"  Multipliers: {multipliers}")
    print(f"  Runs/model: {n_runs}")
    print()

    jobs_to_train, jobs_to_reuse = build_training_plan(
        proposals, periods, n_runs, multipliers, main_predictions_dir)

    print(f"  Training plan:")
    print(f"    Jobs to train:  {len(jobs_to_train)}")
    print(f"    Jobs to reuse:  {len(jobs_to_reuse)} (existing 5H predictions)")
    print(f"    Total models:   {len(jobs_to_train) + len(jobs_to_reuse)}")
    print()

    # Phase 1: Copy reused predictions
    print("-" * 72)
    print("PHASE 1: Copying existing 5H predictions")
    print("-" * 72)
    copy_reused_predictions(jobs_to_reuse, ensemble_preds_dir)

    # Phase 2: Train new models
    if jobs_to_train and not _shutdown_requested:
        print()
        print("-" * 72)
        print("PHASE 2: Training new ensemble models")
        print("-" * 72)

        n_gpus = resolve_n_gpus(args) if hasattr(args, 'n_gpus') else 0
        if n_gpus >= 2:
            train_ensemble_models_parallel(
                jobs_to_train, csv_path, ensemble_preds_dir, args, n_gpus)
        else:
            train_ensemble_models_sequential(
                jobs_to_train, csv_path, ensemble_preds_dir, args)

    # Phase 3: Aggregate and evaluate
    if _shutdown_requested:
        print("\n[SHUTDOWN] Skipping aggregation phase.")
        return

    print()
    print("-" * 72)
    print("PHASE 3: Ensemble aggregation and evaluation")
    print("-" * 72)

    # We'll collect results for the final summary table
    all_results = []

    for period in periods:
        if _shutdown_requested:
            break

        dataset = load_dataset(dataset_name, period)
        train_series_list = dataset.get_training_series()

        print(f"\n  Period: {period}")
        print(f"  {'─' * 60}")

        for ens_name, ens in proposals.items():
            if _shutdown_requested:
                break

            members = ens["members"]
            member_str = ", ".join(f"{c}({p})" for c, p in members)

            for condition in ["horizon_only", "arch_only", "combined"]:
                for agg_method in ["median", "mean"]:
                    # Check if already computed
                    ens_type = f"{condition}"
                    if summary_exists(summary_path, ens_name, ens_type, period, agg_method):
                        print(f"    [SKIP] {ens_name} / {condition} / {agg_method} — already computed")
                        continue

                    preds_list, targets, n_loaded, n_diverged = load_ensemble_predictions(
                        members, condition, period, multipliers, n_runs,
                        ensemble_preds_dir)

                    if not preds_list or targets is None:
                        print(f"    [WARN] {ens_name} / {condition} / {agg_method}: "
                              f"no predictions found (loaded={n_loaded}, diverged={n_diverged})")
                        continue

                    agg_preds = aggregate_predictions(preds_list, method=agg_method)
                    metrics = compute_ensemble_metrics(
                        agg_preds, targets, dataset, train_series_list)

                    # Best single model for comparison
                    best_s_smape, best_s_mase, best_s_owa = find_best_single_model_metrics(
                        members, period, n_runs, ensemble_preds_dir,
                        dataset, train_series_list)

                    # Improvement calculations
                    if math.isfinite(best_s_owa) and best_s_owa > 0:
                        improv_vs_single = (best_s_owa - metrics["owa"]) / best_s_owa * 100
                    else:
                        improv_vs_single = float("nan")

                    # Save summary row
                    row = {
                        "ensemble_name": ens_name,
                        "ensemble_type": ens_type,
                        "period": period,
                        "aggregation": agg_method,
                        "member_configs": member_str,
                        "n_members": len(members),
                        "n_models": n_loaded,
                        "ensemble_smape": f"{metrics['smape']:.6f}",
                        "ensemble_mase": f"{metrics['mase']:.6f}",
                        "ensemble_mae": f"{metrics['mae']:.6f}",
                        "ensemble_mse": f"{metrics['mse']:.6f}",
                        "ensemble_owa": f"{metrics['owa']:.6f}",
                        "best_single_smape": f"{best_s_smape:.6f}" if math.isfinite(best_s_smape) else "nan",
                        "best_single_mase": f"{best_s_mase:.6f}" if math.isfinite(best_s_mase) else "nan",
                        "best_single_owa": f"{best_s_owa:.6f}" if math.isfinite(best_s_owa) else "nan",
                        "improvement_vs_single_pct": f"{improv_vs_single:.4f}" if math.isfinite(improv_vs_single) else "nan",
                        "improvement_vs_paper_horizon_pct": "",  # Filled in post-hoc
                    }
                    append_summary(summary_path, row)

                    # Collect for display
                    all_results.append({
                        "ensemble": ens_name,
                        "condition": condition,
                        "period": period,
                        "agg": agg_method,
                        "owa": metrics["owa"],
                        "smape": metrics["smape"],
                        "mase": metrics["mase"],
                        "n_models": n_loaded,
                        "best_single_owa": best_s_owa,
                        "improv": improv_vs_single,
                    })

                    if agg_method == "median":
                        print(f"    {ens_name:20s} {condition:14s} "
                              f"median: OWA={metrics['owa']:.4f} "
                              f"sMAPE={metrics['smape']:.4f} MASE={metrics['mase']:.4f} "
                              f"({n_loaded} models, {n_diverged} diverged)")

    # Final summary table
    if all_results:
        _print_summary_table(all_results, periods)
        _backfill_paper_horizon_improvement(summary_path, all_results)


def _backfill_paper_horizon_improvement(summary_path, all_results):
    """Compute improvement_vs_paper_horizon_pct using Paper-Horizon median as baseline."""
    # Find Paper-Horizon combined median OWA per period
    paper_horizon_owa = {}
    for r in all_results:
        if r["ensemble"] == "Paper-Horizon" and r["condition"] == "combined" and r["agg"] == "median":
            paper_horizon_owa[r["period"]] = r["owa"]

    if not paper_horizon_owa:
        return

    # Re-read the summary CSV, update the improvement column, rewrite
    if not os.path.exists(summary_path):
        return

    rows = []
    with open(summary_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            period = row.get("period", "")
            owa_str = row.get("ensemble_owa", "")
            baseline = paper_horizon_owa.get(period)
            if baseline and owa_str and owa_str != "nan":
                try:
                    owa = float(owa_str)
                    improv = (baseline - owa) / baseline * 100
                    row["improvement_vs_paper_horizon_pct"] = f"{improv:.4f}"
                except (ValueError, ZeroDivisionError):
                    pass
            rows.append(row)

    lock_path = summary_path + ".lock"
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            with open(summary_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=ENSEMBLE_SUMMARY_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)


def _print_summary_table(all_results, periods):
    """Print a compact summary table to stdout."""
    print()
    print("=" * 90)
    print("ENSEMBLE RESULTS SUMMARY (median aggregation)")
    print("=" * 90)

    for period in periods:
        period_results = [r for r in all_results
                          if r["period"] == period and r["agg"] == "median"]
        if not period_results:
            continue

        print(f"\n  {period}:")
        print(f"  {'Ensemble':<20s} {'Condition':<14s} {'Models':>6s} "
              f"{'OWA':>8s} {'sMAPE':>8s} {'MASE':>8s} "
              f"{'Best1':>8s} {'Improv%':>8s}")
        print(f"  {'─' * 84}")

        for r in sorted(period_results, key=lambda x: x["owa"]):
            best_str = f"{r['best_single_owa']:.4f}" if math.isfinite(r['best_single_owa']) else "N/A"
            improv_str = f"{r['improv']:.2f}%" if math.isfinite(r['improv']) else "N/A"
            print(f"  {r['ensemble']:<20s} {r['condition']:<14s} {r['n_models']:>6d} "
                  f"{r['owa']:>8.4f} {r['smape']:>8.4f} {r['mase']:>8.4f} "
                  f"{best_str:>8s} {improv_str:>8s}")

    print()
    print("=" * 90)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Architecture-Diversity Ensemble Experiment for N-BEATS Lightning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, choices=["m4"],
                        help="Dataset to use (currently only m4 supported)")
    parser.add_argument("--periods", nargs="+", default=None,
                        help="M4 periods to run (default: all)")
    parser.add_argument("--ensemble-name", default=None,
                        help="Run a single ensemble (default: all)")
    parser.add_argument("--n-runs", type=int, default=N_RUNS_ENSEMBLE,
                        help=f"Runs per (config, multiplier) (default: {N_RUNS_ENSEMBLE})")
    parser.add_argument("--max-epochs", type=int, default=100,
                        help="Max training epochs (default: 100)")
    parser.add_argument("--accelerator", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Accelerator (default: auto)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (default: 0)")
    parser.add_argument("--n-gpus", type=int, default=None,
                        help="Number of GPUs for parallel training (default: auto-detect)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size override (tuned values take precedence)")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable W&B logging")
    parser.add_argument("--wandb-project", default="nbeats-lightning",
                        help="W&B project name")

    args = parser.parse_args()
    run_ensemble_experiment(args)


if __name__ == "__main__":
    main()
