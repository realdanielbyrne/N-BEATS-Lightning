#!/usr/bin/env python3
"""Run smol_replacement_paper variants two-at-a-time, one per GPU.

Each variant runs as an independent single-process ``accelerate launch`` job
pinned to one GPU via ``CUDA_VISIBLE_DEVICES``. Pairs run concurrently; the
next pair starts after both finish. This is the recommended launcher for the
smol replacement paper experiment -- the prior 2-GPU DDP recipe wastes
per-GPU compute on these ~135M models.

Completion is detected by the presence of
``trainedmodels/<exp>/<variant>/training_manifest.json``.

Usage:
    .venv/bin/python scripts/run_smol_replacement_parallel.py [CONFIG] [options]

The new 1-GPU recipe requires ``training.grad_accum_steps`` in the YAML to be
double the prior 2-GPU value so the per-step token count stays identical. See
the comment header in ``scripts/experiments/smol_replacement_paper.yaml``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from artifact_paths import (  # noqa: E402
    CHECKPOINTS_DIR,
    EVALS_DIR,
    RUNS_DIR,
    TRAINED_MODELS_DIR,
    apply_data_drive_env_defaults,
)


REPO_ROOT = _SCRIPTS_DIR.parent
PRETRAIN_SCRIPT = REPO_ROOT / "scripts" / "pretrain_smol_replacement.py"
DEFAULT_ACCELERATE = REPO_ROOT / ".venv" / "bin" / "accelerate"


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "Install the experiments extra: pip install -e '.[experiments]'"
        ) from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def variant_outputs(exp_name: str, variant: str) -> dict[str, Path]:
    return {
        "trainedmodel": TRAINED_MODELS_DIR / exp_name / variant,
        "checkpoint": CHECKPOINTS_DIR / exp_name / variant,
        "run": RUNS_DIR / exp_name / variant,
        "eval": EVALS_DIR / exp_name / variant,
    }


def is_complete(exp_name: str, variant: str) -> bool:
    return (
        TRAINED_MODELS_DIR / exp_name / variant / "training_manifest.json"
    ).is_file()


def clear_variant(exp_name: str, variant: str) -> list[Path]:
    cleared: list[Path] = []
    for path in variant_outputs(exp_name, variant).values():
        if path.exists():
            shutil.rmtree(path)
            cleared.append(path)
    return cleared


def build_command(
    *,
    accelerate_bin: str,
    config: Path,
    variant: str,
    token_budget: int | None,
) -> list[str]:
    cmd = [
        accelerate_bin,
        "launch",
        "--num_processes",
        "1",
        str(PRETRAIN_SCRIPT),
        str(config),
        "--variant",
        variant,
    ]
    if token_budget is not None:
        cmd += ["--token-budget", str(token_budget)]
    return cmd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run smol_replacement_paper variants in parallel, one per GPU. "
            "Each pair runs concurrently; the next pair starts after both "
            "finish."
        )
    )
    p.add_argument(
        "config",
        nargs="?",
        default="scripts/experiments/smol_replacement_paper.yaml",
        type=Path,
    )
    p.add_argument(
        "--gpus",
        default="0,1",
        help="Comma-separated GPU ids to alternate across (default: 0,1).",
    )
    p.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Run only these variant names (overrides completion-skip).",
    )
    p.add_argument(
        "--skip",
        nargs="+",
        default=(),
        help="Variant names to skip even if not yet complete.",
    )
    p.add_argument(
        "--restart-variant",
        nargs="+",
        default=(),
        help="Clear these variants' outputs and rerun them.",
    )
    p.add_argument(
        "--force-variant",
        nargs="+",
        default=(),
        help="Run these variants even if they're already complete (no clear).",
    )
    p.add_argument(
        "--token-budget",
        type=int,
        default=None,
        help="Override training.token_budget (passed through to pretrain).",
    )
    p.add_argument(
        "--accelerate-bin",
        default=str(DEFAULT_ACCELERATE),
        help="Path to the accelerate executable (default: .venv/bin/accelerate).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned schedule and commands without launching anything.",
    )
    return p.parse_args()


def select_variants(
    cfg: dict, exp_name: str, args: argparse.Namespace
) -> list[str]:
    all_variants = list(cfg.get("variants", {}).keys())
    if not all_variants:
        raise RuntimeError(f"No variants defined in {args.config}")

    if args.only:
        unknown = [v for v in args.only if v not in all_variants]
        if unknown:
            raise ValueError(f"Unknown variant(s) in --only: {unknown}")
        candidates = list(args.only)
    else:
        candidates = list(all_variants)

    skip = set(args.skip)
    restart = set(args.restart_variant)
    force = set(args.force_variant) | restart

    selected: list[str] = []
    for v in candidates:
        if v in skip:
            print(f"  skip  {v}: --skip", flush=True)
            continue
        if v in force or args.only:
            selected.append(v)
            continue
        if is_complete(exp_name, v):
            print(f"  skip  {v}: already complete", flush=True)
            continue
        selected.append(v)
    return selected


def main() -> int:
    args = parse_args()

    if not PRETRAIN_SCRIPT.is_file():
        raise FileNotFoundError(f"Pretrain script not found: {PRETRAIN_SCRIPT}")
    if not Path(args.accelerate_bin).is_file():
        raise FileNotFoundError(
            f"accelerate binary not found: {args.accelerate_bin}"
        )

    cfg = load_yaml(args.config)
    exp_name = cfg["experiment"]["name"]

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise ValueError(f"--gpus must list at least one id, got {args.gpus!r}")

    print(f"Experiment: {exp_name}")
    print(f"GPUs: {gpus}")
    print(f"Config: {args.config}")
    print()

    if args.restart_variant:
        for v in args.restart_variant:
            if args.dry_run:
                paths = [p for p in variant_outputs(exp_name, v).values() if p.exists()]
                print(f"  would clear {v}: {len(paths)} dirs")
            else:
                cleared = clear_variant(exp_name, v)
                print(f"  cleared {v}: {len(cleared)} dirs")

    selected = select_variants(cfg, exp_name, args)
    if not selected:
        print("Nothing to run.")
        return 0

    n_gpus = len(gpus)
    pairs: list[list[str]] = [
        selected[i : i + n_gpus] for i in range(0, len(selected), n_gpus)
    ]

    print(f"Pending variants ({len(selected)}): {selected}")
    print(f"Pair schedule ({len(pairs)} pairs of up to {n_gpus}):")
    for i, pair in enumerate(pairs):
        slot_to_variant = {gpus[j]: v for j, v in enumerate(pair)}
        print(f"  pair {i+1}: {slot_to_variant}")
    print()

    if args.dry_run:
        for i, pair in enumerate(pairs):
            print(f"--- pair {i+1} ---")
            for j, variant in enumerate(pair):
                gpu = gpus[j]
                cmd = build_command(
                    accelerate_bin=args.accelerate_bin,
                    config=args.config,
                    variant=variant,
                    token_budget=args.token_budget,
                )
                log_path = RUNS_DIR / exp_name / variant / "parallel_launcher.log"
                print(f"  CUDA_VISIBLE_DEVICES={gpu} \\")
                print(f"    " + " ".join(cmd))
                print(f"    # log: {log_path}")
        return 0

    base_env = apply_data_drive_env_defaults(os.environ)

    overall_start = time.time()
    failures: list[tuple[str, int]] = []

    for i, pair in enumerate(pairs):
        print(f"=== pair {i+1}/{len(pairs)}: {pair} ===", flush=True)
        procs: list[tuple[str, str, subprocess.Popen, Path, object]] = []
        for j, variant in enumerate(pair):
            gpu = gpus[j]
            log_dir = RUNS_DIR / exp_name / variant
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "parallel_launcher.log"
            log_fh = log_path.open("a", encoding="utf-8")
            log_fh.write(
                f"\n=== launcher {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"variant={variant} gpu={gpu} ===\n"
            )
            log_fh.flush()
            cmd = build_command(
                accelerate_bin=args.accelerate_bin,
                config=args.config,
                variant=variant,
                token_budget=args.token_budget,
            )
            env = dict(base_env)
            env["CUDA_VISIBLE_DEVICES"] = gpu
            print(
                f"  launching variant={variant} gpu={gpu} log={log_path}",
                flush=True,
            )
            proc = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )
            procs.append((variant, gpu, proc, log_path, log_fh))

        pair_start = time.time()
        for variant, gpu, proc, log_path, log_fh in procs:
            ret = proc.wait()
            log_fh.close()
            elapsed = time.time() - pair_start
            print(
                f"  done variant={variant} gpu={gpu} returncode={ret} "
                f"elapsed={elapsed:.0f}s log={log_path}",
                flush=True,
            )
            if ret != 0:
                failures.append((variant, ret))

        if failures:
            print(
                f"\nHalting: {len(failures)} variant(s) failed in pair "
                f"{i+1}: {failures}",
                flush=True,
            )
            return 1

    overall_elapsed = time.time() - overall_start
    print(
        f"\nAll {len(selected)} variants completed in "
        f"{overall_elapsed/3600:.2f} h."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
