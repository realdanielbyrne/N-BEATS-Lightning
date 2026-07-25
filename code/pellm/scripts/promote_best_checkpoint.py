#!/usr/bin/env python3
"""Promote the lowest-eval-loss checkpoint to the trainedmodels/ directory.

Pretraining saves the *last-step* model to ``trainedmodels/<exp>/<variant>/``.
For from-scratch runs the last step is usually within noise of the best-val
checkpoint (cosine LR decays to ~0), but if a variant overfits late, the
last-step model can be measurably worse. This script reads the per-eval
metrics and promotes the lowest-loss snapshot in-place.

Usage:
    # Dry-run all completed variants in an experiment (default: smol_replacement_paper)
    .venv/bin/python scripts/promote_best_checkpoint.py --dry-run

    # Apply to all variants
    .venv/bin/python scripts/promote_best_checkpoint.py --apply

    # Apply to a single variant
    .venv/bin/python scripts/promote_best_checkpoint.py --apply --variant trendwavelet_db3_32

    # Use symlinks instead of copies (faster, but trainedmodels/ depends on
    # checkpoints/ remaining on disk — safer to copy by default)
    .venv/bin/python scripts/promote_best_checkpoint.py --apply --link

The promoted-from token count and old/new loss are recorded in
``training_manifest.json`` under a ``best_checkpoint_promotion`` key for
traceability. Re-running is idempotent: if the manifest already records a
promotion to the same token count, the script skips.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from artifact_paths import CHECKPOINTS_DIR, EVALS_DIR, TRAINED_MODELS_DIR

_HF_MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
# Files in trainedmodels/<variant>/ that we must NOT overwrite during promotion
# (they describe the run, not the model weights).
_PRESERVE_FILES = ("training_manifest.json", "recipe_drift_note.json")


@dataclass(frozen=True)
class EvalEntry:
    tokens: int
    loss: float
    perplexity: float
    source: str  # "eval_<tokens>.json" or "summary.json"


def collect_evals(eval_dir: Path) -> list[EvalEntry]:
    """Read all per-eval JSONs and summary.json for a variant."""

    entries: list[EvalEntry] = []
    for path in sorted(eval_dir.glob("eval_*.json")):
        with path.open() as f:
            data = json.load(f)
        entries.append(
            EvalEntry(
                tokens=int(data["tokens"]),
                loss=float(data["loss"]),
                perplexity=float(data["perplexity"]),
                source=path.name,
            )
        )

    summary = eval_dir / "summary.json"
    if summary.exists():
        with summary.open() as f:
            data = json.load(f)
        entries.append(
            EvalEntry(
                tokens=int(data["tokens"]),
                loss=float(data["loss"]),
                perplexity=float(data["perplexity"]),
                source="summary.json",
            )
        )

    return entries


def discover_variants(exp: str) -> list[str]:
    eval_root = EVALS_DIR / exp
    if not eval_root.is_dir():
        raise SystemExit(f"No evals found at {eval_root}")
    return sorted(p.name for p in eval_root.iterdir() if p.is_dir())


def load_manifest(output_dir: Path) -> dict:
    manifest_path = output_dir / "training_manifest.json"
    if not manifest_path.exists():
        return {}
    with manifest_path.open() as f:
        return json.load(f)


def write_manifest(output_dir: Path, manifest: dict) -> None:
    with (output_dir / "training_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def promote_files(
    src_hf_dir: Path, dst_dir: Path, *, link: bool, dry_run: bool
) -> list[str]:
    """Replace HF model weight/config files in *dst_dir* from *src_hf_dir*.

    Preserves ``training_manifest.json`` and ``recipe_drift_note.json``.
    Returns the list of filenames touched.
    """

    touched: list[str] = []
    for fname in _HF_MODEL_FILES:
        src = src_hf_dir / fname
        if not src.exists():
            continue
        dst = dst_dir / fname
        touched.append(fname)
        if dry_run:
            continue
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if link:
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)
    return touched


def promote_variant(
    exp: str,
    variant: str,
    *,
    apply: bool,
    link: bool,
) -> dict:
    """Find best-loss checkpoint for a variant and (optionally) promote it.

    Returns a result dict with keys: status, best_tokens, best_loss,
    final_tokens, final_loss, action.
    """

    eval_dir = EVALS_DIR / exp / variant
    output_dir = TRAINED_MODELS_DIR / exp / variant

    if not eval_dir.is_dir():
        return {"status": "skip", "reason": f"no eval_dir at {eval_dir}"}
    if not output_dir.is_dir():
        return {"status": "skip", "reason": f"no output_dir at {output_dir}"}

    entries = collect_evals(eval_dir)
    if not entries:
        return {"status": "skip", "reason": f"no eval files in {eval_dir}"}

    best = min(entries, key=lambda e: e.loss)
    summary_entry = next((e for e in entries if e.source == "summary.json"), None)

    result: dict = {
        "variant": variant,
        "best_tokens": best.tokens,
        "best_loss": best.loss,
        "best_perplexity": best.perplexity,
        "best_source": best.source,
        "final_tokens": summary_entry.tokens if summary_entry else None,
        "final_loss": summary_entry.loss if summary_entry else None,
        "final_perplexity": summary_entry.perplexity if summary_entry else None,
    }

    # Already the final model — nothing to do.
    if best.source == "summary.json":
        result["status"] = "noop"
        result["action"] = "best is final-step model; trainedmodels/ unchanged"
        return result

    # Idempotency check: skip if manifest already records this promotion.
    manifest = load_manifest(output_dir)
    prior = manifest.get("best_checkpoint_promotion") or {}
    if prior.get("source_tokens") == best.tokens:
        result["status"] = "noop"
        result["action"] = (
            f"already promoted from tokens={best.tokens} (manifest records it)"
        )
        return result

    src_hf = CHECKPOINTS_DIR / exp / variant / f"tokens_{best.tokens}" / "hf_model"
    if not src_hf.is_dir():
        result["status"] = "error"
        result["action"] = f"missing source snapshot {src_hf}"
        return result

    touched = promote_files(src_hf, output_dir, link=link, dry_run=not apply)
    result["files"] = touched

    if apply:
        manifest["best_checkpoint_promotion"] = {
            "source_tokens": best.tokens,
            "source_path": str(src_hf),
            "promoted_loss": best.loss,
            "promoted_perplexity": best.perplexity,
            "previous_final_loss": (
                summary_entry.loss if summary_entry else None
            ),
            "previous_final_perplexity": (
                summary_entry.perplexity if summary_entry else None
            ),
            "method": "symlink" if link else "copy",
        }
        write_manifest(output_dir, manifest)
        result["status"] = "promoted"
        result["action"] = (
            f"promoted tokens_{best.tokens}/hf_model -> {output_dir} ({len(touched)} files, {'symlink' if link else 'copy'})"
        )
    else:
        result["status"] = "would_promote"
        result["action"] = (
            f"would promote tokens_{best.tokens}/hf_model -> {output_dir} ({len(touched)} files)"
        )

    return result


def format_result(r: dict) -> str:
    if r["status"] == "skip":
        return f"  skip   {r.get('variant', '?')}: {r['reason']}"
    if r["status"] == "noop":
        return (
            f"  noop   {r['variant']}: best=tokens_{r['best_tokens']} "
            f"loss={r['best_loss']:.4f} ppl={r['best_perplexity']:.3f}  ({r['action']})"
        )
    if r["status"] == "error":
        return f"  ERROR  {r['variant']}: {r['action']}"
    # would_promote / promoted
    delta_loss = (
        r["final_loss"] - r["best_loss"]
        if r["final_loss"] is not None
        else None
    )
    delta_ppl = (
        r["final_perplexity"] - r["best_perplexity"]
        if r["final_perplexity"] is not None
        else None
    )
    tag = "PROMOTE" if r["status"] == "promoted" else "would-promote"
    line = (
        f"  {tag:<14} {r['variant']}: best=tokens_{r['best_tokens']} "
        f"loss={r['best_loss']:.4f} ppl={r['best_perplexity']:.3f}"
    )
    if delta_loss is not None:
        line += (
            f"  (final loss={r['final_loss']:.4f} ppl={r['final_perplexity']:.3f}, "
            f"Δloss={delta_loss:+.4f} Δppl={delta_ppl:+.3f})"
        )
    return line


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        default="smol_replacement_paper",
        help="Experiment name (default: smol_replacement_paper)",
    )
    parser.add_argument(
        "--variant",
        action="append",
        help="Variant name (repeatable). Default: all variants in experiment.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Perform the promotion.")
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Default. Print what would change without modifying files.",
    )
    parser.add_argument(
        "--link",
        action="store_true",
        help=(
            "Use symlinks instead of copies. Faster but ties trainedmodels/ to "
            "checkpoints/ remaining on disk."
        ),
    )
    args = parser.parse_args()

    apply = args.apply  # if neither --apply nor --dry-run, default is dry-run
    variants = args.variant or discover_variants(args.experiment)

    print(f"Experiment: {args.experiment}")
    print(f"Mode: {'APPLY' if apply else 'dry-run'}  Method: {'symlink' if args.link else 'copy'}")
    print(f"Variants ({len(variants)}): {variants}\n")

    results = []
    for variant in variants:
        result = promote_variant(
            args.experiment, variant, apply=apply, link=args.link
        )
        result.setdefault("variant", variant)
        results.append(result)
        print(format_result(result))

    n_promoted = sum(1 for r in results if r["status"] == "promoted")
    n_would = sum(1 for r in results if r["status"] == "would_promote")
    n_noop = sum(1 for r in results if r["status"] == "noop")
    n_skip = sum(1 for r in results if r["status"] == "skip")
    n_err = sum(1 for r in results if r["status"] == "error")
    print(
        f"\nSummary: promoted={n_promoted}, would_promote={n_would}, "
        f"noop={n_noop}, skip={n_skip}, error={n_err}"
    )
    if not apply and n_would:
        print("Re-run with --apply to perform the promotions.")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
