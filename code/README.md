# Anonymous Code Supplement — Wavelets and Autoencoders are All You Need

This is the anonymized code archive accompanying the NeurIPS 2026 submission
*Wavelets and Autoencoders are All You Need*. It contains two independent
Python packages, each with its own dependency set, experiment launcher,
configurations, and pytest suite. The split keeps the time-series
environment free of HuggingFace `transformers` and vice versa.

## Abstract

A 0.48M-parameter model matches N-BEATS on the M4 forecasting benchmark using up to 95% fewer parameters. Applied inside a Transformer, the same structural change produces a Llama-3.2-1B variant that beats its unmodified base on perplexity. We replace the Fourier-style basis in N-BEATS with discrete wavelet transforms, which localize features in time and frequency, and route the block trunk through a learnable-gate autoencoder bottleneck. Together these cut N-BEATS parameter counts by **60–95%** while *matching or beating* its published M4 sMAPEs without ensembling. A single **0.48M-parameter** TrendWavelet model (`TWAELG_10s_ld32_db3`) lands top-5 on M4-Yearly, M4-Daily, and M4-Hourly under the paper-sample protocol, giving **12–80× compression at near-zero accuracy cost** versus 19M–43M-parameter baselines. The blocks transfer cleanly to other architectures. On NHiTS the parameter savings carry over to Weather and Traffic. In Transformer LLMs the effect sharpens. A from-scratch SmolLM2-135M-class model with wavelet and AE replacements for its SwiGLU MLPs and attention projections runs at **half the parameters** (67.2M vs 134.5M) and Pareto-dominates a pure-AE baseline. In a separate experiment, replacing one MLP block in Llama-3.2-1B with the same AE-LG bottleneck and recovering accuracy through autoencoder pretraining and knowledge-distillation fine-tuning shrinks that block by **~90%** and lowers whole-model perplexity to **18.81**, beating the unmodified Llama-3.2-1B baseline of **19.45** (−3.3%). What looks like a forecasting-specific compression trick is a general structural prior. The freed parameter budget can be redeposited into deeper stacks, richer composite bases, or wider models.

## Layout

```text
code/
├── README.md                  This file
├── nbeats/                    Time-series forecasting package
│   ├── pyproject.toml         nbeats_anon metadata (anonymized)
│   ├── requirements.txt       Pip-installable dependencies (forecasting only)
│   ├── conftest.py            Adds src/ to sys.path for pytest
│   ├── README.md              Detailed install / data / experiment guide
│   ├── src/nbeats_anon/       Library: blocks, models, loaders, losses
│   ├── experiments/           run_from_yaml.py + all paper YAML configs
│   └── tests/                 Pytest suite for the forecasting package
└── pellm/                     Transformer experiments package
    ├── pyproject.toml         pellm metadata (anonymized)
    ├── README.md              Detailed install / data / experiment guide
    ├── pellm/                 Library: PE-Llama, custom attention/MLP layers
    ├── scripts/               finetune.py, run_from_yaml.py, eval scripts
    │   └── experiments/       YAML configs for transformer experiments
    ├── tests/                 Pytest suite for the transformer package
    ├── docs/                  PELLM design notes
    └── evals/                 Pre-computed lm-eval-harness results
```

## Which package covers which paper section

| Paper section / table | Package |
| --- | --- |
| Sections 3–4, Appendix B (M4, Tourism, Traffic, Weather, Milk, NHiTS) | `nbeats/` |
| Section on parameter-efficient Llama (PELLM) | `pellm/` |

## Installation

The two packages are installed independently into separate virtual
environments. Each subdirectory contains its own README with the full set
of commands, data-access notes, smoke tests, and reproduction recipes.

### Forecasting (`nbeats/`)

Python 3.12–3.14, PyTorch ≥ 2.1.

```bash
cd nbeats
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest tests/ -v        # validate install
```

Full instructions, the M4 data download recipe, and the YAML-driven
experiment commands are in [`nbeats/README.md`](nbeats/README.md).

### Transformer (`pellm/`)

Python ≥ 3.10, PyTorch ≥ 2.1, Transformers ≥ 4.40.

```bash
cd pellm
python -m venv .venv && source .venv/bin/activate
pip install -e ".[experiments]"
pytest tests/ -v        # validate install
```

Full instructions, the Hugging Face / Llama 3.2 access steps, and the
PELLM experiment commands are in [`pellm/README.md`](pellm/README.md).

## Notes for reviewers

- All author, institutional, repository-URL, and email metadata have been
  scrubbed across both packages. If any oversight remains, please flag it
  via the OpenReview comment system.
- Hardware: every experiment in the paper was run on a single consumer GPU
  (16–24 GB VRAM) or Apple Silicon (MPS). Sub-1M-parameter forecasting
  configurations also train comfortably on CPU.
- Archive size: well below the 100 MB NeurIPS supplementary limit. Heavy
  artefacts (raw experiment logs, training notebooks, model checkpoints,
  M4 raw CSVs) are not bundled; the per-package READMEs document how to
  download or regenerate them.
