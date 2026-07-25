# PEN-BEATS Parameter Efficient N-BEATS

A [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) implementation of [N-BEATS](https://arxiv.org/pdf/1905.10437.pdf) (Oreshkin et al., 2019) and `NHiTSNet`, extended with wavelet and autoencoder block families. Published as [`lightningnbeats`](https://pypi.org/project/lightningnbeats/) on PyPI.

This library preserves the doubly residual stack architecture from the original paper and adds composable block families: orthonormal discrete wavelet transform (DWT) bases, polynomial–wavelet hybrids, and autoencoder backbones with learned-gate and variational variants. Blocks are freely composable via the `stack_types` list, enabling everything from paper-faithful reproductions to heavily compressed parameter-efficient stacks.

The companion research paper, *Wavelets and Autoencoders are All You Need*, lives at [NBEATS-Explorations/paper.md](NBEATS-Explorations/paper.md) and contains the full empirical analysis, ablation evidence, and reproducibility artifacts. This README covers the library: installation, block families, hyperparameters, dataloaders, and usage patterns.

## Requirements

- **Python** ≥ 3.12, < 3.15
- **PyTorch** ≥ 2.1.0
- **Lightning** ≥ 2.1.0

Supports CUDA, Apple MPS, Intel XPU, and CPU. `lightningnbeats.get_best_accelerator()` returns the highest-priority device available.

## Parameter efficiency

The headline feature of this library is **60–95% parameter reduction** with no accuracy penalty on standard benchmarks.

- **`AERootBlock`** replaces the paper's 4-layer FC trunk with an hourglass encoder–decoder (`input → U/2 → latent_dim → U/2 → U`).
- **`AERootBlockLG`** adds a learned `sigmoid(gate) ⊙ z` mask after the latent layer, allowing the network to discover effective latent dimensionality during training. This means `latent_dim` can be kept small (e.g., 16–32) without accuracy loss.

**Empirical result on M4**: a `TrendWaveletAELG` stack with `latent_dim=32`, `wavelet_type='db3'`, and 10 stacks lands within ~0.5% sMAPE of 19–43M-parameter paper baselines while using **0.48M parameters** (12–80× compression). The same AE backbones transfer cleanly to `NHiTSNet` and other architectures.

For high-compression regimes, prefer `AERootBlockLG` over plain `AERootBlock`: the gate removes the need to hand-tune `latent_dim`, and at matched dimensions LG consistently matches or beats the ungated AE backbone across M4 periods.

## Installation

```bash
pip install lightningnbeats         # from PyPI
# or, from source for development:
git clone https://github.com/realdanielbyrne/N-BEATS-Lightning.git
cd N-BEATS-Lightning
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### CUDA

PyPI hosts a CPU-only `torch` wheel. To use a CUDA build, install the package first, then upgrade `torch` from PyTorch's CUDA index:

```bash
pip install -r requirements-cuda.txt                       # CUDA 12.8 (recommended)
# or pick a specific build:
pip install torch --index-url https://download.pytorch.org/whl/cu128 --upgrade
pip install torch --index-url https://download.pytorch.org/whl/cu121 --upgrade
pip install torch --index-url https://download.pytorch.org/whl/cu118 --upgrade
```

Match your driver at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

## Quickstart

```python
from lightningnbeats import NBeatsNet, get_best_accelerator
from lightningnbeats.loaders import TimeSeriesDataModule
import lightning.pytorch as pl
import pandas as pd

milk = pd.read_csv('src/lightningnbeats/data/Milk/milk.csv', index_col=0).values.flatten()
forecast_length, backcast_length = 6, 24

model = NBeatsNet(
    stack_types=['Trend', 'Seasonality'],
    backcast_length=backcast_length,
    forecast_length=forecast_length,
    n_blocks_per_stack=3,
    thetas_dim=5,
    share_weights=True,
)

dm = TimeSeriesDataModule(
    train_data=milk[:-forecast_length],
    val_data=milk[-(forecast_length + backcast_length):],
    backcast_length=backcast_length,
    forecast_length=forecast_length,
    batch_size=64,
    shuffle=True,
)

trainer = pl.Trainer(accelerator=get_best_accelerator(), max_epochs=100)
trainer.fit(model, datamodule=dm)
```

`stack_types` is required — there is no default architecture. Any combination of registered block names is legal; blocks compose freely in any order.

## Block reference

Every entry in [`constants.BLOCKS`](src/lightningnbeats/constants.py) is a valid `stack_types` string and maps to a concrete class in [`src/lightningnbeats/blocks/blocks.py`](src/lightningnbeats/blocks/blocks.py).

### Backbones

| Backbone | Topology | Notes |
|---|---|---|
| `RootBlock` | input → U → U → U → U | Paper-faithful 4-layer FC trunk. |
| `AERootBlock` | input → U/2 → d → U/2 → U | Hourglass; `latent_dim` controls bottleneck `d`. |
| `AERootBlockLG` | same as AE | Adds learned `sigmoid(γ) ⊙ z` gate; allows smaller `d` without accuracy loss. |
| `AERootBlockVAE` | μ/logσ heads, reparameterization | KL loss accumulated and added during `training_step`, scaled by `kl_weight` (default `0.1`). |

### Block families

Each backbone is paired with the same set of basis-expansion heads:

| Family | RootBlock | AERootBlock | AERootBlockLG | AERootBlockVAE |
| --- | --- | --- | --- | --- |
| Generic (paper) | `Generic` | `GenericAE` | `GenericAELG` | `GenericVAE` |
| Bottleneck Generic | `BottleneckGeneric` | `BottleneckGenericAE` | `BottleneckGenericAELG` | `BottleneckGenericVAE` |
| Backcast-only AE | `GenericAEBackcast` | `GenericAEBackcastAE` | `GenericAEBackcastAELG` | `GenericAEBackcastVAE` |
| Branch-specific AE | `AutoEncoder` | `AutoEncoderAE` | `AutoEncoderAELG` | `AutoEncoderVAE` |
| Trend (polynomial) | `Trend` | `TrendAE` | `TrendAELG` | `TrendVAE` |
| Seasonality (Fourier) | `Seasonality` | `SeasonalityAE` | `SeasonalityAELG` | `SeasonalityVAE` |
| Wavelet (DWT) | `WaveletV3` | `WaveletV3AE` | `WaveletV3AELG` | `WaveletV3VAE` |
| Trend + Wavelet | `TrendWavelet` | `TrendWaveletAE` | `TrendWaveletAELG` | — |
| Trend + Wavelet + Generic | `TrendWaveletGeneric` | `TrendWaveletGenericAE` | `TrendWaveletGenericAELG` | `TrendWaveletGenericVAE` |

`WaveletV3` is the recommended public API: pass `wavelet_type` (e.g. `'haar'`, `'db3'`, `'coif2'`, `'sym10'`). Family-named subclasses (`HaarWaveletV3`, `DB3WaveletV3`, `Coif2WaveletV3`, `Symlet10WaveletV3`, …) remain as compatibility wrappers and are listed in `BLOCKS`. V1/V2 wavelet families were removed for instability; V2 results are preserved under `experiments/results/` for reference.

For short forecast targets, prefer short-support wavelets (`haar`, `db2`, `db3`). When `pywt.dwt_max_level(...) == 0`, `_WaveletGeneratorV3` keeps `level=0` (approximation-only orthonormal basis) rather than forcing an invalid level-1 decomposition.

### Composing stacks

```python
# Pure generic (paper-equivalent at depth 30)
NBeatsNet(stack_types=['Generic'] * 30, ...)

# Interpretable (Trend + Seasonality)
NBeatsNet(stack_types=['Trend', 'Seasonality'], n_blocks_per_stack=3, share_weights=True, ...)

# Alternating Trend + Wavelet (RootBlock)
NBeatsNet(stack_types=['Trend', 'DB3WaveletV3'] * 5, ...)

# Unified TrendWavelet (single composite block)
NBeatsNet(stack_types=['TrendWavelet'] * 30, wavelet_type='sym10', trend_thetas_dim=3, basis_dim=H, ...)

# Sub-1M parameter compact model
NBeatsNet(
    stack_types=['TrendWaveletAELG'] * 10,
    wavelet_type='db3', trend_thetas_dim=3, basis_dim=H, latent_dim=32,
    share_weights=True, ...,
)

# Asymmetric wavelets for L >> H
NBeatsNet(
    stack_types=['WaveletV3AELG'] * 10,
    wavelet_type='coif2',
    backcast_wavelet_type='sym20', forecast_wavelet_type='coif2',
    ...,
)
```

> **Convention.** Place `Trend*` stacks before `Wavelet*` / `Generic*` / `Seasonality*` stacks. The block-ordering ablation (paper Appendix B.5) found Trend-first either neutral or strongly preferred on every dataset; reversing it can 2–3× the OWA on VAE backbones.

## Hyperparameters

### Width and dimension parameters

| Parameter | Default | Applies to | Description |
|---|---|---|---|
| `g_width` | 512 | Generic / BottleneckGeneric / GenericAEBackcast / Wavelet families | Hidden units for Generic-type and Wavelet blocks. |
| `s_width` | 2048 | Seasonality / SeasonalityAE | Hidden units for Seasonality blocks (larger because Fourier basis is dense). |
| `t_width` | 256 | Trend / TrendAE / TrendWavelet / TrendWaveletGeneric families | Hidden units for Trend-type blocks. |
| `ae_width` | 512 | AutoEncoder / AutoEncoderAE | Hidden units for branch-specific AutoEncoder blocks. |
| `latent_dim` | 5 | AE / LG / VAE backbones | Bottleneck width `d` in the encoder–decoder. In `AutoEncoderAE` and `GenericAEBackcastAE` it controls the branch-local AE latents. |
| `thetas_dim` | 3 | BottleneckGeneric / GenericAEBackcast* | Rank of the explicit theta/basis-coefficient projection (bottleneck factorization). |
| `trend_thetas_dim` | 3 | Trend*and TrendWavelet* families | Polynomial order (number of trend basis functions). |
| `basis_dim` | 32 | WaveletV3 and TrendWavelet* families | Wavelet basis rank `k` per path. |
| `forecast_basis_dim` | `None` | Wavelet families | Optional override for forecast-path basis rank. When `None`, both paths use `basis_dim`. |
| `generic_dim` | — | TrendWaveletGeneric* | Rank of the learned generic branch in the three-way decomposition. |

### Recommended settings

- **Stack ordering** — Place `Trend*` stacks before `Wavelet*` / `Generic*` / `Seasonality*` stacks. The block-ordering ablation found Trend-first either neutral or strongly preferred on every dataset; reversing it can 2–3× the OWA on VAE backbones.
- **`active_g`** — Default `False` (paper-faithful). Use `'forecast'` (`active_g='forecast'`) specifically on M4-Hourly and for some Generic convergence rescues; it is not a global default and can regress on other periods.
- **High compression** — Use `AERootBlockLG` with `latent_dim=16–32` and `share_weights=True` for the best parameter-efficiency frontier. A 10-stack `TrendWaveletAELG` configuration is the recommended starting point for <1M-parameter targets.
- **AE vs LG** — `AERootBlockLG` matches or beats plain `AERootBlock` at matched `latent_dim`. Prefer LG when `latent_dim` is the binding constraint.
- **VAE backbones** — The VAE family is strictly unusable on M4 (55–68 sMAPE collapse). Reserve for exploratory or non-M4 use cases only.
- **Wavelet shortlist for M4** — `haar`, `db3`, `coif2`, `sym10`. `coif3` produces no per-period SOTA and can be omitted from default sweeps.
- **Short targets** — Prefer `haar`, `db2`, `db3` when `forecast_length` is small; long-support families (`sym20`, `db20`, `coif10`) degrade to approximation-only bases.

### Selected extensions (cross-references to the paper)

These knobs are repository extensions and not part of Oreshkin et al. (2019). See the paper's appendices for empirical evidence.

- **`skip_distance` / `skip_alpha`** — ResNet-style reinjection of the original input every `skip_distance` stacks, scaled by `skip_alpha` (float or `'learnable'`). Useful for `GenericAELG` at depth ≥ 20; off by default.
- **`sum_losses`** — Adds `0.25 · L(backcast, 0)` to the loss to push backcasts toward full reconstruction. Off by default; reserved for convergence studies.
- **Asymmetric wavelet families** — `backcast_wavelet_type` and `forecast_wavelet_type` override `wavelet_type` per path. Useful when `L ≫ H` (e.g., Traffic-96: L=480, H=96).
- **Tiered basis offsets** (`basis_offset`) — Cycle each stack through a different SVD spectrum slice of the orthonormal wavelet basis. Strongest on M4-Daily under paper-sample; regresses on Weekly.

## Custom losses

In addition to the standard `torch.nn` losses, the package registers `SMAPELoss`, `MAPELoss`, `MASELoss` (accepts `seasonal_period`), and `NormalizedDeviationLoss`. Select by name via the `loss=` argument.

## Data and dataloaders

The library provides three `LightningDataModule` implementations tailored to different dataset layouts. Pick the one that matches your data orientation.

### `TimeSeriesDataModule` — single univariate series

For a single time series (e.g., the bundled Milk dataset). Splits the series into sliding windows and performs an 80/20 random train/validation split.

```python
from lightningnbeats.loaders import TimeSeriesDataModule
import pandas as pd

series = pd.read_csv('milk.csv', index_col=0).values.flatten()
forecast_length, backcast_length = 6, 24

dm = TimeSeriesDataModule(
    train_data=series[:-forecast_length],
    val_data=series[-(forecast_length + backcast_length):],
    backcast_length=backcast_length,
    forecast_length=forecast_length,
    batch_size=64,
)
```

### `RowCollectionTimeSeriesDataModule` — row-oriented collections

For datasets where **rows = series** and **columns = time observations**. Each row is an independent time series; windows are sampled across all rows. The validation split takes the last `backcast_length + forecast_length` columns of every series. This matches the M4 competition format.

```python
from lightningnbeats.data import M4Dataset
from lightningnbeats.loaders import RowCollectionTimeSeriesDataModule

m4 = M4Dataset('Yearly', 'All')
backcast_length = m4.forecast_length * 5

dm = RowCollectionTimeSeriesDataModule(
    data=m4.train_data,
    backcast_length=backcast_length,
    forecast_length=m4.forecast_length,
    batch_size=2048,
    split_ratio=0.8,
)
```

For test-time evaluation, use `RowCollectionTimeSeriesTestModule` to concatenate the last `backcast_length` observations from the training matrix with the first `forecast_length` test observations:

```python
from lightningnbeats.loaders import RowCollectionTimeSeriesTestModule

test_dm = RowCollectionTimeSeriesTestModule(
    train_data=m4.train_data,
    test_data=m4.test_data,
    backcast_length=backcast_length,
    forecast_length=m4.forecast_length,
    batch_size=2048,
)
```

### `ColumnarCollectionTimeSeriesDataModule` — column-oriented collections

For datasets where **columns = series** and **rows = time observations**. This is the layout used by Tourism, Traffic, and Weather. Supports optional normalization (`normalization_style='global'` for per-column z-score, `'window'` for RevIN-style per-window z-score), dedicated validation splits, and N-BEATS paper-style insample constraints.

```python
from lightningnbeats.data import TrafficDataset
from lightningnbeats.loaders import (
    ColumnarCollectionTimeSeriesDataModule,
    ColumnarCollectionTimeSeriesTestDataModule,
)

traffic = TrafficDataset(horizon=96)
backcast_length = 96 * 5

dm = ColumnarCollectionTimeSeriesDataModule(
    dataframe=traffic.train_data,
    backcast_length=backcast_length,
    forecast_length=traffic.forecast_length,
    batch_size=1024,
)

test_dm = ColumnarCollectionTimeSeriesTestDataModule(
    train_data=traffic.train_data,
    test_data=traffic.test_data,
    backcast_length=backcast_length,
    forecast_length=traffic.forecast_length,
    batch_size=1024,
)
```

## Usage examples

### Standard N-BEATS interpretable (Trend + Seasonality)

```python
from lightningnbeats import NBeatsNet, get_best_accelerator
from lightningnbeats.loaders import TimeSeriesDataModule
import lightning.pytorch as pl
import pandas as pd

milk = pd.read_csv('src/lightningnbeats/data/Milk/milk.csv', index_col=0).values.flatten()
H, L = 6, 24

model = NBeatsNet(
    stack_types=['Trend', 'Seasonality'],
    backcast_length=L,
    forecast_length=H,
    n_blocks_per_stack=3,
    thetas_dim=5,
    share_weights=True,
)

dm = TimeSeriesDataModule(
    train_data=milk[:-H],
    val_data=milk[-(H + L):],
    backcast_length=L,
    forecast_length=H,
    batch_size=64,
)

trainer = pl.Trainer(accelerator=get_best_accelerator(), max_epochs=100)
trainer.fit(model, datamodule=dm)
```

### High-compression `TrendWaveletAELG`

```python
from lightningnbeats import NBeatsNet, get_best_accelerator
from lightningnbeats.loaders import TimeSeriesDataModule
import lightning.pytorch as pl
import pandas as pd

milk = pd.read_csv('src/lightningnbeats/data/Milk/milk.csv', index_col=0).values.flatten()
H, L = 6, 24

model = NBeatsNet(
    stack_types=['TrendWaveletAELG'] * 10,
    backcast_length=L,
    forecast_length=H,
    wavelet_type='db3',
    trend_thetas_dim=3,
    basis_dim=H,
    latent_dim=32,
    share_weights=True,
)

dm = TimeSeriesDataModule(
    train_data=milk[:-H],
    val_data=milk[-(H + L):],
    backcast_length=L,
    forecast_length=H,
    batch_size=64,
)

trainer = pl.Trainer(accelerator=get_best_accelerator(), max_epochs=100)
trainer.fit(model, datamodule=dm)
```

### M4-style row-oriented dataset

```python
from lightningnbeats import NBeatsNet, get_best_accelerator
from lightningnbeats.data import M4Dataset
from lightningnbeats.loaders import (
    RowCollectionTimeSeriesDataModule,
    RowCollectionTimeSeriesTestModule,
)
import lightning.pytorch as pl

m4 = M4Dataset('Yearly', 'All')
H = m4.forecast_length
L = H * 5

model = NBeatsNet(
    stack_types=['Trend', 'DB3WaveletV3'] * 5,
    backcast_length=L,
    forecast_length=H,
)

dm = RowCollectionTimeSeriesDataModule(
    data=m4.train_data,
    backcast_length=L,
    forecast_length=H,
    batch_size=2048,
)

test_dm = RowCollectionTimeSeriesTestModule(
    train_data=m4.train_data,
    test_data=m4.test_data,
    backcast_length=L,
    forecast_length=H,
    batch_size=2048,
)

trainer = pl.Trainer(accelerator=get_best_accelerator(), max_epochs=50)
trainer.fit(model, datamodule=dm)
trainer.test(model, datamodule=test_dm)
```

### Traffic / Weather column-oriented dataset

```python
from lightningnbeats import NBeatsNet, get_best_accelerator
from lightningnbeats.data import TrafficDataset
from lightningnbeats.loaders import (
    ColumnarCollectionTimeSeriesDataModule,
    ColumnarCollectionTimeSeriesTestDataModule,
)
import lightning.pytorch as pl

traffic = TrafficDataset(horizon=96)
H = traffic.forecast_length
L = H * 5

model = NBeatsNet(
    stack_types=['TrendWaveletAELG'] * 10,
    backcast_length=L,
    forecast_length=H,
    wavelet_type='db3',
    trend_thetas_dim=3,
    basis_dim=H,
    latent_dim=32,
    share_weights=True,
)

dm = ColumnarCollectionTimeSeriesDataModule(
    dataframe=traffic.train_data,
    backcast_length=L,
    forecast_length=H,
    batch_size=1024,
)

test_dm = ColumnarCollectionTimeSeriesTestDataModule(
    train_data=traffic.train_data,
    test_data=traffic.test_data,
    backcast_length=L,
    forecast_length=H,
    batch_size=1024,
)

trainer = pl.Trainer(accelerator=get_best_accelerator(), max_epochs=50)
trainer.fit(model, datamodule=dm)
trainer.test(model, datamodule=test_dm)
```

## Experiments

Reproducing or extending the paper's sweeps is driven by YAML through [`experiments/run_from_yaml.py`](experiments/run_from_yaml.py):

```bash
python experiments/run_from_yaml.py experiments/configs/nbeats_g.yaml --dry-run
python experiments/run_from_yaml.py experiments/configs/nbeats_g.yaml
python experiments/run_from_yaml.py experiments/configs/genericae_pure.yaml          # successive halving
python experiments/run_from_yaml.py experiments/configs/trendae_ae_alternating.yaml
python experiments/run_from_yaml.py experiments/configs/nbeats_g.yaml --analyze-only # no retraining
```

The launcher supports two-pass studies, atomic claim files for parallel workers, search/successive-halving rounds, custom CSV columns, and full CLI overrides. Schema reference: [`experiments/configs/schema.md`](experiments/configs/schema.md). Canonical training-protocol YAMLs (sliding, paper-sample MultiStepLR, paper-sample plateau) and their reproducibility CSVs are listed in [paper.md Appendix B.9](NBEATS-Explorations/paper.md).

Examples in [`examples/`](examples/) (`#%%` cell markers, runnable as scripts or in Jupyter):

```bash
python examples/M4AllBlks.py
python examples/TourismAllBlks.py
```

## Testing

```bash
pytest tests/                         # all
pytest tests/test_blocks.py -v        # block shapes / registries
```

Conventions and troubleshooting: [TESTING.md](TESTING.md). CI does not run tests; no linter or formatter is configured.

## Adding a new block

1. Create the class in [`src/lightningnbeats/blocks/blocks.py`](src/lightningnbeats/blocks/blocks.py), inheriting from `RootBlock`, `AERootBlock`, `AERootBlockLG`, or `AERootBlockVAE`. `forward()` must return `(backcast, forecast)`.
2. Add the class name to `BLOCKS` in [`constants.py`](src/lightningnbeats/constants.py).
3. If a new width parameter is needed, extend the dispatch in `NBeatsNet.create_stack()`.
4. Add a shape test; the parametrized `TestAllBlocksOutputShapes` picks up new entries automatically.

## Citation

If the wavelet / AE block families or the parameter-efficiency results are useful in your work, please cite *Wavelets and Autoencoders are All You Need* — see [NBEATS-Explorations/paper.md](NBEATS-Explorations/paper.md) for the canonical text and reproducibility pointers.
