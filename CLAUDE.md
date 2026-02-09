# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyTorch Lightning implementation of the N-BEATS (Neural Basis Expansion Analysis for Time Series) forecasting algorithm, published as the `lightningnbeats` PyPI package. Extends the original paper with wavelet basis expansion blocks, autoencoder variants, bottleneck generic blocks, and fully customizable stack compositions.

## Build & Install

```bash
pip install -e .                    # editable install from source
pip install lightningnbeats          # install from PyPI
python -m build                      # build distribution package
```

Dependencies are defined in `pyproject.toml`. Uses setuptools as build backend.

## Running Examples

Examples are in `examples/` and are designed to run as scripts or Jupyter cell-by-cell (`#%%` markers):
```bash
cd examples
python M4AllBlks.py       # M4 dataset benchmark across all block types
python TourismAllBlks.py  # Tourism dataset benchmark
```

## Testing

Tests are in `tests/` and use pytest:
```bash
pytest tests/                        # run all tests
pytest tests/ -v                     # verbose output
pytest tests/test_blocks.py          # run specific test file
```

## Architecture

### Package Structure (`src/lightningnbeats/`)

- **`models.py`** - `NBeatsNet(pl.LightningModule)`: the main model class. Accepts a `stack_types` list of strings to define architecture. Handles forward pass with backward/forward residual connections, training/validation/test steps, loss configuration, and optimizer setup.
- **`blocks/blocks.py`** - All block implementations. This is the largest file. Block hierarchy:
  - `RootBlock(nn.Module)` - base: 4 FC layers with activation. Parent of `Generic`, `BottleneckGeneric`, `Seasonality`, `Trend`, `AutoEncoder`, `GenericAEBackcast`, `Wavelet`, `AltWavelet`.
  - `AERootBlock(nn.Module)` - autoencoder base: FC layers organized as encoder-decoder. Parent of `GenericAE`, `BottleneckGenericAE`, `TrendAE`, `SeasonalityAE`, `AutoEncoderAE`, `GenericAEBackcastAE`.
  - Wavelet blocks (`HaarWavelet`, `DB2Wavelet`, etc.) are thin subclasses of `Wavelet` or `AltWavelet` that only set the wavelet type.
- **`loaders.py`** - PyTorch Lightning DataModules and Datasets:
  - `TimeSeriesDataModule` / `TimeSeriesDataset` - single univariate series
  - `RowCollectionTimeSeriesDataModule` / `RowCollectionTimeSeriesDataset` - collection where rows=series, cols=observations (M4 format)
  - `ColumnarCollectionTimeSeriesDataModule` / `ColumnarTimeSeriesDataset` - collection where cols=series, rows=observations (Tourism format)
  - Test variants: `RowCollectionTimeSeriesTestModule`, `ColumnarCollectionTimeSeriesTestDataModule`
- **`losses.py`** - Custom loss functions: `SMAPELoss`, `MAPELoss`, `MASELoss`, `NormalizedDeviationLoss`
- **`constants.py`** - Valid string values for `ACTIVATIONS`, `LOSSES`, `OPTIMIZERS`, `BLOCKS`. Block types and losses are resolved by string lookup.
- **`data/M4/`** - M4Dataset loader with bundled CSV data files

### Key Design Patterns

- **String-based block dispatch**: `NBeatsNet.create_stack()` uses `getattr(b, stack_type)(...)` to instantiate blocks by name. Valid names are in `constants.BLOCKS`.
- **All blocks return `(backcast, forecast)` tuples**. The forward pass subtracts backcast from input (residual) and adds forecast to output.
- **`active_g` parameter**: Non-standard extension that applies activation to the final linear layers of Generic-type blocks. Helps convergence. Default `False` (paper-faithful).
- **Weight sharing**: When `share_weights=True`, blocks within a stack reuse the first block's parameters.
- **`sum_losses`**: Feature adding weighted backcast reconstruction loss (1/4 weight) to forecast loss. Penalizes non-zero residuals, pushing backcasts to fully reconstruct the input.
- **Generic vs BottleneckGeneric**: `Generic` matches the paper (single linear projection to target length). `BottleneckGeneric` uses a two-stage projection through `thetas_dim` bottleneck (rank-d factorized basis expansion).

### Width Parameters

Different block types use different width parameters: `g_width` (Generic/BottleneckGeneric, default 512), `s_width` (Seasonality, default 2048), `t_width` (Trend, default 256), `ae_width` (AutoEncoder, default 512). The `create_stack` method uses `stack_type in [...]` checks to select the appropriate width.

## CI/CD

GitHub Actions workflow (`.github/workflows/python-publish.yml`) publishes to PyPI on GitHub release using `pypa/gh-action-pypi-publish`.
