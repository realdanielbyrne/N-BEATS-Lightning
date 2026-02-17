# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyTorch Lightning implementation of the N-BEATS (Neural Basis Expansion Analysis for Time Series) forecasting algorithm, published as the `lightningnbeats` PyPI package. Extends the original paper with wavelet basis expansion blocks, autoencoder variants, bottleneck generic blocks, and fully customizable stack compositions.

### Requirements

- **Python** >= 3.12, < 3.15
- **PyTorch** >= 2.1.0
- **Lightning** >= 2.1.0

Supports multiple accelerators: **CUDA** (NVIDIA GPUs), **MPS** (Apple Silicon via Metal Performance Shaders), and **CPU**. The `get_best_accelerator()` utility in `__init__.py` detects the best available accelerator (CUDA > MPS > CPU).

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
python examples/M4AllBlks.py       # M4 dataset benchmark across all block types
python examples/TourismAllBlks.py  # Tourism dataset benchmark
```

The `experiments/run_experiments.py` script runs systematic benchmarks with multiple seeds across datasets:

```bash
python experiments/run_experiments.py --dataset m4 --part 1 --periods Yearly --max-epochs 50
python experiments/run_experiments.py --dataset traffic --part 1 --periods Traffic-96 --max-epochs 100
python experiments/run_experiments.py --dataset m4 --part 2 --periods Yearly Monthly --max-epochs 100
python experiments/run_experiments.py --dataset m4 --part all
python experiments/run_experiments.py --part 6 --max-epochs 100
```

`--dataset`: `m4` (default), `traffic`, or `weather`. `--part 1`: Block-type benchmark (paper baselines + novel blocks at 30-stack scale). `--part 2`: Ablation studies (active_g, sum_losses, activations). `--part 6`: Convergence study (ignores `--dataset`; runs across both M4-Yearly and Weather-96 with random seeds). `--convergence-config`: filter to a single config for parallel Part 6 execution. `--periods`: one or more of `Yearly`, `Quarterly`, `Monthly`, `Weekly`, `Daily`, `Hourly` for M4; `Traffic-96` for Traffic; `Weather-96` for Weather. Results are saved to dataset-specific subdirectories under `experiments/results/<dataset>/`.

## Testing

Tests are in `tests/` and use pytest:

```bash
pytest tests/                        # run all tests
pytest tests/ -v                     # verbose output
pytest tests/test_blocks.py          # run specific test file
pytest tests/test_blocks.py -k "TestGenericArchitecture"  # run single test class
pytest tests/test_blocks.py -k "test_output_shapes"       # run single test method
```

Test files: `test_blocks.py` (block shapes, attributes, registries), `test_loaders.py` (DataModule setup, splits), `test_models.py` (width selection, optimizer dispatch, forward pass, sum_losses). Note: CI does not run tests before publishing. No linter or formatter is configured.

## Architecture

### Package Structure (`src/lightningnbeats/`)

- **`models.py`** — `NBeatsNet(pl.LightningModule)`: the main model class. Accepts a `stack_types` list of strings to define architecture. Handles forward pass with backward/forward residual connections, training/validation/test steps, loss configuration, and optimizer setup.
- **`blocks/blocks.py`** — All block implementations (~1086 lines, the largest file). Two parallel inheritance hierarchies:
  - `RootBlock(nn.Module)` — Standard backbone: 4 FC layers with activation. Parent of `Generic`, `BottleneckGeneric`, `Seasonality`, `Trend`, `AutoEncoder`, `GenericAEBackcast`, `Wavelet`, `AltWavelet`, and all concrete wavelet subclasses.
  - `AERootBlock(nn.Module)` — Autoencoder backbone: encoder (units → units/2 → latent_dim) then decoder (latent_dim → units/2 → units). Parent of `GenericAE`, `BottleneckGenericAE`, `TrendAE`, `SeasonalityAE`, `AutoEncoderAE`, `GenericAEBackcastAE`.
  - Wavelet blocks (`HaarWavelet`, `DB2Wavelet`, etc.) are thin subclasses that only set the wavelet type string. `Wavelet` uses a square basis with learned downsampling; `AltWavelet` uses a rectangular basis with direct output.
  - Basis generators (`_SeasonalityGenerator`, `_TrendGenerator`, `_WaveletGenerator`, `_AltWaveletGenerator`) produce non-trainable basis matrices registered as buffers.
- **`loaders.py`** — PyTorch Lightning DataModules and Datasets. Two data layout conventions:
  - **Row-oriented** (M4 format): rows = series, cols = time observations. `RowCollectionTimeSeriesDataModule` splits by time dimension; validation = last `backcast + forecast` columns.
  - **Columnar** (Tourism format): cols = series, rows = time observations. `ColumnarCollectionTimeSeriesDataModule` supports `no_val` mode. Short series are padded with zeros.
  - `TimeSeriesDataModule` / `TimeSeriesDataset` — single univariate series with 80/20 random split.
  - Test variants (`RowCollectionTimeSeriesTestModule`, `ColumnarCollectionTimeSeriesTestDataModule`) concatenate train tail + test head for evaluation.
- **`losses.py`** — Custom loss functions: `SMAPELoss`, `MAPELoss`, `MASELoss`, `NormalizedDeviationLoss`
- **`constants.py`** — String registries: `ACTIVATIONS`, `LOSSES`, `OPTIMIZERS`, `BLOCKS`. All configuration is resolved by string lookup against these lists.
- **`data/benchmark_dataset.py`** — `BenchmarkDataset(ABC)`: abstract base class all datasets implement. Interface: `train_data`, `test_data`, `forecast_length`, `frequency`, `name`, `supports_owa`, `compute_owa()`, `get_training_series()`.
- **`data/M4/`** — `M4Dataset(BenchmarkDataset)` loader with bundled CSV data files for all 6 M4 periods. Includes Naive2 baseline constants and OWA computation.
- **`data/Traffic/`** — `TrafficDataset(BenchmarkDataset)` loader for PeMS Traffic (862 sensors, hourly). Downloads from Hugging Face on first use, caches at `~/.cache/lightningnbeats/Traffic/`. Parameterized by `horizon` (96, 192, 336, 720).
- **`data/Weather/`** — `WeatherDataset(BenchmarkDataset)` loader for Weather (21 meteorological indicators, 10-min intervals). Downloads from Hugging Face on first use, caches at `~/.cache/lightningnbeats/Weather/`. Parameterized by `horizon` (96, 192, 336, 720).

### Key Design Patterns

- **`stack_types` is required**: `NBeatsNet` raises `ValueError` if `stack_types` is not provided. There is no default architecture.
- **String-based block dispatch**: `NBeatsNet.create_stack()` uses `getattr(b, stack_type)(...)` to instantiate blocks by name. Valid names must appear in `constants.BLOCKS`.
- **All blocks return `(backcast, forecast)` tuples**. The forward pass subtracts backcast from input (residual) and adds forecast to output.
- **`active_g` parameter**: Non-standard extension that applies activation to the final linear layers of Generic-type blocks. Default `False` (paper-faithful).
- **Weight sharing**: When `share_weights=True`, blocks within a stack reuse the first block's parameters.
- **`sum_losses`**: Adds weighted backcast reconstruction loss (0.25 × loss vs zeros) to forecast loss, pushing backcasts to fully reconstruct the input.
- **Generic vs BottleneckGeneric**: `Generic` matches the paper (single linear projection to target length). `BottleneckGeneric` projects through `thetas_dim` bottleneck first (rank-d factorized basis expansion).

### Width Parameter Mapping

The `create_stack` method selects hidden layer width by block type:

| Width param | Default | Block types |
|---|---|---|
| `g_width` | 512 | `Generic`, `BottleneckGeneric`, `GenericAE`, `BottleneckGenericAE`, `GenericAEBackcast`, `GenericAEBackcastAE`, all wavelet blocks |
| `s_width` | 2048 | `Seasonality`, `SeasonalityAE` |
| `t_width` | 256 | `Trend`, `TrendAE` |
| `ae_width` | 512 | `AutoEncoder`, `AutoEncoderAE` |

### Adding a New Block Type

1. Create the block class in `blocks/blocks.py`, inheriting from `RootBlock` or `AERootBlock`. Must implement `forward()` returning `(backcast, forecast)`.
2. Add the class name string to the `BLOCKS` list in `constants.py`.
3. If the block needs a new width parameter, add the mapping in `NBeatsNet.create_stack()` in `models.py`.
4. Add a shape test in `tests/test_blocks.py` — the parametrized `TestAllBlocksOutputShapes` will automatically cover it if it's in `BLOCKS`.

### Adding a New Dataset

1. Create a new directory under `src/lightningnbeats/data/<DatasetName>/` with `__init__.py` and a dataset module.
2. Create a class inheriting from `BenchmarkDataset` in `data/benchmark_dataset.py`. Must set `train_data` (DataFrame, columnar), `test_data`, `forecast_length`, `frequency`, and `name`.
3. Override `supports_owa = True` and `compute_owa()` if Naive2 baselines exist.
4. Add the import to `src/lightningnbeats/data/__init__.py`.
5. In `experiments/run_experiments.py`: add a horizons dict (e.g. `NEW_HORIZONS`), add to `DATASET_DEFAULTS`, extend `load_dataset()`, and add the dataset choice to the CLI `--dataset` argument.

## CI/CD

GitHub Actions workflow (`.github/workflows/python-publish.yml`) publishes to PyPI on GitHub release using `pypa/gh-action-pypi-publish`. No tests or linting run in CI.
