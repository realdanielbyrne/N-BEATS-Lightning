# N-BEATS Lightning

N-BEATS Lightning is an implementation of [N-BEATS](https://arxiv.org/pdf/1905.10437.pdf) time series forecasting architecture in [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) with some additional experimental features such as Wavelet Basis Expansion blocks and completely customizable stacks.  

## N-BEATS Algorithm

N-BEATS, Neural Basis Expansion Analysis for Time Series, is a neural network based model for univariate time series forecasting. It was proposed by Boris N. Oreshkin and his co-authors at ElementAI in 2019. N-BEATS consists of a deep stack of fully connected layers with backward and forward residual connections. The model can learn a set of basis functions that can decompose any time series into interpretable components, such as trend and seasonality. N-BEATS can also handle a wide range of forecasting problems without requiring any domain-specific modifications, scaling, or feature engineering. N-BEATS has achieved state-of-the-art performance on several benchmark datasets, such as M3, M4, and TOURISM. This repository provides an implementation of N-BEATS in PyTorch Lightning, along with the code to reproduce the experimental results using the M4 and Tourism datasets which are included as a reference in this repository.

Here are some key points about the N-BEATS algorithm:

- **Block Architecture**:
  N-BEATS consists of a stack of fully connected neural networks called "blocks." Each block processes the input time series data and outputs a set of forecasts along with a backcast, which is the reconstructed version of the input.

- **Generic and Interpretable Blocks**:
  There are two types of blocks within N-BEATS: Generic and Interpretable. Generic blocks are designed to learn the underlying patterns in the data automatically, while Interpretable blocks incorporate prior knowledge about the data and are structured to provide insights into the learned patterns.

- **Stacked Ensemble**:
  The blocks are stacked together in an ensemble, and their forecasts are combined to produce the final prediction. This ensemble approach allows N-BEATS to handle a wide range of time series forecasting problems effectively.

- **Parameter Sharing and Scalability**:
  N-BEATS is designed with parameter sharing across the blocks, which promotes scalability and efficiency in training and inference.

- **Fast Learning**:
  N-BEATS is a fast learner, and it can be trained in a few epochs on a single GPU. This makes it easy to experiment with different hyperparameters and architectures. It generally settles quickly into a relative minimum. Since many models can be trained quickly, it is easy to build an ensemble of differnt models to improve performance and generalization.

The N-BEATS algorithm is a powerful tool for time series forecasting, providing a blend of automatic learning, interpretability, and robust performance across different domains.

## Requirements

- **Python** >= 3.12
- **PyTorch** >= 2.1.0
- **Lightning** >= 2.1.0

## Getting Started

### Installation

Download the source from the [github repository](https://github.com/realdanielbyrne/N-BEATS-Lightning) or install it as a pip package to use in your project using the following command:

```bash
pip install lightningnbeats
```

To install from source for development:

```bash
git clone https://github.com/realdanielbyrne/N-BEATS-Lightning.git
cd N-BEATS-Lightning
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

### Simple Example

The following is a simple example of how to use this model.

#### Load Data

First load the required libraries and your data.

```python
# Import necessary libraries
from lightningnbeats import NBeatsNet
from lightningnbeats.loaders import *
import pandas as pd

# Load the milk.csv dataset
milk = pd.read_csv('src/data/milk.csv', index_col=0)
milkvals = milk.values.flatten() 
```

#### Define the model and its hyperparameters

Define the model by defining the architecture in the `stack_types` parameter.  The `stack_types` parameter is simply a list of strings that specify the type of block to use in each stack.  The following block types are available:

- Generic
- GenericAE
- GenericAEBackcast
- GenericAEBackcastAE
- Trend
- TrendAE
- Seasonality
- SeasonalityAE
- AutoEncoder
- AutoEncoderAE
- HaarWavelet
- DB2Wavelet
- DB2AltWavelet
- DB3Wavelet
- DB3AltWavelet
- DB4Wavelet
- DB4AltWavelet
- DB10Wavelet
- DB10AltWavelet
- DB20Wavelet
- DB20AltWavelet
- Coif1Wavelet
- Coif1AltWavelet
- Coif2Wavelet
- Coif2AltWavelet
- Coif3Wavelet
- Coif10Wavelet
- Symlet2Wavelet
- Symlet2AltWavelet
- Symlet3Wavelet  
- Symlet10Wavelet
- Symlet20Wavelet

This implementation extends the design original paper with several additional block types and by allowing any combination blocks in any order simply by specifying the block types in the stack_types parameter.  

```python
forecast_length = 6
backcast_length = 4 * forecast_length
batch_size = 64
n_stacks = 6

interpretable_milkmodel = NBeatsNet(
  stack_types=['Trend', 'Seasonality'],
  backcast_length = backcast_length,
  forecast_length = forecast_length, 
  n_blocks_per_stack = 3,
  thetas_dim = 5,    
  t_width=256,  
  s_width=2048,
  share_weights = True
)
```

This model will forecast 6 steps into the future. The common practice is to use a multiple of the forecast horizon for the backcast length.  In this case, we will use 4 times the forecast horizon.

Larger batch sizes will result in faster training, but may require more memory.  The number of blocks per stack is a hyperparameter that can be tuned.  The share_weights parameter is set to True to share weights across the blocks. Gerneally deeper stacks do not result in more accurate predictions as the model saturates pretty quickly.  To improve accuracy it is best to build moultiple models from differnt architectures and combine them in an ensemble by taking the forecast from each and using something like the mean or median of the combined results.

#### Define a Pytorch Lightning DataModule

Instantiate one of the predefined PyTorch Lightning Time Series Data Modules in this repository to help organize and load your data.

- *TimeSeriesDataModule* - A PyTorch Lightning DataModule that takes a univariate time series as input and returns batches of samples of the time series. This is the most basic DataModule and is useful for single univariate time series data.
- *RowCollectionTimeSeriesDataModule* - A PyTorch Lightning DataModule accepts a dataset that is a collection of time series organized into rows where each row represents a time series, and each column represents subsequent observations. For instance this is how the M4 dataset is organized.
- *ColumnarCollectionTimeSeriesDataModule* - A PyTorch Datamodule that takes a collection of time series as input and returns batches of samples. The input dataset is a collection of time series organized such that columns represent individual time series and rows represent subsequent observations. This is how the Tourism dataset is organized.

```python
dm = TimeSeriesDataModule(
  train_data = milkvals[:-forecast_length],
  val_data = milkvals[-(forecast_length + backcast_length):],
  batch_size = batch_size,
  backcast_length = backcast_length,
  forecast_length = forecast_length,
  shuffle = True
)

```

#### Define a Pytorch Lightning ModelCheckpoint (optional)

Define a Pytorch Ligntning ModelCheckpoint callback to save the best model during training.  The model checkpoints will be saved to the default  `lightning/logs` directory unless otherwise specified.  

```python
i_chk_callback = ModelCheckpoint(
  save_top_k = 2, # save top 2 models
  monitor = "val_loss", # monitor validation loss as evaluation 
  mode = "min",
  filename = "{name}-{epoch:02d}-{val_loss:.2f}",
)

# Define a tensorboard logger
i_tb_logger = pl_loggers.TensorBoardLogger(save_dir="lightning_logs/", name=i_name)

```

#### Define a Pytorch Lightning Trainer and train the model

Use a standard Pytorch Lightning Trainer to train the model.  Be careful not to overtrain, as performance will begin to degrade after a certain point.  The N-BEATS architecture tends to settly fairly quickly into a relative minimum.   If the model is not performing well, try a different architecture or hyperparameters.  It is best to build an ensemble of models and combine the results to improve accuracy.  This can be done by taking the mean or median of the forecast from each model in the ensemble.

```python
interpretable_trainer =  pl.Trainer(
  accelerator='auto' # use GPU if available
  ,max_epochs=100
  ,callbacks=[i_chk_callback]  
  ,logger=[i_tb_logger]
)

interpretable_trainer.fit(interpretable_milkmodel, datamodule=dm)
interpretable_trainer.validate(interpretable_milkmodel, datamodule=dm) 
```

#### Hardware Acceleration (GPU Support)

This package supports multiple hardware accelerators. The PyTorch Lightning Trainer with `accelerator='auto'` will automatically select the best available device in this priority order: **CUDA GPU** > **Apple MPS** > **CPU**.

| Accelerator | Platform | Notes |
| --- | --- | --- |
| **CUDA** | NVIDIA GPUs (Linux/Windows) | Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) and the matching [PyTorch CUDA build](https://pytorch.org/get-started/locally/) |
| **MPS** | Apple Silicon (macOS) | Metal Performance Shaders — available out of the box on M1/M2/M3/M4 Macs with PyTorch >= 2.1.0 |
| **CPU** | Any | Fallback when no GPU is available |

You can check which accelerator is available on your system:

```python
from lightningnbeats import get_best_accelerator
print(get_best_accelerator())  # Returns 'cuda', 'mps', or 'cpu'
```

Or check individually:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available:  {torch.backends.mps.is_available()}")
```

The `get_best_accelerator()` utility function returns a string compatible with Lightning's `accelerator` parameter:

```python
from lightningnbeats import NBeatsNet, get_best_accelerator
import lightning.pytorch as pl

accelerator = get_best_accelerator()
trainer = pl.Trainer(accelerator=accelerator, max_epochs=100)
```

### Running Examples

Example scripts are in the `examples/` directory and can be run as standalone scripts or cell-by-cell in Jupyter (using `#%%` markers):

```bash
cd examples
python M4AllBlks.py       # M4 dataset benchmark across all block types
python TourismAllBlks.py  # Tourism dataset benchmark
```

### Running Tests

Tests use pytest:

```bash
pytest tests/              # run all tests
pytest tests/ -v           # verbose output
pytest tests/test_blocks.py  # run a specific test file
```

## N-BEATS Extensions and Variations

This repository provides an implementation of N-BEATS that has been extended to include extended features which can be used to augment the basic design with more advanced features.

### ActiveG

This parameter when enabled applies the model's activation function to the linear funtions (gb and gf) which are found by the network in the last layer of each block using the functions' parameters found in the preceding layer. The parameter `active_g` is not a feature found in the original N-Beats paper.

You can enable this feature by setting `active_g` to `True`.  Enabling this activation function helps the Generic model converge.  Generally this results in a comparably accurate model in fewer training cycles.  

Also, Generic models as defined in the original paper have a tendency to not converge.  This is likely due to the sucessive linear layers without activation functions in the final two layers of a standard Generic block. The fix or this would typically be to retrain, or add or remove a stack or a layer. However, enabling this parameter usualy fixes the problem without the need to modify any parameters or to adjust the chosen architecture.

The intuition behind the inclusion of this parameter is that the Generic model as originally designed connects two layers of Linear fully conected layers, the first is ostensibly designed to find the parameters of an expansion polynomial function and the second to find the functions that best fit the forecast and backcast outputs of the block using the parameters found in the prior layer. However, linear layers without activations are not able to learn non-linear functions.  This parameter allows the model to learn non-linear functions by applying the activation function to the linear functions found by the model in the last layer of each block.  

This modification to the original specification is however consistent with the original design.  Take for instance the Interpretable arcitecture as defined in the original paper.  The basis functions used in the Trend, Seasonality, and (in this version) the Wavlet blocks are by their nature non-linear functions. In the original design then, leaving activation funcitons off of the final two layers as depicted in the paper, is actually inconsistent with the Interpretible architecture. Therefore, this feature modifies the Generic model to be more similar in structure to the other basis function blocks.

### Wavelet Basis Expansion Blocks

This repository constains a number of experimental Wavelet Basis Expansion Blocks. Wavelet basis expansion is a mathematical technique used to represent signals or functions in terms of simpler, fixed building blocks called wavelets. Unlike Fourier transforms, which use sine and cosine functions as basis elements, wavelets can be localized in both time and frequency. This means they can represent both the frequency content of a signal and when these frequencies occur.

This method is particularly useful for analyzing functions or signals that contain features at multiple scales.  The multi-resolution analysis capability of wavelets is particularly suited to capturing the essence of time series data then, which can have complex, hierarchical structures due to the presence of trends, seasonal effects, cycles, and irregular fluctuations.

Wavelet blocks can be used in isolation or in combination with other blocks freely. For instance

```python
n_stacks = 5
stack_types = ['DB3Wavelet'] * n_stacks # 5 stacks of DB3Wavelet blocks
stack_types = ['Trend','DB3Wavelet'] * n_stacks # 5 stacks of 1 Trend and 1 DB3Wavelet
stack_types = ['DB3Wavelet','Generic'] # 5 stacks of 1 DB3Wavelet followed by 1 Generic
```

The Wavelet blocks avaiavlable in this repository are as follows:

- HaarWavelet
- DB2Wavelet
- DB2AltWavelet
- DB3Wavelet
- DB3AltWavelet
- DB4Wavelet
- DB4AltWavelet
- DB10Wavelet
- DB10AltWavelet
- DB20Wavelet
- DB20AltWavelet
- Coif1Wavelet
- Coif1AltWavelet
- Coif2Wavelet
- Coif2AltWavelet
- Coif3Wavelet
- Coif10Wavelet
- Symlet2Wavelet
- Symlet2AltWavelet
- Symlet3Wavelet  
- Symlet10Wavelet
- Symlet20Wavelet

### AutoEncoder Block

The AutoEncoder Block utilizes an AutoEncoider structure in both the forecast and backcast branches in the N-BEATS architecture.  The AutoEncoder block is useful for noisey time series data like Electric generation or in highly varied datasets like the M4.   It struggles with simpler more predictable datasets like the Milk Production Dataset taht rely on mostly trend.  However, combining this block with a Trend block often eliminates this problem.  

Like any other blocks in this implementation, the AutoEncoder block can be used in isolation or in combination with other blocks freely. For instance

```python
n_stacks = 5
stack_types = ['AutoEncoder'] * n_stacks # 5 stacks of AutoEncoder blocks 
stack_types = ['Trend','AutoEncoder'] * n_stacks # 5 stacks of 1 Trend block followed by 1 AutoEncoder block
```

### GenericAEBackcast

The GenericAEBackcast block is a Generic block that uses an AutoEncoder structure in only the backcast branch of the N-BEATS architecture.  This block is useful for noisey time series data like Electric generation or in highly varied datasets like the M4.   It doesn't struggle like the AutoEncoder block does with simpler more predictable datasets like the Milk Production Dataset.  It is genreally more accurate than the AutoEncoder block, and it settles on a solution faster.  

```python
n_stacks = 5
stack_types = ['GenericAEBackcast'] * n_stacks # 5 stacks of GenericAEBackcast blocks 
stack_types = ['Trend','GenericAEBackcast'] * n_stacks # 5 stacks of 1 Trend and 1 GenericAEBackcast block
```

### AERootBlock and its Variations

The base N-Beats model consists of a stack of fully connected layers at the head of every block before the signal is spilt into backcast and forecast branches.The AERootBlock is a variation of the root block that uses an AutoEncoder structure as opposed to a uniform stack of fully connected layers.  The AERootBlock is useful for noisey time series data like Electric generation or in highly varied datasets like the M4.

The AERootblock is the parent class of the following blocks:

- GenericAE
- GenericAEBackcastAE
- TrendAE
- SeasonalityAE
- AutoEncoderAE

The AERootBlock cannot be used in isolation since it is oly the first half of the blocks that are derived from it.
However, it's children blocks can be used in isolation or in combination with other blocks freely. For instance

```python
n_stacks = 5
stack_types = ['GenericAE'] * n_stacks # 5 stacks of GenericAE blocks
stack_types = ['TrendAE','GenericAE'] * n_stacks # 5 stacks of 1 TrendAE and 1 GenericAE block
```

### Sum Losses

The sum_losses experimental feature is an experimental feature which takes in consideration the sum of the losses of the forecast branch and reconstruction loss in the backcast branch of the model. The original implementation only considers the loss in the forecast branch.  If enabled, the total loss is defined as the sum of the forecast loss and a 1/4 of the backcast loss.

This feature is not included in the original N-Beats paper.  However, it is included in this implementation as an experimental feature.  It is not clear if this feature improves the performance of the model since more experimentation is needed.  To enable this feature set `sum_losses` to `True` in the model definition.  

```python
n_stacks = 5
TrendAutoEncoder_milkmodel = NBeatsNet(
  stack_types=['Trend', 'AutoEncoder']*n_stacks,
  backcast_length = backcast_length,
  forecast_length = forecast_length, 
  n_blocks_per_stack = 1,
  share_weights = True, 
  sum_losses = True
)
```
