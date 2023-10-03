# N-BEATS Lightning

This repository provides an implementation of N-BEATS in [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/). PyTorch Lightning is a lightweight PyTorch wrapper for high-performance AI research. It provides a high-level interface for PyTorch that makes it easier to train models, while still giving you the flexibility to customize your training loop. 

The N-BEATS implementation in this repository is based on the original [paper](https://arxiv.org/pdf/1905.10437.pdf.).  This model is designed to be easy to use and extend, and so you can apply N-BEATS to your own time series data and explore its capabilities.

## N-BEATS Algorithm

N-BEATS is a neural network based model for univariate time series forecasting. It stands for Neural Basis Expansion Analysis for Time Series. It was proposed by Boris N. Oreshkin and his co-authors at Element AI in 2019. N-BEATS consists of a deep stack of fully connected layers with backward and forward residual connections. The model can learn a set of basis functions that can decompose any time series into interpretable components, such as trend and seasonality. N-BEATS can also handle a wide range of forecasting problems without requiring any domain-specific modifications or feature engineering. N-BEATS has achieved state-of-the-art performance on several benchmark datasets, such as M3, M4, and TOURISM. This repository provides an implementation of N-BEATS in PyTorch Lightning, along with the code to reproduce the experimental results using the M4 dataset which is included as a reference in this repository. 

Here are some key points about the N-BEATS algorithm:

- **Block Architecture**:
  N-BEATS consists of a stack of fully connected neural networks called "blocks." Each block processes the input time series data and outputs a set of forecasts along with a backcast, which is the reconstructed version of the input.

- **Generic and Interpretable Blocks**:
  There are two types of blocks within N-BEATS: Generic and Interpretable. Generic blocks are designed to learn the underlying patterns in the data automatically, while Interpretable blocks incorporate prior knowledge about the data and are structured to provide insights into the learned patterns.

- **Stacked Ensemble**:
  The blocks are stacked together in an ensemble, and their forecasts are combined to produce the final prediction. This ensemble approach allows N-BEATS to handle a wide range of time series forecasting problems effectively.

- **Parameter Sharing and Scalability**:
  N-BEATS is designed with parameter sharing across the blocks, which promotes scalability and efficiency in training and inference.

- **Performance**:
  N-BEATS has shown state-of-the-art performance on a variety of benchmark time series forecasting datasets, making it a robust choice for many forecasting applications.

The N-BEATS algorithm is a powerful tool for time series forecasting, providing a blend of automatic learning, interpretability, and robust performance across different domains.

## Getting Started

**Installation**
```bash
  pip install nbeats-pytorch-lightning
```

First load the required libraries and your data.

```python
# Import necessary libraries
from nbeats_lightning.nbeats import *
from nbeats_lightning.loaders import *
import pandas as pd

# Load the milk.csv dataset
milk = pd.read_csv('data/milk.csv', index_col=0)
milkval = milk.values.flatten() # flat numpy array
milk.head()
```

Define the model and its hyperparameters. This model will forecast 6 steps into the future. The common practice is to use a multiple of teh forecast horizon for the backcast length.  In this case, we will use 4 times the forecast horizon. Larger batch sizes will result in faster training, but may require more memory.  The number of blocks per stack is a hyperparameter that can be tuned.  The share_weights parameter is set to True to share weights across the blocks.

```python
# Define hyperparameters
forecast_length = 6
backcast_length = 4 * forecast_length
batch_size = 64
n_blocks_per_stack = 3


# An Interpretable N-Beats Model, 
#  - 2 stacks (Fixed at 2 stacks)
#     - 3 trend(256) blocks in first stack (default size)
#     - 3 seasonality(2048) in second stack (default size)
interpretable_milkmodel = NBeatsNet(
  backcast = backcast_length,
  forecast = forecast_length, 
  generic_architecture = False,
  n_blocks_per_stack = n_blocks_per_stack,
  share_weights = True  
)

```

Train the model. The model will be trained for 500 epochs.  The model will be trained on the GPU if one is available.

```python
interpretable_trainer =  pl.Trainer(
  accelerator='auto' # use GPU if available
  ,max_epochs=500
)

interpretable_trainer.fit(interpretable_milkmodel, datamodule=dm)
interpretable_trainer.validate(interpretable_milkmodel, datamodule=dm)  
```

## Using CUDA

If you have a CUDA capable GPU, you will want to install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) and the [PyTorch](https://pytorch.org/get-started/locally/) version that works with the toolkit.  Currently PyTorch only supports CUDA versions 11.7 and 11.8.  Installing these pacakged will allow you to train your model on the GPU.  You can check if you have a CUDA capable GPU by running the following command in your terminal:

```bash
  $ nvidia-smi
```

or in python environment

```python
  import torch
  torch.cuda.is_available()
```

## N-BEATS Extensions and Variations in this Repository

This repository provides an implementation of N-BEATS in PyTorch Lightning. The implementation is based on the original [paper](https://arxiv.org/pdf/1905.10437.pdf). However, the implementation in this repository has been extended to include the following features:

### ActiveG

This parameter when enabled applies the model's activation funtion to the linear funtions (gb and gf) which are found by the network in the last layer of each block using the functions' parameters found in the preceding layer. The parameter `active_g` is not a feature found in the original N-Beats paper.

You can enable this feature by setting `active_g` to `True`.  Enabling this activation function helps the Generic model converge.  Generally this results in a comparably accurate model in fewer training cycles.  Also, Generic models might sometimes not converge at all.  The fix or this would typically be to add or remove a stack, layers, or units per layer, to give the model more/less capacity or to just try retraining. However, enabling this parameter usualy fixes the problem without the need to modify any other parameters.

The intuition behind the inclusion of this parameter is that the generic model as originally designed connects two layers 
of Linear fully conencted nodes, the first to find the parameters of an expansion polynomial function and the second to find the functions that best fit the forecast and backcast outputs of the block. However, linear layers without activations are not able to learn non-linear functions.  This parameter allows the model to learn non-linear functions by applying the activation function to the linear functions found by the model in the last layer of each block.  This is concsistent with the interpretable arcitecture since the basis functions are also non-linear, and so this feature allows the interpretable and generic models to be more similar.