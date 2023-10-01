# N-BEATS Algorithm

N-BEATS is a neural network based model for univariate time series forecasting. It stands for Neural Basis Expansion Analysis for Time Series. It was proposed by Boris N. Oreshkin and his co-authors at Element AI in 2019¹. N-BEATS consists of a deep stack of fully connected layers with backward and forward residual connections. The model can learn a set of basis functions that can decompose any time series into interpretable components, such as trend and seasonality. N-BEATS can also handle a wide range of forecasting problems without requiring any domain-specific modifications or feature engineering. N-BEATS has achieved state-of-the-art performance on several benchmark datasets, such as M3, M4, and TOURISM. This repository provides an implementation of N-BEATS in PyTorch Lightning, along with the code to reproduce the experimental results using the M4 dataset which is included as a reference in this repository. You can also use this implementation to apply N-BEATS to your own time series data and explore its capabilities. 

## N-BEATS Algorithm
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

## N-BEATS Lightning

This repository provides an implementation of N-BEATS in [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/). PyTorch Lightning is a lightweight PyTorch wrapper for high-performance AI research. It provides a high-level interface for PyTorch that makes it easier to train models, while still giving you the flexibility to customize your training loop. The N-BEATS implementation in this repository is based on the original paper.  It is designed to be easy to use and extend, so you can apply N-BEATS to your own time series data and explore its capabilities.