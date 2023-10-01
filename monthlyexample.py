#%%
import pandas as pd

# Load the Monthly-train.csv and M4-info.csv files
monthly_data = pd.read_csv('/mnt/data/Monthly-train.csv')
metadata = pd.read_csv('/mnt/data/M4-info.csv')

# Display the first few rows of each dataset
(monthly_data.head(), metadata.head())


#%%
import os

# Get the size of the Monthly-train.csv file
file_size = os.path.getsize('/mnt/data/Monthly-train.csv') / (1024 * 1024)  # size in MB
file_size

#%%
# Load a small portion of the Monthly-train.csv file to understand its structure
monthly_data_sample = pd.read_csv('/mnt/data/Monthly-train.csv', nrows=5)

# Load the M4-info.csv file to understand its structure
metadata_sample = pd.read_csv('/mnt/data/M4-info.csv')

(monthly_data_sample, metadata_sample.head())

#%%
import matplotlib.pyplot as plt
import numpy as np

# Function to plot individual time series
def plot_time_series(data, ts_ids):
    plt.figure(figsize=(10, 6))
    for ts_id in ts_ids:
        # Get the time series data
        ts_data = data[data['V1'] == ts_id].iloc[:, 1:].squeeze()
        # Drop any NaN values
        ts_data = ts_data.dropna()
        # Create a time axis
        time_axis = pd.date_range(start='1/1/2000', periods=len(ts_data), freq='M')
        plt.plot(time_axis, ts_data, label=ts_id)
    plt.title('Time Series Plot')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# Randomly select a few time series IDs
np.random.seed(0)  # for reproducibility
random_ts_ids = np.random.choice(monthly_data_sample['V1'], 3)

# Plot the time series
plot_time_series(monthly_data_sample, random_ts_ids)

#%%
import matplotlib.pyplot as plt
import numpy as np

# Function to plot individual time series
def plot_time_series(data, ts_ids):
    plt.figure(figsize=(10, 6))
    for ts_id in ts_ids:
        # Get the time series data
        ts_data = data[data['V1'] == ts_id].iloc[:, 1:].squeeze()
        # Drop any NaN values
        ts_data = ts_data.dropna()
        # Create a time axis
        time_axis = pd.date_range(start='1/1/2000', periods=len(ts_data), freq='M')
        plt.plot(time_axis, ts_data, label=ts_id)
    plt.title('Time Series Plot')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# Randomly select a few time series IDs
np.random.seed(0)  # for reproducibility
random_ts_ids = np.random.choice(monthly_data_sample['V1'], 3)

# Plot the time series
plot_time_series(monthly_data_sample, random_ts_ids)

#%%
# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(monthly_values, bins=50, edgecolor='k', alpha=0.7)
plt.title('Histogram of Monthly Time Series Values')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

#%%
# Function to process data in chunks and collect values for box plot
def process_chunks_for_boxplot(file_path, chunksize=1000):
    values = []
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        # Drop the time series identifier column and NaN values
        chunk_values = chunk.iloc[:, 1:].values
        # Flatten the array and filter out NaN values
        chunk_values = chunk_values[~np.isnan(chunk_values)]
        values.extend(chunk_values)
    return values

# Collect values for box plot
monthly_values = process_chunks_for_boxplot('/mnt/data/Monthly-train.csv')

# Create box plot
plt.figure(figsize=(10, 6))
plt.boxplot(monthly_values, vert=False)
plt.title('Box Plot of Monthly Time Series Values')
plt.xlabel('Value')
plt.yticks([])  # Hide the y-axis ticks as there's only one box
plt.grid(True)
plt.show()

#%%
# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(monthly_values, bins=50, edgecolor='k', alpha=0.7)
plt.title('Histogram of Monthly Time Series Values')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

#%%
import seaborn as sns

# Function to process data in chunks and collect data for heatmap
def process_chunks_for_heatmap(file_path, num_ts=10, chunksize=1000):
    heatmap_data = None
    ts_count = 0
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        # Stop if we have collected enough time series
        if ts_count >= num_ts:
            break
        # Drop the time series identifier column and NaN values
        chunk = chunk.iloc[:, 1:].dropna(axis=1, how='all')
        # If we haven't collected enough time series yet, add the current chunk to the heatmap data
        if heatmap_data is None:
            heatmap_data = chunk
        else:
            heatmap_data = pd.concat([heatmap_data, chunk], axis=0)
        ts_count += len(chunk)
    return heatmap_data

# Collect data for heatmap
heatmap_data = process_chunks_for_heatmap('/mnt/data/Monthly-train.csv')

# Create heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap='viridis', yticklabels=False, xticklabels=500)
plt.title('Heatmap of Monthly Time Series Values')
plt.xlabel('Time')
plt.ylabel('Time Series')
plt.show()

#%%
from statsmodels.tsa.seasonal import seasonal_decompose

# Function to perform seasonal decomposition on a selected time series
def decompose_time_series(data, ts_id):
    # Get the time series data
    ts_data = data[data['V1'] == ts_id].iloc[:, 1:].squeeze().dropna()
    # Create a time axis
    time_axis = pd.date_range(start='1/1/2000', periods=len(ts_data), freq='M')
    # Create a time series object
    ts_obj = pd.Series(data=ts_data.values, index=time_axis)
    # Perform seasonal decomposition
    decomposition = seasonal_decompose(ts_obj, model='multiplicative')
    return decomposition

# Perform seasonal decomposition on a randomly selected time series
decomposition = decompose_time_series(monthly_data_sample, random_ts_ids[0])

# Plot the seasonal decomposition
plt.figure(figsize=(10, 8))
plt.subplot(411)
plt.plot(decomposition.trend, label='Trend')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(decomposition.seasonal, label='Seasonal')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(decomposition.residual, label='Residual')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(decomposition.observed, label='Observed')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

#%%
# Re-run the seasonal decomposition

# Perform seasonal decomposition on a randomly selected time series
decomposition = decompose_time_series(monthly_data_sample, random_ts_ids[0])

# Plot the seasonal decomposition
plt.figure(figsize=(10, 8))
plt.subplot(411)
plt.plot(decomposition.trend, label='Trend')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(decomposition.seasonal, label='Seasonal')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(decomposition.residual, label='Residual')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(decomposition.observed, label='Observed')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

#%%
pip install statsmodels


#%%
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Assume df is your data frame and 'ts' is the column with your time series data
# Create a time series object
ts_obj = pd.Series(data=df['ts'].values, index=pd.date_range(start='1/1/2000', periods=len(df), freq='M'))

# Perform seasonal decomposition
decomposition = seasonal_decompose(ts_obj, model='multiplicative')

# Plot the seasonal decomposition
plt.figure(figsize=(10, 8))
plt.subplot(411)
plt.plot(decomposition.trend, label='Trend')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(decomposition.seasonal, label='Seasonal')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(decomposition.residual, label='Residual')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(decomposition.observed, label='Observed')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

#%%
from IPython.display import Markdown as md

# Introduction Section
introduction = """
# Introduction to N-BEATS with Pytorch Lightning

N-BEATS is a neural network based time series forecasting algorithm that was introduced in a paper by Oreshkin, et al. The algorithm has shown strong performance on a variety of time series forecasting benchmarks.

In this notebook, we will explore the Pytorch Lightning implementation of the N-BEATS algorithm, and apply it to the Monthly dataset from the M4 Time series analysis competition.

## N-BEATS Algorithm
N-BEATS consists of a stack of fully connected neural network blocks, where each block is trained to forecast the target time series. The unique aspect of N-BEATS is its ability to handle a wide range of time series forecasting problems without requiring any prior knowledge about the data.

## M4 Time Series Analysis Competition
The M4 competition is a well-known benchmark in the time series forecasting community. It provides a large dataset of time series from various domains, which allows for a thorough evaluation of forecasting algorithms.

## Dataset
The dataset we will be using is the Monthly dataset from the M4 competition. This dataset contains monthly time series from various domains.

Let's start by loading and exploring the dataset.
"""

# Display the introduction section
md(introduction)


#%%
# Dataset Section
dataset_section = """
# Dataset

The Monthly dataset from the M4 competition contains a collection of time series data with monthly frequency. Each time series represents a unique sequence of data points collected or recorded at a monthly interval.

## Loading the Dataset

Let's start by loading the Monthly dataset and taking a look at the first few rows.

```python
import pandas as pd

# Load the Monthly dataset
monthly_data = pd.read_csv('path_to/Monthly-train.csv')

# Display the first few rows of the dataset
monthly_data.head()
Exploratory Data Analysis (EDA)

# Exploratory Data Analysis (EDA)
Basic exploratory data analysis can provide insights into the structure and properties of the dataset.

# Summary statistics
summary_stats = monthly_data.describe()

# Missing values
missing_values = monthly_data.isnull().sum()

# Number of unique time series
num_unique_ts = monthly_data['V1'].nunique()

summary_stats, missing_values, num_unique_ts
"""
# Display the dataset section
md(dataset_section)


#%% 
# Model Implementation Section
model_implementation_section = """
# Model Implementation

The N-BEATS algorithm is implemented using PyTorch Lightning, which is a high-level interface for PyTorch. PyTorch Lightning provides a structured framework for training, evaluating, and testing models, which simplifies the process of implementing complex models like N-BEATS.

## N-BEATS Model Structure

Let's explore the structure of the N-BEATS model as implemented in the `nbeats.py` file.

```python
# Assuming nbeats.py is located in the current directory
# Load the nbeats.py file and display its content
with open('path_to/nbeats.py', 'r') as file:
    nbeats_code = file.read()

print(nbeats_code)
This code will load and display the content of the nbeats.py file, which contains the implementation of the N-BEATS model. The model is defined as a PyTorch Lightning module, which allows for easy training, evaluation, and testing.

Model Components

The N-BEATS model consists of several components, including:

Stack: A stack is a sequence of blocks, where each block is a small neural network. The outputs of the blocks are combined to produce the final forecast.
Block: A block is a small neural network that forecasts the target time series or the residuals of the previous blocks.
Backcast and Forecast: The model produces a backcast, which is used to train the model, and a forecast, which is the output of the model.
Each of these components is implemented as a separate class or function in the nbeats.py file.

In the next section, we will train the N-BEATS model on the Monthly dataset and evaluate its performance.
 """
#Display the model implementation section
md(model_implementation_section)




# Training and Evaluation Section

training_evaluation_section = """
# Training and Evaluation

The `m4_example.py` file provides a script for training and evaluating the N-BEATS model on the Monthly dataset from the M4 competition. It sets up the data loaders, the model, the optimizer, and the training loop, and evaluates the model on the test data.

## Training the Model

Let's take a look at the code for training the N-BEATS model.

```python
  # Assuming m4_example.py is located in the current directory
  # Load the m4_example.py file and display its content
  with open('path_to/m4_example.py', 'r') as file:
      m4_example_code = file.read()

  print(m4_example_code)

You can run the above code to display the content of the m4_example.py file, which provides a script for training the N-BEATS model.

Evaluating the Model

The m4_example.py file also includes code for evaluating the model on the test data from the M4 competition. It computes the forecast for each time series in the test data and compares it to the ground truth to compute the error.

You can execute the m4_example.py file in your own environment to train and evaluate the N-BEATS model on the Monthly dataset.

In the next section, we will take a look at the results and conclude our exploration of the N-BEATS model and the Monthly dataset.

"""
#Display the training and evaluation section
md(training_evaluation_section)

#%%

# Results and Conclusion Section
results_conclusion_section = """
# Results and Conclusion

After training and evaluating the N-BEATS model on the Monthly dataset, you will obtain forecasts for each time series in the test data. You can visualize these forecasts and compare them to the ground truth to assess the performance of the model.

## Interpreting the Results

You can use various metrics to evaluate the accuracy of the forecasts, such as the Mean Absolute Error (MAE), Mean Squared Error (MSE), or Symmetric Mean Absolute Percentage Error (sMAPE). These metrics provide a quantitative measure of the accuracy of the forecasts.

```python
# Assuming predictions is an array of forecasts and ground_truth is an array of true values
# Compute evaluation metrics
mae = np.mean(np.abs(predictions - ground_truth))
mse = np.mean((predictions - ground_truth)**2)
smape = 100 * np.mean(2 * np.abs(predictions - ground_truth) / (np.abs(predictions) + np.abs(ground_truth)))

mae, mse, smape

Conclusion

In this notebook, we have introduced the Pytorch Lightning implementation of the N-BEATS algorithm, explored the Monthly dataset from the M4 competition, and discussed the training and evaluation of the model using the m4_example.py script. The N-BEATS model is a powerful tool for time series forecasting, and its modular architecture allows for easy adaptation to different forecasting problems.

Feel free to further explore the model, experiment with different configurations, and apply it to other time series forecasting problems.
"""
 
#Display the results and conclusion section
md(results_conclusion_section)

