import pandas as pd
import numpy as np
from nbeats_lightning.nbeats import *                   
from nbeats_lightning.loaders import *
from nbeats_lightning.losses import *
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import loggers as pl_loggers

def get_columnar_datamodules(
  df_train, 
  df_holdout, 
  backcast_length:int=8, 
  forecast_length:int=4, 
  batch_size:int=1024, 
  no_val:bool=False):
  
  dm = ColumnarCollectionTimeSeriesDataModule(
    df_train, 
    backcast_length=backcast_length,
    forecast_length=forecast_length,    
    no_val=no_val,
    batch_size=batch_size)

  test_dm = ColumnarCollectionTimeSeriesTestDataModule(
    df_train,
    df_holdout,
    backcast_length=backcast_length,
    forecast_length=forecast_length, 
    batch_size=batch_size)
  
  return dm, test_dm


def get_trainer(name, max_epochs:int=100, subdirectory :str="", **kwargs):
  """Returns a Pytorch Lightning Trainer object

  Args:
      name (string): The model name to be used for logging and checkpointing
      max_epocs (int, optional): The maximum number of epochs to train. Defaults to 100.
      subdirectory (string, optional): The subdirectory to save the logs. 
      Path to logs is always ./lightning_logs/{subdirectory}.

  Returns:
       pl.Trainer: A Pytorch Lightning Trainer object
  """
  # Define a model checkpoint callback
  chk_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="best-checkpoint",
    save_top_k = 1, 
    monitor = "val_loss", # monitor validation loss as evaluation 
    mode = "min"
  )

  # Define a tensorboard loger
  tb_logger = pl_loggers.TensorBoardLogger(save_dir="lightning_logs/tourism", name=name)

  # Train the generic model
  trainer =  pl.Trainer(
    accelerator='auto' # use GPU if available
    ,max_epochs=max_epochs
    ,callbacks=[chk_callback]  
    ,logger=[tb_logger],
    **kwargs
  )
  
  return trainer


def filter_dataset_by_removing_short_columnar_ts(data, backcast_length:int= 8, forecast_length:int=4):
  
  train_data = data.iloc[:-forecast_length, :]
  holdout_data = data.iloc[-forecast_length:, :].reset_index(drop=True)
  
  print("Filtering NANs, and sequences that are too short.")
  # Minimum length for each time series to be usable (backcast + forecast)
  min_length = forecast_length + backcast_length

  # Identify columns (time series) that have enough data points for training and validation
  valid_columns = [col for col in data.columns if train_data[col].dropna().shape[0] >= min_length]
  invalid_columns = [col for col in data.columns if train_data[col].dropna().shape[0] < min_length]
  
  # Create a new DataFrame with only the valid columns
  train_data  = train_data[valid_columns]

  print ("Dataframe shape of valid train entries :",train_data.shape)
  print("Dataframe shape of invalid entries :",data[invalid_columns].shape)
  print ("Dataframe shape of holdout entries :",holdout_data.shape)
  
  return train_data, holdout_data


def fill_columnar_ts_gaps(data, backcast_length:int= 8, forecast_length:int=4):
  """
  Fills gaps in time series data to ensure that each series in the DataFrame
  meets a minimum length requirement. The minimum length is determined by the
  sum of the backcast and forecast lengths. Missing values in series that
  don't meet the minimum length are filled with the median of the available
  values in that specific series.
  
  Parameters:
  - data (pd.DataFrame): The DataFrame containing time series data. Each
    column represents an individual time series.
  - backcast_length (int): The number of past observations required for each
    time series. Default is 8.
  - forecast_length (int): The number of future observations required for each
    time series. Default is 4.
  
  Side Effects:
  - Modifies the input DataFrame in-place to fill missing values in columns
    that are shorter than the minimum length.
    
  Prints:
  - Dataframe shape of train entries
  - Dataframe shape of test entries
  
  """
  print("Creating train and holdout data sets.")
  train_data = data.iloc[:-forecast_length, :]
  holdout_data = data.iloc[-forecast_length:, :].reset_index(drop=True)
  
  #print("Imputing sequences that are too short with median.\n")
  # Minimum length for each time series to be usable (backcast + forecast)
  min_length = forecast_length + backcast_length
  train_length = train_data.shape[0]
  
  # Identify columns (time series) that have fewer data points than min_length
  invalid_columns = [col for col in train_data.columns if train_data[col].dropna().shape[0] < min_length]
  print("Number of columns with invalid length. Missing values are imputed with median: ", len(invalid_columns))
  
  # Handle invalid columns
  for col in invalid_columns:
      # Calculate median for this specific train_data column.
      series_median = train_data[col].median()  
      
      missing_count = min_length - train_data[col].dropna().shape[0]
      
      if missing_count > 0:
          # Calculate how many values should be imputed           
          backfill_count = min(backcast_length, missing_count)          
          
          # Backfill and forecast using median
          backfill_values = [series_median] * backfill_count          
          non_na_values = train_data[col].dropna().tolist()
          
          # Create new series
          new_series = backfill_values + non_na_values
          nan_values = [np.nan] * (train_length - len(new_series))
          new_series = nan_values + new_series
          
          # Replace the column in DataFrame
          train_data[col] = pd.Series(new_series)

  
  print("Dataframe shape of train entries: ", train_data.shape)
  print("Dataframe shape of holdout entries: ", holdout_data.shape)
  return train_data, holdout_data
