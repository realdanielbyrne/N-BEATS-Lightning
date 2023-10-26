import pandas as pd
import numpy as np
from nbeats_lightning.nbeats import *                   
from nbeats_lightning.loaders import *
from nbeats_lightning.losses import *
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import loggers as pl_loggers
import seaborn as sns
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

# M4 info file
m4_info_path  = "data/M4/M4-info.csv"
yearly_tourism_data_path = "data/tourism1/tourism_data.csv"
mth_qtr_tourism_data_path = 'data/tourism2/tourism2_revision2.csv'

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
  
  callbacks=[chk_callback]
  # Define a tensorboard loger
  tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"lightning_logs/{subdirectory}", name=name)

  # Train the generic model
  trainer =  pl.Trainer(
    accelerator='auto' # use GPU if available
    ,max_epochs=max_epochs
    ,callbacks=callbacks  
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

def plot_tourism_data(df):
  # Visualization 1: Plotting a few random time series in the same plot
  plt.figure(figsize=(14, 6))
  sample_columns = df.sample(n=5, axis=1).columns
  for col in sample_columns:
      plt.plot(df[col].dropna().values, label=f"Time Series {col}")
  plt.title('Random Sample of 5 Time Series')
  plt.xlabel('Observations')
  plt.ylabel('Value')
  plt.legend()
  plt.show()


  # Visualization 2: Plotting the longest and shortest valid time series
  min_length = 10 + 4
  valid_lengths = [(col, len(df[col].dropna())) for col in df.columns if len(df[col].dropna()) >= min_length]
  sorted_valid_lengths = sorted(valid_lengths, key=lambda x: x[1])
  shortest_col, longest_col = sorted_valid_lengths[0][0], sorted_valid_lengths[-1][0]

  plt.figure(figsize=(14, 6))
  plt.plot(df[shortest_col].dropna().values, label=f"Shortest Valid Time Series ({shortest_col})")
  plt.plot(df[longest_col].dropna().values, label=f"Longest Valid Time Series ({longest_col})")
  plt.title('Comparison of the Shortest and Longest Valid Time Series')
  plt.xlabel('Observations')
  plt.ylabel('Value')
  plt.legend()
  plt.show()

  # Visualization 3: Histogram of the lengths of valid time series
  valid_lengths_values = [length for _, length in sorted_valid_lengths]

  plt.figure(figsize=(14, 6))
  plt.hist(valid_lengths_values, bins=20, alpha=0.7, color='blue')
  plt.title('Histogram of Lengths of Valid Time Series')
  plt.xlabel('Length')
  plt.ylabel('Frequency')
  plt.show()

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

  print(f"Number of columns with invalid length before imputation: {len(invalid_columns)}")
  print(f"Dataframe shape of train entries: {train_data.shape}")
  print(f"Dataframe shape of holdout entries: {holdout_data.shape}")
  return train_data, holdout_data

def get_M4infofile_info(info_file, seasonal_period, forecast_multiplier, category = 'All'):
  
  """
    Reads the M4_info file and returns the frequency, forecast, backcast, and
    indicies of the target time series for the specified category, if any. The
    indicies are used to identify the target rows in the M4 training and test
    data files.

  Returns:
    (int,int,int,[int]): frequency, forecast_length, backcast_length, indicies
  """
  
  data_info  = pd.read_csv(info_file, index_col=0)
  data_id_info = data_info.loc[seasonal_period[0] + f"{1}"]
  frequency = data_id_info.Frequency
  forecast_length = data_id_info.Horizon
  backcast_length = data_id_info.Horizon * forecast_multiplier
  
  if category is not 'All':
    mask = (data_info.index.str.startswith(seasonal_period[0])) & (data_info.category == category)
    category_subset = data_info[mask]
    indicies = category_subset[mask].index
    
  else:
    indicies = None
  
  return frequency, forecast_length, backcast_length, indicies

def get_row_dms(
      train_data, 
      test_data, 
      backcast_length:int, 
      forecast_length:int=4, 
      batch_size:int=1024, 
      split_ratio:float=0.8):
  
  dm = TimeSeriesCollectionDataModule(
    train_data=train_data, 
    backcast_length=backcast_length, 
    forecast_length=forecast_length, 
    batch_size=batch_size, 
    split_ratio=split_ratio
    )

  test_dm = TimeSeriesCollectionTestModule(
    test_data=test_data, 
    train_data=train_data,
    backcast_length=backcast_length, 
    forecast_length=forecast_length, 
    batch_size=batch_size
  )
  
  return dm, test_dm

def get_columnar_dms(
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

def load_m4_train_data(train_file, debug = False, indicies=None):
    """
    Loads the training data from the M4 dataset specified by the train_file parameter.

    Args:
        train_file (string): Path to the M4 training data file. 
        debug (bool, optional): Limits training data to first 1000 rows. 
                                Defaults to False.
        indicies (Index, optional): Indexes of selected category.
                                Defaults to None.

    Returns:
        Tensor : Train data
    """
    all_train_data = pd.read_csv(train_file, index_col=0)
          
    if indicies is not None:
      all_train_data = all_train_data.loc[indicies]
    
    if debug:
      all_train_data[:1000]
      
    return all_train_data

def load_m4_test_data(test_file, debug = False, indicies = None):
    """
    Loads the test data from the M4 dataset specified by the test_file parameter.

    Args:
        test_file (string): Path to the M4 test data file.
        debug (bool, optional): Limits data to first 100 rowes Defaults to False.
        indicies (Index, optional): Indexes of selected category. Defaults to None.

    Returns:
        _type_: _description_
    """
    all_test_data = pd.read_csv(test_file, index_col=0)
    
    if indicies is not None:
      all_test_data = all_test_data.loc[indicies]
      
    if debug:
      all_test_data[:1000]
            
    return all_test_data

def get_tourism_data(data_freq:str="Yearly"):

  if data_freq == "Yearly":
    print("Loading yearly data...\n")
    file_path = yearly_tourism_data_path    
  elif data_freq == "Monthly":
    print("Loading monthly data...\n")
    file_path = mth_qtr_tourism_data_path    
  elif data_freq == "Quarterly":
    print("Loading quarterly data...\n")
    file_path = mth_qtr_tourism_data_path    
  else:
    raise ValueError("Invalid data frequency. Valid options are 'Yearly', 'Monthly', or 'Quarterly'")    
        
  df = pd.read_csv(file_path)
  df.dropna(how='all', inplace=True)
  
  if data_freq == "Monthly":
    df = df.iloc[:, :365] # first 366 columns are monthly data
  elif data_freq == "Quarterly":
    df = df.iloc[:, -427:] # last 427 columns are quarterly data
    
  return df

def fill_rowwise_ts_gaps(data, backcast_length:int= 8, forecast_length:int=4):
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

