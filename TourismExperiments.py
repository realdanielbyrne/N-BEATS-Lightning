#%%
import pandas as pd
import numpy as np
from nbeats_lightning.nbeats import *                   
from nbeats_lightning.loaders import *
from nbeats_lightning.losses import *
from nbeats_lightning.constants import BLOCKS
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import loggers as pl_loggers
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
tqdm.pandas()
import tensorboard
import warnings
warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('medium')
import pywt
from scipy.signal import resample
from scipy.interpolate import interp1d


#%%

def get_tourism_data(data_freq:str="yearly"):
  yearly_path = 'data/tourism1/tourism_data.csv'
  monthly_quarterly_path = 'data/tourism2/tourism2_revision2.csv'
  
  if data_freq == "yearly":
    print("Loading yearly data...\n")
    file_path = yearly_path
    dataset_id = "Yrly"
  elif data_freq == "monthly":
    print("Loading monthly data...\n")
    file_path = monthly_quarterly_path
    dataset_id = "Mth"
  else:
    print("Loading quarterly data...\n")
    file_path = monthly_quarterly_path
    dataset_id = "Qtr"
        
  df = pd.read_csv(file_path)
  df.dropna(how='all', inplace=True)
  
  if data_freq == "monthly":
    df = df.iloc[:, :365] # first 366 columns are monthly data
  elif data_freq == "quarterly":
    df = df.iloc[:, -427:] # last 427 columns are quarterly data
    
  return df, dataset_id

def filter_dataset_by_removing_short_ts(data, backcast_length:int= 8, forecast_length:int=4):
  print("Filtering NANs, and sequences that are too short.")
  # Minimum length for each time series to be usable (backcast + forecast)
  min_length = forecast_length + backcast_length

  # Identify columns (time series) that have enough data points for training and validation
  valid_columns = [col for col in data.columns if data[col].dropna().shape[0] >= min_length]
  invalid_columns = [col for col in data.columns if data[col].dropna().shape[0] < min_length]
  
  # Create a new DataFrame with only the valid columns
  valid_data  = data[valid_columns]


  train_data = valid_data.iloc[:-forecast_length, :]
  test_data = valid_data.iloc[-forecast_length:, :].reset_index(drop=True)

  print ("Dataframe shape of valid train entries :",train_data.shape)
  print("Dataframe shape of invalid entries :",data[invalid_columns].shape)
  print ("Dataframe shape of valid test entries :",test_data.shape)
  
  return train_data, test_data

def fill_time_series_gaps(data, backcast_length:int= 8, forecast_length:int=4):
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
  #print("Handling NaNs and sequences that are too short.\n")
  
  # Minimum length for each time series to be usable (backcast + forecast)
  min_length = forecast_length + backcast_length
  
  # Identify columns (time series) that have fewer data points than min_length
  invalid_columns = [col for col in data.columns if data[col].dropna().shape[0] < min_length]
  print("Number of columns with invalid length. Missing values are imputed with median: ", len(invalid_columns))
  
  # Handle invalid columns
  for col in invalid_columns:
      series_median = data[col].median()  # Calculate median for this specific column
      
      missing_count = min_length - data[col].dropna().shape[0]
      
      if missing_count > 0:
          # Calculate how many values should be imputed           
          backfill_count = min(backcast_length, missing_count)
          forecast_count = missing_count - backfill_count
          
          # Backfill and forecast using median
          backfill_values = [series_median] * backfill_count
          forecast_values = [series_median] * forecast_count
          
          non_na_values = data[col].dropna().tolist()
          
          # Create new series
          new_series = backfill_values + non_na_values + forecast_values
          
          # Replace the column in DataFrame
          data[col] = pd.Series(new_series)
  
  # Create train and test data
  train_data = data.iloc[:-forecast_length, :]
  holdout_data = data.iloc[-forecast_length:, :].reset_index(drop=True)
  
  print("Dataframe shape of train entries: ", train_data.shape)
  print("Dataframe shape of holdout entries: ", holdout_data.shape)
  return train_data, holdout_data

def plot_data(df):
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

def get_dms(df_valid, df_holdout, backcast_length:int, forecast_length:int=4, batch_size:int=1024, no_val:bool=False):
  dm = ColumnarCollectionTimeSeriesDataModule(
    df_valid, 
    backcast_length=backcast_length,
    forecast_length=forecast_length,    
    no_val=no_val,
    batch_size=batch_size)

  test_dm = ColumnarCollectionTimeSeriesTestDataModule(
    df_valid,
    df_holdout,
    backcast_length=backcast_length,
    forecast_length=forecast_length, 
    batch_size=batch_size)
  
  return dm, test_dm

def get_trainer(name, **kwargs):
  """Returns a Pytorch Lightning Trainer object

  Args:
      name (string): The model name to be used for logging and checkpointing

  Returns:
       pl.Trainer: A Pytorch Lightning Trainer object which defaults the
  """
  # Define a model checkpoint callback
  chk_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="best-checkpoint",
    save_top_k = 1, # save top 2 models
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


    

#%%
# parameters
fast_dev_run = False
batch_size = 1024
max_epochs = 200
loss = 'SMAPELoss'
viz = False

no_val=True


# Load the data
periods = {"yearly":[8,4], "monthly":[72,24], "quarterly":[24,8]}
#periods = {"monthly":[72,24], "quarterly":[24,8]}

for p, lengths in periods.items():
  backcast_length = lengths[0] 
  forecast_length = lengths[1]  
  #print (f"Backcast Length: {backcast_length}, Forecast Length: {forecast_length}\n")
  
  df, dataset_id = get_tourism_data(p)
  print(f"Dataset ID: {dataset_id}")
  
  df_valid, df_holdout = fill_time_series_gaps(df, backcast_length, forecast_length)
  dm, test_dm = get_dms(df_valid, df_holdout, backcast_length, forecast_length, batch_size, no_val)

  if viz:
    plot_data(df_valid)
  
  blocks_to_test = {
    # Mth leaders
    #"TrendSeasonality":["TrendBlock","SeasonalityBlock"],
    #"SeasonalityTrend":["SeasonalityBlock","TrendBlock"], #potentially same as TrendSeasonality    
    #"GenericAEBackcast":["GenericAEBackcastBlock"],
    #"TrendGeneric":["TrendBlock","GenericBlock"],
    #"GenericCoif2":["GenericBlock","Coif2Block"],
    #"TrendAutoEncoder":["TrendBlock","AutoEncoderBlock"],

    # Qtr leaders
    #"TrendSeasonality":["TrendBlock","SeasonalityBlock"],
    #"TrendGenericAeBackcast":["TrendBlock","GenericAEBackcastBlock"],    
    #"TrendGenericGenericAEBackcastDB2":["TrendBlock","GenericBlock","GenericAEBackcastBlock","DB2Block"],    
    #"TrendAutoEncoder":["TrendBlock","AutoEncoderBlock"],
    #"SeasonalityTrend":["SeasonalityBlock","TrendBlock"], #potentially same as TrendSeasonality
    #"TrendGeneric":["TrendBlock","GenericBlock"],
    
    # Yrly leaders
    #"SeasonalityTrend":["SeasonalityBlock","TrendBlock"], #potentially same as TrendSeasonality    
    #"TrendSeasonality":["TrendBlock","SeasonalityBlock"],
    #"TrendCoif2":["TrendBlock","Coif2Block"],
    #"TrendGenericGenericAEBackcastDB2":["TrendBlock","GenericBlock","GenericAEBackcastBlock","DB2Block"],        
    #"TrendCoif1":["TrendBlock","Coif1Block"],
    #"GenericHaar":["GenericBlock","HaarBlock"], # nan on mth
    #"TrendHaar":["TrendBlock","HaarBlock"],
    #"TrendDB2":["TrendBlock","DB2Block"],
    #"TrendGeneric":["TrendBlock","GenericBlock"],
    #"GenericDB2":["GenericBlock","DB2Block"],
    #"DB4":["DB4Block"], # nan on mthly test
    #"TrendGenericAeBackcast":["TrendBlock","GenericAEBackcastBlock"],    
    #"Haar":["HaarBlock"], 
    "DB2":["DB2Block"],
    "DB4":["DB4Block"],
    "TrendDB3":["TrendBlock","DB3Block"],
    #s"TrendDB4":["TrendBlock","DB4Block"],
    #"TrendSym10":["TrendBlock","Sym10Block"],
    #"Sym10Generic":["Sym10Block","GenericBlock"],
    
    #"TrendSeasonalityCoif2DB2Generic":["TrendBlock","SeasonalityBlock","Coif2Block","DB2Block","GenericBlock"],
    
    }  

  for key,value in blocks_to_test.items():
    
    n_stacks = 20//len(value)    
    thetas_dim = 5
    bps = 1
    active_g = True
    share_w = True
    latent = 4
    g_width = 512
    s_width = 2048
    t_width = 256
    ae_width = 512
    sum_losses = False

    b_type = key
    stack_types = value * n_stacks
    
    model = NBeatsNet (
      backcast = backcast_length,
      forecast = forecast_length, 
      stack_types = stack_types,
      n_blocks_per_stack = bps,
      share_weights = share_w, # share initial weights
      thetas_dim = thetas_dim,
      loss = loss,
      g_width = g_width,
      s_width = s_width,
      t_width = t_width,
      active_g = active_g,
      ae_width = ae_width,
      latent_dim = latent,
      sum_losses = sum_losses      
    ) 
    print(model)

    name = f"{key}-{dataset_id}-{thetas_dim=}" 
    

    trainer = get_trainer(name, fast_dev_run = fast_dev_run)
    trainer.fit(model, datamodule=dm)
    trainer.validate(model, datamodule=dm)
    trainer.test(model, datamodule=test_dm)




# %%
