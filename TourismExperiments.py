#%%
import pandas as pd
import numpy as np
from nbeats_lightning.nbeats import *
from nbeats_lightning.loaders import *
from nbeats_lightning.losses import *
from nbeats_lightning.constants import *
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


#%%
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

def filter_dataset(data, backcast_length, forecast_length:int=4):
  print("Filtering NANs, and sequences that are too short.")
  # Minimum length for each time series to be usable (backcast + forecast)
  min_length = forecast_length + backcast_length

  # Identify columns (time series) that have enough data points for training and validation
  valid_columns = [col for col in data.columns if data[col].dropna().shape[0] >= min_length]
  invalid_columns = [col for col in data.columns if data[col].dropna().shape[0] < min_length]
  
  # Create a new DataFrame with only the valid columns
  valid_data  = data[valid_columns]

  train_data = valid_data.iloc[:-4, :]
  test_data = valid_data.iloc[-4:, :].reset_index(drop=True)

  print ("Dataframe shape of valid train entries :",train_data.shape)
  print("Dataframe shape of invalid entries :",data[invalid_columns].shape)
  print ("Dataframe shape of valid test entries :",test_data.shape)
  
  return train_data, test_data

def get_dms(df_valid, df_holdout, backcast_length:int, forecast_length:int=4, batch_size:int=1024, split_ratio:float=0.8):
  dm = ColumnarCollectionTimeSeriesDataModule(
    df_valid, 
    backcast_length=backcast_length,
    forecast_length=forecast_length,
    split_ratio=split_ratio,
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

def get_tourism_data(data_freq:str="yearly"):
  yearly_path = 'data/tourism1/tourism_data.csv'
  monthly_quarterly_path = 'data/tourism2/tourism2_revision2.csv'
  
  if data_freq == "yearly":
    print("Loading yearly data")
    file_path = yearly_path
    dataset_id = "TourismYr"
  elif data_freq == "monthly":
    print("Loading monthly data")
    file_path = monthly_quarterly_path
    dataset_id = "TourismMth"
  else:
    print("Loading quarterly data")
    file_path = monthly_quarterly_path
    dataset_id = "TourismQtr"
        
  df = pd.read_csv(file_path)
  df.dropna(how='all', inplace=True)
  
  if data_freq == "monthly":
    df = df.iloc[:, :365] # first 366 columns are monthly data
  elif data_freq == "quarterly":
    df = df.iloc[:, -427:] # last 427 columns are quarterly data
    
  return df, dataset_id
    

#%%
# parameters
backcast_length = 8
forecast_length = 4
fast_dev_run = False
batch_size = 1024
max_epochs = 200
loss = 'SMAPELoss'
viz = False

split_ratio = 1.0
if split_ratio == 1.0:
  no_val = True

# Load the data
df, dataset_id = get_tourism_data("yearly")
df_valid, df_holdout = filter_dataset(df, backcast_length, forecast_length)
dm, test_dm = get_dms(df_valid, df_holdout, backcast_length, forecast_length, batch_size, split_ratio)

if viz:
  plot_data(df_valid)
  
#%%
blocks_to_test = {
  "Generic":["GenericBlock"],
  "GenericAE":["GenericAEBlock"],
  "GenericAEBackcast":["GenericAEBackcastBlock"],
  "GenericAEBackcastAE":["GenericAEBackcastAEBlock"],
  "TrendBlock":["TrendBlock"],
  "TrendAE":["TrendAEBlock"],
  "AutoEncoder":["AutoEncoderBlock"],
  "AutoEncoderAE":["AutoEncoderAEBlock"],
  "DB1":["DB1Block"],
  "DB2":["DB2Block"],
  "DB3":["DB3Block"],
  "DB4":["DB4Block"],
  "Haar":["HaarBlock"],
  "TrendSeasonality":["TrendBlock","SeasonalityBlock"],
  "TrendDB2":["TrendBlock","DB2Block"],
  "TrendAEDB2":["TrendAEBlock","DB2Block"],
  "SeasonalityDB2":["SeasonalityBlock","DB2Block"],
  "TrendGeneric":["TrendBlock","GenericBlock"],
  "SeasonalityGeneric":["SeasonalityBlock","GenericBlock"],
  "AutoencoderDB2":["AutoEncoderBlock","DB2Block"],
  "AutoencoderAEDB2":["AutoEncoderAEBlock","DB2Block"],
}

for key,value in blocks_to_test.items():
  
  n_stacks = 20//len(value)
  thetas_dim = 4
  bps = 1
  active_g = True
  share_w = False
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
    no_val = no_val,
    ae_width = ae_width,
    latent_dim = latent,
    sum_losses = sum_losses
  ) 

  name = f"{key}-{n_stacks=}" 
  print(f"Model Name :{name}")

  trainer = get_trainer(name, fast_dev_run = fast_dev_run)
  trainer.fit(model, datamodule=dm)
  trainer.validate(model, datamodule=dm)
  trainer.test(model, datamodule=test_dm)


# %%
