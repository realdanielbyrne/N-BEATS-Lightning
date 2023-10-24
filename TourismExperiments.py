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
from utils.utils import *

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


#%%
# parameters
fast_dev_run = False
batch_size = 1024
max_epochs = 175
loss = 'SMAPELoss'
viz = False
no_val=False


# Load the data
#periods = {"yearly":[8,4], "monthly":[72,24], "quarterly":[24,8]}
periods = {"yearly":[8,4], "quarterly":[24,8]}

for p, lengths in periods.items():
  backcast_length = lengths[0] 
  forecast_length = lengths[1]  
  #print (f"Backcast Length: {backcast_length}, Forecast Length: {forecast_length}\n")
  
  df, dataset_id = get_tourism_data(p)
  print(f"Dataset ID: {dataset_id}")
  
  df_valid, df_holdout = fill_columnar_ts_gaps(df, backcast_length, forecast_length)
  dm, test_dm = get_columnar_datamodules(df_valid, df_holdout, backcast_length, forecast_length, batch_size, no_val)

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
    "TrendSeasonalityControl":["TrendBlock","SeasonalityBlock"],
    "TrendSeasonality":["TrendBlock","SeasonalityBlock"],
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
    "WaveletStack":["WaveletStackBlock"], #DB3
    "Haar":["HaarBlock"], # DB1
    #"DB2":["DB2Block"],
    "DB3":["DB2Block"],
    "TrendDB3":["TrendBlock","DB3Block"],
    #"DB4":["DB4Block"],
    #"DB20":["DB20Block"],
    #"Symlet3Block":["Symlet3Block"],
    #"Coif3Block":["Coif3Block"],
    
    #"TrendDB4":["TrendBlock","DB4Block"],
    #"TrendSym10":["TrendBlock","Sym10Block"],
    #"Sym10Generic":["Sym10Block","GenericBlock"],
    
    #"TrendSeasonalityCoif2DB2Generic":["TrendBlock","SeasonalityBlock","Coif2Block","DB2Block","GenericBlock"],
    
    }  

  for key,value in blocks_to_test.items():
    
    if key == "TrendSeasonalityControl":
      n_stacks = 2
    else:
      n_stacks = 10//len(value)    
    
    if key == "TrendSeasonalityControl":
      thetas_dim = 5
    elif key == "TrendSeasonality":
      thetas_dim = 5
    else:
      thetas_dim = 32
      
    
    if key == "TrendSeasonalityControl":
      bps = 3
    else:
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
    

    name = f"{key}-{dataset_id}-{thetas_dim=}" 
    print(name + '\n')

    trainer = get_trainer(name, max_epochs, subdirectory='tourism',fast_dev_run = fast_dev_run)
    trainer.fit(model, datamodule=dm)
    trainer.validate(model, datamodule=dm)
    trainer.test(model, datamodule=test_dm)


# %%
