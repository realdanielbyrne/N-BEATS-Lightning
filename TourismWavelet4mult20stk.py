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
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
tqdm.pandas()
import tensorboard
import warnings
warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('medium')
import pywt
from scipy.signal import resample
from scipy.interpolate import interp1d
import seaborn as sns
from utils.utils import *


#%%
# Training parameters
batch_size = 2048
max_epochs = 250
fast_dev_run = False
no_val=False
debug = False
dataset_id = 'Tourism'

# Define stacks, by creating a list.  
# Stacks will be created in the order they appear in the list.
stacks_to_test = [
    ["HaarWavelet"],
    ["DB2Wavelet"],
    ["DB3Wavelet"],
    ["DB4Wavelet"],
    ["DB10Wavelet"],
    ["DB20Wavelet"],
    ["Coif1Wavelet"],
    ["Coif2Wavelet"],
    ["Coif3Wavelet"],
    ["Coif10Wavelet"], 
    ["Symlet2Wavelet"],
    ["Symlet3Wavelet"],
    ["WaveletStack"],
    ["AltWavelet"]
  ]

horizon_mult = 4
periods = {"Yearly":[horizon_mult*4,4], "Monthly":[horizon_mult*24,24], "Quarterly":[horizon_mult*8,8]}
for seasonal_period, lengths in periods.items():
  
  backcast_length = lengths[0] 
  forecast_length = lengths[1]  
  
  # load data
  df = get_tourism_data(seasonal_period)  
  df_valid, df_holdout = fill_columnar_ts_gaps(df, backcast_length, forecast_length)
  
  
  for s in stacks_to_test:
    
    n_stacks = 20
    n_stacks = n_stacks//len(s)  
    stack_types = s * n_stacks
    basis = 128
      
    model = NBeatsNet (
      backcast_length = backcast_length,
      forecast_length = forecast_length, 
      stack_types = stack_types,
      n_blocks_per_stack = 1,
      share_weights = True, 
      thetas_dim = 5,      
      loss = 'SMAPELoss',
      active_g = True,
      latent_dim = 12,
      basis_dim = basis
    ) 
    
    
    model_id="".join(s)
    model_name = f"{model_id}-{seasonal_period}[{backcast_length},{forecast_length}]{basis=}-Wave4Horizon" 
    print(f'{model_name=}\n\n')


    trainer = get_trainer(model_name, max_epochs, subdirectory=dataset_id, no_val=no_val, fast_dev_run = fast_dev_run)
    dm, test_dm = get_columnar_dms(df_valid, df_holdout, backcast_length, forecast_length, batch_size, no_val)
    
    trainer.fit(model, datamodule=dm)
    trainer.validate(model, datamodule=dm)
    trainer.test(model, datamodule=test_dm)
    print('### TEST END ###\n\n\n')




# %%
