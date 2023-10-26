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
import seaborn as sns
from utils.utils import *


#%%
# Training parameters
batch_size = 2048
max_epochs = 100
loss = 'SMAPELoss'
fast_dev_run = False
no_val=False
split_ratio = .9
forecast_multiplier = 7
debug = False
dataset_id = 'M4'

# Define stacks, by creating a list.  
# Stacks will be created in the order they appear in the list.
stacks_to_test = [
    ["Generic"],
    ["Trend","Seasonality"],
    ["AltWaveletBlock"], 
    ["WaveletStack"],
    ["DB3"],
    ["GenericAEBackcast"],
    ["Trend","GenericAEBackcast"],
    ["Trend","Generic"],
    ["Trend","AutoEncoder"],
  ]


periods = ["Yearly","Quarterly","Monthly","Weekly","Daily","Hourly"]
for seasonal_period in periods:
  train_file = f"data/M4/Train/{seasonal_period}-train.csv"
  test_file  = f"data/M4/Test/{seasonal_period}-test.csv"
  category = 'All' 

  # load data
  frequency, forecast_length, backcast_length, indicies = get_M4infofile_info (
                      m4_info_path, seasonal_period, forecast_multiplier, category)
  train_data = load_m4_train_data(train_file, debug, indicies)
  test_data = load_m4_test_data(test_file, debug, indicies)


  for s in stacks_to_test:
    n_stacks = 12
    n_stacks = n_stacks//len(s)  
    stack_types = s * n_stacks
      
    model = NBeatsNet (
      backcast_length = backcast_length,
      forecast_length = forecast_length, 
      stack_types = stack_types,
      n_blocks_per_stack = 1,
      share_weights = True, 
      thetas_dim = 5,      
      loss = 'SMAPELoss',
      active_g = True,
      latent_dim = 4,
      basis_dim = 128
    ) 
    
    model_id="".join(s)
    name = f"{model_id}{seasonal_period}{category}[{backcast_length},{forecast_length}]" 
    print(f"Model Name : {name}\n")
    
    
    trainer = get_trainer(name, max_epochs, subdirectory=dataset_id, fast_dev_run = fast_dev_run)
    dm, test_dm = get_row_dms(train_data, test_data, backcast_length, forecast_length, batch_size, split_ratio)

    trainer.fit(model, datamodule=dm)
    trainer.validate(model, datamodule=dm)  
    trainer.test(model, datamodule=test_dm)
  
