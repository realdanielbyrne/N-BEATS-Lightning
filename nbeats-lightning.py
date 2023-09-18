#%%
from models.nbeats import *
from models.loaders import *
from models.losses import *
import tensorboard as tb

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd

import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import loggers as pl_loggers

from tqdm.notebook import tqdm
tqdm.pandas()


#%%
# Set M4 data parameters

# Hourly, Daily, Weekly, Monthly, Quarterly, Yearly
seasonal_period= "Monthly"

# Any index will do since we are just loading a representative entry from the M4Info.csv file to gather data parameters
data_index = 1 

# The backcast lenght is determined by multiplying the forecast horizon by an integer multiplier
forecast_multiplier = 3


#%%
def get_trainer(max_epochs=100, fast_dev_run=False, val_nepoch=1, chk_cb = None, logger = None):
  trainer =  pl.Trainer(
    accelerator='auto'
    ,max_epochs=max_epochs   
    ,fast_dev_run=fast_dev_run
    ,logger=[logger]
    ,check_val_every_n_epoch=val_nepoch 
    ,callbacks=[chk_cb]  
  )
  return trainer


def get_M4infofile_info(info_file,seasonal_period, forecast_multiplier, data_index = 1):
  
  data_info  = pd.read_csv(info_file, index_col=0)
  data_id_info = data_info.loc[seasonal_period[0] + f"{data_index}"]
  category = data_id_info.category
  frequency = data_id_info.Frequency
  forecast = data_id_info.Horizon
  backcast = data_id_info.Horizon * forecast_multiplier
  
  return category, frequency, forecast, backcast


#%%
# Load data
info_file  = "data/M4/M4-info.csv"
train_file = f"data/M4/Train/{seasonal_period}-train.csv"
test_file  = f"data/M4/Test/{seasonal_period}-test.csv"

# Get data parameters from M4Info.csv file
category, frequency, forecast, backcast = get_M4infofile_info(
  info_file, seasonal_period, forecast_multiplier)


#%%
# Set model hyperparameters
optimizer = 'adam'
hidden_layer_units = 512
share_weights_in_stack = False
learning_rate = 1e-5
thetas_dim = 5
stack_blocks = 1

# Set trainer hyperparameters
batch_size = 1024 # N-BEATS paper uses 1024
val_nepoch = 1 # perform a validation check every n epochs
max_epochs = 2
train = True # set to True to train the model
test = False # set to True to test the model
split_ratio = 0.8
fast_dev_run = True  # set to True to run a single batch through the model for debugging purposes
debug = True # set to True t limit the size of the dataset for debugging purposes
chkpoint = None # set to checkpoint path if you want to load a previous model
#loss = SMAPELoss() # Any Pytorch loss function will do.  N-BEATS paper uses MAPELoss, SMAPELoss, and MASELoss
loss = MASELoss()

# set precision to 32 bit
torch.set_float32_matmul_precision('medium')

# Define model architecture
# trend blocks should generally be paired with seasonality blocks, but 1:1 pairing 
# doesn't seem to be necessary. The N-BEATS paper shows that ensembles of different
# varations in stack architecture offer the best performance gains.
stack_types =[NBeatsNet.GENERIC_BLOCK,
              NBeatsNet.GENERIC_BLOCK,
              NBeatsNet.GENERIC_BLOCK,
              NBeatsNet.GENERIC_BLOCK,
              NBeatsNet.GENERIC_BLOCK,
              NBeatsNet.GENERIC_BLOCK,
              NBeatsNet.GENERIC_BLOCK,
              NBeatsNet.GENERIC_BLOCK,
              NBeatsNet.GENERIC_BLOCK,              
              NBeatsNet.GENERIC_BLOCK              
              ]

if chkpoint is not None:  
  model = NBeatsNet.load_from_checkpoint(chkpoint)
else:
  model = NBeatsNet(
    loss_fn = loss,  
    optimizer_name = optimizer,
    stack_types = stack_types,
    n_forecast = forecast, 
    n_backcast = backcast,
    learning_rate = learning_rate,
    thetas_dim = thetas_dim,
    blocks_per_stack = stack_blocks,
    share_weights_in_stack = share_weights_in_stack,
    hidden_layer_units = hidden_layer_units,
    no_val = False
  )


#%%
# define a tensorboard loger
name = f"n-beats-{loss}-{seasonal_period}-{backcast}-{forecast}-{frequency}" 
tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/", name=name)

chk_callback = ModelCheckpoint(
  save_top_k=3,
  monitor="val_loss",
  mode="min",
  filename="{name}-{epoch:02d}-{val_loss:.2f}",
)

trainer = get_trainer(
                      max_epochs=max_epochs, 
                      fast_dev_run=fast_dev_run, 
                      val_nepoch=val_nepoch, 
                      chk_cb=chk_callback, 
                      logger=tb_logger)
    
#%%
# Train
if train:

  dmc = TimeSeriesCollectionDataModule(
    train_file=train_file, 
    backcast=backcast, 
    forecast=forecast, 
    batch_size=batch_size, 
    split_ratio=split_ratio
    )
  trainer.fit(model, datamodule=dmc, ckpt_path=chkpoint)
  trainer.validate(model, datamodule=dmc)
  
if test:
  
  test_module = TimeSeriesCollectionTestModule(
    test_file=test_file, 
    train_file=train_file,
    backcast=backcast, 
    forecast=forecast, 
    batch_size=batch_size
  )
  trainer.test(model, datamodule=test_module)
  



# %%
