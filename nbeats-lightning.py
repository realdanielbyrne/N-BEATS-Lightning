#%%
from models.nbeats import *
from models.loaders import *

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


#%% Begining of Notebook Cell
# Set M4 data parameters

# Hourly, Daily, Weekly, Monthly, Quarterly, Yearly
seasonal_period= "Monthly"

# Any index will do since we are just loading a representative entry from the M4Info.csv file to gather data parameters
data_index = 1 

# The backcast length is determined by multiplying the forecast horizon by an integer multiplier
forecast_multiplier = 5

# set to a category name to train on a single M4 category
category = 'Macro' 


#%% Begining of Notebook Cell
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


def get_M4infofile_info(info_file, seasonal_period, forecast_multiplier, category = None):
  
  data_info  = pd.read_csv(info_file, index_col=0)
  data_id_info = data_info.loc[seasonal_period[0] + f"{1}"]
  frequency = data_id_info.Frequency
  forecast = data_id_info.Horizon
  backcast = data_id_info.Horizon * forecast_multiplier
  
  if category is not None:
    mask = (data_info.index.str.startswith(seasonal_period[0])) & (data_info.category == category)
    category_subset = data_info[mask]
    indicies = category_subset[mask].index
    
  else:
    indicies = None
  
  return frequency, forecast, backcast, indicies


#%% Begining of Notebook Cell
# Load data
info_file  = "data/M4/M4-info.csv"
train_file = f"data/M4/Train/{seasonal_period}-train.csv"
test_file  = f"data/M4/Test/{seasonal_period}-test.csv"

# Get data parameters from M4Info.csv file
frequency, forecast, backcast, indicies = get_M4infofile_info(
  info_file, seasonal_period, forecast_multiplier, category)


#%% Begining of Notebook Cell

# Set model hyperparameters
optimizer = 'adam'
loss = 'smape'
hidden_layer_units = 512
share_weights_in_stack = False
learning_rate = 1e-5
thetas_dim = 5
stack_blocks = 1

# Set trainer hyperparameters
batch_size = 1024 # N-BEATS paper uses 1024
val_nepoch = 1 # perform a validation check every n epochs
max_epochs = 100
train = False # set to True to train the model
test = True # set to True to test the model
split_ratio = 0.8
fast_dev_run = False  # set to True to run a single batch through the model for debugging purposes
debug = False # set to True to limit the size of the dataset for debugging purposes
chkpoint = "logs/n-beats-smape-MonthlyMacro-90-18-12/version_0/checkpoints/name=0-epoch=99-val_loss=8.29.ckpt" # set to checkpoint path if you want to load a previous model

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

#%% Begining of Notebook Cell

# Load a previous model or create a new one
if chkpoint is not None:  
  model = NBeatsNet.load_from_checkpoint(chkpoint)
else:
  model = NBeatsNet(
    loss = loss,  
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
  print (model)


#%% Begining of Notebook Cell

# the model name is derived from the parameters used to train the model
name = f"n-beats-{loss}-{seasonal_period}{category}-{backcast}-{forecast}-{frequency}" 
print("Model Name : ", name)

# define a tensorboard loger
tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/", name=name)

# define a model checkpoint callback
chk_callback = ModelCheckpoint(
  save_top_k=2,
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
#%% Begining of Notebook Cell

def load_train_data(train_file, debug = False, indicies=None):
    """Loads the training data from the M4 dataset specified by the train_file parameter.

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

def load_test_data(test_file, debug = False, indicies = None):
    """Loads the test data from the M4 dataset specified by the test_file parameter.

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

train_data = load_train_data(train_file, debug, indicies)
test_data = load_test_data(test_file, debug, indicies)
    
#%% Begining of Notebook Cell

# Train and runa final validation on the model
if train:

  dmc = TimeSeriesCollectionDataModule(
    train_data=train_data, 
    backcast=backcast, 
    forecast=forecast, 
    batch_size=batch_size, 
    split_ratio=split_ratio
    )
  trainer.fit(model, datamodule=dmc, ckpt_path=chkpoint)
  trainer.validate(model, datamodule=dmc)

#%% Begining of Notebook Cell

# Test the model with the in-sample data from the trianing set 
# merged with the out-of-sample data from the test set
if test:
  
  test_module = TimeSeriesCollectionTestModule(
    test_data=test_data, 
    train_data=train_data,
    backcast=backcast, 
    forecast=forecast, 
    batch_size=batch_size
  )
  trainer.test(model, datamodule=test_module)
  


# %%
