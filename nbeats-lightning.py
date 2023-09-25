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
chkpoint = None # set to checkpoint path if you want to load a previous model

optimizer_name = 'adamw'
loss = 'smape'
generic_architecture = False
weight_share = True
learning_rate = 1e-5
thetas_dim = 2
blocks_p_stack = 3
stacks = 2

# Set trainer hyperparameters
batch_size = 1024 # N-BEATS paper uses 1024
val_nepoch = 1 # perform a validation check every n epochs
max_epochs = 31
train = True # set to True to train the model
test = True # set to True to test the model
split_ratio = 0.8
fast_dev_run = False  # set to True to run a single batch through the model for debugging purposes
debug = False # set to True to limit the size of the dataset for debugging purposes

# Set precision to 32 bit
torch.set_float32_matmul_precision('medium')

chkpoint = "logs/n-beats-interpretive-smape-MonthlyMacro-adamw-90-18-2-3-2-shared"

#%% Begining of Notebook Cell

# Load a previous model or create a new one
if chkpoint is not None:  
  model = NBeatsNet.load_from_checkpoint(chkpoint)
else:
  model = NBeatsNet(
    backcast = backcast,
    forecast = forecast, 
    generic_architecture = generic_architecture,
    n_blocks_per_stack = blocks_p_stack,
    n_stacks = stacks,
    share_weights_in_stack = weight_share,
    thetas_dim = thetas_dim,
    learning_rate = learning_rate,
    loss = loss,  
    no_val = False,
    optimizer_name = optimizer_name,
    frequency=frequency
  )
  print (model)


#%% Begining of Notebook Cell
if generic_architecture:
  arch = 'generic'
else :
  arch = 'interpretive'

if weight_share:
  shared = 'shared'
else:
  shared = 'unshared'

# the model name is derived from the parameters used to train the model
name = f"n-beats-{arch}-{loss}-{seasonal_period}{category}-{optimizer_name}-{backcast}-{forecast}-{stacks}-{blocks_p_stack}-{thetas_dim}-{shared}" 
print("Model Name :", name)

# define a tensorboard loger
tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/", name=name)

# define a model checkpoint callback
chk_callback = ModelCheckpoint(
  save_top_k = 2,
  monitor = "val_loss",
  mode = "min",
  filename = "{name}-{epoch:02d}-{val_loss:.2f}",
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

# Train and run a final validation on the model
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
d = train_data.loc['M579'].dropna()
d = d[-backcast:].values

# %%
y = test_data.loc['M579'].dropna().values
y_hat = model.predict(d)
# %%
#plot
import matplotlib.pyplot as plt
data_info  = pd.read_csv(info_file, index_col=0)

start_date = pd.to_datetime(data_info.loc['M579'].StartingDate, format="%d-%m-%y %H:%M")

dates_d = pd.date_range(start=start_date, periods=len(d), freq='M')
dates_y_hat_y = pd.date_range(start=dates_d[-1] + pd.DateOffset(months=1), periods=len(y_hat), freq='M')

# Create a new figure
plt.figure(figsize=(10, 6))

# Plotting all the series on the same plot
plt.plot(dates_d, d, label='In-Sample (d)', linestyle='-', marker='o')
plt.plot(dates_y_hat_y, y_hat, label='Model Prediction (y_hat)', linestyle='-', marker='o')
plt.plot(dates_y_hat_y, y, label='Actual (y)', linestyle='-', marker='o')

# Adding legend, labels, and title
plt.legend(loc='upper right')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('M579 Prediction vs Actual')

# Display the plot
plt.show()

# %%
