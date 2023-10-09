#%% Import necessary libraries
from nbeats_lightning.nbeats import *
from nbeats_lightning.loaders import *
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import loggers as pl_loggers
from statsmodels.tsa.seasonal import seasonal_decompose
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
tqdm.pandas()


import tensorboard
import warnings
warnings.filterwarnings('ignore')

# Load the milk.csv dataset
milk = pd.read_csv('data/milk.csv', index_col=0)
milkval = milk.values.flatten() # flat numpy array


#%% Define the Generic N-Beats Models
# Generic hyperparameters
forecast_length = 6
backcast_length = 4 * forecast_length
batch_size = 64
n_stacks = 6

blocks_per_stack = 1
v_width = 512



# Create a simple pytorch dataloader
dm = TimeSeriesDataModule(
  data=milkval,
  batch_size=batch_size,
  backcast=backcast_length,
  forecast=forecast_length)

# A Generic N-Beats Model.  
# - 6 stacks
#   - 1 generic() block per stack 
#     - 5 waveform theta parameters per block

stack_types = [NBeatsNet.VAE] * n_stacks
vae = NBeatsNet (
  backcast = backcast_length,
  forecast = forecast_length, 
  stack_types = stack_types,
  n_blocks_per_stack = blocks_per_stack,
  n_stacks = n_stacks,
  share_weights = False, # share initial weights
  v_width = v_width)

print(vae)

# Define a model checkpoint callback
vae_chk_callback = ModelCheckpoint(
  save_top_k = 2, # save top 2 models
  monitor = "val_loss", # monitor validation loss as evaluation 
  mode = "min",
  filename = "{name}-{epoch:02d}-{val_loss:.2f}",
)

# Define a tensorboard loger
name = f"Vae-[{backcast_length}-{forecast_length}]-s[{n_stacks}-{blocks_per_stack}]-w[{v_width}]" 
print("Model Name :", name)
g1_tb_logger = pl_loggers.TensorBoardLogger(save_dir="lightning_logs/", name=name)


# Train the generic model
vae_trainer =  pl.Trainer(
  accelerator='auto' # use GPU if available
  ,max_epochs=1000
  ,callbacks=[vae_chk_callback]  
  ,logger=[g1_tb_logger]
)


#%%
vae_trainer.fit(vae, datamodule=dm)
vae_trainer.validate(vae, datamodule=dm)