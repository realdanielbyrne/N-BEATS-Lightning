#%% Import necessary libraries
from nbeats_lightning.nbeats import *
from nbeats_lightning.loaders import *
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import loggers as pl_loggers

from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
tqdm.pandas()


import tensorboard
import warnings
warnings.filterwarnings('ignore')

# Load the milk.csv dataset
df = pd.read_csv('data/Electric_Production.csv', index_col=0)
df_val = df.values.flatten() # flat numpy array
torch.set_float32_matmul_precision('medium')


forecast_length = 6
backcast_length = 4 * forecast_length
batch_size = 64

# Create a simple pytorch dataloader
dm = TimeSeriesDataModule(
  data=df_val,
  batch_size=batch_size,
  backcast=backcast_length,
  forecast=forecast_length)

#%% ###########################################################################################
# Generic Model
n_stacks = 6
bps = 1
width = 512
thetas_dim = 5
loss='SMAPELoss'
active_g = True
share_w = False

stack_types = ["GenericBlock"] * n_stacks
generic = NBeatsNet (
  backcast = backcast_length,
  forecast = forecast_length, 
  stack_types = stack_types,
  n_blocks_per_stack = bps,
  share_weights = share_w, # share initial weights
  thetas_dim=thetas_dim,
  loss=loss,
  g_width = width,
  active_g = active_g)

name = f"Generic-[{backcast_length}-{forecast_length}]-{n_stacks=}-{width=}-{active_g=}-{bps=}-{thetas_dim=}-{loss}" 
print("Model Name :", name)

# Define a model checkpoint callback
generic_chk_callback = ModelCheckpoint(
  save_top_k = 1, # save top 2 models
  monitor = "val_loss", # monitor validation loss as evaluation 
  mode = "min",
  filename = "{name}-{epoch:02d}-{val_loss:.2f}",
)

# Define a tensorboard loger
generic_tb_logger = pl_loggers.TensorBoardLogger(save_dir="lightning_logs/", name=name)


# Train the generic model
generic_trainer =  pl.Trainer(
  accelerator='auto' # use GPU if available
  ,max_epochs=1000
  ,callbacks=[generic_chk_callback]  
  ,logger=[generic_tb_logger]
)

generic_trainer.fit(generic, datamodule=dm)
generic_trainer.validate(generic, datamodule=dm)



#%% ###########################################################################################
# Generic AE Model
n_stacks = 10
bps = 1
width = 768
thetas_dim = 5
loss='SMAPELoss'
active_g = True
latent = 5
share_w = False


stack_types = ["GenericAEBlock"] * n_stacks
genericAE = NBeatsNet (
  backcast = backcast_length,
  forecast = forecast_length, 
  stack_types = stack_types,
  n_blocks_per_stack = bps,  
  share_weights = share_w, # share initial weights
  thetas_dim=thetas_dim,
  loss=loss,
  g_width = width,
  active_g=active_g,
  latent_dim = latent)

#print(genericAE)
name = f"GenericAE-[{backcast_length}-{forecast_length}]-{n_stacks=}-{width=}-{active_g=}-{bps=}-{thetas_dim=}-{loss}-{latent=}" 
print("Model Name :", name)

# Define a model checkpoint callback
genericAE_chk_callback = ModelCheckpoint(
  save_top_k = 1, # save top 2 models
  monitor = "val_loss", # monitor validation loss as evaluation 
  mode = "min",
  filename = "{name}-{epoch:02d}-{val_loss:.2f}",
)

# Define a tensorboard loger
print("Model Name :", name)
genericAE_tb_logger = pl_loggers.TensorBoardLogger(save_dir="lightning_logs/", name=name)


# Train the generic model
genericAE_trainer =  pl.Trainer(
  accelerator='auto' # use GPU if available
  ,max_epochs=1000
  ,callbacks=[genericAE_chk_callback]  
  ,logger=[genericAE_tb_logger]
)

genericAE_trainer.fit(genericAE, datamodule=dm)
genericAE_trainer.validate(genericAE, datamodule=dm)


#%% ###########################################################################################
# Interpretable Model
bps = 3
s_width = 2048
t_width = 512
thetas_dim = 5
share_w = True
loss='SMAPELoss'


stack_types = ["TrendBlock","SeasonalityBlock"] 
interp = NBeatsNet (
  backcast = backcast_length,
  forecast = forecast_length, 
  stack_types = stack_types,
  n_blocks_per_stack = bps,
  share_weights = share_w, # share initial weights
  thetas_dim=thetas_dim,
  loss=loss,
  s_width = s_width,
  t_width = t_width,
  active_g=True)

#print(interp)
name = f"Interpretable-[{backcast_length}-{forecast_length}]-{bps=}-{s_width=}-{t_width=}-{thetas_dim=}-{loss}"
print("Model Name :", name)

# Define a model checkpoint callback
interp_chk_callback = ModelCheckpoint(
  save_top_k = 1, # save top 2 models
  monitor = "val_loss", # monitor validation loss as evaluation 
  mode = "min",
  filename = "{name}-{epoch:02d}-{val_loss:.2f}",
)

# Define a tensorboard loger
interp_tb_logger = pl_loggers.TensorBoardLogger(save_dir="lightning_logs/", name=name)


# Train the generic model
interp_trainer =  pl.Trainer(
  accelerator='auto' # use GPU if available
  ,max_epochs=1000
  ,callbacks=[interp_chk_callback]  
  ,logger=[interp_tb_logger]
)

interp_trainer.fit(interp, datamodule=dm)
interp_trainer.validate(interp, datamodule=dm)

#%% ###########################################################################################
# Interpretable AE Model
bps = 5
s_width = 2048
t_width = 512
thetas_dim = 5
share_w = True
loss='SMAPELoss'
latent = 5

stack_types = ["TrendAEBlock","SeasonalityAEBlock"] 
iae = NBeatsNet (
  backcast = backcast_length,
  forecast = forecast_length, 
  stack_types = stack_types,
  n_blocks_per_stack = bps,
  share_weights = share_w, # share initial weights
  thetas_dim=thetas_dim,
  loss=loss,
  s_width = s_width,
  t_width = t_width,
  active_g=True,
  latent_dim=latent)

#print(iae)
name = f"InterpretableAE-[{backcast_length}-{forecast_length}]-{bps=}-{s_width=}-{t_width=}-{thetas_dim=}-{loss}-{latent=}"
print("Model Name :", name)

# Define a model checkpoint callback
iae_chk_callback = ModelCheckpoint(
  save_top_k = 1, # save top 2 models
  monitor = "val_loss", # monitor validation loss as evaluation 
  mode = "min",
  filename = "{name}-{epoch:02d}-{val_loss:.2f}",
)

# Define a tensorboard loger
iae_tb_logger = pl_loggers.TensorBoardLogger(save_dir="lightning_logs/", name=name)


# Train the generic model
iae_trainer =  pl.Trainer(
  accelerator='auto' # use GPU if available
  ,max_epochs=1000
  ,callbacks=[iae_chk_callback]  
  ,logger=[iae_tb_logger]
)

iae_trainer.fit(iae, datamodule=dm)
iae_trainer.validate(iae, datamodule=dm)


#%% ###########################################################################################
# AutoEncoder Model
n_stacks = 6
bps = 1
width = 512
thetas_dim = 5
loss='SMAPELoss'
active_g = True
share_w = False

stack_types = ["AutoEncoderBlock"] * n_stacks
ae = NBeatsNet (
  backcast = backcast_length,
  forecast = forecast_length, 
  stack_types = stack_types,
  n_blocks_per_stack = bps,
  share_weights = share_w, # share initial weights
  thetas_dim=thetas_dim,
  loss=loss,
  v_width = width,
  active_g = active_g)

#print(ae)
name = f"AutoEncoder-[{backcast_length}-{forecast_length}]-{n_stacks=}-{width=}-{active_g=}-{bps=}-{thetas_dim=}-{loss}" 
print("Model Name :", name)

# Define a model checkpoint callback
ae_chk_callback = ModelCheckpoint(
  save_top_k = 1, # save top 2 models
  monitor = "val_loss", # monitor validation loss as evaluation 
  mode = "min",
  filename = "{name}-{epoch:02d}-{val_loss:.2f}",
)

# Define a tensorboard loger
ae_tb_logger = pl_loggers.TensorBoardLogger(save_dir="lightning_logs/", name=name)


# Train the generic model
ae_trainer =  pl.Trainer(
  accelerator='auto' # use GPU if available
  ,max_epochs=1000
  ,callbacks=[ae_chk_callback]  
  ,logger=[ae_tb_logger]
)

ae_trainer.fit(ae, datamodule=dm)
ae_trainer.validate(ae, datamodule=dm)


#%% ###########################################################################################
# AutoEncoderAE Model
n_stacks = 10
bps = 1
width = 768
thetas_dim = 5
loss='SMAPELoss'
active_g = True
share_w = False
latent = 5

stack_types = ["AutoEncoderAEBlock"] * n_stacks
aeae = NBeatsNet (
  backcast = backcast_length,
  forecast = forecast_length, 
  stack_types = stack_types,
  n_blocks_per_stack = bps,
  share_weights = share_w, # share initial weights
  thetas_dim=thetas_dim,
  loss=loss,
  v_width = width,
  active_g=active_g,
  latent_dim=latent)

#print(aeae)

# Define a model checkpoint callback
aeae_chk_callback = ModelCheckpoint(
  save_top_k = 1, # save top 2 models
  monitor = "val_loss", # monitor validation loss as evaluation 
  mode = "min",
  filename = "{name}-{epoch:02d}-{val_loss:.2f}",
)

# Define a tensorboard loger
name = f"AutoEncoderAE-[{backcast_length}-{forecast_length}]-{n_stacks=}-{width=}-{active_g=}-{bps=}-{thetas_dim=}-{loss}-{latent=}" 
print("Model Name :", name)
generic_tb_logger = pl_loggers.TensorBoardLogger(save_dir="lightning_logs/", name=name)


# Train the generic model
aeae_trainer =  pl.Trainer(
  accelerator='auto' # use GPU if available
  ,max_epochs=1000
  ,callbacks=[generic_chk_callback]  
  ,logger=[generic_tb_logger]
)

aeae_trainer.fit(aeae, datamodule=dm)
aeae_trainer.validate(aeae, datamodule=dm)

#%% ###########################################################################################
# GenericAEBackcastBlock Model
n_stacks = 6
bps = 1
width = 512
thetas_dim = 5
loss='SMAPELoss'
active_g = True
share_w = False


stack_types = ["GenericAEBackcastBlock"] * n_stacks
ae_bckcast = NBeatsNet (
  backcast = backcast_length,
  forecast = forecast_length, 
  stack_types = stack_types,
  n_blocks_per_stack = bps,
  share_weights = share_w, # share initial weights
  thetas_dim=thetas_dim,
  loss=loss,
  v_width = width,
  active_g=active_g)

#print(ae_bckcast)
name = f"GenericAEBackcastBlock-[{backcast_length}-{forecast_length}]-{n_stacks=}-{width=}-{active_g=}-{bps=}-{thetas_dim=}-{loss}" 
print("Model Name :", name)

# Define a model checkpoint callback
ae_bckcast_chk_callback = ModelCheckpoint(
  save_top_k = 1, # save top 2 models
  monitor = "val_loss", # monitor validation loss as evaluation 
  mode = "min",
  filename = "{name}-{epoch:02d}-{val_loss:.2f}",
)

# Define a tensorboard loger
ae_bckcast_tb_logger = pl_loggers.TensorBoardLogger(save_dir="lightning_logs/", name=name)


# Train the generic model
ae_bckcast_trainer =  pl.Trainer(
  accelerator='auto' # use GPU if available
  ,max_epochs=1000
  ,callbacks=[ae_bckcast_chk_callback]  
  ,logger=[ae_bckcast_tb_logger]
)

ae_bckcast_trainer.fit(ae_bckcast, datamodule=dm)
ae_bckcast_trainer.validate(ae_bckcast, datamodule=dm)

#%% ###########################################################################################
# GenericAEBackcastAEBlock Model
n_stacks = 10
bps = 1
width = 768
thetas_dim = 5
loss='SMAPELoss'
active_g = True
share_w = False
latent = 5

stack_types = ["GenericAEBackcastAEBlock"] * n_stacks
ae_bckcast_ae = NBeatsNet (
  backcast = backcast_length,
  forecast = forecast_length, 
  stack_types = stack_types,
  n_blocks_per_stack = bps,
  share_weights = share_w, # share initial weights
  thetas_dim=thetas_dim,
  loss=loss,
  v_width = width,
  active_g=active_g)

#print(ae_bckcast)
name = f"GenericAEBackcastAEBlock-[{backcast_length}-{forecast_length}]-{n_stacks=}-{width=}-{active_g=}-{bps=}-{thetas_dim=}-{loss}-{latent=}" 
print("Model Name :", name)

# Define a model checkpoint callback
ae_bckcast_ae_chk_callback = ModelCheckpoint(
  save_top_k = 1, # save top 2 models
  monitor = "val_loss", # monitor validation loss as evaluation 
  mode = "min",
  filename = "{name}-{epoch:02d}-{val_loss:.2f}",
)

# Define a tensorboard loger
ae_bckcast_ae_tb_logger = pl_loggers.TensorBoardLogger(save_dir="lightning_logs/", name=name)


# Train the generic model
ae_bckcast_ae_bckcast_trainer =  pl.Trainer(
  accelerator='auto' # use GPU if available
  ,max_epochs=1000
  ,callbacks=[ae_bckcast_ae_chk_callback]  
  ,logger=[ae_bckcast_ae_tb_logger]
)

ae_bckcast_ae_bckcast_trainer.fit(ae_bckcast_ae, datamodule=dm)
ae_bckcast_ae_bckcast_trainer.validate(ae_bckcast_ae, datamodule=dm)

