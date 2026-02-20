#%%
from lightningnbeats import NBeatsNet
from lightningnbeats.loaders import *
from lightningnbeats.losses import *
from tqdm.notebook import tqdm
tqdm.pandas()
import tensorboard
import torch
import warnings
warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('medium')
from utils import *


#%%
# Training parameters
batch_size = 2048
max_epochs = 75
no_val=False
dataset_id = 'Tourism'

# Define stacks, by creating a list.  
# Stacks will be created in the order they appear in the list.
stacks_to_test = [
    #["Generic"],
    #["BottleneckGeneric"],
    #["Trend","Seasonality"],
    #["Trend","Generic"], 
    #["TrendAE","SeasonalityAE"], 
    #["GenericAE"],
    #["GenericAEBackcast"],
    #["GenericAEBackcastAE"],
    ["AutoEncoder","Generic"],
    #["AutoEncoderAE"],
    # V3 Wavelet blocks (orthonormal DWT basis)
    #["HaarWaveletV3"],
    #["DB2WaveletV3"],
    #["DB3WaveletV3"],
    #["DB4WaveletV3"],
    #["DB10WaveletV3"],
    #["DB20WaveletV3"],
    #["Coif1WaveletV3"],
    #["Coif2WaveletV3"],
    #["Coif3WaveletV3"],
    #["Coif10WaveletV3"],
    #["Symlet2WaveletV3"],
    #["Symlet3WaveletV3"],
    #["Symlet10WaveletV3"],
    #["Symlet20WaveletV3"],
  ]


periods = {"Yearly":[8,4], "Monthly":[72,24], "Quarterly":[24,8]}
for seasonal_period, lengths in periods.items():
  
  backcast_length = lengths[0] 
  forecast_length = lengths[1]  
  
  # load data
  df = get_tourism_data(seasonal_period)  
  train_data, test_data = fill_columnar_ts_gaps(df, backcast_length, forecast_length)
  
  for s in stacks_to_test:
    n_stacks = 6
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
      loss = 'MAPELoss',
      active_g = True,
      latent_dim = 4,
      basis_dim = basis,
      learning_rate = 1e-3,
      no_val=no_val,
    ) 
    
    
    model_id="".join(s)
    name = f"{model_id}-{seasonal_period}[{backcast_length},{forecast_length}]{basis=}-Allblocks" 
    print(f"Model Name : {name}\n")

    trainer = get_trainer(name, max_epochs, subdirectory=dataset_id, no_val=no_val)
    dm = ColumnarCollectionTimeSeriesDataModule(train_data, backcast_length=backcast_length, forecast_length=forecast_length, batch_size=batch_size,no_val=no_val)
    test_dm = ColumnarCollectionTimeSeriesTestDataModule(train_data, test_data, backcast_length=backcast_length, forecast_length=forecast_length, batch_size=batch_size)
    
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=test_dm)
    model = NBeatsNet.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    trainer.test(model, datamodule=test_dm)



# %%
