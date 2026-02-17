#%%
from lightningnbeats.models import *                   
from lightningnbeats.loaders import *
from lightningnbeats.losses import *
from lightningnbeats.data import M4Dataset
from tqdm.notebook import tqdm
tqdm.pandas()
import tensorboard
import warnings
import torch
print("CUDA Available: ",torch.cuda.is_available())
warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('medium')
from utils import *


#%%
# Training parameters
batch_size = 2048
max_epochs = 40
loss = 'SMAPELoss'
fast_dev_run = False

no_val = False
forecast_multiplier = 5
debug = False
dataset_id = 'M4'

# Select a category or All = "Micro","Macro","Industry","Finance","Demographic","Other"
category = 'All'

periods = ["Yearly","Quarterly","Monthly","Weekly","Daily","Hourly"]

# Define stacks, by creating a list.  
# Stacks will be created in the order they appear in the list.
stacks_to_test = [
    ["Generic"],
    ["BottleneckGeneric"],
    ["Trend","Seasonality"],
    ["Trend","Generic"], 
    ["TrendAE","SeasonalityAE"], 
    ["GenericAE"],
    ["GenericAEBackcast"],
    ["GenericAEBackcastAE"],
    ["AutoEncoder"],
    ["AutoEncoderAE"],
    # V1 Wavelet blocks
    ["HaarWavelet"],
    ["HaarAltWavelet"],
    ["DB2Wavelet"],
    ["DB2AltWavelet"],
    ["DB3Wavelet"],
    ["DB3AltWavelet"],
    ["DB4Wavelet"],
    ["DB4AltWavelet"],
    ["DB10Wavelet"],
    ["DB10AltWavelet"],
    ["DB20AltWavelet"],
    ["Coif1Wavelet"],
    ["Coif1AltWavelet"],
    ["Coif2Wavelet"],
    ["Coif2AltWavelet"],
    ["Coif3Wavelet"],
    ["Coif3AltWavelet"],
    ["Coif10Wavelet"],
    ["Coif10AltWavelet"],
    ["Symlet2Wavelet"],
    ["Symlet2AltWavelet"],
    ["Symlet3Wavelet"],
    ["Symlet10Wavelet"],
    ["Symlet20Wavelet"],
    # V2 Wavelet blocks (numerically stabilized)
    ["HaarWaveletV2"],
    ["HaarAltWaveletV2"],
    ["DB2WaveletV2"],
    ["DB2AltWaveletV2"],
    ["DB3WaveletV2"],
    ["DB3AltWaveletV2"],
    ["DB4WaveletV2"],
    ["DB4AltWaveletV2"],
    ["DB10WaveletV2"],
    ["DB10AltWaveletV2"],
    ["DB20AltWaveletV2"],
    ["Coif1WaveletV2"],
    ["Coif1AltWaveletV2"],
    ["Coif2WaveletV2"],
    ["Coif2AltWaveletV2"],
    ["Coif3WaveletV2"],
    ["Coif3AltWaveletV2"],
    ["Coif10WaveletV2"],
    ["Coif10AltWaveletV2"],
    ["Symlet2WaveletV2"],
    ["Symlet2AltWaveletV2"],
    ["Symlet3WaveletV2"],
    ["Symlet10WaveletV2"],
    ["Symlet20WaveletV2"],
    # V3 Wavelet blocks (orthonormal DWT basis)
    ["HaarWaveletV3"],
    ["DB2WaveletV3"],
    ["DB3WaveletV3"],
    ["DB4WaveletV3"],
    ["DB10WaveletV3"],
    ["DB20WaveletV3"],
    ["Coif1WaveletV3"],
    ["Coif2WaveletV3"],
    ["Coif3WaveletV3"],
    ["Coif10WaveletV3"],
    ["Symlet2WaveletV3"],
    ["Symlet3WaveletV3"],
    ["Symlet10WaveletV3"],
    ["Symlet20WaveletV3"],
  ]

for seasonal_period in periods:
  # load data
  m4 = M4Dataset(seasonal_period, category)
  backcast_length = m4.forecast_length * forecast_multiplier  
  train_data = m4.train_data
  test_data = m4.test_data
  print(f"Train Data Shape: {train_data.shape}")
  print(f"Test Data Shape: {test_data.shape}")
  
  
  for s in stacks_to_test:
    n_stacks = 8
    n_stacks = n_stacks//len(s)  
    stack_types = s * n_stacks
    basis = 128
      
    model = NBeatsNet (
      backcast_length = backcast_length,
      forecast_length = m4.forecast_length, 
      stack_types = stack_types,
      n_blocks_per_stack = 1,
      share_weights = False, 
      thetas_dim = 5,      
      loss = 'SMAPELoss',
      active_g = True,
      latent_dim = 4,
      basis_dim = basis,
      learning_rate = 1e-4,
      no_val=no_val,
    ) 
    
    model_id="".join(s)
    name = f"{model_id}{seasonal_period}{category}[{backcast_length},{m4.forecast_length}]-{basis=}" 
    print(f"Model Name : {name}\n")
    
    
    trainer = get_trainer(name, max_epochs, subdirectory=dataset_id, no_val=no_val)
    dm = ColumnarCollectionTimeSeriesDataModule(train_data, backcast_length=backcast_length, forecast_length=m4.forecast_length, batch_size=batch_size,no_val=no_val)
    test_dm = ColumnarCollectionTimeSeriesTestDataModule(train_data, test_data, backcast_length=backcast_length, forecast_length=m4.forecast_length, batch_size=batch_size)
    
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=test_dm)
    model = NBeatsNet.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    trainer.test(model, datamodule=test_dm)


# %%
