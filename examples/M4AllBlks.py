#%%
from nbeatslightning.models import *                   
from nbeatslightning.loaders import *
from nbeatslightning.losses import *
from nbeatslightning.data import M4Dataset
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

# Select a category or All = "Micro","Macro","Industry","Finance","Demographic","Ot
category = 'All'

#periods = ["Yearly","Quarterly","Monthly","Weekly","Daily","Hourly"]
periods = ["Yearly"]

# Define stacks, by creating a list.  
# Stacks will be created in the order they appear in the list.
stacks_to_test = [
    ["Generic"],
    ["Trend","Seasonality"], 
    ["TrendAE","SeasonalityAE"], 
    ["GenericAE"],
    ["GenericAEBackcast"],
    ["GenericAEBackcastAE"],
    ["AutoEncoder"],
    ["AutoEncoderAE"],
    ["HaarWavelet"],
    ["DB2Wavelet"],
    ["DB2AltWavelet"],
    ["DB3Wavelet"],
    ["DB4Wavelet"],
    ["DB10Wavelet"],
    ["Coif1Wavelet"],
    ["Coif2Wavelet"],
    ["Coif2AltWavelet"],
    ["Coif3Wavelet"],
    ["Coif10Wavelet"], 
    ["Symlet2Wavelet"],
    ["Symlet2AltWavelet"],
    ["Symlet3Wavelet"],
    ["Symlet10Wavelet"],
    ["Symlet20Wavelet"],
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
