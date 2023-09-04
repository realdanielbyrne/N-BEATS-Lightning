#%%
from time import time
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import pathlib

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.nn import functional as F
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset
from utils import *
import pathlib

torch.set_float32_matmul_precision('medium')
from lightning.pytorch import loggers as pl_loggers

#%%

# Hourly, Daily, Weekly, Daily, Monthly, Quarterly, or Yearly
seasonal_period= "Daily"
data_index = 1 

scrpt_path = pathlib.Path(__file__).parent.resolve()
cwd = pathlib.Path().resolve()

train_file = pathlib.Path(scrpt_path,f"../data/M4/Train/{seasonal_period}-train.csv")
test_file = pathlib.Path(scrpt_path, f"../data/M4/Test/{seasonal_period}-test.csv")
train_data = pd.read_csv(train_file, index_col=0)
test_data = pd.read_csv(test_file, index_col=0)

data_info = pd.read_csv(pathlib.Path(scrpt_path,"../data/M4/M4-info.csv"), index_col=0)
data_id_info = data_info.loc[seasonal_period[0] + f"{data_index}"]
freq = data_id_info.Frequency

#%%
class Block(nn.Module):
  def __init__(self, units, thetas_dim, backcast_length=10, forecast_length=5, share_thetas=False):
    super(Block, self).__init__()
    self.units = units
    self.thetas_dim = thetas_dim
    self.backcast_length = backcast_length
    self.forecast_length = forecast_length
    self.share_thetas = share_thetas
    self.fc1 = nn.Linear(backcast_length, units)
    self.fc2 = nn.Linear(units, units)
    self.fc3 = nn.Linear(units, units)
    self.fc4 = nn.Linear(units, units)
    
    
    if share_thetas:
        self.forecast_linear = self.backcast_linear = nn.Linear(units, thetas_dim, bias=True)
    else:
        self.backcast_linear = nn.Linear(units, thetas_dim, bias=True)
        self.forecast_linear = nn.Linear(units, thetas_dim, bias=True)

  def forward(self, x):
    x = squeeze_last_dim(x)    
    # N-Beats paper specifies ReLU activation for all hidden layers.
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = F.relu(self.fc4(x))
    return x

  def __str__(self):
    block_type = type(self).__name__
    return f'{block_type}(units={self.units}, thetas_dim={self.thetas_dim}, ' \
            f'backcast_length={self.backcast_length}, forecast_length={self.forecast_length}, ' \
            f'share_thetas={self.share_thetas}) at @{id(self)}'

class GenericBlock(Block):
  def __init__(self, units, thetas_dim, backcast_length=10, forecast_length=5, nb_harmonics=None):
    super(GenericBlock, self).__init__(units, thetas_dim, backcast_length, forecast_length, share_thetas= False)

    self.backcast_fc = nn.Linear(thetas_dim, backcast_length, bias=True)
    self.forecast_fc = nn.Linear(thetas_dim, forecast_length, bias=True)

  def forward(self, x):
    x = super(GenericBlock, self).forward(x)
    theta_b = self.backcast_linear(x)
    theta_f = self.forecast_linear(x)
    backcast = self.backcast_fc(theta_b)  
    forecast = self.forecast_fc(theta_f)  

    return backcast, forecast

class _SeasonalityGenerator(nn.Module):
  def __init__(self, target_length):
    super().__init__()
    half_minus_one = int(target_length / 2 - 1)
    cos_vectors = [
        torch.cos(torch.arange(target_length) / target_length * 2 * np.pi * i)
        for i in range(1, half_minus_one + 1)
    ]
    sin_vectors = [
        torch.sin(torch.arange(target_length) / target_length * 2 * np.pi * i)
        for i in range(1, half_minus_one + 1)
    ]

    # basis is of size (2 * int(target_length / 2 - 1) + 1, target_length)
    basis = torch.stack(
        [torch.ones(target_length)] + cos_vectors + sin_vectors, dim=1
    ).T

    self.basis = nn.Parameter(basis, requires_grad=False)

  def forward(self, x):
    return torch.matmul(x, self.basis)

class SeasonalityBlock(Block):
  def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, nb_harmonics=None):
    if nb_harmonics:
      super(SeasonalityBlock, self).__init__(units, nb_harmonics, device, backcast_length,forecast_length, share_thetas=True)
    else:
      super(SeasonalityBlock, self).__init__(units, forecast_length, device, backcast_length,forecast_length, share_thetas=True)

    self.backcast_g = _SeasonalityGenerator(thetas_dim, backcast_length)
    self.forecast_g = _SeasonalityGenerator(thetas_dim, forecast_length)

  def forward(self, x):
    x = super(SeasonalityBlock, self).forward(x)
    # linear compression
    backcast = self.backcast_linear(x)
    forecast = self.forecast_linear(x)
    
    # fourier expansion
    backcast = self.backcast_g(backcast)  
    forecast = self.forecast_g(forecast)
    
    return backcast, forecast

class _TrendGenerator(nn.Module):
  def __init__(self, expansion_coefficient_dim, target_length):
    super().__init__()

    # basis is of size (expansion_coefficient_dim, target_length)
    basis = torch.stack(
        [
            (torch.arange(target_length) / target_length) ** i
            for i in range(expansion_coefficient_dim)
        ],
        dim=1,
    ).T

    self.basis = nn.Parameter(basis, requires_grad=False)

  def forward(self, x):
    return torch.matmul(x, self.basis)

class TrendBlock(Block):
  def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, nb_harmonics=None):
    super(TrendBlock, self).__init__(units, thetas_dim, device, backcast_length,forecast_length, share_thetas=True)
    self.backcast_g = _TrendGenerator(thetas_dim, backcast_length)
    self.forecast_g = _TrendGenerator(thetas_dim, forecast_length)

  def forward(self, x):
    x = super(TrendBlock, self).forward(x)
    
    # linear compression
    backcast = self.backcast_linear(x)
    forecast = self.forecast_linear(x)
    
    #  Vandermonde basis expansion
    backcast = self.backcast_g(backcast)    
    forecast = self.forecast_g(forecast)
            
    return backcast, forecast


class NBeatsNet(pl.LightningModule):
  SEASONALITY_BLOCK = 'seasonality'
  TREND_BLOCK = 'trend'
  GENERIC_BLOCK = 'generic'
    
  def __init__(
      self,
      loss,
      stack_types =(GENERIC_BLOCK,GENERIC_BLOCK,GENERIC_BLOCK),
      blocks_per_stack:int = 1, # N-BEATS paper best results Generic 1 blk/stk, Trend 3 blk/stk, Seasonality 3 blk/stk
      n_forecast:int = 6,  # forecast (H)orizon
      n_backcast:int = 30, # default of 5*forecast_length or 5H , N-BEATS paper tested 2H,3H,4H,5H,6H,7H          
      thetas_dim: int = 5, # Output of FC layer in each of the forecast and backcast branches of each block
                            # N-Beats paper doesn't specify
      share_weights_in_stack:bool = True, # Generic model prefers no weight sharing, while interpretable model does.
      hidden_layer_units:int = 512,
      learning_rate: float = 1e-5,
      optimizer: str = 'adam',
      nb_harmonics = None,
      no_val:bool = False
    ):
    super(NBeatsNet, self).__init__()
    self.loss = loss
    self.stack_types = stack_types
    self.n_backcast = n_backcast
    self.n_forecast = n_forecast
    self.thetas_dim = thetas_dim
    self.share_weights_in_stack = share_weights_in_stack
    self.hidden_layer_units = hidden_layer_units
    self.learning_rate = learning_rate
    self.blocks_per_stack = blocks_per_stack
    self.optimizer_name = optimizer
    self.nb_harmonics = nb_harmonics
    self.no_val = no_val
    self.save_hyperparameters()
    
    print('| N-Beats')
    self.stacks = nn.ModuleList()
    for stack_id in range(len(self.stack_types)):
      self.stacks.append(self.create_stack(stack_id))
    
  
  def create_stack(self, stack_id):
    stack_type = self.stack_types[stack_id]
    print(f'| --  Stack {stack_type.title()} (#{stack_id}) (share_weights_in_stack={self.share_weights_in_stack})')
    blocks = nn.ModuleList()
    
    for block_id in range(self.blocks_per_stack):
        block_init = NBeatsNet.select_block(stack_type)
        if self.share_weights_in_stack and block_id != 0:
            block = blocks[-1]  # pick up the last one when we share weights.
        else:
            block = block_init(
                self.hidden_layer_units, self.thetas_dim,
                self.n_backcast, self.n_forecast,
                self.nb_harmonics
            )
        print(f'     | -- {block}')
        blocks.append(block)   

    return blocks
  


  def save(self, filename: str):
    torch.save(self.state_dict(), filename)

  @classmethod
  def load(cls, filename: str, map_location=None):
    model = cls()
    model.load_state_dict(torch.load(filename, map_location=map_location))
    return model

  @staticmethod
  def select_block(block_type):
    if block_type == NBeatsNet.SEASONALITY_BLOCK:
      return SeasonalityBlock
    elif block_type == NBeatsNet.TREND_BLOCK:
      return TrendBlock
    else:
      return GenericBlock

  def forward(self, backcast):
    backcast = squeeze_last_dim(backcast)
    forecast = torch.zeros(size=(backcast.size()[0], self.n_forecast,),device=self.device)  # maybe batch size here.
    for stack_id in range(len(self.stacks)):
        for block_id in range(len(self.stacks[stack_id])):
          b, f = self.stacks[stack_id][block_id](backcast)
          backcast = backcast - b
          forecast = forecast + f
    return backcast, forecast

  def training_step(self, batch, batch_idx):
    x, y = batch
    _, forecast = self(x)
    
    if self.loss == 'mase':
      loss = self.loss(forecast, squeeze_last_dim(y), x)
    else:
      loss = self.loss(forecast, squeeze_last_dim(y))
    
    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, batch_idx):
    if self.no_val:
      return None
  
    x, y = batch
    _, forecast = self(x)
    
    if self.loss == 'mase':
      loss = self.loss(forecast, squeeze_last_dim(y), x)
    else:
      loss = self.loss(forecast, squeeze_last_dim(y))
      
    self.log('val_loss', loss)
    return loss

  def test_step(self, batch, batch_idx):
    x, y = batch
    _, forecast = self(x)
    
    if self.loss == 'mase':
      loss = self.loss(forecast, squeeze_last_dim(y), x)
    else:      
      loss = self.loss(forecast, squeeze_last_dim(y))
      
    self.log('test_loss', loss)
    return loss

  def configure_optimizers(self):
    if self.optimizer_name == 'adam':
      optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
    elif self.optimizer_name == 'sgd':
      optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
    elif self.optimizer_name == 'rmsprop':
      optimizer = optim.RMSprop(self.parameters(), lr=self.learning_rate)
    else:
      raise ValueError(f'Unknown optimizer name: {optimizer}.')
    return optimizer

  def predict(self, x, return_backcast=False):
    self.eval()
    backcast, forecast = self(torch.tensor(x, dtype=torch.float))
    backcast = backcast.detach().cpu().numpy()
    forecast = forecast.detach().cpu().numpy()
    if len(x.shape) == 3:
      backcast = np.expand_dims(backcast, axis=-1)
      forecast = np.expand_dims(forecast, axis=-1)
    if return_backcast:
        return backcast
    return forecast

  
class TimeSeriesDataset(Dataset):
  def __init__(self, data, backcast, forecast):
      super(TimeSeriesDataset, self).__init__()
      self.data = data
      self.backcast = backcast
      self.forecast = forecast
      self.items = []

      total_len = self.backcast + self.forecast
      for row in range(self.data.shape[0]):
          for col_start in range(0, self.data.shape[1] - total_len + 1):
              seq = self.data[row, col_start:col_start + total_len]
              if not np.isnan(seq).any():
                  self.items.append((row, col_start))
  
  def __len__(self):
    return len(self.items)

  def __getitem__(self, idx):
    row, col = self.items[idx]
    x = self.data[row, col:col+self.backcast]
    y = self.data[row, col+self.backcast:col+self.backcast+self.forecast]

    return torch.FloatTensor(x), torch.FloatTensor(y)    
      
class TimeSeriesCollectionDataModule(pl.LightningDataModule):
  def __init__(self, 
               train_data, 
               backcast=10, 
               forecast=2, 
               batch_size=512, 
               split_ratio=0.8):
    
      super(TimeSeriesCollectionDataModule, self).__init__()
      self.train_data = train_data
      self.backcast = backcast
      self.forecast = forecast
      self.batch_size = batch_size
      self.split_ratio = split_ratio

  def setup(self, stage:str=None):      

      # shuffle rows      
      all_train_data = train_data.sample(frac=1).reset_index(drop=True)
      train_rows = int(self.split_ratio * len(all_train_data))
      
      self.train_data = all_train_data.iloc[:train_rows].values      
      self.val_data = all_train_data.iloc[train_rows:].values
      
      self.train_dataset = TimeSeriesDataset(self.train_data, self.backcast, self.forecast)
      self.val_dataset = TimeSeriesDataset(self.val_data, self.backcast, self.forecast)    
    
  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle = True, num_workers=0)

  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
   
class TimeSeriesCollectionTestModule(pl.LightningDataModule):
  def __init__(self, 
               train_data,
               test_data,
               backcast=70, 
               forecast=14, 
               batch_size=512
               ):
    
      super(TimeSeriesCollectionTestModule, self).__init__()
      self.train_data = train_data.values
      self.test_data = test_data.values
      self.backcast = backcast
      self.forecast = forecast
      self.batch_size = batch_size

  def setup(self, stage:str=None):      
        
      # Create test data by concatenating last `backcast` samples from 
      # train_data and first `forecast` samples from test_data
      test_data_sequences = []      
      for train_row, test_row in zip(self.train_data, self.test_data):
        train_row = train_row[~np.isnan(train_row)]
        sequence = np.concatenate((train_row[-self.backcast:], test_row[:self.forecast]))
        test_data_sequences.append(sequence)
        
      self.test_data = np.array(test_data_sequences)     
      self.test_dataset = TimeSeriesDataset(self.test_data, self.backcast, self.forecast)
    
  def test_dataloader(self):
    return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle = True, num_workers=0)

    
def smape(preds, targets):
  # flatten
  targets = torch.reshape(targets, (-1,))
  preds = torch.reshape(preds, (-1,))
  return torch.mean(2.0 * torch.abs(targets - preds) / (torch.abs(targets) + torch.abs(preds)))
          

def get_trainer(max_epochs=100, dev=False, val_nepoch=5):
  tb_logger = pl_loggers.TensorBoardLogger(save_dir="../logs/", name="nbeats_smape")
  csv_logger = pl_loggers.CSVLogger(save_dir="../logs/", name="nbeats_csv")
  loggers = [tb_logger, csv_logger]
  
  trainer =  pl.Trainer(
    accelerator='auto'
    ,max_epochs=max_epochs   
    ,fast_dev_run=dev
    ,logger=[tb_logger]
    ,check_val_every_n_epoch=val_nepoch   
    )
  return trainer

def select_loss(loss_fn, freq: int = 1):
  if loss_fn == 'smape':       
    return smape
  
  elif loss_fn == 'mase':
    return MAsE(freq)
  
  elif loss_fn == 'mse':      
    return nn.MSELoss()
  else:
    raise ValueError(f'Unknown loss name: {loss_fn}.')
  

#%%
## hyperparameters
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

forecast_multiplier = 5
forecast = data_id_info.Horizon
backcast = data_id_info.Horizon * forecast_multiplier
batch_size = 1024
split_ratio = 0.8
train = True
max_epochs = 300
dev = True
chkpoint = None #"../logs/nbeats/version_6/checkpoints/epoch=499-step=80500.ckpt"
val_nepoch = 5
loss = select_loss('smape', freq)

print("Script File Path\t", scrpt_path)
print("Train File Path \t", train_file)
print("Test File Path  \t", test_file)
print("Seasonal Period \t", seasonal_period)
print("Forecast        \t", forecast)
print("Forecast Mult   \t", forecast_multiplier)
print("Backcast        \t", backcast)
print("Frequency       \t", freq)
print("Batch Size      \t", batch_size)
print("Split Ratio     \t", split_ratio)
print("max_epochs      \t", max_epochs)
print("chkpoint        \t", chkpoint)
print("fast_dev_run    \t", dev)
print("val_nepoch      \t", val_nepoch)
print("\n")


model = NBeatsNet(
  stack_types=stack_types,
  n_forecast=forecast, 
  n_backcast=backcast,
  loss=loss,  
  learning_rate = 1e-5,
  thetas_dim = 5,
  blocks_per_stack = 1,
  share_weights_in_stack = False,
  hidden_layer_units = 512,
  optimizer='adam',
  no_val = False 
)

if chkpoint is not None:
  chkpoint = pathlib.Path(scrpt_path,chkpoint)
  model = NBeatsNet.load_from_checkpoint(chkpoint)
    
#%%

if train:
  if dev:
    train_data = train_data.iloc[:1000]
    test_data = test_data.iloc[:1000]

  dmc = TimeSeriesCollectionDataModule(
    train_data=train_data, 
    backcast=backcast, 
    forecast=forecast, 
    batch_size=batch_size, 
    split_ratio=split_ratio
    )
  
  trainer = get_trainer(max_epochs=max_epochs, dev=dev, val_nepoch=val_nepoch)
  
  trainer.fit(model, datamodule=dmc, ckpt_path=chkpoint)
  trainer.validate(model, datamodule=dmc)
else:
  test_module = TimeSeriesCollectionTestModule(
  test_data=test_data, 
  backcast=backcast, 
  forecast=forecast, 
  batch_size=batch_size)
  trainer = get_trainer(max_epochs=max_epochs, dev=dev, val_nepoch=val_nepoch)
  trainer.test(model, datamodule=test_module)
  



# %%
