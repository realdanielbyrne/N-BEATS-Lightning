#%%
from time import time
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.nn import functional as F
import lightning.pytorch as pl
import pathlib

from lightning.pytorch import loggers as pl_loggers


def squeeze_last_dim(tensor):
  if len(tensor.shape) == 3 and tensor.shape[-1] == 1:  # (128, 10, 1) => (128, 10).
      return tensor[..., 0]
  return tensor

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
      loss_fn: nn.Module,
      stack_types =(GENERIC_BLOCK,GENERIC_BLOCK,GENERIC_BLOCK),
      n_backcast:int = 30, # default of 5*forecast_length or 5H , N-BEATS paper tested 2H,3H,4H,5H,6H,7H          
      n_forecast:int = 6,  # forecast (H)orizon
      thetas_dim:int = 5, # Output of FC layer in each of the forecast and backcast branches of each block
      blocks_per_stack:int = 1, # N-BEATS paper best results Generic 1 blk/stk, Trend 3 blk/stk, Seasonality 3 blk/stk
      share_weights_in_stack:bool = True, # Generic model prefers no weight sharing, while interpretable model does.
      hidden_layer_units:int = 512,
      learning_rate: float = 1e-5,
      optimizer: str = 'adam',
      nb_harmonics = None,
      no_val:bool = False,
    ):
    super(NBeatsNet, self).__init__()
    self.stack_types = stack_types
    self.n_backcast = n_backcast
    self.n_forecast = n_forecast
    self.thetas_dim = thetas_dim
    self.blocks_per_stack = blocks_per_stack
    self.share_weights_in_stack = share_weights_in_stack
    self.hidden_layer_units = hidden_layer_units
    self.learning_rate = learning_rate
    self.optimizer_name = optimizer
    self.nb_harmonics = nb_harmonics
    self.no_val = no_val
    self.loss_fn = loss_fn
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
        block_init = self.select_block(stack_type)
        if self.share_weights_in_stack and block_id != 0:
            block = blocks[-1]  # pick up the last one when we share weights.
        else:
            block = block_init(
                self.hidden_layer_units, self.thetas_dim,
                self.n_backcast, self.n_forecast,
                self.nb_harmonics
            )
        blocks.append(block)   

    return blocks
  
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
    loss = self.loss_fn(forecast, squeeze_last_dim(y))
    
    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, batch_idx):
    if self.no_val:
      return None
    
    x, y = batch
    _, forecast = self(x)
    loss = self.loss_fn(forecast, squeeze_last_dim(y))
    
    self.log('val_loss', loss)
    return loss

  def test_step(self, batch, batch_idx):
    x, y = batch
    _, forecast = self(x)
    loss = self.loss_fn(forecast, squeeze_last_dim(y))
      
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



# %%
