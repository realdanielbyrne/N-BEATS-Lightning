from time import time
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import lightning.pytorch as pl
from .losses import MASELoss, SMAPELoss, MAPELoss


def squeeze_last_dim(tensor):
  if len(tensor.shape) == 3 and tensor.shape[-1] == 1:  # (128, 10, 1) => (128, 10).
      return tensor[..., 0]
  return tensor

class Block(nn.Module):
  def __init__(self, backcast, units):
    super(Block, self).__init__()
    self.units = units
    self.backcast = backcast
    
    self.fc1 = nn.Linear(backcast, units)
    self.fc2 = nn.Linear(units, units)
    self.fc3 = nn.Linear(units, units)
    self.fc4 = nn.Linear(units, units)

  def forward(self, x): 
    x = squeeze_last_dim(x)
    x = F.leaky_relu(self.fc1(x))
    x = F.leaky_relu(self.fc2(x))
    x = F.leaky_relu(self.fc3(x))
    x = F.leaky_relu(self.fc4(x))
    return x
  
class GenericBlock(Block):
  def __init__(self, units, backcast=10, forecast=5, thetas_dim = 5, share_thetas= False):
    super(GenericBlock, self).__init__(backcast, units)
    
    if share_thetas:
        self.backcast_linear = self.forecast_linear = nn.Linear(units, thetas_dim, bias=False)
    else:
        self.backcast_linear = nn.Linear(units, thetas_dim, bias=False)
        self.forecast_linear = nn.Linear(units, thetas_dim, bias=False)
        
    self.backcast_g = nn.Linear(thetas_dim, backcast, bias=True)
    self.forecast_g = nn.Linear(thetas_dim, forecast, bias=True)

  def forward(self, x):
    x = super(GenericBlock, self).forward(x)
    theta_b = self.backcast_linear(x)
    theta_f = self.forecast_linear(x)
    backcast = self.backcast_g(theta_b)  
    forecast = self.forecast_g(theta_f)  

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
  def __init__(self, units, backcast=10, forecast=5):
    super(SeasonalityBlock, self).__init__(backcast, units)

    self.backcast_linear = nn.Linear(units, 2 * int(backcast / 2 - 1) + 1, bias = False)
    self.forecast_linear = nn.Linear(units, 2 * int(forecast / 2 - 1) + 1, bias = False)
      
    self.backcast_g = _SeasonalityGenerator(backcast)
    self.forecast_g = _SeasonalityGenerator(forecast)

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
  def __init__(self, units, backcast=10, forecast=5, thetas_dim = 5, share_thetas= True):
    super(TrendBlock, self).__init__(backcast, units)
    if share_thetas:
        self.backcast_linear = self.forecast_linear = nn.Linear(units, thetas_dim, bias=False)
    else:
        self.backcast_linear = nn.Linear(units, thetas_dim, bias=False)
        self.forecast_linear = nn.Linear(units, thetas_dim, bias=False)
        
    self.backcast_g = _TrendGenerator(thetas_dim, backcast)
    self.forecast_g = _TrendGenerator(thetas_dim, forecast)

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
  SEASONALITY = 'seasonality'
  TREND = 'trend'
  GENERIC = 'generic'
    
  def __init__(
      self,
      backcast:int = 30, # default of 5*forecast_length or 5H , N-BEATS paper tested 2H,3H,4H,5H,6H,7H          
      forecast:int = 6,  # forecast (H)orizon
      generic_architecture:bool = True,
      n_blocks_per_stack:int = 1, # Number of stacks N-BEATS paper: Generic 1 blk/stk, 2 Stacks (1 Trend 3 blks, 1 Seasonality 3 blks)
      n_stacks:int = 2, # number of stacks
      g_width:int = 512, # the width of the fully connected layers in the blocks comprising the stacks of the generic model
      s_width:int = 2048, # the width of the fully connected layers in the blocks comprising the seasonality stack of the interpretable model.
      t_width:int = 256, # the width of the fully connected layers in the blocks comprising the trend stack of the interpretable model
      share_weights_in_stack:bool = False, # Generic model prefers no weight sharing, while interpretable model does.
      thetas_dim:int = 5, # 5 for generic, 2 or 3 for interpretable tomimic trend
      learning_rate: float = 1e-5,  
      loss: str = 'smape', # 'mape', 'smape', 'mase'
      no_val:bool = False,  # set to True to skip validation during training ( when using the entire test set for training)
      optimizer_name:str = 'adamw', # 'adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adamw'
      frequency:int = 1, # frequency of the data
      # https://openreview.net/forum?id=r1ecqn4YwB
    ):
    
    super(NBeatsNet, self).__init__()
    self.backcast = backcast
    self.forecast = forecast
    self.n_blocks_per_stack = n_blocks_per_stack
    self.n_stacks = n_stacks
    self.share_weights_in_stack = share_weights_in_stack
    self.g_width = g_width
    self.s_width = s_width
    self.t_width = t_width
    self.thetas_dim = thetas_dim
    self.learning_rate = learning_rate
    self.no_val = no_val
    self.optimizer_name = optimizer_name
    self.frequency = frequency
    self.loss = loss
    self.generic_architecture = generic_architecture
    self.loss_fn = self.configure_loss()    
    self.save_hyperparameters()    
                        
    if generic_architecture:  
      self.stack_types = [NBeatsNet.GENERIC] * self.n_stacks
    else:
      self.stack_types = [NBeatsNet.TREND, NBeatsNet.SEASONALITY]
         
    self.stacks = nn.ModuleList()
    for stack_id in range(len(self.stack_types)):
      self.stacks.append(self.create_stack(stack_id))  
    
    self.stacks[-1][-1].backcast_linear.requires_grad_(False)
    self.stacks[-1][-1].backcast_g.requires_grad_(False)
           
  def create_stack(self, stack_id):
    stack_type = self.stack_types[stack_id]
    blocks = nn.ModuleList()
    
    for block_id in range(self.n_blocks_per_stack):
        if self.share_weights_in_stack and block_id != 0:
            # share initial weights across blocks
            block = blocks[-1]  
        else:
            if self.generic_architecture:
                block = GenericBlock(
                    self.g_width, self.backcast, self.forecast, self.thetas_dim
                )
            elif stack_type == NBeatsNet.SEASONALITY:
                block = SeasonalityBlock(
                    self.s_width, self.backcast, self.forecast
                )
            elif stack_type == NBeatsNet.TREND:
                block = TrendBlock(
                    self.t_width, self.backcast, self.forecast, self.thetas_dim
                )
                
        blocks.append(block)   

    return blocks

  def forward(self, backcast):
    backcast = squeeze_last_dim(backcast)
    forecast = torch.zeros(size=(backcast.size()[0], self.forecast,),device=self.device)  
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
    
    self.log('train_loss', loss, prog_bar=True)
    return loss

  def validation_step(self, batch, batch_idx):
    if self.no_val:
      return None
    
    x, y = batch
    _, forecast = self(x)
    loss = self.loss_fn(forecast, squeeze_last_dim(y))
    
    self.log('val_loss', loss, prog_bar=True)
    return loss

  def test_step(self, batch, batch_idx):
    x, y = batch
    _, forecast = self(x)
    loss = self.loss_fn(forecast, squeeze_last_dim(y))
      
    self.log('test_loss', loss)
    return loss

  def configure_loss(self):
    if self.loss == 'mase':
      return MASELoss(self.frequency)
    if self.loss == 'mape':
      return MAPELoss()
    if self.loss == 'smape':
      return SMAPELoss()
    else:
      raise ValueError(f"Unknown loss function name:  Please select one of 'mase', 'mape', or 'smape'.")
    
  def configure_optimizers(self):
    if self.optimizer_name == 'adam':
      opti = optim.Adam(self.parameters(), lr=self.learning_rate)
    elif self.optimizer_name == 'sgd':
      opti = optim.SGD(self.parameters(), lr=self.learning_rate)
    elif self.optimizer_name == 'rmsprop':
      opti = optim.RMSprop(self.parameters(), lr=self.learning_rate)
    elif self.optimizer_name == 'adagrad':
      opti = optim.Adagrad(self.parameters(), lr=self.learning_rate)
    elif self.optimizer_name == 'adadelta':
      opti = optim.Adadelta(self.parameters(), lr=self.learning_rate)
    elif self.optimizer_name == 'adamw':
      opti = optim.AdamW(self.parameters(), lr=self.learning_rate)
    else:
      raise ValueError(f"Unknown optimizer name: {self.optimizer_name}. Please select one of 'adam', 'sgd', or 'rmsprop'.")
    return opti

  def predict(self, x, return_backcast=False):
    self.eval()
    backcast, forecast = self(torch.tensor(x, dtype=torch.float))
    backcast = backcast.detach().cpu().numpy()
    forecast = forecast.detach().cpu().numpy()

    if return_backcast:
        return backcast
    return forecast

