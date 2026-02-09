from time import time
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import lightning.pytorch as pl

from .losses import *
from .blocks import blocks as b
from .constants import LOSSES, OPTIMIZERS, BLOCKS


class NBeatsNet(pl.LightningModule):
  SEASONALITY = 'seasonality'
  TREND = 'trend'
  GENERIC = 'generic'
  AE = 'ae'
    
  def __init__(
      self,
      backcast_length:int, 
      forecast_length:int,  
      stack_types:list = None,
      n_blocks_per_stack:int = 1, 
      g_width:int = 512, 
      s_width:int = 2048, 
      t_width:int = 256, 
      ae_width:int = 512,
      share_weights:bool = False, 
      thetas_dim:int = 5, 
      learning_rate: float = 1e-3,
      loss: str = 'SMAPELoss', 
      no_val:bool = False,  
      optimizer_name:str = 'Adam', # 'Adam', 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'AdamW'
      activation:str = 'ReLU', # 'ReLU', 'RReLU', 'PReLU', 'ELU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU', 'Sigmoid', 'GELU'
      frequency:int = 1, 
      active_g:bool = False,
      latent_dim:int = 5,
      sum_losses:bool = False,
      basis_dim:int = 32
    ):

    """A PyTorch Lightning module for the N-BEATS network for time series forecasting.

    N-Beats is based on the idea of neural basis expansion, where the input time series
    is decomposed into a linear combination of basis functions learned by the network.
    The network consists of multiple stacks of blocks, each block containing a fully
    connected layer followed by an activation function and a linear layer. The output
    of each block is added to the input of the next block (backward residual
    connection) and also to the final output of the stack (forward residual
    connection). The final output of the stack is split into two parts: the backcast,
    which is used to reconstruct the input time series, and the forecast, which is used
    to predict the future values.
    
    Based on the paper by Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2019).
    N-BEATS: Neural basis expansion analysis for interpretable time series forecasting.
    arXiv preprint arXiv:1905.10437. https://openreview.net/forum?id=r1ecqn4YwB

    Attributes
    ----------
    SEASONALITY : str
        Constant for seasonality stack type.
    TREND : str
        Constant for trend stack type.
    GENERIC : str
        Constant for generic stack type.

    Parameters
    ----------
    backcast_length : int, optional
        The length of the historical data.  It is customary to use a multiple of the 
        forecast (H)orizon (2H,3H,4H,5H,...).
    forecast_length : int, optional
        The length of the forecast horizon.
    generic_architecture : bool, optional
        If True, use the generic architecture, otherwise use the interpretable
        architecture. 
        Default True.
    n_blocks_per_stack : int, optional
        The number of blocks per stack. 
        Default 1.
    n_stacks : int, optional
        The number of stacks, by default 5 when Generic, else fixed at 2  (1 Seasonal
        + 1 Trend) when interpretable.
    g_width, s_width, t_width : int, optional
        The width of the fully connected layers in a Generic block(g), Seasonal
        Block(s), or Trend Block(t). 
        Default (g = 512, s = 2048, t = 256).
    share_weights : bool, optional
        If True, share initial weights across blocks.
    thetas_dim : int, optional
        The dimensionality of the waveform generator parameters in the Generic and
        Trend blocks.
    learning_rate : float, optional
        The learning rate for the optimizer, by default 1e-5.
    loss : str, optional
        The loss function to use, defined in LOSSES.
    no_val : bool, optional
        If True, skip validation during training. 
        Default False.
    optimizer_name : str, optional
        The name of the optimizer to use. Allowed methDefined in OPTIMIZERS. 
        Default 'Adam'.
    activation : str, optional
        The activation function to use.  Defined in ACTIVATIONS. 
        Default : 'ReLU'.
    frequency : int, optional
        The frequency of the data.  Used only when MASELoss is used as teh loss funtion. Default 1.
    activate_g : bool, optional
        If True, the function implemented by the waveform generators in the forecast anb backcast blocks has n activation function applied to the output.  
        This feature is not defined in the original design of N-BEATS.  However, since both the Trend and Seasonality blocks apply non-linear functions
        to the waveform parameters generated in he preceeding layer, applying an activation function here mirrors that structure.  I've found that this
        approch improved convergence of GENERIC models.  Default : False.
    sum_losses : bool, optional
        If True, the total loss is defined as forecast_loss + 1/4 Backcast_loss.  This is an experimental feature. Default False.
    latent_dim : int, optional
        The dimensionality of the latent space in the AutoEncoder blocks. Default 5.
    basis_dim : int, optional
        The dimensionality of the basis space in the Wavelet blocks. Default 32.
    
    Inputs
    ------
    stack_input of shape `(batch_size, input_chunk_length)`
        Tensor containing the input sequence.

    Outputs
    -------
    stack_residual of shape `(batch_size, input_chunk_length)`
        Tensor containing the 'backcast' of the block, which represents an approximation of `x`
        given the constraints of the functional space determined by `g`.
    stack_forecast of shape `(batch_size, output_chunk_length)`
        Tensor containing the forward forecast of the stacks.
    """    
  
    super(NBeatsNet, self).__init__()
    self.backcast_length = backcast_length
    self.forecast_length = forecast_length
    self.n_blocks_per_stack = n_blocks_per_stack
    self.share_weights = share_weights
    self.g_width = g_width
    self.s_width = s_width
    self.t_width = t_width
    self.ae_width = ae_width
    self.thetas_dim = thetas_dim
    self.learning_rate = learning_rate
    self.no_val = no_val
    self.optimizer_name = optimizer_name
    self.frequency = frequency
    self.loss = loss
    self.activation = activation
    self.active_g = active_g
    self.sum_losses = sum_losses
    self.latent_dim = latent_dim
    self.basis_dim = basis_dim
    self.loss_fn = self.configure_loss()    
    
    
    self.save_hyperparameters()   
      
    if stack_types is not None:
      self.stack_types = stack_types
      self.n_stacks = len(stack_types)                              
    else:
      raise ValueError("Stack architecture must be specified.")
          
    self.stacks = nn.ModuleList()
    for stack_type in self.stack_types:
      self.stacks.append(self.create_stack(stack_type))  
               
           
  def create_stack(self, stack_type):
    
    blocks = nn.ModuleList()
    if (stack_type not in BLOCKS):
        raise ValueError(f"Unknown stack type: {stack_type}. Please select one of {BLOCKS}")      
    
    for block_id in range(self.n_blocks_per_stack):
        if self.share_weights and block_id != 0:
          # share weights across blocks
          block = blocks[-1]  
        else:           
          if stack_type in ["Generic", "BottleneckGeneric", "GenericAE", "BottleneckGenericAE", "GenericAEBackcast", "GenericAEBackcastAE"]:
            units = self.g_width
          elif stack_type in ["Seasonality", "SeasonalityAE"]:
            units = self.s_width
          elif stack_type in ["Trend", "TrendAE"]:
            units = self.t_width
          elif stack_type in ["AutoEncoder", "AutoEncoderAE"]:
            units = self.ae_width
          else:
            units = self.g_width

          if (stack_type == "GenericAE" or
              stack_type == "BottleneckGenericAE" or
              stack_type == "SeasonalityAE" or
              stack_type == "AutoEncoderAE" or
              stack_type == "TrendAE"):
            block = getattr(b,stack_type)(
                units, self.backcast_length, self.forecast_length, self.thetas_dim, 
                self.share_weights, self.activation, self.active_g, self.latent_dim)
          elif "Wavelet" in stack_type:
            block = getattr(b,stack_type)(
                units, self.backcast_length, self.forecast_length, self.basis_dim, 
                self.share_weights, self.activation, self.active_g)
          else:
            block = getattr(b,stack_type)(
                units, self.backcast_length, self.forecast_length, self.thetas_dim, 
                self.share_weights, self.activation, self.active_g) 
                            
        blocks.append(block)   

    return blocks

  def forward(self, x):
    x = torch.squeeze(x,-1)    
    y = torch.zeros(
            x.shape[0],
            self.forecast_length,
            device=x.device,
            dtype=x.dtype)
    
    for stack_id in range(len(self.stacks)):
        for block_id in range(len(self.stacks[stack_id])):
          x_hat, y_hat = self.stacks[stack_id][block_id](x)
          x = x - x_hat
          y = y + y_hat
    stack_residual = x
    stack_forecast = y
    return stack_residual, stack_forecast

  
  def training_step(self, batch, batch_idx):
    """Training step for the NBeatsNet model.

    Args:
        batch (Tensor): Batch of training data.
        batch_idx (Tensor): Index of training batch.

    Returns:
        Tensor: Loss for the training step.
    """
    # run forward pass
    x, y = batch
    backcast, forecast = self(x)
    
    # calculate loss
    loss = self.loss_fn(forecast, y)
    
    if self.sum_losses:
      backcast_loss = self.loss_fn(backcast, torch.zeros_like(backcast))
      loss = loss + backcast_loss * 0.25

    self.log('train_loss', loss, prog_bar=True)
    return loss

  def validation_step(self, batch, batch_idx):
    if self.no_val:
      return None
    
    x, y = batch
    backcast, forecast = self(x)
    loss = self.loss_fn(forecast, y)
    
    if self.sum_losses:
      backcast_loss = self.loss_fn(backcast, torch.zeros_like(backcast))
      loss = loss + backcast_loss * 0.25

    self.log('val_loss', loss, prog_bar=True)
    return loss
  
  def test_step(self, batch, batch_idx):
    x, y = batch
    backcast, forecast = self(x)
    loss = self.loss_fn(forecast, y)
    
    if self.sum_losses:
      backcast_loss = self.loss_fn(backcast, torch.zeros_like(backcast))
      loss = loss + backcast_loss * 0.25

    self.log('test_loss', loss)
    return loss

  def predict_step(self, batch, batch_idx, dataloader_idx=None):
      self.eval()
      _, forecast = self(batch)
      return forecast.detach().cpu().numpy()
    
  def configure_loss(self):
    if self.loss not in LOSSES:
        raise ValueError(f"Unknown loss function name: {self.loss}. Please select one of {LOSSES}")
    if self.loss == 'MASELoss':
        return MASELoss(self.frequency)
    if self.loss == 'MAPELoss':
        return MAPELoss()
    if self.loss == 'SMAPELoss':
        return SMAPELoss()
    if self.loss == 'NormalizedDeviationLoss':
        return NormalizedDeviationLoss()
    else:
        return getattr(nn, self.loss)()
    
  def configure_optimizers(self):
    if self.optimizer_name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer name: {self.optimizer_name}. Please select one of {OPTIMIZERS}")
    
    optimizer = getattr(optim, self.optimizer_name)(self.parameters(), lr=self.learning_rate)
    return optimizer
    # scheduler = {
    #   'scheduler': StepLR(optimizer, step_size=10, gamma=0.1),
    #   'interval': 'epoch',  # could be 'step' if you want to update the learning rate at every optimization step
    #   'monitor': 'val_loss',  # Metric to monitor for performance improvement. Only necessary if `reduce_on_plateau` scheduler is used
    # }
    # return {'optimizer': optimizer, 'lr_scheduler': scheduler}


