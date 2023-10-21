
import torch
import pytorch_lightning as pl
from torch import nn
from torch.autograd import Variable
from ..constants import ACTIVATIONS
import numpy as np
import pywt
from scipy.signal import resample
from ..blocks import _SeasonalityGenerator, squeeze_last_dim, _TrendGenerator

class AERootBlock(nn.Module):
  def __init__(self, backcast_length, units, activation='ReLU', latent_dim=5):
    """The AERootBlock class is the basic building block of the N-BEATS network.  
    It consists of a stack of fully connected layers organized as an Autoencoder.
    It serves as the base class for the GenericAEBlock, SeasonalityAEBlock, 
    and the TrendAEBlock classes.
    
    Args:
        backcast_length (int): 
          The length of the historical data.  It is customary to use a multiple 
          of the forecast_length (H)orizon (2H,3H,4H,5H,...).
        units (int): 
          The width of the fully connected layers in the blocks comprising the 
          stacks of the generic model.
        activation (str, optional): 
          The activation function applied to each of the fully connected Linear 
          layers. Defaults to 'ReLU'.
        latent_dim (int, optional):
          The dimensionality of the latent space. Defaults to 5.

    Raises:
          ValueError: If the activation function is not in ACTIVATIONS.
    """
    super(AERootBlock, self).__init__()
    self.units = units
    self.backcast_length = backcast_length
    self.latent_dim = latent_dim
    
    if not activation in ACTIVATIONS:
      raise ValueError(f"'{activation}' is not in {ACTIVATIONS}")
    
    self.activation = getattr(nn, activation)()    
    
    self.fc1 = nn.Linear(backcast_length, units//2)
    self.fc2 = nn.Linear(units//2, latent_dim)
    self.fc3 = nn.Linear(latent_dim, units//2)
    self.fc4 = nn.Linear(units//2, units)
    

  def forward(self, x): 
    x = squeeze_last_dim(x)
    x = self.activation(self.fc1(x))
    x = self.activation(self.fc2(x))
    x = self.activation(self.fc3(x))
    x = self.activation(self.fc4(x))
    return x

class SeasonalityAEBlock(AERootBlock):
  def __init__(self, units, backcast_length, forecast_length,  thetas_dim=5, 
               share_weights = False, activation='ReLU', active_g:bool = False, latent_dim = 5):
    """The SeasonalityAEBlock is the basic building block of the N-BEATS network.  
    It consists of a stack of fully connected layers defined the parent class Block, 
    followed by a linear layer, which generates the parameters of a Fourier expansion.

    The SeasonalityAEBlock is an AutoEncoder version of the SeasonalityBlock where the presplit
    section of the network is an AutoEncoder.    

    Args:
        units (int): 
          The width of the fully connected layers in the blocks comprising the parent 
          class Block.
        backcast_length (int): 
          The length of the historical data.  It is customary to use a multiple of 
          the forecast_length (H)orizon (2H,3H,4H,5H,...).
        forecast_length (int): 
          The length of the forecast_length horizon.
        activation (str, optional): 
          The activation function passed to the parent class Block. Defaults to 'LeakyReLU'.
    """
    super(SeasonalityAEBlock, self).__init__(backcast_length, units, activation, latent_dim = latent_dim)

    self.backcast_linear = nn.Linear(units, 2 * int(backcast_length / 2 - 1) + 1, bias = False)
    self.forecast_linear = nn.Linear(units, 2 * int(forecast_length / 2 - 1) + 1, bias = False)
      
    self.backcast_g = _SeasonalityGenerator(backcast_length)
    self.forecast_g = _SeasonalityGenerator(forecast_length)

  def forward(self, x):
    x = super(SeasonalityAEBlock, self).forward(x)
    # linear compression
    backcast_thetas = self.backcast_linear(x)
    forecast_thetas = self.forecast_linear(x)
    
    # fourier expansion
    backcast = self.backcast_g(backcast_thetas)  
    forecast = self.forecast_g(forecast_thetas)
    
    return backcast, forecast
    
    return backcast_length, forecast_length

     
class GenericAEBackcastAEBlock(AERootBlock):  
  def __init__(self, 
                units:int,
                backcast_length:int,
                forecast_length:int,
                thetas_dim:int,
                share_weights:bool,
                activation:str = 'ReLU',
                active_g:bool = False,
                latent_dim:int = 5):
    
      super(GenericAEBackcastAEBlock, self).__init__(backcast_length, units, activation,latent_dim=latent_dim)
      
      self.units = units
      self.thetas_dim = thetas_dim
      self.share_weights = share_weights
      self.activation = getattr(nn, activation)()
      self.backcast_length = backcast_length
      self.forecast_length = forecast_length
      self.active_g = active_g        
      
      self.forecast_linear = nn.Linear(units, thetas_dim)
      self.forecast_g = nn.Linear(thetas_dim, forecast_length, bias = False)
      
      # Encoders
      self.b_encoder = nn.Sequential(
          nn.Linear(units, thetas_dim),
          nn.ReLU(),
      )
      
      # Decoders
      self.b_decoder = nn.Sequential(
          nn.Linear(thetas_dim, units),
          nn.ReLU(),
          nn.Linear(units, backcast_length),          
      )

  def forward(self, x):
    x = super(GenericAEBackcastAEBlock, self).forward(x)
    b = self.b_encoder(x)
    b = self.b_decoder(b)

    theta_f = self.forecast_linear(x)
    f = self.forecast_g(theta_f)
    
    # N-BEATS paper does not apply activation here, but Generic models will not converge sometimes without it
    if self.active_g:
      b = self.activation(b)
      f = self.activation(f)
    return b,f     
  
class AutoEncoderAEBlock(AERootBlock):  
  def __init__(self, 
                units:int,
                backcast_length:int,
                forecast_length:int,
                thetas_dim:int,
                share_weights:bool,
                activation:str = 'ReLU',
                active_g:bool = False,
                latent_dim:int = 5):
    
    super(AutoEncoderAEBlock, self).__init__(backcast_length, units, activation, latent_dim=latent_dim)
    
    self.units = units
    self.thetas_dim = thetas_dim
    self.share_weights = share_weights
    self.activation = getattr(nn, activation)()
    self.backcast_length = backcast_length
    self.forecast_length = forecast_length      
    self.active_g = active_g  
    
    # Encoders
    if share_weights:
      self.b_encoder = self.f_encoder = nn.Sequential(
        nn.Linear(units, thetas_dim),
        nn.ReLU(),
      )
    else:
      self.b_encoder = nn.Sequential(
          nn.Linear(units, thetas_dim),
          nn.ReLU(),
      )      
      self.f_encoder = nn.Sequential(
          nn.Linear(units, thetas_dim),
          nn.ReLU(),
      )      

    self.b_decoder = nn.Sequential(
        nn.Linear(thetas_dim, units),
        nn.ReLU(),
        nn.Linear(units, backcast_length),
    )
    self.f_decoder = nn.Sequential(
        nn.Linear(thetas_dim, units),
        nn.ReLU(),
        nn.Linear(units, forecast_length),
    )

  def forward(self, x):
    x = super(AutoEncoderAEBlock, self).forward(x)
    b = self.b_encoder(x)
    b = self.b_decoder(b)
    
    f = self.f_encoder(x)
    f = self.f_decoder(f)

    # N-BEATS paper does not apply activation here, but Generic models will not converge sometimes without it
    if self.active_g:
      b = self.activation(b)
      f = self.activation(f)        
  
    return b,f          

class GenericAEBlock(AERootBlock):
  def __init__(self, 
               units:int, 
               backcast_length:int, 
               forecast_length:int, 
               thetas_dim:int = 5, 
               share_weights:bool= False, 
               activation:str = 'ReLU', 
               active_g:bool = False,
               latent_dim:int = 5):
    """The GenericAEBlock is the basic building block of the N-BEATS network.  
    It consists of a stack of fully connected layers, followed by
    two linear layers. The first, backcast_linear, generates the parameters 
    of a waveform generator, which is implemented by the function 
    defined in the next layer. These two layers can also be thought of os a 
    compression and expansion layer or rudimentary AutoEncoder.

    The GenericAEBlock is an AutoEncoder version of the GenericBlock where the presplit
    section of the network is an AutoEncoder.
    
    Args:
        units (int): 
          The width of the fully connected layers in the blocks comprising 
          the stacks of the generic model
        backcast_length (int): 
          The length of the historical data.  It is customary to use a 
          multiple of the forecast_length (H)orizon (2H,3H,4H,5H,...).
        forecast_length (int): 
          The length of the forecast_length horizon.
        thetas_dim (int, optional): 
          The dimensionality of the wavefor generator parameters. Defaults 
          to 5.
        share_weights (bool, optional): 
          If True, the initial weights of the Linear laers are shared. 
          Defaults to False.
        activation (str, optional): ß
          The activation function used in the parent class Block, and 
          optionally as the non-linear activation of the backcast_g and 
          forecast_g layers. Defaults to 'ReLU'.
        active_g (bool, optional): 
          This parameter when enabled applies the model's activation 
          funtion to the linear funtions (gb and gf) which are found by 
          the network in the last layer of each block using the theta parameters
          found in the preceding layer. Enabling this activation function 
          seems to help the Generic model converge. 
          The parameter `active_g` is not a feature found in the original 
          N-Beats paper. Defaults to False.
    """
    super(GenericAEBlock, self).__init__(backcast_length, units, activation, latent_dim=latent_dim)
    
    if share_weights:
        self.backcast_linear = self.forecast_linear = nn.Linear(units, thetas_dim)
    else:
        self.backcast_linear = nn.Linear(units, thetas_dim)
        self.forecast_linear = nn.Linear(units, thetas_dim)
        
    self.backcast_g = nn.Linear(thetas_dim, backcast_length, bias = False) 
    self.forecast_g = nn.Linear(thetas_dim, forecast_length, bias = False)
    self.active_g = active_g
    
  def forward(self, x):
    x = super(GenericAEBlock, self).forward(x)
    theta_b = self.backcast_linear(x)
    theta_f = self.forecast_linear(x)
    
    backcast_length = self.backcast_g(theta_b)
    forecast_length = self.forecast_g(theta_f)
    
    # N-BEATS paper does not apply activation here, but Generic models will not converge sometimes without it
    if self.active_g:
      backcast_length = self.activation(backcast_length)
      forecast_length = self.activation(forecast_length)

    return backcast_length, forecast_length  


class TrendAEBlock(AERootBlock):
  def __init__(self, units, backcast_length, forecast_length, thetas_dim, 
               share_weights = True, activation='LeakyReLU', active_g:bool = False, latent_dim = 5):
    """The TrendAEBlock implements the function whose parameters are generated by the _TrendGenerator block.  
    The TrendAEBlock is an AutoEncoder version of the TrendBlock where the presplit section of the network
    is an AutoEncoder.

    Args:
        units (int): 
          The width of the fully connected layers in the blocks comprising the parent class Block.
        backcast_length (int): 
          The length of the historical data.  It is customary to use a multiple of the forecast_length (H)orizon (2H,3H,4H,5H,...).
        forecast_length (int): 
          The length of the forecast_length horizon.
        thetas_dim (int): 
          The dimensionality of the _TrendGenerator polynomial.
        share_weights (bool, optional): 
          If True, the inital weights of the Linear layers are shared. Defaults to True.
        activation (str, optional): 
          The activation function passed to the parent class Block. Defaults to 'LeakyReLU'.
    """
    super(TrendAEBlock, self).__init__(backcast_length, units, activation, latent_dim=latent_dim)
    if share_weights:
        self.backcast_linear = self.forecast_linear = nn.Linear(units, thetas_dim)
    else:
        self.backcast_linear = nn.Linear(units, thetas_dim)
        self.forecast_linear = nn.Linear(units, thetas_dim)
        
    self.backcast_g = _TrendGenerator(thetas_dim, backcast_length)
    self.forecast_g = _TrendGenerator(thetas_dim, forecast_length)

  def forward(self, x):
    x = super(TrendAEBlock, self).forward(x)
    
    # linear compression
    backcast_thetas = self.backcast_linear(x)
    forecast_thetas = self.forecast_linear(x)
    
    #  Vandermonde basis expansion
    backcast = self.backcast_g(backcast_thetas)    
    forecast = self.forecast_g(forecast_thetas)
            
    return backcast, forecast

