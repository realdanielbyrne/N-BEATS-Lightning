import torch
import pytorch_lightning as pl
from torch import nn
from torch.autograd import Variable
from ..constants import ACTIVATIONS
import numpy as np

def squeeze_last_dim(tensor):
  if len(tensor.shape) == 3 and tensor.shape[-1] == 1:  # (128, 10, 1) => (128, 10).
      return tensor[..., 0]
  return tensor

class Block(nn.Module):
  def __init__(self, backcast, units, activation='ReLU'):
    """The Block class is the basic building block of the N-BEATS network.  
    It consists of a stack of fully connected layers.It serves as the base 
    class for the GenericBlock, SeasonalityBlock, and TrendBlock classes.

    Args:
        backcast (int): 
          The length of the historical data.  It is customary to use a 
          multiple of the forecast (H)orizon (2H,3H,4H,5H,...).
        units (int): 
          The width of the fully connected layers in the blocks comprising 
          the stacks of the generic model
        activation (str, optional): 
          The activation function applied to each of the fully connected 
          Linear layers. Defaults to 'ReLU'.

    Raises:
          ValueError: If the activation function is not in ACTIVATIONS.
    """
    super(Block, self).__init__()
    self.units = units
    self.backcast = backcast
    
    if not activation in ACTIVATIONS:
      raise ValueError(f"'{activation}' is not in {ACTIVATIONS}")
    
    self.activation = getattr(nn, activation)()    
    
    self.fc1 = nn.Linear(backcast, units)
    self.fc2 = nn.Linear(units, units)
    self.fc3 = nn.Linear(units, units)
    self.fc4 = nn.Linear(units, units)

  def forward(self, x): 
    x = squeeze_last_dim(x)
    x = self.activation(self.fc1(x))
    x = self.activation(self.fc2(x))
    x = self.activation(self.fc3(x))
    x = self.activation(self.fc4(x))
    return x

class AERootBlock(nn.Module):
  def __init__(self, backcast, units, activation='ReLU', latent_dim=5):
    """The AERootBlock class is the basic building block of the N-BEATS network.  
    It consists of a stack of fully connected layers organized as an Autoencoder.
    It serves as the base class for the GenericAEBlock, SeasonalityAEBlock, 
    and the TrendAEBlock classes.
    
    Args:
        backcast (int): 
          The length of the historical data.  It is customary to use a multiple 
          of the forecast (H)orizon (2H,3H,4H,5H,...).
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
    self.backcast = backcast
    self.latent_dim = latent_dim
    
    if not activation in ACTIVATIONS:
      raise ValueError(f"'{activation}' is not in {ACTIVATIONS}")
    
    self.activation = getattr(nn, activation)()    
    
    self.fc1 = nn.Linear(backcast, units//2)
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


class AutoEncoderBlock(Block):  
  def __init__(self, 
                units:int,
                backcast:int,
                forecast:int,
                thetas_dim:int,
                share_weights:bool,
                activation:str = 'ReLU',
                active_g:bool = False):
    
      super(AutoEncoderBlock, self).__init__(backcast, units, activation)
      
      self.units = units
      self.thetas_dim = thetas_dim
      self.share_weights = share_weights
      self.activation = getattr(nn, activation)()
      self.backcast = backcast
      self.forecast = forecast
      self.active_g = active_g      
      
      # Encoders
      self.b_encoder = nn.Sequential(
          nn.Linear(units, thetas_dim),
          nn.ReLU(),
      )
      
      self.f_encoder = nn.Sequential(
          nn.Linear(units, thetas_dim),
          nn.ReLU(),
      )  
      
      # Decoders
      self.b_decoder = nn.Sequential(
          nn.Linear(thetas_dim, units),
          nn.ReLU(),
          nn.Linear(units, backcast),
      )
      
      self.f_decoder = nn.Sequential(
          nn.Linear(thetas_dim, units),
          nn.ReLU(),
          nn.Linear(units, forecast),
      )

  def forward(self, x):
      x = super(AutoEncoderBlock, self).forward(x)
      b = self.b_encoder(x)
      b = self.b_decoder(b)
      
      f = self.f_encoder(x)
      f = self.f_decoder(f)
      
      # N-BEATS paper does not apply activation here, but Generic models will not converge sometimes without it
      if self.active_g:
        b = self.activation(b)
        f = self.activation(f)

      return b,f          

class GenericAEBackcastBlock(Block):  
  def __init__(self, 
                units:int,
                backcast:int,
                forecast:int,
                thetas_dim:int,
                share_weights:bool,
                activation:str = 'ReLU',
                active_g:bool = False):
    
      super(GenericAEBackcastBlock, self).__init__(backcast, units, activation)
      
      self.units = units
      self.thetas_dim = thetas_dim
      self.share_weights = share_weights
      self.activation = getattr(nn, activation)()
      self.backcast = backcast
      self.forecast = forecast
      self.active_g = active_g        
      
      self.forecast_linear = nn.Linear(units, thetas_dim)
      self.forecast_g = nn.Linear(thetas_dim, forecast, bias = False)
      
      # Encoders
      self.b_encoder = nn.Sequential(
          nn.Linear(units, thetas_dim),
          nn.ReLU(),
      )
      
      # Decoders
      self.b_decoder = nn.Sequential(
          nn.Linear(thetas_dim, units),
          nn.ReLU(),
          nn.Linear(units, backcast),          
      )

  def forward(self, x):
    x = super(GenericAEBackcastBlock, self).forward(x)
    b = self.b_encoder(x)
    b = self.b_decoder(b)

    theta_f = self.forecast_linear(x)
    f = self.forecast_g(theta_f)
    
    # N-BEATS paper does not apply activation here, but Generic models will not converge sometimes without it
    if self.active_g:
      b = self.activation(b)
      f = self.activation(f)
      return b,f 
    
class GenericAEBackcastAEBlock(AERootBlock):  
  def __init__(self, 
                units:int,
                backcast:int,
                forecast:int,
                thetas_dim:int,
                share_weights:bool,
                activation:str = 'ReLU',
                active_g:bool = False,
                latent_dim:int = 5):
    
      super(GenericAEBackcastAEBlock, self).__init__(backcast, units, activation,latent_dim=latent_dim)
      
      self.units = units
      self.thetas_dim = thetas_dim
      self.share_weights = share_weights
      self.activation = getattr(nn, activation)()
      self.backcast = backcast
      self.forecast = forecast
      self.active_g = active_g        
      
      self.forecast_linear = nn.Linear(units, thetas_dim)
      self.forecast_g = nn.Linear(thetas_dim, forecast, bias = False)
      
      # Encoders
      self.b_encoder = nn.Sequential(
          nn.Linear(units, thetas_dim),
          nn.ReLU(),
      )
      
      # Decoders
      self.b_decoder = nn.Sequential(
          nn.Linear(thetas_dim, units),
          nn.ReLU(),
          nn.Linear(units, backcast),          
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
                backcast:int,
                forecast:int,
                thetas_dim:int,
                share_weights:bool,
                activation:str = 'ReLU',
                active_g:bool = False,
                latent_dim:int = 5):
    
    super(AutoEncoderAEBlock, self).__init__(backcast, units, activation, latent_dim=latent_dim)
    
    self.units = units
    self.thetas_dim = thetas_dim
    self.share_weights = share_weights
    self.activation = getattr(nn, activation)()
    self.backcast = backcast
    self.forecast = forecast      
    self.active_g = active_g  
    
    # Encoders
    self.b_encoder = nn.Sequential(
        nn.Linear(units, thetas_dim),
        nn.ReLU(),
    )
    
    self.f_encoder = nn.Sequential(
        nn.Linear(units, thetas_dim),
        nn.ReLU(),
    )  
          
    # Decoders
    self.b_decoder = nn.Sequential(
        nn.Linear(thetas_dim, units),
        nn.ReLU(),
        nn.Linear(units, backcast),
    )
    
    self.f_decoder = nn.Sequential(
        nn.Linear(thetas_dim, units),
        nn.ReLU(),
        nn.Linear(units, forecast),
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


class GenericBlock(Block):
  def __init__(self, 
               units:int, 
               backcast:int, 
               forecast:int, 
               thetas_dim:int = 5, 
               share_weights:bool= False, 
               activation:str = 'ReLU', 
               active_g:bool = False):
    """The Generic Block is the basic building block of the N-BEATS network.  
    It consists of a stack of fully connected layers, followed by
    two linear layers. The first, backcast_linear, generates the parameters 
    of a waveform generator, which is implemented by the function 
    defined in the next layer. These two layers can also be thought of os a 
    compression and expansion layer or rudimentary AutoEncoder.

    Args:
        units (int): 
          The width of the fully connected layers in the blocks comprising 
          the stacks of the generic model
        backcast (int): 
          The length of the historical data.  It is customary to use a 
          multiple of the forecast (H)orizon (2H,3H,4H,5H,...).
        forecast (int): 
          The length of the forecast horizon.
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
    super(GenericBlock, self).__init__(backcast, units, activation)
    
    if share_weights:
        self.backcast_linear = self.forecast_linear = nn.Linear(units, thetas_dim)
    else:
        self.backcast_linear = nn.Linear(units, thetas_dim)
        self.forecast_linear = nn.Linear(units, thetas_dim)
        
    self.backcast_g = nn.Linear(thetas_dim, backcast, bias = False) 
    self.forecast_g = nn.Linear(thetas_dim, forecast, bias = False)
    self.active_g = active_g
    
  def forward(self, x):
    x = super(GenericBlock, self).forward(x)
    theta_b = self.backcast_linear(x)
    theta_f = self.forecast_linear(x)
    
    backcast = self.backcast_g(theta_b)
    forecast = self.forecast_g(theta_f)
    
    # N-BEATS paper does not apply activation here, but Generic models will not converge sometimes without it
    if self.active_g:
      backcast = self.activation(backcast)
      forecast = self.activation(forecast)

    return backcast, forecast
  
class GenericAEBlock(AERootBlock):
  def __init__(self, 
               units:int, 
               backcast:int, 
               forecast:int, 
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
        backcast (int): 
          The length of the historical data.  It is customary to use a 
          multiple of the forecast (H)orizon (2H,3H,4H,5H,...).
        forecast (int): 
          The length of the forecast horizon.
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
    super(GenericAEBlock, self).__init__(backcast, units, activation, latent_dim=latent_dim)
    
    if share_weights:
        self.backcast_linear = self.forecast_linear = nn.Linear(units, thetas_dim)
    else:
        self.backcast_linear = nn.Linear(units, thetas_dim)
        self.forecast_linear = nn.Linear(units, thetas_dim)
        
    self.backcast_g = nn.Linear(thetas_dim, backcast, bias = False) 
    self.forecast_g = nn.Linear(thetas_dim, forecast, bias = False)
    self.active_g = active_g
    
  def forward(self, x):
    x = super(GenericAEBlock, self).forward(x)
    theta_b = self.backcast_linear(x)
    theta_f = self.forecast_linear(x)
    
    backcast = self.backcast_g(theta_b)
    forecast = self.forecast_g(theta_f)
    
    # N-BEATS paper does not apply activation here, but Generic models will not converge sometimes without it
    if self.active_g:
      backcast = self.activation(backcast)
      forecast = self.activation(forecast)

    return backcast, forecast  

class _SeasonalityGenerator(nn.Module):
  def __init__(self, forecast):
    """Generates the basis for the Fourier expansion of the Seasonality Block.

    Args:
        forecast (int): The length of the forecast horizon.
    """
    super().__init__()
    half_minus_one = int(forecast / 2 - 1)
    cos_vectors = [
        torch.cos(torch.arange(forecast) / forecast * 2 * np.pi * i)
        for i in range(1, half_minus_one + 1)
    ]
    sin_vectors = [
        torch.sin(torch.arange(forecast) / forecast * 2 * np.pi * i)
        for i in range(1, half_minus_one + 1)
    ]

    # basis is of size (2 * int(target_length / 2 - 1) + 1, target_length)
    basis = torch.stack(
        [torch.ones(forecast)] + cos_vectors + sin_vectors, dim=1
    ).T

    self.basis = nn.Parameter(basis, requires_grad=False)

  def forward(self, x):
    return torch.matmul(x, self.basis)

class SeasonalityBlock(Block):
  def __init__(self, units, backcast, forecast,  thetas_dim=5, 
               share_weights = False, activation='ReLU',active_g:bool = False):
    """The Seasonality Block is the basic building block of the N-BEATS network.  
    It consists of a stack of fully connected layers defined the parent class Block, 
    followed by a linear layer, which generates the parameters of a Fourier expansion.

    Args:
        units (int): 
          The width of the fully connected layers in the blocks comprising the parent 
          class Block.
        backcast (int): 
          The length of the historical data.  It is customary to use a multiple of 
          the forecast (H)orizon (2H,3H,4H,5H,...).
        forecast (int): 
          The length of the forecast horizon.
        activation (str, optional): 
          The activation function passed to the parent class Block. Defaults to 'LeakyReLU'.
    """
    super(SeasonalityBlock, self).__init__(backcast, units, activation)

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

class SeasonalityAEBlock(AERootBlock):
  def __init__(self, units, backcast, forecast,  thetas_dim=5, 
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
        backcast (int): 
          The length of the historical data.  It is customary to use a multiple of 
          the forecast (H)orizon (2H,3H,4H,5H,...).
        forecast (int): 
          The length of the forecast horizon.
        activation (str, optional): 
          The activation function passed to the parent class Block. Defaults to 'LeakyReLU'.
    """
    super(SeasonalityAEBlock, self).__init__(backcast, units, activation, latent_dim = latent_dim)

    self.backcast_linear = nn.Linear(units, 2 * int(backcast / 2 - 1) + 1, bias = False)
    self.forecast_linear = nn.Linear(units, 2 * int(forecast / 2 - 1) + 1, bias = False)
      
    self.backcast_g = _SeasonalityGenerator(backcast)
    self.forecast_g = _SeasonalityGenerator(forecast)

  def forward(self, x):
    x = super(SeasonalityAEBlock, self).forward(x)
    # linear compression
    backcast = self.backcast_linear(x)
    forecast = self.forecast_linear(x)
    
    # fourier expansion
    backcast = self.backcast_g(backcast)  
    forecast = self.forecast_g(forecast)
    
    return backcast, forecast


class _TrendGenerator(nn.Module):
  def __init__(self, expansion_coefficient_dim, target_length):
    """ Trend model. A typical characteristic of trend is that most of the time it is a
    monotonic function, or at least a slowly varying function. In order to mimic this 
    behaviour this block implements a low pass filter, or slowly varying function of a 
    small degree polynomial across forecast window.

    Args:
        expansion_coefficient_dim (int): The dimensionality of the expansion coefficients used in 
        the Vandermonde expansion.  The N-BEATS paper uses 2 or 3, but any positive integer can be used. 
        5 is also a reasonalbe choice.
        target_length (int): The length of the forecast horizon.
    """
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
  def __init__(self, units, backcast, forecast, thetas_dim, 
               share_weights = True, activation='LeakyReLU', active_g:bool = False):
    """The Trend Block implements the function whose parameters are generated by the _TrendGenerator block.  

    Args:
        units (int): 
          The width of the fully connected layers in the blocks comprising the parent class Block.
        backcast (int): 
          The length of the historical data.  It is customary to use a multiple of the forecast (H)orizon (2H,3H,4H,5H,...).
        forecast (int): 
          The length of the forecast horizon.
        thetas_dim (int): 
          The dimensionality of the _TrendGenerator polynomial.
        share_weights (bool, optional): 
          If True, the inital weights of the Linear layers are shared. Defaults to True.
        activation (str, optional): 
          The activation function passed to the parent class Block. Defaults to 'LeakyReLU'.
    """
    super(TrendBlock, self).__init__(backcast, units, activation)
    if share_weights:
        self.backcast_linear = self.forecast_linear = nn.Linear(units, thetas_dim)
    else:
        self.backcast_linear = nn.Linear(units, thetas_dim)
        self.forecast_linear = nn.Linear(units, thetas_dim)
        
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

class TrendAEBlock(AERootBlock):
  def __init__(self, units, backcast, forecast, thetas_dim, 
               share_weights = True, activation='LeakyReLU', active_g:bool = False, latent_dim = 5):
    """The TrendAEBlock implements the function whose parameters are generated by the _TrendGenerator block.  
    The TrendAEBlock is an AutoEncoder version of the TrendBlock where the presplit section of the network
    is an AutoEncoder.

    Args:
        units (int): 
          The width of the fully connected layers in the blocks comprising the parent class Block.
        backcast (int): 
          The length of the historical data.  It is customary to use a multiple of the forecast (H)orizon (2H,3H,4H,5H,...).
        forecast (int): 
          The length of the forecast horizon.
        thetas_dim (int): 
          The dimensionality of the _TrendGenerator polynomial.
        share_weights (bool, optional): 
          If True, the inital weights of the Linear layers are shared. Defaults to True.
        activation (str, optional): 
          The activation function passed to the parent class Block. Defaults to 'LeakyReLU'.
    """
    super(TrendAEBlock, self).__init__(backcast, units, activation, latent_dim=latent_dim)
    if share_weights:
        self.backcast_linear = self.forecast_linear = nn.Linear(units, thetas_dim)
    else:
        self.backcast_linear = nn.Linear(units, thetas_dim)
        self.forecast_linear = nn.Linear(units, thetas_dim)
        
    self.backcast_g = _TrendGenerator(thetas_dim, backcast)
    self.forecast_g = _TrendGenerator(thetas_dim, forecast)

  def forward(self, x):
    x = super(TrendAEBlock, self).forward(x)
    
    # linear compression
    backcast = self.backcast_linear(x)
    forecast = self.forecast_linear(x)
    
    #  Vandermonde basis expansion
    backcast = self.backcast_g(backcast)    
    forecast = self.forecast_g(forecast)
            
    return backcast, forecast
