from time import time
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import lightning.pytorch as pl
from .losses import MASELoss, SMAPELoss, MAPELoss


ACTIVATIONS = [
    "ReLU",
    "RReLU",
    "PReLU",
    "ELU",
    "Softplus",
    "Tanh",
    "SELU",
    "LeakyReLU",
    "Sigmoid",
    "GELU"
]

LOSSES = [
    "MAPELoss",
    "SMAPELoss",
    "MASELoss",
    "MSELoss",
    "L1Loss",
    "SmoothL1Loss",
    "BCEWithLogitsLoss", 
    "BCELoss",
    "CrossEntropyLoss",
    "NLLLoss",
    "PoissonNLLLoss",
    "KLDivLoss",
    "BCEWithLogitsLoss",
    "MarginRankingLoss",
    "HingeEmbeddingLoss",
    "MultiLabelMarginLoss",
    "CosineEmbeddingLoss",
    "MultiMarginLoss",
    "TripletMarginLoss",
    "TripletMarginWithDistanceLoss"        
]

OPTIMIZERS = [
  "Adam",
  "SGD",
  "RMSprop",
  "Adagrad",
  "Adadelta",
  "AdamW"
]

class NBeatsNet(pl.LightningModule):
  SEASONALITY = 'seasonality'
  TREND = 'trend'
  GENERIC = 'generic'
    
  def __init__(
      self,
      backcast:int, # default of 5*forecast_length or 5H , N-BEATS paper tested 2H,3H,4H,5H,6H,7H          
      forecast:int,  # forecast (H)orizon
      generic_architecture:bool = True,
      n_blocks_per_stack:int = 1, # Number of stacks N-BEATS paper: Generic 1 blk/stk, 2 Stacks (1 Trend 3 blks, 1 Seasonality 3 blks)
      n_stacks:int = 5, # number of stacks
      g_width:int = 512, # the width of the fully connected layers in the blocks comprising the stacks of the generic model
      s_width:int = 2048, # the width of the fully connected layers in the blocks comprising the seasonality stack of the interpretable model.
      t_width:int = 256, # the width of the fully connected layers in the blocks comprising the trend stack of the interpretable model
      share_weights:bool = False, # Generic model prefers no weight sharing, while interpretable model does.
      thetas_dim:int = 5, # 5 for generic, 2 or 3 for interpretable tomimic trend
      learning_rate: float = 1e-5,  
      loss: str = 'smape', # 'mape', 'smape', 'mase'
      no_val:bool = False,  # set to True to skip validation during training ( when using the entire test set for training)
      optimizer_name:str = 'AdamW', # 'adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adamw'
      activation:str = 'LeakyReLU', # 'ReLU', 'RReLU', 'PReLU', 'ELU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU', 'Sigmoid', 'GELU'
      frequency:int = 1, # frequency of the data
      activate_g:bool = True, # activate the g function in the generic model.  Sometime generic model willnot coverge without this.
      sum_losses:bool = False, # sum the backcast and forecast losses

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
    backcast : int, optional
        The length of the historical data.  It is customary to use a multiple of the 
        forecast (H)orizon (2H,3H,4H,5H,...).
    forecast : int, optional
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
    g_width, s_width, s_width : int, optional
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
        Default 'AdamW'.
    activation : str, optional
        The activation function to use.  Defined in ACTIVATIONS. 
        Default : 'LeakyReLU'.
    frequency : int, optional
        The frequency of the data.  Used only when MASELoss is used as teh loss funtion. Default 1.
    activate_g : bool, optional
        If True, the function implemented by the waveform generators in the forecast anb backcast blocks has n activation function applied to the output.  
        This feature is not defined in the original design of N-BEATS.  However, since both the Trend and Seasonality blocks apply non-linear functions
        to the waveform parameters generated in he preceeding layer, applying an activation function here mirrors that structure.  I've found that this
        approch improved convergence of GENERIC models.  Default : TRUE.
    sum_losses : bool, optional
        If True, the total loss is defined as forecast_loss + 1/2 Backcast_loss.  This is an experimental feature. Default False.
    
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
        Tensor containing the forward forecast of the stack.
    """    
  
    super(NBeatsNet, self).__init__()
    self.backcast = backcast
    self.forecast = forecast
    self.n_blocks_per_stack = n_blocks_per_stack
    self.n_stacks = n_stacks
    self.share_weights = share_weights
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
    self.activation = activation
    self.activate_g = activate_g
    self.sum_losses = sum_losses
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
        if self.share_weights and block_id != 0:
            # share initial weights across blocks
            block = blocks[-1]  
        else:
            if self.generic_architecture:
                block = GenericBlock(
                    self.g_width, self.backcast, self.forecast, self.thetas_dim, self.share_weights, self.activation, self.activate_g
                )
            elif stack_type == NBeatsNet.SEASONALITY:
                block = SeasonalityBlock(
                    self.s_width, self.backcast, self.forecast, self.activation
                )
            elif stack_type == NBeatsNet.TREND:
                block = TrendBlock(
                    self.t_width, self.backcast, self.forecast, self.thetas_dim, self.share_weights, self.activation
                )
                
        blocks.append(block)   

    return blocks

  def forward(self, backcast):
    backcast = squeeze_last_dim(backcast)    
    forecast = torch.zeros(
            backcast.shape[0],
            self.forecast,
            device=backcast.device,
            dtype=backcast.dtype)
    
    for stack_id in range(len(self.stacks)):
        for block_id in range(len(self.stacks[stack_id])):
          b, f = self.stacks[stack_id][block_id](backcast)
          backcast = backcast - b
          forecast = forecast + f
    return backcast, forecast

  def training_step(self, batch, batch_idx):
    x, y = batch
    backcast, forecast = self(x)
    loss = self.loss_fn(forecast, squeeze_last_dim(y))
    backcast_loss = self.loss_fn(backcast, squeeze_last_dim(x))
    if self.sum_losses:
      loss = loss + backcast_loss*0.5
    self.log('train_loss', loss, prog_bar=True)
    return loss

  def validation_step(self, batch, batch_idx):
    if self.no_val:
      return None
    
    x, y = batch
    backcast, forecast = self(x)
    loss = self.loss_fn(forecast, squeeze_last_dim(y))
    backcast_loss = self.loss_fn(backcast, squeeze_last_dim(x))
    if self.sum_losses:
      loss = loss + backcast_loss * 0.5    
    
    self.log('val_loss', loss, prog_bar=True)
    return loss

  def test_step(self, batch, batch_idx):
    x, y = batch
    backcast, forecast = self(x)
    loss = self.loss_fn(forecast, squeeze_last_dim(y))
    backcast_loss = self.loss_fn(backcast, squeeze_last_dim(x))
    if self.sum_losses:
      loss = loss + backcast_loss * 0.5
      
    self.log('test_loss', loss)
    return loss

  def predict_step(self, batch, batch_idx, dataloader_idx=None):
      self.eval()
      _, forecast = self(batch)
      return forecast.detach().cpu().numpy()
    
  def configure_loss(self):
    if self.loss not in LOSSES:
        raise ValueError(f"Unknown loss function name: {self.loss}. Please select one of {LOSSES}")
    if self.loss == 'MAPELoss':
        return MASELoss(self.frequency)
    if self.loss == 'MAPELoss':
        return MAPELoss()
    if self.loss == 'SMAPELoss':
        return SMAPELoss()
    else:
        return getattr(nn, self.loss)()
    
  def configure_optimizers(self):
    if self.optimizer_name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer name: {self.optimizer_name}. Please select one of {OPTIMIZERS}")
    
    opti = getattr(optim, self.optimizer_name)(self.parameters(), lr=self.learning_rate)
    return opti


def squeeze_last_dim(tensor):
  if len(tensor.shape) == 3 and tensor.shape[-1] == 1:  # (128, 10, 1) => (128, 10).
      return tensor[..., 0]
  return tensor

class Block(nn.Module):
  def __init__(self, backcast, units, activation='LeakyReLU'):
    """The Block class is the basic building block of the N-BEATS network.  It consists of a stack of fully connected layers.
    It serves as the base class for the GenericBlock, SeasonalityBlock, and TrendBlock classes.

    Args:
        backcast (int): The length of the historical data.  It is customary to use a multiple of the forecast (H)orizon (2H,3H,4H,5H,...).
        units (int): The width of the fully connected layers in the blocks comprising the stacks of the generic model
        activation (str, optional): The activation function applied to each of the fully connected Linear layers. Defaults to 'LeakyReLU'.

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
  
class GenericBlock(Block):
  def __init__(self, 
               units:int, 
               backcast:int, 
               forecast:int, 
               thetas_dim:int = 5, 
               share_weights:bool= False, 
               activation:str='LeakyReLU', 
               activate_g:bool=True):
    """The Generic Block is the basic building block of the N-BEATS network.  It consists of a stack of fully connected layers, followed by
    two linear layers. The first, backcast_linear, generates the parameters of a waveform generator, which is implemented by the function 
    defined in the next layer. These two layers can also be thought of os a compression and expansion layer or rudimentary AutoEncoder.

    Args:
        units (int): The width of the fully connected layers in the blocks comprising the stacks of the generic model
        backcast (int): The length of the historical data.  It is customary to use a multiple of the forecast (H)orizon (2H,3H,4H,5H,...).
        forecast (int): The length of the forecast horizon.
        thetas_dim (int, optional): The dimensionality of the wavefor generator parameters. Defaults to 5.
        share_weights (bool, optional): If True, the initial weights of the Linear laers are shared. Defaults to False.
        activation (str, optional): The activation function used in the parent class Block, and optionally as the non-linear activation of
        the backcast_g and forecast_g layers.
        Defaults to 'LeakyReLU'.
        activate_g (bool, optional): _description_. Defaults to True.
    """
    super(GenericBlock, self).__init__(backcast, units, activation)
    
    if share_weights:
        self.backcast_linear = self.forecast_linear = nn.Linear(units, thetas_dim)
    else:
        self.backcast_linear = nn.Linear(units, thetas_dim)
        self.forecast_linear = nn.Linear(units, thetas_dim)
        
    self.backcast_g = nn.Linear(thetas_dim, backcast)
    self.forecast_g = nn.Linear(thetas_dim, forecast)
    self.activate_g = activate_g
    
  def forward(self, x):
    x = super(GenericBlock, self).forward(x)
    theta_b = self.backcast_linear(x)
    theta_f = self.forecast_linear(x)
    
    backcast = self.backcast_g(theta_b)
    forecast = self.forecast_g(theta_f)
    
    # N-BEATS paper does not aply activation here, but Generic models will not converge sometimes without it
    if self.activate_g:
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
  def __init__(self, units, backcast, forecast, activation='LeakyReLU'):
    """The Seasonality Block is the basic building block of the N-BEATS network.  
    It consists of a stack of fully connected layers defined the parent class Block, 
    followed by a linear layer, which generates the parameters of a Fourier expansion.

    Args:
        units (int ): The width of the fully connected layers in the blocks comprising the parent class Block.
        backcast (int): The length of the historical data.  It is customary to use a multiple of the forecast (H)orizon (2H,3H,4H,5H,...).
        forecast (int): The length of the forecast horizon.
        activation (str, optional): The activation function passed to the parent class Block. Defaults to 'LeakyReLU'.
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
  def __init__(self, units, backcast, forecast, thetas_dim, share_weights = True, activation='LeakyReLU'):
    """The Trend Block implements the function whose parameters are generated by the _TrendGenerator block.  
    

    Args:
        units (int ): The width of the fully connected layers in the blocks comprising the parent class Block.
        backcast (int): The length of the historical data.  It is customary to use a multiple of the forecast (H)orizon (2H,3H,4H,5H,...).
        forecast (int): The length of the forecast horizon.
        thetas_dim (int): The dimensionality of the _TrendGenerator polynomial.
        share_weights (bool, optional): If True, the inital weights of the Linear layers are shared. Defaults to True.
        activation (str, optional): The activation function passed to the parent class Block. Defaults to 'LeakyReLU'.
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
