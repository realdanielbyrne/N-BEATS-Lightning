import torch
import pytorch_lightning as pl
from torch import nn
from torch.autograd import Variable


def squeeze_last_dim(tensor):
  if len(tensor.shape) == 3 and tensor.shape[-1] == 1:  # (128, 10, 1) => (128, 10).
      return tensor[..., 0]
  return tensor

class Block(nn.Module):
  def __init__(self, backcast, units, activation='ReLU'):
    """The Block class is the basic building block of the N-BEATS network.  It consists of a stack of fully connected layers.
    It serves as the base class for the GenericBlock, SeasonalityBlock, and TrendBlock classes.

    Args:
        backcast (int): 
          The length of the historical data.  It is customary to use a multiple of the forecast (H)orizon (2H,3H,4H,5H,...).
        units (int): 
          The width of the fully connected layers in the blocks comprising the stacks of the generic model
        activation (str, optional): 
          The activation function applied to each of the fully connected Linear layers. Defaults to 'ReLU'.

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


class VAEBlock(Block):
    def __init__(self, 
                 units:int,
                 backcast:int,
                 forecast:int,
                 thetas_dim:int,
                 share_weights:bool,
                 activation:str = 'ReLU'):
      
        super(VAEBlock, self).__init__(backcast, units, activation)
        
        self.units = units
        self.thetas_dim = thetas_dim
        self.activation = getattr(nn, activation)()
        
        # Encoders
        ef1 = nn.Linear(units, units//2),
        ef2 = nn.Linear(units//2, int(thetas_dim*2))  # 20 for mean and 20 for log variance

        
        self.encoderforward = nn.ModuleList(
            nn.Linear(units, units//2),
            nn.Linear(units//2, int(thetas_dim*2))  # 20 for mean and 20 for log variance
        )
              
        # Decoders
        self.decoderback = nn.ModuleList(
            nn.Linear(thetas_dim, units//2),
            nn.Linear(units//2, units),
            nn.Linear(units, backcast),
        )
        
        self.decoderforward = nn.ModuleList(
            nn.Linear(thetas_dim, units//2),
            nn.Linear(units//2, units),

            nn.Linear(units, forecast),
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = Variable(torch.randn(std.size()))
        return mu + eps * std
    
    def forward(self, x):
        b = self.encoderback(x)
        f = self.encoderforward(x)
        
        mu_b, logvar_b = torch.chunk(b, 2, dim=1)
        mu_f, logvar_f = torch.chunk(f, 2, dim=1)
        
        z = self.reparameterize(mu_b, logvar_b)
        z = self.reparameterize(mu_f, logvar_f)
        
        d_b = self.decoderback(z)
        d_f = self.decoderforward(z)
        
        #kl_divergence_b = -0.5 * torch.sum(1 + logvar_b - mu_b.pow(2) - logvar_b.exp())
        #kl_divergence_f = -0.5 * torch.sum(1 + logvar_f - mu_f.pow(2) - logvar_f.exp())

        return d_b, d_f #, kl_divergence_b, kl_divergence_f

  
class GenericBlock(Block):
  def __init__(self, 
               units:int, 
               backcast:int, 
               forecast:int, 
               thetas_dim:int = 5, 
               share_weights:bool= False, 
               activation:str = 'ReLU', 
               active_g:bool = False):
    """The Generic Block is the basic building block of the N-BEATS network.  It consists of a stack of fully connected layers, followed by
    two linear layers. The first, backcast_linear, generates the parameters of a waveform generator, which is implemented by the function 
    defined in the next layer. These two layers can also be thought of os a compression and expansion layer or rudimentary AutoEncoder.

    Args:
        units (int): 
          The width of the fully connected layers in the blocks comprising the stacks of the generic model
        backcast (int): 
          The length of the historical data.  It is customary to use a multiple of the forecast (H)orizon (2H,3H,4H,5H,...).
        forecast (int): 
          The length of the forecast horizon.
        thetas_dim (int, optional): 
          The dimensionality of the wavefor generator parameters. Defaults to 5.
        share_weights (bool, optional): 
          If True, the initial weights of the Linear laers are shared. Defaults to False.
        activation (str, optional): ß
          The activation function used in the parent class Block, and optionally as the non-linear activation of
          the backcast_g and forecast_g layers. Defaults to 'ReLU'.
        active_g (bool, optional): 
          This parameter when enabled applies the model's activation funtion to the linear 
          funtions (gb and gf) which are found by the network in the last layer of each block using the theta parameters
          found in the preceding layer. Enabling this activation function seems to help the Generic model converge. 
          The parameter `active_g` is not a feature found in the original N-Beats paper. Defaults to False.
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
    
    # N-BEATS paper does not aply activation here, but Generic models will not converge sometimes without it
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
  def __init__(self, units, backcast, forecast, activation='LeakyReLU'):
    """The Seasonality Block is the basic building block of the N-BEATS network.  
    It consists of a stack of fully connected layers defined the parent class Block, 
    followed by a linear layer, which generates the parameters of a Fourier expansion.

    Args:
        units (int): 
          The width of the fully connected layers in the blocks comprising the parent class Block.
        backcast (int): 
          The length of the historical data.  It is customary to use a multiple of the forecast (H)orizon (2H,3H,4H,5H,...).
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