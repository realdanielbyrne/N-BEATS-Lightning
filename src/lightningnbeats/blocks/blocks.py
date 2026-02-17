import torch
from torch import nn
from ..constants import ACTIVATIONS
import numpy as np
import pywt
from scipy.interpolate import interp1d

def squeeze_last_dim(tensor):
  if len(tensor.shape) == 3 and tensor.shape[-1] == 1:  # (128, 10, 1) => (128, 10).
      return tensor[..., 0]
  return tensor
class RootBlock(nn.Module):
  def __init__(self, backcast_length, units, activation='ReLU'):
    """The Block class is the basic building block of the N-BEATS network.
    It consists of a stack of fully connected layers.It serves as the base
    class for the GenericBlock, SeasonalityBlock, and TrendBlock classes.

    Args:
        backcast_length (int):
          The length of the historical data.  It is customary to use a
          multiple of the forecast_length (H)orizon (2H,3H,4H,5H,...).
        units (int):
          The width of the fully connected layers in the blocks comprising
          the stacks of the generic model
        activation (str, optional):
          The activation function applied to each of the fully connected
          Linear layers. Defaults to 'ReLU'.

    Raises:
          ValueError: If the activation function is not in ACTIVATIONS.
    """
    super(RootBlock, self).__init__()
    self.units = units
    self.backcast_length = backcast_length

    if activation not in ACTIVATIONS:
      raise ValueError(f"'{activation}' is not in {ACTIVATIONS}")

    self.activation = getattr(nn, activation)()

    self.fc1 = nn.Linear(backcast_length, units)
    self.fc2 = nn.Linear(units, units)
    self.fc3 = nn.Linear(units, units)
    self.fc4 = nn.Linear(units, units)

  def forward(self, x):
    x = self.activation(self.fc1(x))
    x = self.activation(self.fc2(x))
    x = self.activation(self.fc3(x))
    x = self.activation(self.fc4(x))
    return x

class AutoEncoder(RootBlock):
  def __init__(self,
                units:int,
                backcast_length:int,
                forecast_length:int,
                thetas_dim:int,
                share_weights:bool,
                activation:str = 'ReLU',
                active_g:bool = False):

      super(AutoEncoder, self).__init__(backcast_length, units, activation)

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
      x = super(AutoEncoder, self).forward(x)
      b = self.b_encoder(x)
      b = self.b_decoder(b)

      f = self.f_encoder(x)
      f = self.f_decoder(f)

      # N-BEATS paper does not apply activation here, but Generic models will not converge sometimes without it
      if self.active_g:
        if self.active_g != 'forecast':
          b = self.activation(b)
        if self.active_g != 'backcast':
          f = self.activation(f)

      return b,f

class Generic(RootBlock):
  def __init__(self,
               units:int,
               backcast_length:int,
               forecast_length:int,
               thetas_dim:int = 5,
               share_weights:bool= False,
               activation:str = 'ReLU',
               active_g:bool = False):
    """Paper-faithful Generic Block as defined in Oreshkin et al. (2019).

    Uses a single linear layer producing theta of size (backcast_length + forecast_length),
    then slices into backcast and forecast components. This matches the paper's formulation
    exactly: 4 FC+ReLU layers followed by one linear projection, with no intermediate
    bottleneck dimension.

    Args:
        units (int):
          The width of the fully connected layers.
        backcast_length (int):
          The length of the historical data.
        forecast_length (int):
          The length of the forecast horizon.
        thetas_dim (int, optional):
          Not used in paper-faithful Generic (kept for API compatibility). Defaults to 5.
        share_weights (bool, optional):
          If True, the initial weights of the Linear layers are shared.
          Defaults to False.
        activation (str, optional):
          The activation function used in the FC layers. Defaults to 'ReLU'.
        active_g (bool, optional):
          If True, applies activation after the basis expansion.
          Not a feature in the original paper. Defaults to False.
    """
    super(Generic, self).__init__(backcast_length, units, activation)

    self.backcast_length = backcast_length
    self.forecast_length = forecast_length
    self.active_g = active_g

    self.theta_b_fc = nn.Linear(units, backcast_length, bias=False)
    self.theta_f_fc = nn.Linear(units, forecast_length, bias=False)

  def forward(self, x):
    x = super(Generic, self).forward(x)

    backcast = self.theta_b_fc(x)
    forecast = self.theta_f_fc(x)

    if self.active_g:
      if self.active_g != 'forecast':
        backcast = self.activation(backcast)
      if self.active_g != 'backcast':
        forecast = self.activation(forecast)

    return backcast, forecast


class BottleneckGeneric(RootBlock):
  def __init__(self,
               units:int,
               backcast_length:int,
               forecast_length:int,
               thetas_dim:int = 5,
               share_weights:bool= False,
               activation:str = 'ReLU',
               active_g:bool = False):
    """Bottleneck Generic Block — a novel extension of the paper's Generic block.

    Uses a two-stage projection through an intermediate thetas_dim bottleneck:
    units → thetas_dim → target_length. This is equivalent to a rank-d factorization
    of the basis expansion matrix, where d = thetas_dim, providing a tunable knob
    to control basis complexity.

    Args:
        units (int):
          The width of the fully connected layers in the blocks comprising
          the stacks of the generic model.
        backcast_length (int):
          The length of the historical data.
        forecast_length (int):
          The length of the forecast horizon.
        thetas_dim (int, optional):
          The dimensionality of the bottleneck (rank of factorized basis).
          Defaults to 5.
        share_weights (bool, optional):
          If True, the initial weights of the Linear layers are shared.
          Defaults to False.
        activation (str, optional):
          The activation function used in the parent class Block, and
          optionally as the non-linear activation of the backcast_g and
          forecast_g layers. Defaults to 'ReLU'.
        active_g (bool, optional):
          If True, applies activation after the basis expansion.
          Not a feature in the original paper. Defaults to False.
    """
    super(BottleneckGeneric, self).__init__(backcast_length, units, activation)

    if share_weights:
        self.backcast_linear = self.forecast_linear = nn.Linear(units, thetas_dim)
    else:
        self.backcast_linear = nn.Linear(units, thetas_dim)
        self.forecast_linear = nn.Linear(units, thetas_dim)

    self.backcast_g = nn.Linear(thetas_dim, backcast_length, bias = False)
    self.forecast_g = nn.Linear(thetas_dim, forecast_length, bias = False)
    self.active_g = active_g

  def forward(self, x):
    x = super(BottleneckGeneric, self).forward(x)
    theta_b = self.backcast_linear(x)
    theta_f = self.forecast_linear(x)

    backcast = self.backcast_g(theta_b)
    forecast = self.forecast_g(theta_f)

    if self.active_g:
      if self.active_g != 'forecast':
        backcast = self.activation(backcast)
      if self.active_g != 'backcast':
        forecast = self.activation(forecast)

    return backcast, forecast


class GenericAEBackcast(RootBlock):
  def __init__(self,
                units:int,
                backcast_length:int,
                forecast_length:int,
                thetas_dim:int,
                share_weights:bool,
                activation:str = 'ReLU',
                active_g:bool = False):

      super(GenericAEBackcast, self).__init__(backcast_length, units, activation)

      self.units = units
      self.thetas_dim = thetas_dim
      self.share_weights = share_weights
      self.activation = getattr(nn, activation)()
      self.backcast_length = backcast_length
      self.forecast_length = forecast_length
      self.active_g = active_g

      self.forecast_linear = nn.Linear(units, self.thetas_dim)
      self.forecast_g = nn.Linear(self.thetas_dim, forecast_length, bias = False)

      self.b_encoder = nn.Linear(units, self.thetas_dim)
      self.b_decoder = nn.Linear(self.thetas_dim, units)
      self.backcast_g = nn.Linear(units, backcast_length)

  def forward(self, x):
    x = super(GenericAEBackcast, self).forward(x)
    b = self.activation(self.b_encoder(x))
    b = self.activation(self.b_decoder(b))
    b = self.backcast_g(b)

    theta_f = self.forecast_linear(x)
    f = self.forecast_g(theta_f)

    # N-BEATS paper does not apply activation here, but Generic models will not converge sometimes without it
    if self.active_g:
      if self.active_g != 'forecast':
        b = self.activation(b)
      if self.active_g != 'backcast':
        f = self.activation(f)
    return b, f

class _SeasonalityGenerator(nn.Module):
  def __init__(self, len):
    """Generates the basis for the Fourier basic expansion of the Seasonality Block.

    Args:
        forecast_length (int): The length of the forecast_length horizon.
    """
    super().__init__()
    half_minus_one = int(len / 2 - 1)
    cos_vectors = [
        torch.cos(torch.arange(len) / len * 2 * np.pi * i)
        for i in range(1, half_minus_one + 1)
    ]
    sin_vectors = [
        torch.sin(torch.arange(len) / len * 2 * np.pi * i)
        for i in range(1, half_minus_one + 1)
    ]

    # basis is of size (2 * int(forecast_length / 2 - 1) + 1, forecast_length)
    basis = torch.stack(
        [torch.ones(len)] + cos_vectors + sin_vectors, dim=1
    ).T

    self.basis = nn.Parameter(basis, requires_grad=False)

  def forward(self, x):
    return torch.matmul(x, self.basis)

class Seasonality(RootBlock):
  def __init__(self, units, backcast_length, forecast_length,  thetas_dim=5,
               share_weights = False, activation='ReLU', active_g:bool = False):

    super(Seasonality, self).__init__(backcast_length, units, activation)

    self.backcast_linear = nn.Linear(units, 2 * int(backcast_length / 2 - 1) + 1, bias = False)
    self.forecast_linear = nn.Linear(units, 2 * int(forecast_length / 2 - 1) + 1, bias = False)

    self.backcast_g = _SeasonalityGenerator(backcast_length)
    self.forecast_g = _SeasonalityGenerator(forecast_length)

  def forward(self, x):
    x = super(Seasonality, self).forward(x)
    # linear compression
    backcast_thetas = self.backcast_linear(x)
    forecast_thetas = self.forecast_linear(x)

    # fourier expansion
    backcast = self.backcast_g(backcast_thetas)
    forecast = self.forecast_g(forecast_thetas)

    return backcast, forecast

class _TrendGenerator(nn.Module):
  def __init__(self, thetas_dim, target_length):
    """ Trend model. A typical characteristic of trend is that most of the time it is a
    monotonic function, or at least a slowly varying function. In order to mimic this
    behaviour this block implements a low pass filter, or slowly varying function of a
    small degree polynomial across forecast_length window.

    Args:
        expansion_coefficient_dim (int): The dimensionality of the expansion coefficients used in
        the Vandermonde expansion.  The N-BEATS paper uses 2 or 3, but any positive integer can be used.
        5 is also a reasonalbe choice.
        target_length (int): The length of the forecast_length horizon.
    """
    super().__init__()

    # basis is of size (expansion_coefficient_dim, target_length)
    basis = torch.stack(
        [
            (torch.arange(target_length) / target_length) ** i
            for i in range(thetas_dim)
        ],
        dim=1,
    ).T

    self.basis = nn.Parameter(basis, requires_grad=False)

  def forward(self, x):
    return torch.matmul(x, self.basis)

class Trend(RootBlock):
  def __init__(self, units, backcast_length, forecast_length, thetas_dim,
               share_weights = False, activation='ReLU', active_g:bool = False):
    """The Trend Block implements the function whose parameters are generated by the _TrendGenerator block.

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
          If True, the inital weights of the Linear layers are shared. Defaults to False.
        activation (str, optional):
          The activation function passed to the parent class Block. Defaults to 'ReLU'.
    """
    super(Trend, self).__init__(backcast_length, units, activation)
    self.share_weights = share_weights
    if self.share_weights:
        self.backcast_linear = self.forecast_linear = nn.Linear(units, thetas_dim)
    else:
        self.backcast_linear = nn.Linear(units, thetas_dim)
        self.forecast_linear = nn.Linear(units, thetas_dim)

    self.backcast_g = _TrendGenerator(thetas_dim, backcast_length)
    self.forecast_g = _TrendGenerator(thetas_dim, forecast_length)

  def forward(self, x):
    x = super(Trend, self).forward(x)

    # linear compression
    backcast_thetas = self.backcast_linear(x)
    forecast_thetas = self.forecast_linear(x)

    #  Vandermonde basis expansion
    backcast = self.backcast_g(backcast_thetas)
    forecast = self.forecast_g(forecast_thetas)

    return backcast, forecast

class _AltWaveletGenerator(nn.Module):
  def __init__(self, N, target_length, wavelet_type='db2'):
    super().__init__()

    wavelet = pywt.Wavelet(wavelet_type)
    phi, psi, x = wavelet.wavefun(level=10)


    # Create an interpolation function
    interp_phi = interp1d(x, phi, kind='linear')
    interp_psi = interp1d(x, psi, kind='linear')

    # Define new x-values where you want to sample the wavelet function
    new_x = np.linspace(min(x), max(x), N)

    # Get the wavelet function values at these new x-values
    new_phi = interp_phi(new_x)
    new_psi = interp_psi(new_x)


    # Initialize basis matrix
    W = np.zeros((N, target_length))

    # Populate basis matrix half with phi and half with psi
    for i in range(target_length):
        W[:, i] = np.roll(new_phi, i)[:N] if i < target_length//2 else np.roll(new_psi, i - target_length//2)[:N]


    self.basis = nn.Parameter(torch.tensor(W, dtype=torch.float32), requires_grad=False)

  def forward(self, x):
    return torch.matmul(x, self.basis)

class AltWavelet(RootBlock):
  def __init__(self, units, backcast_length, forecast_length,  basis_dim=32,
              share_weights = False, activation='ReLU', active_g:bool = False, wavelet_type='db2'):
    super(AltWavelet, self).__init__(backcast_length, units, activation)

    self.activation = getattr(nn, activation)()
    self.wavelet_type = wavelet_type
    self.share_weights = share_weights

    if share_weights:
      self.backcast_linear = self.forecast_linear = nn.Linear(units, basis_dim)
    else:
      self.backcast_linear = nn.Linear(units, basis_dim)
      self.forecast_linear = nn.Linear(units, basis_dim)

    self.backcast_g = _AltWaveletGenerator(basis_dim, backcast_length, wavelet_type=wavelet_type)
    self.forecast_g = _AltWaveletGenerator(basis_dim, forecast_length, wavelet_type=wavelet_type)


  def forward(self, x):
    x = super(AltWavelet, self).forward(x)

    # Wavelet basis expansion
    b = self.backcast_linear(x)
    f = self.forecast_linear(x)
    b = self.backcast_g(b)
    f = self.forecast_g(f)

    return b, f

class _WaveletGenerator(nn.Module):
  def __init__(self, basis_dim, wavelet_type='db3'):
    super().__init__()

    wavelet = pywt.Wavelet(wavelet_type)
    phi, psi, x = wavelet.wavefun(level=10)


    # Create an interpolation function
    interp_phi = interp1d(x, phi, kind='linear')
    interp_psi = interp1d(x, psi, kind='linear')

    # Define new x-values where you want to sample the wavelet function
    new_x = np.linspace(min(x), max(x), basis_dim)

    # Get the wavelet function values at these new x-values
    new_phi = interp_phi(new_x)
    new_psi = interp_psi(new_x)


    # Initialize basis matrix
    W = np.zeros((basis_dim, basis_dim))

    # Populate basis matrix half with phi and half with psi
    for i in range(basis_dim):
        W[:, i] = np.roll(new_phi, i)[:basis_dim] if i < basis_dim//2 else np.roll(new_psi, i - basis_dim//2)[:basis_dim]


    self.basis = nn.Parameter(torch.tensor(W, dtype=torch.float32), requires_grad=False)

  def forward(self, x):
    return torch.matmul(x, self.basis)


class Wavelet(RootBlock):
  def __init__(self, units, backcast_length, forecast_length,  basis_dim=32,
              share_weights = False, activation='ReLU', active_g:bool = False, wavelet_type='db3'):
    super(Wavelet, self).__init__(backcast_length, units, activation)

    self.activation = getattr(nn, activation)()
    self.wavelet_type = wavelet_type
    self.share_weights = share_weights

    if share_weights:
      self.backcast_linear = self.forecast_linear = nn.Linear(units, basis_dim)
    else:
      self.backcast_linear = nn.Linear(units, basis_dim)
      self.forecast_linear = nn.Linear(units, basis_dim)

    self.backcast_g = _WaveletGenerator(basis_dim, wavelet_type=wavelet_type)
    self.forecast_g = _WaveletGenerator(basis_dim, wavelet_type=wavelet_type)
    self.backcast_down_sample = nn.Linear(basis_dim, backcast_length, bias=False)
    self.forecast_down_sample = nn.Linear(basis_dim, forecast_length, bias=False)


  def forward(self, x):
    x = super(Wavelet, self).forward(x)

    # Wavelet basis expansion
    b = self.backcast_linear(x)
    f = self.forecast_linear(x)
    b = self.backcast_g(b)
    b = self.backcast_down_sample(b)
    f = self.forecast_g(f)
    f = self.forecast_down_sample(f)

    return b, f


class HaarWavelet(Wavelet):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights = False, activation='ReLU', active_g:bool = False):
    super(HaarWavelet, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                    share_weights, activation, active_g, wavelet_type='haar')
  def forward(self, x):
    return super(HaarWavelet, self).forward(x)

class HaarAltWavelet(AltWavelet):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights = False, activation='ReLU', active_g:bool = False):
    super(HaarAltWavelet, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                    share_weights, activation, active_g, wavelet_type='haar')
  def forward(self, x):
    return super(HaarAltWavelet, self).forward(x)
class DB2Wavelet(Wavelet):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights = False, activation='ReLU', active_g:bool = False):
    super(DB2Wavelet, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                    share_weights, activation, active_g, wavelet_type='db2')
  def forward(self, x):
    return super(DB2Wavelet, self).forward(x)

class DB2AltWavelet(AltWavelet):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights = False, activation='ReLU', active_g:bool = False):
    super(DB2AltWavelet, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                    share_weights, activation, active_g, wavelet_type='db2')
  def forward(self, x):
    return super(DB2AltWavelet, self).forward(x)

class DB3Wavelet(Wavelet):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights = False, activation='ReLU', active_g:bool = False):
    super(DB3Wavelet, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                    share_weights, activation, active_g, wavelet_type='db3')
  def forward(self, x):
    return super(DB3Wavelet, self).forward(x)

class DB3AltWavelet(AltWavelet):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights = False, activation='ReLU', active_g:bool = False):
    super(DB3AltWavelet, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                    share_weights, activation, active_g, wavelet_type='db3')
  def forward(self, x):
    return super(DB3AltWavelet, self).forward(x)

class DB4Wavelet(Wavelet):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights = False, activation='ReLU', active_g:bool = False):
    super(DB4Wavelet, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                    share_weights, activation, active_g, wavelet_type='db4')
  def forward(self, x):
    return super(DB4Wavelet, self).forward(x)

class DB4AltWavelet(AltWavelet):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights = False, activation='ReLU', active_g:bool = False):
    super(DB4AltWavelet, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                    share_weights, activation, active_g, wavelet_type='db4')
  def forward(self, x):
    return super(DB4AltWavelet, self).forward(x)

class DB10Wavelet(Wavelet):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights = False, activation='ReLU', active_g:bool = False):
    super(DB10Wavelet, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                    share_weights, activation, active_g, wavelet_type='db10')
  def forward(self, x):
    return super(DB10Wavelet, self).forward(x)

class DB10AltWavelet(AltWavelet):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights = False, activation='ReLU', active_g:bool = False):
    super(DB10AltWavelet, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                    share_weights, activation, active_g, wavelet_type='db10')
  def forward(self, x):
    return super(DB10AltWavelet, self).forward(x)
class DB20AltWavelet(AltWavelet):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights = False, activation='ReLU', active_g:bool = False):
    super(DB20AltWavelet, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                    share_weights, activation, active_g, wavelet_type='db20')
  def forward(self, x):
    return super(DB20AltWavelet, self).forward(x)

class Coif1Wavelet(Wavelet):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights = False, activation='ReLU', active_g:bool = False):
    super(Coif1Wavelet, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                    share_weights, activation, active_g, wavelet_type='coif1')
  def forward(self, x):
    return super(Coif1Wavelet, self).forward(x)

class Coif1AltWavelet(AltWavelet):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights = False, activation='ReLU', active_g:bool = False):
    super(Coif1AltWavelet, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                    share_weights, activation, active_g, wavelet_type='coif1')
  def forward(self, x):
    return super(Coif1AltWavelet, self).forward(x)

class Coif2Wavelet(Wavelet):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights = False, activation='ReLU', active_g:bool = False):
    super(Coif2Wavelet, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                    share_weights, activation, active_g, wavelet_type='coif2')
  def forward(self, x):
    return super(Coif2Wavelet, self).forward(x)

class Coif2AltWavelet(AltWavelet):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights = False, activation='ReLU', active_g:bool = False):
    super(Coif2AltWavelet, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                    share_weights, activation, active_g, wavelet_type='coif2')
  def forward(self, x):
    return super(Coif2AltWavelet, self).forward(x)

class Coif3Wavelet(Wavelet):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights = False, activation='ReLU', active_g:bool = False):
    super(Coif3Wavelet, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                    share_weights, activation, active_g, wavelet_type='coif3')
  def forward(self, x):
    return super(Coif3Wavelet, self).forward(x)

class Coif3AltWavelet(AltWavelet):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights = False, activation='ReLU', active_g:bool = False):
    super(Coif3AltWavelet, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                    share_weights, activation, active_g, wavelet_type='coif3')
  def forward(self, x):
    return super(Coif3AltWavelet, self).forward(x)

class Coif10Wavelet(Wavelet):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights = False, activation='ReLU', active_g:bool = False):
    super(Coif10Wavelet, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                    share_weights, activation, active_g, wavelet_type='coif10')
  def forward(self, x):
    return super(Coif10Wavelet, self).forward(x)

class Coif10AltWavelet(AltWavelet):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights = False, activation='ReLU', active_g:bool = False):
    super(Coif10AltWavelet, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                    share_weights, activation, active_g, wavelet_type='coif10')
  def forward(self, x):
    return super(Coif10AltWavelet, self).forward(x)


class Symlet2Wavelet(Wavelet):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights = False, activation='ReLU', active_g:bool = False):
    super(Symlet2Wavelet, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                    share_weights, activation, active_g, wavelet_type='sym2')
  def forward(self, x):
    return super(Symlet2Wavelet, self).forward(x)

class Symlet2AltWavelet(AltWavelet):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights = False, activation='ReLU', active_g:bool = False):
    super(Symlet2AltWavelet, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                    share_weights, activation, active_g, wavelet_type='sym2')
  def forward(self, x):
    return super(Symlet2AltWavelet, self).forward(x)

class Symlet3Wavelet(Wavelet):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights = False, activation='ReLU', active_g:bool = False):
    super(Symlet3Wavelet, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                    share_weights, activation, active_g, wavelet_type='sym3')
  def forward(self, x):
    return super(Symlet3Wavelet, self).forward(x)

class Symlet10Wavelet(Wavelet):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights = False, activation='ReLU', active_g:bool = False):
    super(Symlet10Wavelet, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                    share_weights, activation, active_g, wavelet_type='sym10')
  def forward(self, x):
    return super(Symlet10Wavelet, self).forward(x)

class Symlet20Wavelet(Wavelet):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights = False, activation='ReLU', active_g:bool = False):
    super(Symlet20Wavelet, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                    share_weights, activation, active_g, wavelet_type='sym20')
  def forward(self, x):
    return super(Symlet20Wavelet, self).forward(x)

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

class SeasonalityAE(AERootBlock):
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
    super(SeasonalityAE, self).__init__(backcast_length, units, activation, latent_dim = latent_dim)

    self.backcast_linear = nn.Linear(units, 2 * int(backcast_length / 2 - 1) + 1, bias = False)
    self.forecast_linear = nn.Linear(units, 2 * int(forecast_length / 2 - 1) + 1, bias = False)

    self.backcast_g = _SeasonalityGenerator(backcast_length)
    self.forecast_g = _SeasonalityGenerator(forecast_length)

  def forward(self, x):
    x = super(SeasonalityAE, self).forward(x)
    # linear compression
    backcast_thetas = self.backcast_linear(x)
    forecast_thetas = self.forecast_linear(x)

    # fourier expansion
    backcast = self.backcast_g(backcast_thetas)
    forecast = self.forecast_g(forecast_thetas)

    return backcast, forecast


class GenericAEBackcastAE(AERootBlock):
  def __init__(self,
                units:int,
                backcast_length:int,
                forecast_length:int,
                thetas_dim:int,
                share_weights:bool,
                activation:str = 'ReLU',
                active_g:bool = False,
                latent_dim:int = 5):

      super(GenericAEBackcastAE, self).__init__(backcast_length, units, activation,latent_dim=latent_dim)

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
    x = super(GenericAEBackcastAE, self).forward(x)
    b = self.b_encoder(x)
    b = self.b_decoder(b)

    theta_f = self.forecast_linear(x)
    f = self.forecast_g(theta_f)

    # N-BEATS paper does not apply activation here;
    # however Generic models will not always converge without it
    if self.active_g:
      if self.active_g != 'forecast':
        b = self.activation(b)
      if self.active_g != 'backcast':
        f = self.activation(f)
    return b,f

class AutoEncoderAE(AERootBlock):
  def __init__(self,
                units:int,
                backcast_length:int,
                forecast_length:int,
                thetas_dim:int,
                share_weights:bool,
                activation:str = 'ReLU',
                active_g:bool = False,
                latent_dim:int = 5):
    """_summary_

    Args:
        units (int): The number of inoput and output units
        backcast_length (int): The length of the historical data.
        forecast_length (int): The length of the forecast_length horizon.
        thetas_dim (int): The dimensionality of the compressed latent space in the middle of the autoencoder
        share_weights (bool): The weights of the encoder are shared if True.
        activation (str, optional): _description_. Defaults to 'ReLU'.
        active_g (bool, optional): _description_. Defaults to False.
        latent_dim (int, optional): _description_. Defaults to 5.
    """

    super(AutoEncoderAE, self).__init__(backcast_length, units, activation, latent_dim=latent_dim)

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
        getattr(nn, activation)(),
      )
    else:
      self.b_encoder = nn.Sequential(
          nn.Linear(units, thetas_dim),
          getattr(nn, activation)(),
      )
      self.f_encoder = nn.Sequential(
          nn.Linear(units, thetas_dim),
          getattr(nn, activation)(),
      )

    self.b_decoder = nn.Sequential(
        nn.Linear(thetas_dim, units),
        getattr(nn, activation)(),
        nn.Linear(units, backcast_length),
    )
    self.f_decoder = nn.Sequential(
        nn.Linear(thetas_dim, units),
        getattr(nn, activation)(),
        nn.Linear(units, forecast_length),
    )

  def forward(self, x):
    x = super(AutoEncoderAE, self).forward(x)
    b = self.b_encoder(x)
    b = self.b_decoder(b)

    f = self.f_encoder(x)
    f = self.f_decoder(f)

    # N-BEATS paper does not apply activation here, but Generic models will not converge sometimes without it
    if self.active_g:
      if self.active_g != 'forecast':
        b = self.activation(b)
      if self.active_g != 'backcast':
        f = self.activation(f)

    return b,f

class GenericAE(AERootBlock):
  def __init__(self,
               units:int,
               backcast_length:int,
               forecast_length:int,
               thetas_dim:int = 5,
               share_weights:bool= False,
               activation:str = 'ReLU',
               active_g:bool = False,
               latent_dim:int = 5):
    """Paper-faithful Generic Block with AERootBlock (bottleneck pre-split) backbone.

    Uses direct linear projections from units to target lengths (no intermediate
    thetas_dim bottleneck), matching the paper's Generic formulation.

    Args:
        units (int): Width of the fully connected layers.
        backcast_length (int): Length of the historical data.
        forecast_length (int): Length of the forecast horizon.
        thetas_dim (int, optional): Not used (kept for API compatibility). Defaults to 5.
        share_weights (bool, optional): Defaults to False.
        activation (str, optional): Defaults to 'ReLU'.
        active_g (bool, optional): If True, applies activation after projection. Defaults to False.
        latent_dim (int, optional): Latent dimension for the AE backbone. Defaults to 5.
    """
    super(GenericAE, self).__init__(backcast_length, units, activation, latent_dim=latent_dim)

    self.backcast_length = backcast_length
    self.forecast_length = forecast_length
    self.active_g = active_g

    self.theta_b_fc = nn.Linear(units, backcast_length, bias=False)
    self.theta_f_fc = nn.Linear(units, forecast_length, bias=False)

  def forward(self, x):
    x = super(GenericAE, self).forward(x)

    backcast = self.theta_b_fc(x)
    forecast = self.theta_f_fc(x)

    if self.active_g:
      if self.active_g != 'forecast':
        backcast = self.activation(backcast)
      if self.active_g != 'backcast':
        forecast = self.activation(forecast)

    return backcast, forecast


class BottleneckGenericAE(AERootBlock):
  def __init__(self,
               units:int,
               backcast_length:int,
               forecast_length:int,
               thetas_dim:int = 5,
               share_weights:bool= False,
               activation:str = 'ReLU',
               active_g:bool = False,
               latent_dim:int = 5):
    """Bottleneck Generic Block with AERootBlock (bottleneck pre-split) backbone.

    Uses a two-stage projection through an intermediate thetas_dim bottleneck,
    equivalent to a rank-d factorization of the basis expansion matrix.

    Args:
        units (int): Width of the fully connected layers.
        backcast_length (int): Length of the historical data.
        forecast_length (int): Length of the forecast horizon.
        thetas_dim (int, optional): Bottleneck dimension. Defaults to 5.
        share_weights (bool, optional): Defaults to False.
        activation (str, optional): Defaults to 'ReLU'.
        active_g (bool, optional): If True, applies activation after expansion. Defaults to False.
        latent_dim (int, optional): Latent dimension for the AE backbone. Defaults to 5.
    """
    super(BottleneckGenericAE, self).__init__(backcast_length, units, activation, latent_dim=latent_dim)

    if share_weights:
        self.backcast_linear = self.forecast_linear = nn.Linear(units, thetas_dim)
    else:
        self.backcast_linear = nn.Linear(units, thetas_dim)
        self.forecast_linear = nn.Linear(units, thetas_dim)

    self.backcast_g = nn.Linear(thetas_dim, backcast_length, bias = False)
    self.forecast_g = nn.Linear(thetas_dim, forecast_length, bias = False)
    self.active_g = active_g

  def forward(self, x):
    x = super(BottleneckGenericAE, self).forward(x)
    theta_b = self.backcast_linear(x)
    theta_f = self.forecast_linear(x)

    backcast = self.backcast_g(theta_b)
    forecast = self.forecast_g(theta_f)

    if self.active_g:
      if self.active_g != 'forecast':
        backcast = self.activation(backcast)
      if self.active_g != 'backcast':
        forecast = self.activation(forecast)

    return backcast, forecast


class TrendAE(AERootBlock):
  def __init__(self, units, backcast_length, forecast_length, thetas_dim,
               share_weights = False, activation='ReLU', active_g:bool = False, latent_dim = 5):
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
          If True, the inital weights of the Linear layers are shared. Defaults to False.
        activation (str, optional):
          The activation function passed to the parent class Block. Defaults to 'ReLU'.
    """
    super(TrendAE, self).__init__(backcast_length, units, activation, latent_dim=latent_dim)
    if share_weights:
        self.backcast_linear = self.forecast_linear = nn.Linear(units, thetas_dim)
    else:
        self.backcast_linear = nn.Linear(units, thetas_dim)
        self.forecast_linear = nn.Linear(units, thetas_dim)

    self.backcast_g = _TrendGenerator(thetas_dim, backcast_length)
    self.forecast_g = _TrendGenerator(thetas_dim, forecast_length)

  def forward(self, x):
    x = super(TrendAE, self).forward(x)

    # linear compression
    backcast_thetas = self.backcast_linear(x)
    forecast_thetas = self.forecast_linear(x)

    #  Vandermonde basis expansion
    backcast = self.backcast_g(backcast_thetas)
    forecast = self.forecast_g(forecast_thetas)

    return backcast, forecast



# ---------------------------------------------------------------------------
# V2 Wavelet Blocks — Numerically stabilized implementations
#
# Root causes of V1 instability (confirmed by SVD analysis):
#   - Haar basis is singular (cond=inf), DB3 cond~604K, Coif2 cond~59M
#   - Spectral norms 11-89x amplify signals through the basis matmul
#   - No input normalization before basis expansion
#   - Default Kaiming init mismatched with fixed basis scale
#
# V2 fixes:
#   1. Spectral normalization: basis divided by max singular value (spectral norm=1)
#   2. LayerNorm before linear projection stabilizes input distribution
#   3. Xavier uniform init on projection layers (linear, not ReLU, downstream)
#   4. Output clamping prevents catastrophic amplification across 30 stacks
# ---------------------------------------------------------------------------

class _WaveletGeneratorV2(nn.Module):
  def __init__(self, basis_dim, wavelet_type='db3'):
    super().__init__()

    wavelet = pywt.Wavelet(wavelet_type)
    phi, psi, x = wavelet.wavefun(level=10)

    interp_phi = interp1d(x, phi, kind='linear')
    interp_psi = interp1d(x, psi, kind='linear')
    new_x = np.linspace(min(x), max(x), basis_dim)
    new_phi = interp_phi(new_x)
    new_psi = interp_psi(new_x)

    W = np.zeros((basis_dim, basis_dim))
    for i in range(basis_dim):
      W[:, i] = np.roll(new_phi, i)[:basis_dim] if i < basis_dim // 2 else np.roll(new_psi, i - basis_dim // 2)[:basis_dim]

    # Spectral normalization: divide by largest singular value
    svd_values = np.linalg.svd(W, compute_uv=False)
    max_sv = svd_values[0]
    if max_sv > 1e-8:
      W = W / max_sv

    self.basis = nn.Parameter(torch.tensor(W, dtype=torch.float32), requires_grad=False)

  def forward(self, x):
    return torch.matmul(x, self.basis)


class _AltWaveletGeneratorV2(nn.Module):
  def __init__(self, N, target_length, wavelet_type='db2'):
    super().__init__()

    wavelet = pywt.Wavelet(wavelet_type)
    phi, psi, x = wavelet.wavefun(level=10)

    interp_phi = interp1d(x, phi, kind='linear')
    interp_psi = interp1d(x, psi, kind='linear')
    new_x = np.linspace(min(x), max(x), N)
    new_phi = interp_phi(new_x)
    new_psi = interp_psi(new_x)

    W = np.zeros((N, target_length))
    for i in range(target_length):
      W[:, i] = np.roll(new_phi, i)[:N] if i < target_length // 2 else np.roll(new_psi, i - target_length // 2)[:N]

    # Spectral normalization: divide by largest singular value
    svd_values = np.linalg.svd(W, compute_uv=False)
    max_sv = svd_values[0]
    if max_sv > 1e-8:
      W = W / max_sv

    self.basis = nn.Parameter(torch.tensor(W, dtype=torch.float32), requires_grad=False)

  def forward(self, x):
    return torch.matmul(x, self.basis)


class WaveletV2(RootBlock):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False, wavelet_type='db3'):
    super(WaveletV2, self).__init__(backcast_length, units, activation)

    self.activation = getattr(nn, activation)()
    self.wavelet_type = wavelet_type
    self.share_weights = share_weights

    self.layer_norm = nn.LayerNorm(units)

    if share_weights:
      self.backcast_linear = self.forecast_linear = nn.Linear(units, basis_dim)
    else:
      self.backcast_linear = nn.Linear(units, basis_dim)
      self.forecast_linear = nn.Linear(units, basis_dim)

    self.backcast_g = _WaveletGeneratorV2(basis_dim, wavelet_type=wavelet_type)
    self.forecast_g = _WaveletGeneratorV2(basis_dim, wavelet_type=wavelet_type)
    self.backcast_down_sample = nn.Linear(basis_dim, backcast_length, bias=False)
    self.forecast_down_sample = nn.Linear(basis_dim, forecast_length, bias=False)

    self._init_projection_weights()

  def _init_projection_weights(self):
    nn.init.xavier_uniform_(self.backcast_linear.weight)
    nn.init.xavier_uniform_(self.forecast_linear.weight)
    nn.init.xavier_uniform_(self.backcast_down_sample.weight)
    nn.init.xavier_uniform_(self.forecast_down_sample.weight)

  def forward(self, x):
    x = super(WaveletV2, self).forward(x)
    x = self.layer_norm(x)

    b = self.backcast_linear(x)
    f = self.forecast_linear(x)
    b = self.backcast_g(b)
    b = self.backcast_down_sample(b)
    f = self.forecast_g(f)
    f = self.forecast_down_sample(f)

    b = torch.clamp(b, min=-1e4, max=1e4)
    f = torch.clamp(f, min=-1e4, max=1e4)

    return b, f


class AltWaveletV2(RootBlock):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False, wavelet_type='db2'):
    super(AltWaveletV2, self).__init__(backcast_length, units, activation)

    self.activation = getattr(nn, activation)()
    self.wavelet_type = wavelet_type
    self.share_weights = share_weights

    self.layer_norm = nn.LayerNorm(units)

    if share_weights:
      self.backcast_linear = self.forecast_linear = nn.Linear(units, basis_dim)
    else:
      self.backcast_linear = nn.Linear(units, basis_dim)
      self.forecast_linear = nn.Linear(units, basis_dim)

    self.backcast_g = _AltWaveletGeneratorV2(basis_dim, backcast_length, wavelet_type=wavelet_type)
    self.forecast_g = _AltWaveletGeneratorV2(basis_dim, forecast_length, wavelet_type=wavelet_type)

    self._init_projection_weights()

  def _init_projection_weights(self):
    nn.init.xavier_uniform_(self.backcast_linear.weight)
    nn.init.xavier_uniform_(self.forecast_linear.weight)

  def forward(self, x):
    x = super(AltWaveletV2, self).forward(x)
    x = self.layer_norm(x)

    b = self.backcast_linear(x)
    f = self.forecast_linear(x)
    b = self.backcast_g(b)
    f = self.forecast_g(f)

    b = torch.clamp(b, min=-1e4, max=1e4)
    f = torch.clamp(f, min=-1e4, max=1e4)

    return b, f


# --- V2 Wavelet subclasses (Wavelet backbone) ---

class HaarWaveletV2(WaveletV2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(HaarWaveletV2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                       share_weights, activation, active_g, wavelet_type='haar')
  def forward(self, x):
    return super(HaarWaveletV2, self).forward(x)

class DB2WaveletV2(WaveletV2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(DB2WaveletV2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                      share_weights, activation, active_g, wavelet_type='db2')
  def forward(self, x):
    return super(DB2WaveletV2, self).forward(x)

class DB3WaveletV2(WaveletV2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(DB3WaveletV2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                      share_weights, activation, active_g, wavelet_type='db3')
  def forward(self, x):
    return super(DB3WaveletV2, self).forward(x)

class DB4WaveletV2(WaveletV2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(DB4WaveletV2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                      share_weights, activation, active_g, wavelet_type='db4')
  def forward(self, x):
    return super(DB4WaveletV2, self).forward(x)

class DB10WaveletV2(WaveletV2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(DB10WaveletV2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                       share_weights, activation, active_g, wavelet_type='db10')
  def forward(self, x):
    return super(DB10WaveletV2, self).forward(x)

class Coif1WaveletV2(WaveletV2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(Coif1WaveletV2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                        share_weights, activation, active_g, wavelet_type='coif1')
  def forward(self, x):
    return super(Coif1WaveletV2, self).forward(x)

class Coif2WaveletV2(WaveletV2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(Coif2WaveletV2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                        share_weights, activation, active_g, wavelet_type='coif2')
  def forward(self, x):
    return super(Coif2WaveletV2, self).forward(x)

class Coif3WaveletV2(WaveletV2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(Coif3WaveletV2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                        share_weights, activation, active_g, wavelet_type='coif3')
  def forward(self, x):
    return super(Coif3WaveletV2, self).forward(x)

class Coif10WaveletV2(WaveletV2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(Coif10WaveletV2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                         share_weights, activation, active_g, wavelet_type='coif10')
  def forward(self, x):
    return super(Coif10WaveletV2, self).forward(x)

class Symlet2WaveletV2(WaveletV2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(Symlet2WaveletV2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                          share_weights, activation, active_g, wavelet_type='sym2')
  def forward(self, x):
    return super(Symlet2WaveletV2, self).forward(x)

class Symlet3WaveletV2(WaveletV2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(Symlet3WaveletV2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                          share_weights, activation, active_g, wavelet_type='sym3')
  def forward(self, x):
    return super(Symlet3WaveletV2, self).forward(x)

class Symlet10WaveletV2(WaveletV2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(Symlet10WaveletV2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                           share_weights, activation, active_g, wavelet_type='sym10')
  def forward(self, x):
    return super(Symlet10WaveletV2, self).forward(x)

class Symlet20WaveletV2(WaveletV2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(Symlet20WaveletV2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                           share_weights, activation, active_g, wavelet_type='sym20')
  def forward(self, x):
    return super(Symlet20WaveletV2, self).forward(x)


# --- V2 AltWavelet subclasses (AltWavelet backbone) ---

class HaarAltWaveletV2(AltWaveletV2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(HaarAltWaveletV2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                          share_weights, activation, active_g, wavelet_type='haar')
  def forward(self, x):
    return super(HaarAltWaveletV2, self).forward(x)

class DB2AltWaveletV2(AltWaveletV2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(DB2AltWaveletV2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                         share_weights, activation, active_g, wavelet_type='db2')
  def forward(self, x):
    return super(DB2AltWaveletV2, self).forward(x)

class DB3AltWaveletV2(AltWaveletV2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(DB3AltWaveletV2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                         share_weights, activation, active_g, wavelet_type='db3')
  def forward(self, x):
    return super(DB3AltWaveletV2, self).forward(x)

class DB4AltWaveletV2(AltWaveletV2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(DB4AltWaveletV2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                         share_weights, activation, active_g, wavelet_type='db4')
  def forward(self, x):
    return super(DB4AltWaveletV2, self).forward(x)

class DB10AltWaveletV2(AltWaveletV2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(DB10AltWaveletV2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                          share_weights, activation, active_g, wavelet_type='db10')
  def forward(self, x):
    return super(DB10AltWaveletV2, self).forward(x)

class DB20AltWaveletV2(AltWaveletV2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(DB20AltWaveletV2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                          share_weights, activation, active_g, wavelet_type='db20')
  def forward(self, x):
    return super(DB20AltWaveletV2, self).forward(x)

class Coif1AltWaveletV2(AltWaveletV2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(Coif1AltWaveletV2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                           share_weights, activation, active_g, wavelet_type='coif1')
  def forward(self, x):
    return super(Coif1AltWaveletV2, self).forward(x)

class Coif2AltWaveletV2(AltWaveletV2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(Coif2AltWaveletV2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                           share_weights, activation, active_g, wavelet_type='coif2')
  def forward(self, x):
    return super(Coif2AltWaveletV2, self).forward(x)

class Coif3AltWaveletV2(AltWaveletV2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(Coif3AltWaveletV2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                           share_weights, activation, active_g, wavelet_type='coif3')
  def forward(self, x):
    return super(Coif3AltWaveletV2, self).forward(x)

class Coif10AltWaveletV2(AltWaveletV2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(Coif10AltWaveletV2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                            share_weights, activation, active_g, wavelet_type='coif10')
  def forward(self, x):
    return super(Coif10AltWaveletV2, self).forward(x)

class Symlet2AltWaveletV2(AltWaveletV2):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(Symlet2AltWaveletV2, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                             share_weights, activation, active_g, wavelet_type='sym2')
  def forward(self, x):
    return super(Symlet2AltWaveletV2, self).forward(x)


# ---------------------------------------------------------------------------
# V3 Wavelet Blocks — Orthonormal DWT basis via impulse-response synthesis
#
# Root cause fix for V1/V2 instability: the rolled phi/psi basis construction
# produces ill-conditioned matrices (DB3 cond~604K). V3 replaces this entirely
# with proper DWT impulse-response synthesis + SVD orthogonalization, producing
# a genuinely orthonormal basis where ALL singular values = 1.0 (cond = 1.0).
#
# Architecture: 2-projection pattern matching Seasonality/Trend (no downsampling
# layer needed since the basis is always target_length x target_length).
# ---------------------------------------------------------------------------

class _WaveletGeneratorV3(nn.Module):
  def __init__(self, target_length, wavelet_type='db3', max_decomp_level=5):
    super().__init__()
    basis = self._build_basis(target_length, wavelet_type, max_decomp_level)
    self.basis = nn.Parameter(basis, requires_grad=False)

  @staticmethod
  def _build_basis(target_length, wavelet_type, max_decomp_level):
    import logging

    wavelet = pywt.Wavelet(wavelet_type)
    max_level = pywt.dwt_max_level(target_length, wavelet.dec_len)
    level = max(1, min(max_level, max_decomp_level))

    # Get DWT coefficient structure
    dummy = np.zeros(target_length)
    coeffs = pywt.wavedec(dummy, wavelet_type, level=level)
    coeff_lengths = [len(c) for c in coeffs]

    # Build raw synthesis matrix via impulse responses
    basis_rows = []
    for band_idx, band_len in enumerate(coeff_lengths):
      for j in range(band_len):
        impulse = [np.zeros(l) for l in coeff_lengths]
        impulse[band_idx][j] = 1.0
        reconstructed = pywt.waverec(impulse, wavelet_type)
        basis_rows.append(reconstructed[:target_length])

    raw_basis = np.array(basis_rows, dtype=np.float64)

    # SVD orthogonalization
    U, S, Vt = np.linalg.svd(raw_basis, full_matrices=False)
    tol = S[0] * max(raw_basis.shape) * np.finfo(np.float64).eps
    rank = int(np.sum(S > tol))
    ortho_basis = Vt[:rank, :]  # (rank, target_length), rank == target_length

    if rank < target_length:
      logging.warning(f"WaveletV3 rank-deficient: {rank}/{target_length} for '{wavelet_type}'")

    return torch.tensor(ortho_basis, dtype=torch.float32)

  def forward(self, x):
    return torch.matmul(x, self.basis)


class WaveletV3(RootBlock):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False, wavelet_type='db3'):
    super(WaveletV3, self).__init__(backcast_length, units, activation)
    self.wavelet_type = wavelet_type
    self.share_weights = share_weights
    self.active_g = active_g

    # Linear to target_length coefficients (like Seasonality's n_harmonics)
    if share_weights:
      self.backcast_linear = self.forecast_linear = nn.Linear(units, backcast_length, bias=False)
    else:
      self.backcast_linear = nn.Linear(units, backcast_length, bias=False)
      self.forecast_linear = nn.Linear(units, forecast_length, bias=False)

    self.backcast_g = _WaveletGeneratorV3(backcast_length, wavelet_type=wavelet_type)
    self.forecast_g = _WaveletGeneratorV3(forecast_length, wavelet_type=wavelet_type)

  def forward(self, x):
    x = super(WaveletV3, self).forward(x)
    backcast_thetas = self.backcast_linear(x)
    forecast_thetas = self.forecast_linear(x)
    backcast = self.backcast_g(backcast_thetas)
    forecast = self.forecast_g(forecast_thetas)
    if self.active_g:
      if self.active_g != 'forecast':
        backcast = self.activation(backcast)
      if self.active_g != 'backcast':
        forecast = self.activation(forecast)
    return backcast, forecast


# --- V3 Wavelet subclasses (thin wrappers setting wavelet_type) ---

class HaarWaveletV3(WaveletV3):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(HaarWaveletV3, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                       share_weights, activation, active_g, wavelet_type='haar')
  def forward(self, x):
    return super(HaarWaveletV3, self).forward(x)

class DB2WaveletV3(WaveletV3):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(DB2WaveletV3, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                      share_weights, activation, active_g, wavelet_type='db2')
  def forward(self, x):
    return super(DB2WaveletV3, self).forward(x)

class DB3WaveletV3(WaveletV3):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(DB3WaveletV3, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                      share_weights, activation, active_g, wavelet_type='db3')
  def forward(self, x):
    return super(DB3WaveletV3, self).forward(x)

class DB4WaveletV3(WaveletV3):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(DB4WaveletV3, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                      share_weights, activation, active_g, wavelet_type='db4')
  def forward(self, x):
    return super(DB4WaveletV3, self).forward(x)

class DB10WaveletV3(WaveletV3):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(DB10WaveletV3, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                       share_weights, activation, active_g, wavelet_type='db10')
  def forward(self, x):
    return super(DB10WaveletV3, self).forward(x)

class DB20WaveletV3(WaveletV3):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(DB20WaveletV3, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                       share_weights, activation, active_g, wavelet_type='db20')
  def forward(self, x):
    return super(DB20WaveletV3, self).forward(x)

class Coif1WaveletV3(WaveletV3):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(Coif1WaveletV3, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                        share_weights, activation, active_g, wavelet_type='coif1')
  def forward(self, x):
    return super(Coif1WaveletV3, self).forward(x)

class Coif2WaveletV3(WaveletV3):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(Coif2WaveletV3, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                        share_weights, activation, active_g, wavelet_type='coif2')
  def forward(self, x):
    return super(Coif2WaveletV3, self).forward(x)

class Coif3WaveletV3(WaveletV3):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(Coif3WaveletV3, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                        share_weights, activation, active_g, wavelet_type='coif3')
  def forward(self, x):
    return super(Coif3WaveletV3, self).forward(x)

class Coif10WaveletV3(WaveletV3):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(Coif10WaveletV3, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                         share_weights, activation, active_g, wavelet_type='coif10')
  def forward(self, x):
    return super(Coif10WaveletV3, self).forward(x)

class Symlet2WaveletV3(WaveletV3):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(Symlet2WaveletV3, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                          share_weights, activation, active_g, wavelet_type='sym2')
  def forward(self, x):
    return super(Symlet2WaveletV3, self).forward(x)

class Symlet3WaveletV3(WaveletV3):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(Symlet3WaveletV3, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                          share_weights, activation, active_g, wavelet_type='sym3')
  def forward(self, x):
    return super(Symlet3WaveletV3, self).forward(x)

class Symlet10WaveletV3(WaveletV3):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(Symlet10WaveletV3, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                           share_weights, activation, active_g, wavelet_type='sym10')
  def forward(self, x):
    return super(Symlet10WaveletV3, self).forward(x)

class Symlet20WaveletV3(WaveletV3):
  def __init__(self, units, backcast_length, forecast_length, basis_dim=32,
               share_weights=False, activation='ReLU', active_g: bool = False):
    super(Symlet20WaveletV3, self).__init__(units, backcast_length, forecast_length, basis_dim,
                                           share_weights, activation, active_g, wavelet_type='sym20')
  def forward(self, x):
    return super(Symlet20WaveletV3, self).forward(x)
