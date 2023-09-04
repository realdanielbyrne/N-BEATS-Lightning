import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split

def squeeze_last_dim(tensor):
  if len(tensor.shape) == 3 and tensor.shape[-1] == 1:  # (128, 10, 1) => (128, 10).
      return tensor[..., 0]
  return tensor

class SMAPE(nn.Module):
  """
  Calculates sMAPE

  :param preds: predicted values
  :param targets: actual values
  :return: sMAPE
  """

  def __init__(self):
    super(SMAPE, self).__init__() 
    
  def forward(preds, targets):
    # flatten
    targets = torch.reshape(targets, (-1,))
    preds = torch.reshape(preds, (-1,))
    return torch.mean(2.0 * torch.abs(targets - preds) / (torch.abs(targets) + torch.abs(preds)))

class MAsE(nn.Module):
  """
  Calculates MAsE, the Mean Absolute scaled Error (MAsE) for the M4
  competition. If MAsE is greater than 1, it implies that the given
  forecasts are worse than a simple naïve forecast, while a MAsE 
  less than 1 indicates that the given forecasts are better than 
  the naïve forecast.

  Parameters
  ----------
  :param preds: predicted values
  :param targets: target values
  :param x: insample data
  :param freq:  The data frequency, which indicates how often data is collected
    The M4-info.csv file contains the frequency of each time series.  
  
  :return: MAsE
  """
  def __init__(self, freq):
    super(MAsE, self).__init__()
    if freq is None:
      raise ValueError("freq must be specified")
    self.freq = freq
    
  def forward(self, preds, targets, x):
    y_t = x[:-self.freq]
    y_tm = x[self.freq:]    
    masep = torch.mean(torch.abs(y_t - y_tm))

    return torch.mean(torch.abs(targets - preds)) / masep


class Mape(nn.Module):
  """
    mape = mean(abs((y - yhat) / abs(y)))
    
  """
  def __init__(self):
    super(Mape, self).__init__()
  
  def forward(self, preds, targets):
    # flatten
    targets = torch.reshape(targets, (-1,))
    preds = torch.reshape(preds, (-1,))
    return torch.mean(torch.abs(targets - preds) / torch.abs(targets))