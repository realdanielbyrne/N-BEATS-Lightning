import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.nn import functional as F
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset, random_split

def squeeze_last_dim(tensor):
  if len(tensor.shape) == 3 and tensor.shape[-1] == 1:  # (128, 10, 1) => (128, 10).
      return tensor[..., 0]
  return tensor
