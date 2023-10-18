import torch
from torch import nn


class MASELoss(nn.Module):
  """ Mean Absolute Scaled Error (MASE) PyTorch loss function."""  
  def __init__(self, seasonal_period=1):
      """Initializes MASELoss.

      Args:
          seasonal_period (int, optional): The seasonal period of the data.  
          For instance 1 for Yearly M4 data. Defaults to 1.
      """
      super(MASELoss, self).__init__()
      self.seasonal_period = seasonal_period

  def forward(self, y_pred, y_true):
      """Calculates MASE loss between y_pred and y_true.

      Args:
          y_pred (Tensor): The tensor of predicted values.
          y_true (Tensor): The tensor of true values.

      Returns:
          Tensor: MASE loss
      """
      m = self.seasonal_period
      
      # Calculating the numerator, which is the mean absolute error between the true and predicted values
      numerator = torch.mean(torch.abs(y_true - y_pred), dim=1)
      
      # Calculating the denominator, which is the mean absolute error of the naive forecast
      denominator = torch.mean(torch.abs(y_true[:, m:] - y_true[:, :-m]), dim=1)
      
      # Avoid division by zero by adding a small constant to the denominator
      denominator = torch.where(denominator == 0, torch.ones_like(denominator) * 1e-6, denominator)
      
      # Calculating the MASE for each time series in the batch and then averaging 
      mase = torch.mean(numerator / denominator)
          
      return mase

class SMAPELoss(nn.Module):
  def __init__(self, epsilon=1e-8):
      super(SMAPELoss, self).__init__()
      self.epsilon = epsilon

  def forward(self, y_pred, y_true):
      """Calculates SMAPE loss between y_pred and y_true.

      Args:
          y_pred (Tensor): The tensor of predicted values.
          y_true (Tensor): The tensor of true values.

      Returns:
          Tensor: SMAPE loss
      """
      # Calculate the SMAPE loss according to the formula
      # Adding epsilon to the denominator to avoid division by zero
      denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2.0 + self.epsilon
      diff = torch.abs(y_true - y_pred) / denominator
      
      smape = torch.mean(diff) * 100.0
      
      return smape
    
class MAPELoss(nn.Module):
  def __init__(self):
      super(MAPELoss, self).__init__()

  def forward(self, y_pred, y_true):
      """Calculated MAPE, Mean Absolute Percentage Error (MAPE), loss between y_pred and y_true.

      Args:
          y_pred (torch.Tensor): The tensor of predicted values.
          y_true (torch.Tensor): The tensor of true values.

      Returns:
          Tensor : MAPE loss
      """
      # Avoid division by zero by adding a small constant to the denominator
      denominator = torch.abs(y_true) + 1e-6
      
      # Calculate the MAPE loss according to the formula
      mape = torch.mean(torch.abs((y_true - y_pred) / denominator)) * 100.0
      
      return mape  

class NormalizedDeviationLoss(nn.Module):
  def __init__(self):
    super(NormalizedDeviationLoss, self).__init__()

  def forward(self, y_pred, y_true):
      """
      Computes the Normalized Deviation loss between the predicted and true tensors based on the given mathematical definition:
      
      ND = sum(|y_pred - y_true|) / sum(|y_true|)
      
      Parameters
      ----------
      y_pred (torch.Tensor) : The tensor of predicted values.
          
      y_true (torch.Tensor) : The tensor of true values.
          
      Returns
      -------
      torch.Tensor
          The Normalized Deviation loss.
      """
      abs_deviation = torch.abs(y_pred - y_true)
      sum_abs_deviation = torch.sum(abs_deviation)
      sum_abs_true_value = torch.sum(torch.abs(y_true))
      normalized_deviation = sum_abs_deviation / sum_abs_true_value
      return normalized_deviation
