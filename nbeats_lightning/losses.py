import torch
from torch import nn

class MASELoss(nn.Module):
  """ Mean Absolute Scaled Error (MASE) PyTorch loss function."""  
  def __init__(self, seasonal_period=1):
      """Initializes MASELoss.

      Args:
          seasonal_period (int, optional): The seasonal period of the data.  
          For instance 12 for Yearly M4 data. Defaults to 1.
      """
      super(MASELoss, self).__init__()
      self.seasonal_period = seasonal_period

  def forward(self, y_pred, y_true):
      """Calculates MASE loss between y_pred and y_true.

      Args:
          y_pred (Tensor): predictions
          y_true (Tensor): targets

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
  def __init__(self):
      super(SMAPELoss, self).__init__()

  def forward(self, y_pred, y_true):
      """Calculates SMAPE loss between y_pred and y_true.

      Args:
          y_pred (Tensor): predictions
          y_true (Tensor): targets

      Returns:
          Tensor: SMAPE loss
      """
      # Calculate the SMAPE loss according to the formula
      denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2.0
      diff = torch.abs(y_true - y_pred) / denominator
      
      # Handle the case where both the predicted and true values are 0
      # which would result in a NaN SMAPE value; we set it to 0 in this case
      diff[denominator == 0] = 0.0
      
      smape = torch.mean(diff) * 100.0
      
      return smape
    
class MAPELoss(nn.Module):
  def __init__(self):
      super(MAPELoss, self).__init__()

  def forward(self, y_pred, y_true):
      """Calculated MAPE, Mean Absolute Percentage Error (MAPE), loss between y_pred and y_true.

      Args:
          y_pred (Tensor): predicitons
          y_true (Tensor): targets

      Returns:
          Tensor : MAPE loss
      """
      # Avoid division by zero by adding a small constant to the denominator
      denominator = torch.abs(y_true) + 1e-6
      
      # Calculate the MAPE loss according to the formula
      mape = torch.mean(torch.abs((y_true - y_pred) / denominator)) * 100.0
      
      return mape  
