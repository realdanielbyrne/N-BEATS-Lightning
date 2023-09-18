import torch
from torch import nn

class MASELoss(nn.Module):
    def __init__(self, seasonal_period=1):
        super(MASELoss, self).__init__()
        self.seasonal_period = seasonal_period

    def forward(self, y_pred, y_true):
        m = self.seasonal_period
        
        # Calculating the numerator, which is the mean absolute error between the true and predicted values
        numerator = torch.mean(torch.abs(y_true - y_pred), dim=1)
        
        # Calculating the denominator, which is the mean absolute error of the naive forecast
        denominator = torch.mean(torch.abs(y_true[:, m:] - y_true[:, :-m]), dim=1)
        
        # Avoid division by zero by adding a small constant to the denominator
        denominator = torch.where(denominator == 0, torch.ones_like(denominator) * 1e-6, denominator)
        
        # Calculating the MASE for each time series in the batch
        mase = numerator / denominator
        
        # Averaging the MASE values across the batch to get a single scalar loss value
        return torch.mean(mase)

    def __str__(self):
      name = type(self).__name__
      return name

class SMAPELoss(nn.Module):
    def __init__(self):
        super(SMAPELoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Calculate the SMAPE loss according to the formula
        denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2.0
        diff = torch.abs(y_true - y_pred) / denominator
        
        # Handle the case where both the predicted and true values are 0
        # which would result in a NaN SMAPE value; we set it to 0 in this case
        diff[denominator == 0] = 0.0
        
        smape = torch.mean(diff) * 100.0
        
        return smape

    def __str__(self):
      name = type(self).__name__
      return name     