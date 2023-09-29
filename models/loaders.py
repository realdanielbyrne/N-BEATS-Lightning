from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from torch.utils.data import random_split

class TimeSeriesCollectionDataset(Dataset):
  def __init__(self, 
               data, 
               backcast, 
               forecast):
      """The TimeSeriesCollectionDataset class is a PyTorch Dataset that takes a 
      collection of time series as input and returns a single sample of the time 
      series. Used for training a time series model whose input is a collection 
      of time series.

      Parameters
      ----------
        train_data (numpy.ndarray): 
          The univariate time series data. The data organization is assumed to be a 
          numpy.ndarray with rows representingtime series and columns representing time steps. 
        backcast (int, optional): 
          The length of the historical data.
        forecast (int, optional): 
          The length of the future data to predict.
      """
    
      super(TimeSeriesCollectionDataset, self).__init__()
      self.data = data
      self.backcast = backcast
      self.forecast = forecast
      self.items = []

      total_len = self.backcast + self.forecast
      for row in range(self.data.shape[0]):
          for col_start in range(0, self.data.shape[1] - total_len + 1):
              seq = self.data[row, col_start:col_start + total_len]
              if not np.isnan(seq).any():
                  self.items.append((row, col_start))
  
  def __len__(self):
    return len(self.items)

  def __getitem__(self, idx):
    row, col = self.items[idx]
    x = self.data[row, col:col+self.backcast]
    y = self.data[row, col+self.backcast:col+self.backcast+self.forecast]

    return torch.FloatTensor(x), torch.FloatTensor(y)    
      
class TimeSeriesCollectionDataModule(pl.LightningDataModule):
  def __init__(self, 
               train_data, 
               backcast, 
               forecast, 
               batch_size=1024, 
               split_ratio=0.8,
               debug = False):
    """The TimeSeriesCollectionDataModule class is a PyTorch Lightning DataModule
    used for training a time series model whose input is a collection of time series.
    

    Parameters
    ----------
        train_data (numpy.ndarray): 
          The univariate time series data. The data organization is assumed to be a 
          numpy.ndarray with rows representingtime series and columns representing time steps. 
        backcast (int, optional): 
          The length of the historical data.
        forecast (int, optional): 
          The length of the future data to predict.
        batch_size (int, optional): 
          The batch size. Defaults to 1024.
        split_ratio (float, optional): 
          The ratio of the data to use for training/validation.
        debug (bool, optional): 
          If True, only use a small subset of the data. Defaults False.
    """
          
    super(TimeSeriesCollectionDataModule, self).__init__()
    self.train_data_raw = train_data
    self.backcast = backcast
    self.forecast = forecast
    self.batch_size = batch_size
    self.split_ratio = split_ratio
    self.debug = debug

  def setup(self, stage:str=None):      
        
    shuffled = self.train_data_raw.sample(frac=1).reset_index(drop=True)
    train_rows = int(self.split_ratio * len(shuffled))
    
    self.train_data = shuffled.iloc[:train_rows].values      
    self.val_data = shuffled.iloc[train_rows:].values
          
    self.train_dataset = TimeSeriesCollectionDataset(self.train_data, self.backcast, self.forecast)
    self.val_dataset = TimeSeriesCollectionDataset(self.val_data, self.backcast, self.forecast)    
    
  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle = True)

  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle = False)
   
class TimeSeriesCollectionTestModule(pl.LightningDataModule):
  def __init__(self, 
                train_data,
                test_data,
                backcast, 
                forecast, 
                batch_size=1024):
    """The TimeSeriesCollectionTestModule class is a PyTorch Lightning DataModule
    used for testing a time series model whose input is a collection of time series.
    The final `backcast` samples of each time series in `train_data` are concatenated
    with the first `forecast` samples of the corresponding time series in `test_data`.
    
    Parameters
    ----------
      backcast (int, optional): 
        The length of the historical data.
      forecast (int, optional): 
        The length of the future data to predict.
      batch_size (int, optional): 
        The batch size. Defaults to 1024.
    """

    super(TimeSeriesCollectionTestModule, self).__init__()
    if (isinstance(train_data, pd.DataFrame)):
      self.train_data = train_data.values
    else:
      self.train_data = train_data
    
    if (isinstance(test_data, pd.DataFrame)):  
      self.test_data_raw = test_data.values
    else:
      self.test_data_raw = test_data
    
    self.backcast = backcast
    self.forecast = forecast
    self.batch_size = batch_size

  def setup(self, stage:str=None):      
      
    # Create test data by concatenating last `backcast` samples from 
    # train_data and first `forecast` samples from test_data
      
    test_data_sequences = []      
    for train_row, test_row in zip(self.train_data, self.test_data_raw):
      train_row = train_row[~np.isnan(train_row)]
      sequence = np.concatenate((train_row[-self.backcast:], test_row[:self.forecast]))
      if (sequence.shape[0] == self.backcast + self.forecast):
        test_data_sequences.append(sequence)
      
    self.test_data = np.array(test_data_sequences)     
    self.test_dataset = TimeSeriesCollectionDataset(self.test_data, self.backcast, self.forecast)  
    
  def test_dataloader(self):
    return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle = False, num_workers=0)



class TimeSeriesDataset(Dataset):
    def __init__(self, data, backcast_length, forecast_length):
      """A simple PyTorch Dataset that takes a time series as input and returns a single sample of
      the time series. Used for training a simple time series model.

      Parameters
      ----------
          data (numpy.ndarray): The univariate time series data.
          backcast_length (int ): The length of the historical data.
          forecast_length (int): The length of the future data to predict.
      """
      self.data = data
      self.backcast_length = backcast_length
      self.forecast_length = forecast_length
      self.total_length = backcast_length + forecast_length

    def __len__(self):
      return len(self.data) - self.total_length + 1

    def __getitem__(self, index):
      start_idx = index
      end_idx = index + self.total_length
      x = self.data[start_idx:end_idx - self.forecast_length]
      y = self.data[start_idx + self.backcast_length:end_idx]
      return torch.FloatTensor(x), torch.FloatTensor(y)

class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, data, batch_size, backcast, forecast):
      """The TimeSeriesDataModule class is a PyTorch Lightning DataModule that takes a time series
      as input and returns batches of samples of the time series. Used for training a time series
      model.

      Parameters
      ----------
          data (numpy.ndarray): The univariate time series data.
          batch_size (int): The batch size.
          backcast (int): The length of the historical data.
          forecast (int): The length of the future data to predict.
      """
      super().__init__()
      self.data = data
      self.batch_size = batch_size
      self.backcast = backcast
      self.forecast = forecast

    def setup(self, stage=None):
        dataset = TimeSeriesDataset(self.data, self.backcast, self.forecast)
        if stage == 'fit' or stage is None:
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

class ForecastingDataset(Dataset):
    def __init__(self, historical_data):
      """The ForecastingDataset class is a PyTorch Dataset that takes a historical
      time series as input and returns a single sample of the time series. Used for
      inferencing, predicting, for a time series model.

      Parameters
      ----------
          historical_data (pytorch.tensor): A single time series of historical data.
      """
      super().__init__()
      self.historical_data = historical_data

    def __len__(self):
      return len(self.historical_data)

    def __getitem__(self, idx):
      return self.historical_data[idx]

