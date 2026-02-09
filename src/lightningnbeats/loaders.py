from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from torch.utils.data import random_split


import numpy as np
import torch
from torch.utils.data import Dataset

class RowCollectionTimeSeriesDataset(Dataset):
  def __init__(self, 
               data, 
               backcast_length, 
               forecast_length):
    """
    The RowCollectionTimeSeriesDataset class is a PyTorch Dataset that takes a 
    collection of time series as input and returns a single sample of the time 
    series. Used for training a time series model whose input is a collection 
    of time series organized such that rows represent individual time series and
    columns give the subsequent observations. Each timeset does not have to be the
    the same length. 

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
  
    super(RowCollectionTimeSeriesDataset, self).__init__()
    self.data = data
    self.backcast_length = backcast_length
    self.forecast_length = forecast_length
    self.items = []

    total_len = self.backcast_length + self.forecast_length
    for row in range(self.data.shape[0]):
      col_starts = np.arange(0, self.data.shape[1] - total_len + 1)
      seqs = [self.data[row, start:start + total_len] for start in col_starts]
      valid_indices = [i for i, seq in enumerate(seqs) if not np.isnan(seq).any()]
      self.items.extend([(row, col_starts[i]) for i in valid_indices])
          
                
  def __len__(self):
    return len(self.items)
  
  def __getitem__(self, idx):
    row, col = self.items[idx]
    x = self.data[row, col:col+self.backcast_length]
    y = self.data[row, col+self.backcast_length:col+self.backcast_length+self.forecast_length]
    
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class RowCollectionTimeSeriesDataModule(pl.LightningDataModule):
  def __init__(self, 
                data, 
                backcast_length, 
                forecast_length, 
                batch_size=1024, 
                split_ratio=0.8,
                fill_short_ts=True):
    """The RowCollectionTimeSeriesDataModule class is a PyTorch Lightning DataModule
    is used for training a time series model with a dataset that is a collection of time series
    organized into rows where each row represents a time series, and each column represents
    subsequent observations.    

    Parameters
    ----------
        data (pd.Dataframe): 
          The univariate time series data. The data organization is assumed to be a 
          pandas dataframe with rows representing time series and columns representing time steps. 
        backcast_length (int, optional): 
          The length of the historical data.
        forecast_length (int, optional): 
          The length of the future data to predict.
        batch_size (int, optional): 
          The batch size. Defaults to 1024.
        split_ratio (float, optional): 
          The ratio of the data to use for training/validation.
        fill_short_ts (bool, optional): 
          If True, short training sequences are filled. Defaults to True.
    """      
    super(RowCollectionTimeSeriesDataModule, self).__init__()
    
    self.data = data    
    self.backcast_length = backcast_length
    self.forecast_length = forecast_length
    self.batch_size = batch_size
    self.split_ratio = split_ratio
    self.fill_short_ts = fill_short_ts

  def setup(self, stage:str=None):
    shuffled = self.data.sample(frac=1, axis = 0).reset_index(drop=True)        
    total_len = self.backcast_length + self.forecast_length
    
    # Split the original data into training and validation sets (split by rows, not columns)
    split_idx = int(len(shuffled) * self.split_ratio)
    self.train_data = shuffled.iloc[:split_idx].values
    self.val_data = shuffled.iloc[split_idx:].values

        
    if self.fill_short_ts:
      for dataset in [self.train_data, self.val_data]:
        for row in range(dataset.shape[0]):
          nan_indices = np.isnan(dataset[row])
          row_length = np.sum(~nan_indices)
          elements_to_add = total_len - row_length
          if elements_to_add > 0:            
            fill_val = 0.0 
            
            imputed_values = np.full(elements_to_add, fill_val)
            new_row = np.concatenate([imputed_values, dataset[row][~nan_indices]])
            nan_padding = np.full(dataset.shape[1] - len(new_row), np.nan)
            dataset[row] = np.concatenate([new_row, nan_padding])

    self.train_dataset = RowCollectionTimeSeriesDataset(self.train_data, self.backcast_length, self.forecast_length)
    self.val_dataset = RowCollectionTimeSeriesDataset(self.val_data, self.backcast_length, self.forecast_length)

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

class RowCollectionTimeSeriesTestModule(pl.LightningDataModule):
  def __init__(self, 
                train_data,
                test_data,
                backcast_length, 
                forecast_length, 
                batch_size=1024,
                fill_short_ts:bool=True):
    
    """The RowCollectionTimeSeriesTestModule class is a PyTorch Lightning DataModule
    used for testing a time series model whose input is a collection of time series.
    The final `backcast` samples of each time series in `train_data` are concatenated
    with the first `forecast` samples of the corresponding time series in `test_data`. 
    If a time series in `train_data` is shorter than `backcast`, the missing values are
    imputed with the median of the training row. 
    
    Parameters
    ----------
      backcast_length (int, optional): 
        The length of the historical data.
      forecast_length (int, optional): 
        The length of the future data to predict.
      batch_size (int, optional): 
        The batch size. Defaults to 1024.
    """

    super(RowCollectionTimeSeriesTestModule, self).__init__()
    self.train_data = train_data    
    self.test_data_raw = test_data
    
    self.backcast_length = backcast_length
    self.forecast_length = forecast_length
    self.batch_size = batch_size
    self.fill_short_ts = fill_short_ts

  def setup(self, stage:str=None):      
    # Create test data by concatenating last `backcast` samples from 
    # train_data and first `forecast` samples from test_data
      
    test_data_sequences = []      
    total_len = self.backcast_length + self.forecast_length
    
    for train_row, test_row in zip(self.train_data.values, self.test_data_raw.values):
      train_row = train_row[~np.isnan(train_row)]
      sequence = np.concatenate((train_row[-self.backcast_length:], test_row[:self.forecast_length]))

      if self.fill_short_ts:
        nan_indices = np.isnan(sequence)
        seq_length = np.sum(~nan_indices)
        elements_to_add = total_len - seq_length      
        
        if (elements_to_add > 0):                    
          fill_val = 0.0 
          
          # Create an array of imputed values
          imputed_values = np.full(elements_to_add, fill_val)
                          
          # Insert the imputed values before the first backcast observation
          sequence = np.concatenate([imputed_values, sequence]) 
      
      if (sequence.shape[0] == self.backcast_length + self.forecast_length):
        test_data_sequences.append(sequence)
      
    self.test_data = np.array(test_data_sequences)  
    self.test_dataset = RowCollectionTimeSeriesDataset(self.test_data, self.backcast_length, self.forecast_length)  
    
  def test_dataloader(self):
    return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle = False, num_workers=0)

class TimeSeriesDataset(Dataset):
  def __init__(self, data, backcast_length, forecast_length):
    """A simple PyTorch Dataset that takes a time series as input and returns a single sample of
    the time series. Used for training a simple time series model.

    Parameters
    ----------
        data (numpy.ndarray): The univariate time series data.
        backcast_length (int): The length of the historical data.
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
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class TimeSeriesDataModule(pl.LightningDataModule):
  def __init__(self, data, batch_size, backcast_length, forecast_length):
    """
    The TimeSeriesDataModule class is a PyTorch Lightning DataModule that takes a univariate
    time series as input and returns batches of samples of the time series. 

    Parameters
    ----------
        data (numpy.ndarray): The univariate time series data.
        batch_size (int): The batch size.
        backcast_length (int): The length of the historical data.
        forecast_length (int): The length of the future data to predict.
    """
    super().__init__()
    self.data = data
    self.batch_size = batch_size
    self.backcast_length = backcast_length
    self.forecast_length = forecast_length

  def setup(self, stage=None):
    dataset = TimeSeriesDataset(self.data, self.backcast_length, self.forecast_length)   
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

class ColumnarTimeSeriesDataset(Dataset):
  def __init__(self, dataframe, backcast_length, forecast_length):
    self.backcast_length = backcast_length
    self.forecast_length = forecast_length
    self.min_length = backcast_length + forecast_length

    # Drop columns with insufficient data and convert to dictionary of NumPy arrays
    self.data_dict = {col: self.pad_series(dataframe[col].dropna().values) for col in dataframe.columns }

    # Precompute column indices and starting positions
    self.col_indices = [(col, idx) for col, series in self.data_dict.items() for idx in range(len(series) - self.min_length + 1)]
  
  def pad_series(self, series):
    valid_entries = len(series)
    if valid_entries < self.min_length:
      # Calculate the number of zeros to pad
      pad_size = self.min_length - valid_entries
      # Create a padding array of zeros
      padding = np.zeros(pad_size)
            
      # Pad the series with zeros at the beginning
      series = np.concatenate((padding, series), axis=0)
    return series
    
  def __len__(self):
    return len(self.col_indices)

  def __getitem__(self, idx):
    col, start_idx = self.col_indices[idx]
    series = self.data_dict[col]
    x = torch.from_numpy(series[start_idx:start_idx + self.backcast_length]).float()
    y = torch.from_numpy(series[start_idx + self.backcast_length:start_idx + self.min_length]).float()
    return x, y
    
class ColumnarCollectionTimeSeriesDataModule(pl.LightningDataModule):
  def __init__(self, 
               dataframe,
               backcast_length, 
               forecast_length, 
               batch_size=1024, 
               no_val = False):
    """
    The ColumnarCollectionTimeSeriesDataModule class is a PyTorch Datamodule that takes a 
    collection of time series as input and returns a single sample of the time 
    series. The input dataset is a collection of time series organized such that columns 
    represent individual time series and rows represent subsequent observations. This is how
    the Tourism dataset is organized.

    Args:
        dataframe (Pandas): Pandas dataframe with columns representing time series and rows representing time steps.
        backcast_length (int, optional):  Number of past observations to use for predictio. Defaults to 10.
        forecast_length (int, optional): Number of future observations to predict. Defaults to 4.
        batch_size (int, optional): The batch size. Defaults to 32.
    """
    super().__init__()
    self.backcast_length = backcast_length
    self.forecast_length = forecast_length
    self.batch_size = batch_size
    self.no_val = no_val
  
    self.total_length = backcast_length + forecast_length
    self.dataframe = dataframe

  def setup(self, stage=None):
    if self.no_val:
      train_data = self.dataframe
      val_data = pd.DataFrame()
    else:
      train_data = self.dataframe.iloc[:-self.forecast_length]
      val_data = self.dataframe.iloc[-self.forecast_length-self.backcast_length:]

    self.val_dataset = ColumnarTimeSeriesDataset(val_data, self.backcast_length, self.forecast_length)
    self.train_dataset = ColumnarTimeSeriesDataset(train_data, self.backcast_length, self.forecast_length)

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size=self.batch_size)

class ColumnarCollectionTimeSeriesTestDataModule(pl.LightningDataModule):
  def __init__(self, 
                train_data,
                test_data,
                backcast_length, 
                forecast_length, 
                batch_size=1024):
    """Takes two collections of time series organized into columns
    where each row represents a time step and each column represents
    an individual time series. The module combines the training data
    with the holdout data to create a single test dataset.
    
    Args:
        train_data (pd.Dataframe): The training data.
        test_data (pd.Dataframe): The holdout test data.
        backcast_length (_type_): The length of the historical data.
        forecast_length (_type_): The length of the future data to predict.
        batch_size (int, optional): The batch size. Defaults to 1024.
    """
    super(ColumnarCollectionTimeSeriesTestDataModule, self).__init__()
    
    if backcast_length > len(train_data):
      raise ValueError(f"backcast_length ({backcast_length}) cannot exceed training data length ({len(train_data)})")
    
    self.test_data = pd.concat([train_data.iloc[-backcast_length:], test_data]).reset_index(drop=True)
    self.backcast_length = backcast_length
    self.forecast_length = forecast_length
    self.batch_size = batch_size

  def setup(self, stage:str=None):
    self.test_dataset = ColumnarTimeSeriesDataset(self.test_data, self.backcast_length, self.forecast_length) 

  def test_dataloader(self):
    return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle = False)