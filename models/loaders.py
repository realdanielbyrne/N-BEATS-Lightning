from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl

class TimeSeriesDataset(Dataset):
  def __init__(self, data, backcast, forecast):
      super(TimeSeriesDataset, self).__init__()
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
               backcast=70, 
               forecast=14, 
               batch_size=1024, 
               split_ratio=0.8,
               debug = False):
    
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
            
      self.train_dataset = TimeSeriesDataset(self.train_data, self.backcast, self.forecast)
      self.val_dataset = TimeSeriesDataset(self.val_data, self.backcast, self.forecast)    
    
  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle = True)

  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle = False)
   
class TimeSeriesCollectionTestModule(pl.LightningDataModule):
  def __init__( self, 
                train_data,
                test_data,
                backcast=70, 
                forecast=14, 
                batch_size=512
               ):
    
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
      self.test_dataset = TimeSeriesDataset(self.test_data, self.backcast, self.forecast)  
      
  def test_dataloader(self):
    return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle = False, num_workers=0)
