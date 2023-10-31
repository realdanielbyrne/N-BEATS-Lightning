import pandas as pd

from loaders import *
m4_train_filepath ='Train/Monthly-train.csv'
m4_test_filepath ='Test/Monthly-test.csv'
m4_info_filepath ='M4-info.csv'


class M4Info:
  def __init__(self):
    self.m4_info_filepath = m4_info_filepath
    self.info_df = pd.read_csv(m4_info_filepath)
  
  def get_category_indicies(self, period, category):
    if category is None:
      category = 'All'
      
    if category != 'All':
      mask = (self.info_df.index.str.startswith(period[0])) & (self.info_df.category == category)
      category_subset = self.info_df[mask]
      indicies = category_subset[mask].index
    else:
      indicies = None
    return indicies
  
  def get_period_horizon(self, period):
    data_id_info = self.info_df.loc[period[0] + f"{1}"]
    horizon = data_id_info.Horizon
    return horizon
  
  def get_period_frequency(self, period):
    data_id_info = self.info_df.loc[period[0] + f"{1}"]
    frequency = data_id_info.Frequency
    return frequency
    

class M4Dataset(M4Info):
  def __init__(self, period, category = None):
    self.period = period
    self.category = category
    m4_train_filepath = f"Train/{period}-train.csv"
    m4_test_filepath  = f"Test/{period}-test.csv"
    self.m4_train_filepath  = m4_train_filepath
    self.m4_test_filepath = m4_test_filepath
    self.forecast_length = self.get_period_horizon(self.period)
    self.train_df = self._load_train_data()
    self.test_df = self._load_test_data()
  
  def _load_train_data(self):
    self.train_df = pd.read_csv(self.m4_train_filepath)
          
    if self.category != 'All' or self.category is not None:
      indicies = self.get_category_indicies(self.category)
      self.train_df = self.train_data[indicies]

    return self.train_df
  
  def _load_test_data(self):    
    self.test_df = pd.read_csv(self.m4_test_filepath)
      
    if self.category != 'All' or self.category is not None:
      indicies = self.get_category_indicies(self.category)
      self.test_data = self.test_data[indicies]
  
    return self.test_df
  
  def get_train_dm(self, batch_size:int=1024, split_ratio:float=0.9, horiz_mult:int = 5, imput_short_ts:bool=True):
  
    forecast_length=self.forecast_length
    backcast_length=forecast_length * horiz_mult  
    
    dm = TimeSeriesImputedCollectionDataModule(
      data=self.train_df, 
      forecast_length=forecast_length,
      backcast_length=backcast_length,
      batch_size=batch_size, 
      split_ratio=split_ratio,
      imput_short_ts=imput_short_ts
    ) 
      
    return dm
  
  def get_test_dm(self, batch_size:int=1024, horiz_mult:int = 5, imput_short_ts:bool=True):
    forecast_length=self.forecast_length
    backcast_length=forecast_length * horiz_mult  


    test_dm = TimeSeriesImputedCollectionTestModule(
      test_data=self.test_df, 
      train_data=self.train_df,
      backcast_length=backcast_length, 
      forecast_length=forecast_length, 
      batch_size=batch_size,
      imput_short_ts=imput_short_ts
    )
    
    return test_dm