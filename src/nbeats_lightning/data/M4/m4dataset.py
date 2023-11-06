import pandas as pd
import os

class M4Dataset:
  def __init__(self, period, category = None):
    
    if category is None:
      category = 'All'
      
    self.category = category      
    self.period = period
    
    #get the path to the data directory
    self.info_dir = os.path.dirname(os.path.abspath(__file__))
    self.m4_info_filepath = os.path.join(self.info_dir, "M4-info.csv")

    self.test_dir = os.path.join(self.info_dir, "Test")
    self.train_dir = os.path.join(self.info_dir, "Train")
    self.m4_train_filepath = os.path.join(self.train_dir, f"{period}-train.csv")
    self.m4_test_filepath = os.path.join(self.test_dir, f"{period}-test.csv")
        
    self.info_df = pd.read_csv(self.m4_info_filepath,index_col=0)   
    
    data_id_info = self.info_df.loc[period[0] + f"{1}"]
    self.horizon = self.forecast_length = data_id_info.Horizon
    self.frequency = data_id_info.Frequency
    self.indicies = self._get_category_indicies()
    
    
    self.train_df = self._load_train_data()
    self.test_df = self._load_test_data()
    
  def _get_category_indicies(self):
      
    if self.category != 'All':
      mask = (self.info_df.index.str.startswith(self.period[0])) & (self.info_df.category == self.category)
      category_subset = self.info_df[mask]
      indicies = category_subset[mask].index
    else:
      indicies = None
    return indicies

  def _load_train_data(self):
    self.train_df = pd.read_csv(self.m4_train_filepath, index_col=0)
          
    if self.category != 'All':
      if self.indicies is not None:
        self.train_df = self.train_df[self.indicies]

    return self.train_df
  
  def _load_test_data(self):    
    self.test_df = pd.read_csv(self.m4_test_filepath, index_col=0)
      
    if self.category != 'All':
      if self.indicies is not None:
        self.test_data = self.test_data[self.indicies]
  
    return self.test_df
  
