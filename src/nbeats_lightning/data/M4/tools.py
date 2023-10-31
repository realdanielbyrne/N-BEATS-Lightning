import pandas as pd


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
  def __init__(self, period):
    self.period = period
    m4_train_filepath = f"Train/{period}-train.csv"
    m4_test_filepath  = f"Test/{period}-test.csv"
    self.m4_train_filepath  = m4_train_filepath
    self.m4_test_filepath = m4_test_filepath
  
  def load_train(self):
    self.monthly_train = pd.read_csv(self.m4_train_filepath)
  
  def load_test(self):
    self.monthly_test = pd.read_csv(self.m4_test_filepath)
    
  def train_data(self, category=None):
    if self.monthly_train is None:
      self.load_train()
      
    if category is None:
      category = 'All'
    
    if category != 'All':
      indicies = self.get_category_indicies(category)
      train_data = self.train_data[indicies]
    else:
      train_data = self.train_data

    return train_data
  
  def test_data(self, category=None):
    if self.monthly_test is None:
      self.load_test()
      
    if category is None:
      category = 'All'
    
    if category != 'All':
      indicies = self.get_category_indicies(category)
      test_data = self.test_data[indicies]
    else:
      test_data = self.test_data
    return test_data