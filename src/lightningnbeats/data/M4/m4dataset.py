import pandas as pd
import numpy as np
import os

from ..benchmark_dataset import BenchmarkDataset

# Naive2 baseline values from the M4 competition (Makridakis et al., 2020).
# Used to compute OWA = 0.5 * (sMAPE/sMAPE_Naive2 + MASE/MASE_Naive2).
NAIVE2_SMAPE = {
    "Yearly": 16.342, "Quarterly": 11.012, "Monthly": 14.427,
    "Weekly": 9.161,  "Daily": 3.045,      "Hourly": 18.383,
}
NAIVE2_MASE = {
    "Yearly": 3.974, "Quarterly": 1.371, "Monthly": 1.063,
    "Weekly": 2.777, "Daily": 3.278,     "Hourly": 2.395,
}


class M4Dataset(BenchmarkDataset):
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

    self._load_train_data()
    self._load_test_data()

  @property
  def name(self):
    return f"M4-{self.period}"

  supports_owa = True

  def compute_owa(self, smape, mase):
    """OWA (Overall Weighted Average) as defined in the M4 competition.

    OWA = 0.5 * (sMAPE / sMAPE_Naive2 + MASE / MASE_Naive2)
    """
    return 0.5 * (smape / NAIVE2_SMAPE[self.period] + mase / NAIVE2_MASE[self.period])

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
    self.train_data = self.transform_array(self.train_df.values)


  def _load_test_data(self):
    self.test_df = pd.read_csv(self.m4_test_filepath, index_col=0)

    if self.category != 'All':
      if self.indicies is not None:
        self.test_df = self.test_df[self.indicies]
    self.test_data = self.transform_array(self.test_df.values)


  def transform_array(self, arr):
      # Calculate the maximum valid length of the time series
      valid_lengths = np.array([np.max(np.where(~np.isnan(row))) + 1 for row in arr])
      max_length = np.max(valid_lengths)

      # Initialize a new array with nans
      new_arr = np.full((max_length, len(arr)), np.nan)

      # Populate the new array with the valid values from the input array
      for i, row in enumerate(arr):
          # Count the number of valid (non-nan) entries in the current row
          valid_entries = valid_lengths[i]
          # Fill the corresponding column in the new array with the valid values, aligned at the end
          new_arr[-valid_entries:, i] = row[:valid_entries]

      return pd.DataFrame(new_arr)
