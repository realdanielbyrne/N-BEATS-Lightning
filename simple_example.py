#%%
from models.nbeats import *
from models.loaders import *
import torch
from torch.utils.data import DataLoader
import pandas as pd

import lightning.pytorch as pl
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
tqdm.pandas()

#%%
# Load data
milk = pd.read_csv('data/milk.csv', index_col=0)
print(milk.head())
milkval = milk.values.flatten()  # just keep np array here for simplicity.

#%%
# Define hyperparameters
forecast_length = 6
backcast_length = 5 * forecast_length
batch_size = 100  

#%%

# Create dataloader
dm = TimeSeriesDataModule(
  data=milkval,
  batch_size=batch_size,
  backcast=backcast_length,
  forecast=forecast_length)

# A Generic N-Beats Model
generic_milkmodel = NBeatsNet(
  backcast = backcast_length,
  forecast = forecast_length, 
  generic_architecture = True,
  n_blocks_per_stack = 1,
  n_stacks = 5,
  share_weights = False,
  #g_width = 768,
  activate_g = False,
  sum_losses = False,
  loss="mase")

# An Interpretable N-Beats Model
interpretable_milkmodel = NBeatsNet(
  backcast = backcast_length,
  forecast = forecast_length, 
  generic_architecture = False,
  n_blocks_per_stack = 3,
  n_stacks = 2,  
  share_weights = True,
  sum_losses=False,
  loss="MSELoss")


#%%
# Train the generic model
generic_trainer =  pl.Trainer(
  accelerator='auto'
  ,max_epochs=100 
)

generic_trainer.fit(generic_milkmodel, datamodule=dm)
generic_trainer.validate(generic_milkmodel, datamodule=dm)

#%%
# Train the interpretable model
interpretable_trainer =  pl.Trainer(
  accelerator='auto'
  ,max_epochs=100 
)

interpretable_trainer.fit(interpretable_milkmodel, datamodule=dm)
interpretable_trainer.validate(interpretable_milkmodel, datamodule=dm)


# %%
# Predict
historical_data = torch.tensor(milkval[-backcast_length - forecast_length:- forecast_length], dtype=torch.float32).view(1, -1)

# Create forecasting dataloader
predict_dataset = ForecastingDataset(historical_data)
predict_dataloader = DataLoader(predict_dataset, batch_size=1)

# The model uses a ptl trainer to predict
generic_predictions = generic_trainer.predict(generic_milkmodel, dataloaders=predict_dataloader)
interpretable_predictions = interpretable_trainer.predict(interpretable_milkmodel, dataloaders=predict_dataloader)

# The trainer returns a list of dictionaries, one for each dataloader passed to trainer.predict.  We only want the first and only one.
generic_preds = generic_predictions[0].squeeze()
interpretable_preds = interpretable_predictions[0].squeeze()


#%%
# Plot a subset of the data
predicted_months = pd.date_range(start="1975-07", periods=forecast_length, freq='M').strftime('%Y-%m')
generic_df = pd.DataFrame(generic_preds, index=predicted_months)
interpretable_df = pd.DataFrame(interpretable_preds, index=predicted_months)
milk_subset = milk[milk.index > '1971-01']


plt.figure(figsize=(12,6))
plt.plot(milk_subset.index, milk_subset['milk_production_pounds'], marker='o', linestyle='-', color='b')
plt.plot(generic_df.index, generic_df, marker='o', linestyle='-', color='g')
plt.plot(interpretable_df.index, interpretable_df, marker='o', linestyle='-', color='indigo')
plt.title('Milk Production Over Time (1971-1975)', fontsize=14)
plt.xlabel('Time (Year-Month)', fontsize=12)
plt.ylabel('Milk Production (lbs)', fontsize=12)
plt.grid(True)
plt.xticks([])
plt.tight_layout()
plt.show()


