#%%
# Import necessary libraries
from nbeats_lightning.nbeats import *
from nbeats_lightning.loaders import *
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from tqdm.notebook import tqdm
tqdm.pandas()


#%%
# Load the milk.csv dataset
milk = pd.read_csv('data/milk.csv', index_col=0)
print(milk.head())
milkval = milk.values.flatten()  # just keep np array here for simplicity.


# Display the first few rows of the dataset
milk.head()

"""The dataset consists of two columns:

month: The month during which the data was collected, starting from January 1962.
milk_production_pounds: The amount of milk produced in pounds.
Now, let's create some visualizations to better understand this dataset. We can generate:

A Time Series Plot to observe the trend and seasonality in milk production over time.
A Seasonal Decomposition to break down the time series into its trend, seasonal, and residual components.
A Box Plot to observe the distribution of milk production values.
"""

import matplotlib.pyplot as plt

# Convert the 'month' column to datetime format
milk['month'] = pd.to_datetime(milk.index)

# Create Time Series Plot
plt.figure(figsize=(10, 6))
plt.plot(milk['month'], milk['milk_production_pounds'], label='Milk Production')
plt.title('Time Series Plot of Milk Production')
plt.xlabel('Time')
plt.ylabel('Milk Production (pounds)')
plt.legend()
plt.grid(True)
plt.show()

#%%

"""
The Time Series Plot above illustrates the trend and seasonality in milk production over time.
It's apparent that there's a recurring pattern every year, which suggests a strong seasonal
component. Additionally, there seems to be a general upward trend in milk production over the years.
Next, let's move on to the Seasonal Decomposition to decompose the time series into its trend,
seasonal, and residual components. This will provide a clearer picture of the underlying patterns
in the data.
"""
milk.set_index('month', inplace=True)

# Perform seasonal decomposition
decomposition = seasonal_decompose(milk['milk_production_pounds'], model='multiplicative')

# Plot the seasonal decomposition again with the corrected attribute name
plt.figure(figsize=(10, 8))
plt.subplot(411)
plt.plot(decomposition.trend, label='Trend')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(decomposition.seasonal, label='Seasonal')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(decomposition.resid, label='Residual')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(milk['milk_production_pounds'], label='Observed')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


#%%
"""The Seasonal Decomposition plot breaks down the time series into three components:

Trend: Shows a general upward trend in milk production over the years.
Seasonal: Illustrates a clear seasonal pattern that repeats every year.
Residual: Contains the residual values after the trend and seasonal components have
been removed.
"""
# Create Box Plot
plt.figure(figsize=(10, 6))
plt.boxplot(milk['milk_production_pounds'], vert=False)
plt.title('Box Plot of Milk Production')
plt.xlabel('Milk Production (pounds)')
plt.yticks([])  # Hide the y-axis ticks as there's only one box
plt.grid(True)
plt.show()


# %%
"""Next we will create a PyTorch Lightning DataModule to load the data, and two 
  nbeats_lightning models, one generic, one interpretable, with which to train the data
  and then predict.  We will then compare the predictions of the two models against the
  actual data and to each other.
"""
#%%
# Define hyperparameters
forecast_length = 6
backcast_length = 3 * forecast_length
batch_size = 64

# Create dataloader
dm = TimeSeriesDataModule(
  data=milkval,
  batch_size=batch_size,
  backcast=backcast_length,
  forecast=forecast_length)

# A Generic 512 Units wide N-Beats Model.  5 stacks, 1 block per stack 
generic_milkmodel_5_1_512 = NBeatsNet(
  backcast = backcast_length,
  forecast = forecast_length, 
  generic_architecture = True,
  n_blocks_per_stack = 1,
  n_stacks = 5,
  thetas_dim=4,
  g_width = 128,
  share_weights = False,
  active_g = False,
  sum_losses = False)

# same model as above, but with g-activation 
generic_milkmodel_actg_5_1_512 = NBeatsNet(
  backcast = backcast_length,
  forecast = forecast_length, 
  generic_architecture = True,
  n_blocks_per_stack = 1,
  n_stacks = 5,
  thetas_dim = 4,
  g_width = 128,
  share_weights = False,
  active_g = True,
  sum_losses = False)

# An Interpretable N-Beats Model, 2 stacks, 1 trend(256), 1 seasonality(2048), 3 blocks per stack
interpretable_milkmodel = NBeatsNet(
  backcast = backcast_length,
  forecast = forecast_length, 
  generic_architecture = False,
  n_blocks_per_stack = 3,
  n_stacks = 2,  
  share_weights = True,  
  sum_losses=False
  )



#%%
# Train the generic model
generic_trainer =  pl.Trainer(
  accelerator='auto'
  ,max_epochs=1000,
)

generic_trainer.fit(generic_milkmodel_5_1_512, datamodule=dm)
generic_trainer.save_checkpoint('generic_milkmodel_5_1_512.ckpt')
generic_trainer.validate(generic_milkmodel_5_1_512, datamodule=dm)

#%%
# Train the generic model with g_activation
generic_act_trainer =  pl.Trainer(
  accelerator='auto'
  ,max_epochs=300
)

generic_act_trainer.fit(generic_milkmodel_actg_5_1_512, datamodule=dm)
generic_act_trainer.validate(generic_milkmodel_actg_5_1_512, datamodule=dm)

#%%
# Train the interpretable model
interpretable_trainer =  pl.Trainer(
  accelerator='auto'
  ,max_epochs=200 
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
predicted_months = pd.date_range(start="1975-06", periods=forecast_length, freq='M').strftime('%Y-%m')
generic_df = pd.DataFrame(generic_preds, index=predicted_months)

interpretable_df = pd.DataFrame(interpretable_preds, index=predicted_months)
milk_subset = milk[milk.index > '1971-01']


plt.figure(figsize=(12,6))
#plt.plot(milk_subset.index, milk_subset['milk_production_pounds'], marker='o', linestyle='-', color='b')
plt.plot(generic_df.index, generic_df, marker='o', linestyle='-', color='g')
plt.plot(interpretable_df.index, interpretable_df, marker='o', linestyle='-', color='indigo')
plt.title('Milk Production Over Time (1971-1975)', fontsize=14)
plt.xlabel('Time (Year-Month)', fontsize=12)
plt.ylabel('Milk Production (lbs)', fontsize=12)
plt.grid(True)
plt.xticks([])
plt.tight_layout()
plt.show()




# %%
