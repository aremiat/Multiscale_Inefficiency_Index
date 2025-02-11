import numpy as np
import math
import matplotlib.pyplot as plt

#Import the data
import pandas as pd
#!pip install openpyxl (if necessary)
data_AIR = pd.read_excel('AIR_Prices.xlsx', engine='openpyxl', header=1) #header=1 indicates the right lign where to find the headers
print(data_AIR.head())

#Create a column for the date only, for the time only, and one for the price return
data_AIR['Date only'] = data_AIR["Dates"].dt.strftime("%Y-%m-%d")
data_AIR['Hour_Minute'] = data_AIR["Dates"].dt.strftime("%H:%M")
data_AIR['Return'] = data_AIR['Close'].pct_change()
data_AIR : pd.DataFrame = data_AIR.iloc[1:,:]
print(data_AIR.head())

# Group the data by day (excluding the first line, which is a return between two distinct days)
data_AIR_group = data_AIR.groupby(["Date only"])
#Calculate the standard deviation of the returns for each day as well as the number of observations used
daily_volatility = data_AIR_group["Return"].apply(lambda x:x.std())
daily_count = data_AIR_group["Return"].apply(lambda x:x.count())
#Create a dataframe for the dates and the volatilities
daily_stats_AIR_df = pd.DataFrame({"Date":sorted(list(set(data_AIR["Date only"]))),
                                   "Vol":list(daily_volatility),
                                   "Count":list(daily_count)})
print(daily_stats_AIR_df.head())

daily_stats_AIR_df['1-mn vol'] = daily_stats_AIR_df["Vol"] * np.sqrt(252*512)
print(daily_stats_AIR_df.head())

#Create a new dataframe for the volatilities
volatility_df = pd.DataFrame()

for interval in [2, 3, 30]:  # From 2 to 30 minutes
    
    data_AIR[f'Return {interval}'] = data_AIR['Close'].pct_change(interval)
    # Exclude first observation
    data_AIR : pd.DataFrame = data_AIR.iloc[interval:,:]
    # Calculate the volatility
    daily_stats_AIR_df[f"Vol {interval}"] = data_AIR_group[f'Return {interval}'].apply(lambda x:x.std())
    # Annualize
    daily_stats_AIR_df['1-mn vol'] = daily_stats_AIR_df[f"Vol {interval}"] * np.sqrt(252*512)
    # Number of observations
    ...

# Put the results in daily_stats_AIR_df
...

# Print the result
print(daily_stats_AIR_df.head())

# Select the columns to be displayed
volatility_columns = ['1-mn vol', '2-mn vol', '5-mn vol', '15-mn vol']

# Plot them
plt.title('Time evolution of realized volatility for AIR FP')
...

# Choose a specific day and filter the corresponding data
daily_stats_AIR_df.index = pd.to_datetime(daily_stats_AIR_df.index) # Otherwise not recognized as a date
specific_day = '2022-12-15' # You can try for other dates and will not always get the same shape.
day_data = ...

# Define the time intervals (in minutes) for which we want to plot volatility
# time_intervals = ???

# Extract volatility values for each time interval
...

# Plot volatilities as a function of time scale
...

# Group by 'Hour_Minute' and calculate the standard deviation of returns for each time of day
volatility_by_time = data_AIR.groupby("Hour_Minute")["Return"].std()

volatility_by_time.plot()
plt.title('Vol of 1-mn returns depending on time')
plt.show()

# High-low range
# data_AIR['Log High Low'] = ???
print(data_AIR.head())

# Group the data by day
# Calculate the first two moments of the range
...

# Deduce Parkinson volatilities (with annualization)
...

# Put the results in daily_stats_AIR_df
...

# Compare in a graph realized vol and Parkinson vol
volatility_columns = ['1-mn vol', 'Parkinson1', 'Parkinson2']
plt.title('Time evolution of realized/Parkinson volatility for AIR FP')
...

# Create a dataframe aggregating the OHLC prices in data_AIR at a daily scale
# daily_data = ???

# Add close-to-close absolute arithmetic return (low scale realized vol)
...

# Add low-scale Parkinson volatilities
...

# Annualize
...

# Display
plt.title('Time evolution of low-scale realized/Parkinson volatility for AIR FP')
...

# Some descriptive statistics of the three series
# stats = daily_data[volatility_columns].describe()
print(stats)

# Histogram (alpha drives the transparency)
...

# Scatter plot
import seaborn as sns
...

df = daily_stats_AIR_df
print(daily_stats_AIR_df.head())

df["LogVol"] = np.log(df["1-mn vol"])
print(df.head())
print(len(df))

def compute_variance(log_vols: list, tau: int = 1):
    temp = 0
    for i in range(1,len(log_vols)-tau):
        temp += (log_vols[i+tau] - log_vols[i])**2
    return temp/(len(log_vols)-tau)

variance = compute_variance(df["LogVol"].tolist())
print(variance)

var_dict = []
taus = []
for tau in range(1, 10):
    var_tau = compute_variance(df["LogVol"].tolist(), tau)
    var_dict.append(var_tau)
    taus.append(tau)
print(var_dict)

x = np.array(np.log(taus)).reshape(-1, 1)
y = np.array(np.log(var_dict))

import numpy as np
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x, y)

print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_[0]}")

H = model.coef_[0]/2
print(H)

plt.plot(x, y)
plt.show()

