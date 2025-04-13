# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 10:48:34 2024

@author: Steve
"""
#%%
#loading data from kaggle
from pandas_datareader import data
import matplotlib.pyplot as plt
import datetime as dt
import urllib.request, json


import pandas as pd
import numpy as np

import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv(os.path.join(r"path\to\aal.us.txt"),delimiter = ',', usecols = ['Date','Open','High','Low','Close','Volume'])
print('Loaded data from Kaggle successfully')
#pd.read_csv directly loads data from kaggle into a pandas dataframe

df = df.sort_values('Date')
print(df.head())

#visualising the data

plt.figure(figsize = (18,9))
plt.plot(range(df.shape[0]),(df['High']+df['Low'])/2.0)
plt.xticks(range(0,df.shape[0],19),df['Date'].loc[::19],rotation=45)
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Mid-Price', fontsize = 18)
plt.show()

plt.plot(range(df.shape[0]),(df['High']+df['Low'])/2.0)
plt.xticks(range(0,df.shape[0],99),df['Date'].loc[::99],rotation =45)
plt.xlabel('Date', fontsize = 19)
plt.ylabel('Mid-Price', fontsize =19)
plt.show()

print(df.shape[0])

#.iloc allows to access rows and columns by index position e.g df.iloc[-1 gives last]
"""[]
TO ACCESS THE LAST ROW
print(df['Date'].iloc[-1])

print(df['Date'].iloc[988])

"""
#%%
#now we break the data to test data and training data
#taking mid-price values
high_price = df['High'].loc[:].to_numpy()
low_price = df['Low'].loc[:].to_numpy()
mid_price = (high_price +low_price)/2.0

#of the 989 entries we have, we'll use the first 400 entries for training the data
#next 589 entries for testing the data 

train_data = mid_price[:401]
test_data = mid_price[401:]

print(train_data)
print('Test data:')
print(test_data)


#%%

#total data points we have is 988
#for windowed normalization we will keep window size = 198
#4 windows, so we will lose 4 data pts

scaler = MinMaxScaler()
#significance of (-1,1) in .reshape
# 1 in (-1,1) signifies the number of columns we want data to fit in
#-1 conveys numpy to automatically assign the number of rows  
#https://chatgpt.com/share/672c56bd-5004-800c-8d53-46dbc7b19119
train_data = train_data.reshape(-1,1)
test_data = test_data.reshape(-1,1)


print(test_data)

#%%
smoothing_window_size = 198
print(train_data)

print(test_data)
# Apply MinMax scaling in fixed-size windows
for di in range(0, 792, smoothing_window_size):
    scaler.fit(train_data[di:di + smoothing_window_size, :])  
    train_data[di:di + smoothing_window_size, :] = scaler.transform(train_data[di:di + smoothing_window_size, :])  

# Handle any remaining data dynamically
if train_data.shape[0] > smoothing_window_size:
    scaler.fit(train_data[smoothing_window_size:, :])  # Fit remaining portion
    train_data[smoothing_window_size:, :] = scaler.transform(train_data[smoothing_window_size:, :])
#4 windows - 0-198, 199-396, 397-594, 595-792    
#now remaining data which is out of range, i.e after 792

#in di:di+smoothing_window_size: This part selects a slice of rows from the array.
#di is the starting index (inclusive).
#di + smoothing_window_size is the ending index (exclusive).
#Example: If di = 0 and smoothing_window_size = 3, this selects rows 0, 1, and 2.
#,: This separates the row selection from the column selection.
#: (after the comma): This means all columns of the selected rows should be included.
'''
train_data[di:di+smoothing_window_size, :] selects a block of data:

Rows: From di to di + smoothing_window_size (exclusive).
Columns: All available columns.

scaler.fit(train_data[di:di+smoothing_window_size,:])
train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])
'''
train_data = train_data.reshape(-1)
test_data = scaler.transform(test_data).reshape(-1)
#This line transforms the selected segment using the fitted scaler and replaces it back in the train_data array.
print(train_data)
#now performing exponenetial moving average smoothing
#so data will have a smoother curve than original ragged data
#%%
EMA = 0.0
gamma = 0.1 #smoothing factor
for pt in range(len(train_data)):
    EMA = gamma*(train_data[pt]) + (1-gamma)*EMA
    train_data[pt] = EMA

all_mid_data = np.concatenate([train_data,test_data],axis = 0)
#np.concatenate([train_data, test_data], axis=0) will concatenate these two arrays along the 0th axis, which generally means stacking them vertically (for 2D arrays) or appending them end-to-end (for 1D arrays).



#initializing variables for SMA 

window_size = 23
N = train_data.size
std_avg_predictions = []
std_avg_date = []
mse_error = []

#pred_idx short for predicted value(using mean of values in previous window) at that index

#getting the date
for pred_idx in range(window_size,N):
    
    if pred_idx>=N:
        k = df.loc[pred_idx-1, 'Date']
        date = dt.datetime.strptime(k,'%Y-%m-%d').date() + dt.timedelta(days = 1)
    else:
        date = df.loc[pred_idx,'Date']
        
    #now this is where we actually get the mean
    
    std_avg_predictions.append(np.mean(train_data[pred_idx-window_size:pred_idx]))
    mse_error.append((std_avg_predictions[-1]-train_data[pred_idx])**2)
    std_avg_date.append(date)
    
print(f"MSE Error for Standard Averaging is: , {0.5*np.mean(mse_error):.5f}")
    








