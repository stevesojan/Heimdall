# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 10:48:34 2024

@author: Steve
"""

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

df = pd.read_csv(os.path.join('Stocks','aal.us.txt'),delimiter = ',', usecols = ['Date','Open','High','Low','Close','Volume'])
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
"""
TO ACCESS THE LAST ROW
print(df['Date'].iloc[-1])

print(df['Date'].iloc[988])

"""

#now we break the data to test data and training data
#taking mid-price values
high_price = df['High'].loc[:].to_numpy()
low_price = df['Low'].loc[:].to_numpy()
mid_price = (high_price +low_price)/2.0

#of the 989 entries we have, we'll use the first 400 entries for training the data
#next 589 entries for testing the data 

train_data = mid_price[:401]
test_data = mid_price[401:]






