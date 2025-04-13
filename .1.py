<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 16:06:48 2024

@author: Steve
"""

from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

import urllib.request, json
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


data_source = 'kaggle'
if data_source == 'alphavantage':
    #2 ways of getting data from alphvantage, 
    #one is saving data from alhpavantage website directly to csv file and using that file
    #second is retrieving data from alphavantage website and storing the data in a pandas dataframe
    
    
    
    #storing from alphavantage to csv file
    #4 main steps
    
    api_key = 'AD9C9HJUNS64N2H6'
    ticker = 'AAL'
    
    #in the url remember 4 importnt values:
    #function, e.g TIME_SERIES_DAILY, TIME_SERIES_INTRADAY
    #outputsize = full (gives data of all history)
    #symbol = AAL, HPQ (legitimate abbreviated name of stock we'll be using)
    #apikey  = AD9C9HJUNS64N2H6
    url_string = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&outputsize=full&symbol={ticker}&apikey={api_key}'
    
    '''
    by using placeholder %s for variables symbol and api_key in url_string, 
    the program becomes more general for the api to use for different users
    they need only change the value of the variables symbol and api_key 
    according to what they are working with respectively
    '''
    file_to_save = 'stockmkt_data_%s.csv'%ticker
    
    
    #now the other approach of retrieving the data from the alphavantage link and storing directly in a pandas dataframe
    if not os.path.exists(file_to_save):
        #with is used for better program flow because it closes the resources used right after processing them
        with urllib.request.urlopen(url_string) as url:
            data = json.loads([url.read().decode()])
            data = data['TIME_SERIES_DAILY']
            #json.loads loads the data  of the website into a python dictionary
            #in the next line (of json.loads) we assign the values of the key TIME_SERIES_DAILY Tto the same variable data
            df = pd.Dataframe(columns=['Date','Open','High','Low','Close'])
            for k, v in data.items:
                date = dt.datetime.strptime(k,'%Y-%M-%D')
                data_row = [date.date(), float[v,'1. Open'], float[v,'2. High'], float[v,'3.Low'],float[v,'4.Close']]
                df.loc[-1:] = data_row
                df.index = df.index + 1
        df.to_csv(file_to_save)
        print('Data saved to: %s',(file_to_save))
    else:
        
        df = pd.read_csv(file_to_save)
        print('File Exists, Loading data from csv..Done')
       #now loading from kaggle file
else:
    df = pd.read_csv(os.path.join('Stocks','aal.us.txt'), delimiter = ',', usecols = ['Date','Open','High','Low','Close'])
    print('loaded from Kaggle')
            
        
=======
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 16:06:48 2024

@author: Steve
"""

from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

import urllib.request, json
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


data_source = 'kaggle'
if data_source == 'alphavantage':
    #2 ways of getting data from alphvantage, 
    #one is saving data from alhpavantage website directly to csv file and using that file
    #second is retrieving data from alphavantage website and storing the data in a pandas dataframe
    
    
    
    #storing from alphavantage to csv file
    #4 main steps
    
    api_key = 'AD9C9HJUNS64N2H6'
    ticker = 'AAL'
    
    #in the url remember 4 importnt values:
    #function, e.g TIME_SERIES_DAILY, TIME_SERIES_INTRADAY
    #outputsize = full (gives data of all history)
    #symbol = AAL, HPQ (legitimate abbreviated name of stock we'll be using)
    #apikey  = AD9C9HJUNS64N2H6
    url_string = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&outputsize=full&symbol={ticker}&apikey={api_key}'
    
    '''
    by using placeholder %s for variables symbol and api_key in url_string, 
    the program becomes more general for the api to use for different users
    they need only change the value of the variables symbol and api_key 
    according to what they are working with respectively
    '''
    file_to_save = 'stockmkt_data_%s.csv'%ticker
    
    
    #now the other approach of retrieving the data from the alphavantage link and storing directly in a pandas dataframe
    if not os.path.exists(file_to_save):
        #with is used for better program flow because it closes the resources used right after processing them
        with urllib.request.urlopen(url_string) as url:
            data = json.loads([url.read().decode()])
            data = data['TIME_SERIES_DAILY']
            #json.loads loads the data  of the website into a python dictionary
            #in the next line (of json.loads) we assign the values of the key TIME_SERIES_DAILY Tto the same variable data
            df = pd.Dataframe(columns=['Date','Open','High','Low','Close'])
            for k, v in data.items:
                date = dt.datetime.strptime(k,'%Y-%M-%D')
                data_row = [date.date(), float[v,'1. Open'], float[v,'2. High'], float[v,'3.Low'],float[v,'4.Close']]
                df.loc[-1:] = data_row
                df.index = df.index + 1
        df.to_csv(file_to_save)
        print('Data saved to: %s',(file_to_save))
    else:
        
        df = pd.read_csv(file_to_save)
        print('File Exists, Loading data from csv..Done')
       #now loading from kaggle file
else:
    df = pd.read_csv(os.path.join('Stocks','aal.us.txt'), delimiter = ',', usecols = ['Date','Open','High','Low','Close'])
    print('loaded from Kaggle')
            
        
>>>>>>> 7d86c5933574b43b07e3846d2e0e3fdf8df75867
    