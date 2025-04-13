<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 13:24:57 2024

@author: Steve
"""

#loading data from alphavantage

#first importing all dependencies

from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import urllib, json
import datetime as dt
import numpy as np
import tensorflow as tf
import os 
from sklearn.preprocessing import MinMaxScaler





api_key = 'AD9C9HJUNS64N2H6'
ticker = 'AAL'

url_string = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&outputsize=full&symbol={ticker}&apikey={api_key}'

file_to_save = f'stockmkt_{ticker}.csv'

if not os.path.exists(file_to_save):
    with urllib.request.urlopen(url_string) as url:
        data = json.loads(url.read().decode())
        
        
        data = data['Time Series (Daily)']
        df = pd.DataFrame(columns = ['Date','Open','High','Low','Close','Volume'])
        for k, v in data.items():
            date = dt.datetime.strptime(k,'%Y-%m-%d')
            #  %Y uses 4 digit year fromat & %y uses 2 digit year format, likewise for M,m and D,d 
            data_row = [date.date(),float(v['1. open']), float(v['2. high']), float(v['3. low']), float(v['4. close']), float(v['5. volume'])]
            new_row = pd.DataFrame([data_row], columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df = pd.concat([df, new_row], ignore_index=True)

        df.to_csv(file_to_save)
        print(f'Data Saved to: {file_to_save}')
else:
    print('File Exists, Loading...')
    df = pd.read_csv(file_to_save)
    print('Load Complete')
    
df.sort_values('Date')
df.head(5)
    
    

        
=======
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 13:24:57 2024

@author: Steve
"""

#loading data from alphavantage

#first importing all dependencies

from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import urllib, json
import datetime as dt
import numpy as np
import tensorflow as tf
import os 
from sklearn.preprocessing import MinMaxScaler





api_key = '<your alphavantage api key>'
ticker = 'AAL'

url_string = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&outputsize=full&symbol={ticker}&apikey={api_key}'

file_to_save = f'stockmkt_{ticker}.csv'

if not os.path.exists(file_to_save):
    with urllib.request.urlopen(url_string) as url:
        data = json.loads(url.read().decode())
        
        
        data = data['Time Series (Daily)']
        df = pd.DataFrame(columns = ['Date','Open','High','Low','Close','Volume'])
        for k, v in data.items():
            date = dt.datetime.strptime(k,'%Y-%m-%d')
            #  %Y uses 4 digit year fromat & %y uses 2 digit year format, likewise for M,m and D,d 
            data_row = [date.date(),float(v['1. open']), float(v['2. high']), float(v['3. low']), float(v['4. close']), float(v['5. volume'])]
            new_row = pd.DataFrame([data_row], columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df = pd.concat([df, new_row], ignore_index=True)

        df.to_csv(file_to_save)
        print(f'Data Saved to: {file_to_save}')
else:
    print('File Exists, Loading...')
    df = pd.read_csv(file_to_save)
    print('Load Complete')
    
df.sort_values('Date')
df.head(5)
    
    

        
>>>>>>> 7d86c5933574b43b07e3846d2e0e3fdf8df75867
