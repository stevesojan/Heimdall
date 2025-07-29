#importing the entries of a .csv into a dataframe and sorting them according to data()ascending)
#also set the values of date to be the index

import pandas as pd
import os

df = pd.read_csv("stockmkt_AAL.csv")

#remember it is essential to convert the entries of date column to pd.to_datetime to use the sort_values fucntion effectively orit will just sort in lexicographical order

df["Date"] = pd.to_datetime(df["Date"])

df.sort_values("Date", inplace = True)
df.set_index("Date", inplace= True)

os.makedirs("candlestick_images", exist_ok = True)

