#this file checks the number of unique candlesticks patterns that ta-lib identifies

import os

with open('image_labels.csv', 'r') as f:
    c = f.read()
    print(set(c))

#%%

#working with enumerate

l = ["hdh", "hrh", "hdh", "hrio"]
for i, els in enumerate(l):
    print(i,els)