import mplfinance as mpf
import pandas as pd

# Sample DataFrame (replace with your data)
data = pd.DataFrame({
    'Open': [100, 105, 110, 108, 112],
    'High': [105, 112, 115, 112, 115],
    'Low': [98, 100, 105, 105, 110],
    'Close': [105, 110, 108, 112, 114]
}, index=pd.to_datetime(['2025-05-13', '2025-05-14', '2025-05-15', '2025-05-16', '2025-05-17']))

# Plotting the data
mpf.plot(data, type='candle', style='yahoo', title='Stock Price')