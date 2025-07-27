import os
import pandas as pd
import talib
import mplfinance as mpf

# === Load your stock data ===
def main():
    df = pd.read_csv('stockmkt_AAL.csv')  # Replace with your actual CSV
    df['Date'] = pd.to_datetime(df['Date'])
    #always set the dates in ascending order

    if df['Date'].iloc[0]> df['Date'].iloc[1]:
        df.sort_values('Date', inplace=True)

    df.set_index('Date', inplace=True)

    # === Create directory to save images ===
    output_dir = 'candlestick_images'
    os.makedirs(output_dir, exist_ok=True)

    # === Use TA-Lib to detect candlestick patterns ===
    candlestick_patterns = talib.get_function_groups()['Pattern Recognition']

    # Add each pattern detection column
    for pattern in candlestick_patterns:
        df[pattern] = getattr(talib, pattern)(df['Open'], df['High'], df['Low'], df['Close'])

    # === Generate labeled candlestick charts ===
    window_size = 20
    pattern_dict = {}
    
    my_style = mpf.make_mpf_style(
    base_mpf_style='classic',
    marketcolors=mpf.make_marketcolors(up='red', down='green', edge='inherit', wick='inherit', volume='inherit')
)
    for i in range(window_size, len(df)):
        for pattern in candlestick_patterns:
            if df[pattern].iloc[i] != 0:
                label = f"{pattern}_{'bullish' if df[pattern].iloc[i] > 0 else 'bearish'}"
                filename = f"{label}_{i}.png"
                filepath = os.path.join(output_dir, filename)

                # Plot and save the last 20 rows as a candlestick chart
                mpf.plot(df.iloc[i-window_size:i], type='candle', style=my_style, savefig=filepath)
                pattern_dict[filename] = label
                break  # Only label each image with one pattern

    # === Save labels as CSV ===
    pd.DataFrame.from_dict(pattern_dict, orient='index', columns=['label']).to_csv('image_labels.csv')


if __name__ == "__main__":
    main()  # or your actual code here
