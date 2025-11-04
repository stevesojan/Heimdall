import pandas as pd
import talib
import mplfinance as mpf
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def main():
    df = pd.read_csv('stockmkt_AAL.csv')  # Replace with your actual CSV
    df['Date'] = pd.to_datetime(df['Date'])

    # Ensure ascending order by Date
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)

    # === Create directory to save images ===
    output_dir = 'candlestick_images'
    os.makedirs(output_dir, exist_ok=True)

    # === Use TA-Lib to detect candlestick patterns ===
    candlestick_patterns = talib.get_function_groups()['Pattern Recognition']
    active_patterns = []
    for pattern in candlestick_patterns:
        df[pattern] = getattr(talib, pattern)(df['Open'], df['High'], df['Low'], df['Close'])
        if df[pattern].ne(0).sum() > 0:
            active_patterns.append(pattern)

    print("Active Patterns:", active_patterns)
    # === Plotting Parameters ===
    window_size = 20
    pattern_dict = {}

    my_style = mpf.make_mpf_style(
        base_mpf_style='classic',
        marketcolors=mpf.make_marketcolors(up='green', down='red', edge='inherit', wick='inherit', volume='inherit')
    )

    for i in range(window_size, len(df) - 5):  # leave 5 for trendline
        for pattern in active_patterns:
            if df[pattern].iloc[i] != 0:
                label = f"{pattern}_{'bullish' if df[pattern].iloc[i] > 0 else 'bearish'}"
                filename = f"{label}_{i}.png"
                filepath = os.path.join(output_dir, filename)

                # === Plot with annotation ===
                fig, axlist = mpf.plot(
                    df.iloc[i-window_size:i+5],  # show future 5 candles
                    type='candle',
                    style=my_style,
                    returnfig=True,
                    figsize=(8, 6)
                )

                ax = axlist[0]  # candlestick axis

                # === Rectangle around the pattern candle ===
                pattern_index = window_size - 1  # pattern at end of base window
                candle_date = df.iloc[i].name
                candle_open = df['Open'].iloc[i]
                candle_close = df['Close'].iloc[i]
                candle_high = df['High'].iloc[i]
                candle_low = df['Low'].iloc[i]

                # Draw box from low to high
                rect = Rectangle(
                    (pattern_index - 0.4, candle_low),
                    width=0.8,
                    height=candle_high - candle_low,
                    linewidth=1.5,
                    edgecolor='black',
                    facecolor='none'
                )
                ax.add_patch(rect)

                # === Add Trendline for next 5 candles ===
                future_closes = df['Close'].iloc[i:i+5].values
                trend_x = list(range(pattern_index, pattern_index + 5))
                ax.plot(trend_x, future_closes, linestyle='--', color='blue', linewidth=1.5)

                fig.savefig(filepath)
                plt.close(fig)

                pattern_dict[filename] = label
                break  # Only one pattern per image

    # === Save labels ===
    pd.DataFrame.from_dict(pattern_dict, orient='index', columns=['label']).to_csv('image_labels.csv')

if __name__ == "__main__":
    main()  # or your actual code here
