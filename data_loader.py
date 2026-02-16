import yfinance as yf
import pandas as pd
import os


def download_data(ticker,
                  start="2020-01-01",
                  end="2024-01-01",
                  save=True):

    os.makedirs("data", exist_ok=True)
    filepath = f"data/{ticker}.csv"

    # If file already exists, load it
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        return df

    df = yf.download(ticker, start=start, end=end)

    # Fix multi-index issue
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)

    if save:
        df.to_csv(filepath)

    return df
