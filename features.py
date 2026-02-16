import ta


def add_features(df):

    close = df['Close'].squeeze()

    df['SMA_10'] = ta.trend.sma_indicator(close, window=10)
    df['EMA_10'] = ta.trend.ema_indicator(close, window=10)
    df['RSI'] = ta.momentum.rsi(close, window=14)
    df['MACD'] = ta.trend.macd(close)
    df['Returns'] = close.pct_change()

    df.dropna(inplace=True)

    return df
