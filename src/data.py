import yfinance as yf
import pandas as pd

def get_data(ticker="SPY"):
    df = yf.download(ticker, start="1993-01-01")
    df = df.dropna()

    # flatten multi-index if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df