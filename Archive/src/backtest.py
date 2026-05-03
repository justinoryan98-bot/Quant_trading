import numpy as np

def run_backtest(df):

    df = df.copy()

    df["strategy_return"] = df["signal"].shift(1) * df["returns"]

    df["equity"] = (1 + df["strategy_return"].fillna(0)).cumprod()

    return df