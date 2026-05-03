import pandas as pd

def apply_strategy(df, streak_threshold=4, ma_window=100):

    df = df.copy()

    close = df["Close"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    # returns
    df["returns"] = close.pct_change()

    # trend regime (LIGHT filter only)
    df["ma"] = close.rolling(ma_window).mean()
    df["trend"] = close / df["ma"] - 1

    # direction
    df["up"] = (df["returns"] > 0).astype(int)
    df["down"] = (df["returns"] < 0).astype(int)

    # streaks
    df["up_streak"] = df["up"].groupby((df["up"] == 0).cumsum()).cumcount() + 1
    df["down_streak"] = df["down"].groupby((df["down"] == 0).cumsum()).cumcount() + 1

    # signals
    df["signal"] = 0

    df.loc[df["down_streak"] >= streak_threshold, "signal"] = 1
    df.loc[df["up_streak"] >= streak_threshold, "signal"] = -1

    # LIGHT regime filter only (NOT aggressive)
    df.loc[df["trend"].abs() > 0.05, "signal"] = 0

    return df