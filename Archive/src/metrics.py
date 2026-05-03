import numpy as np

def calculate_metrics(df):

    rets = df["strategy_return"].dropna()

    if len(rets) == 0:
        return {"return": 0, "sharpe": np.nan, "trades": 0}

    total_return = df["equity"].iloc[-1] - 1

    sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() != 0 else np.nan

    trades = df["signal"].diff().abs().sum()

    return {
        "return": total_return,
        "sharpe": sharpe,
        "trades": trades
    }