from src.data import get_data
from src.strategy import apply_strategy
from src.backtest import run_backtest
from src.metrics import calculate_metrics
import numpy as np

df = get_data("SPY")

# -----------------------------
# WALK-FORWARD CONFIG (OPTION 1)
# -----------------------------

trading_days = 252

train_size = 7 * trading_days      # 7 years
test_size = 1 * trading_days       # 1 year
step = test_size // 2              # 6 months overlap

streak_grid = [2, 3, 4, 5, 6, 7]

results = []

for start in range(0, len(df) - train_size - test_size, step):

    train_end = start + train_size
    test_end = train_end + test_size

    train = df.iloc[start:train_end]
    test = df.iloc[train_end:test_end]

    print("\n--- NEW WINDOW ---")
    print(f"Test period: {test.index[0]} to {test.index[-1]}")

    window_results = []

    for s in streak_grid:

        train_strat = apply_strategy(train, s)
        train_strat = run_backtest(train_strat)

        test_strat = apply_strategy(test, s)
        test_strat = run_backtest(test_strat)

        metrics = calculate_metrics(test_strat)

        print(
            f"Streak {s} | Trades: {metrics['trades']:.0f} | "
            f"Return: {metrics['return']:.3f} | Sharpe: {metrics['sharpe']}"
        )

        window_results.append((s, metrics))

    results.append(window_results)

# -----------------------------
# AGGREGATION
# -----------------------------

final = []

for i, s in enumerate(streak_grid):

    rets = [w[i][1]["return"] for w in results if len(w) > i]
    sharpes = [w[i][1]["sharpe"] for w in results if len(w) > i]
    trades = [w[i][1]["trades"] for w in results if len(w) > i]

    final.append({
        "streak": s,
        "return": np.mean(rets),
        "sharpe": np.nanmean(sharpes),
        "trades": np.mean(trades)
    })

print("\nFINAL RESULTS:")
for r in final:
    print(r)

best = max(final, key=lambda x: x["return"])

print("\nBEST PARAMETER:")
print(best)