"""Run a simple moving average strategy backtest."""

from __future__ import annotations

from datetime import datetime

from quant_trading.backtesting import VectorizedBacktestEngine
from quant_trading.data import YFinanceDataProvider
from quant_trading.strategies import MovingAverageStrategy


def main() -> None:
    """Download data, run the strategy, and print summary metrics."""
    tickers = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOG"]
    start_date = datetime(2018, 1, 1)

    data_provider = YFinanceDataProvider()
    price_data = data_provider.get_history(tickers, start=start_date)

    strategy = MovingAverageStrategy()
    backtest = VectorizedBacktestEngine(price_data, strategy)
    result = backtest.run()

    cumulative_return = result.cumulative_returns.iloc[-1]

    print(f"Cumulative return: {cumulative_return:.2%}")
    print(f"Max drawdown: {result.max_drawdown:.2%}")
    print(f"Sharpe ratio: {result.sharpe_ratio:.2f}")


if __name__ == "__main__":
    main()
