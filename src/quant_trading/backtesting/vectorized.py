"""Simple vectorized backtesting engine."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_trading.backtesting.base import BacktestEngine
from quant_trading.strategies.base import Strategy


@dataclass(frozen=True)
class BacktestResult:
    """Container for core backtest outputs."""

    portfolio_returns: pd.Series
    cumulative_returns: pd.Series
    max_drawdown: float
    sharpe_ratio: float


class VectorizedBacktestEngine(BacktestEngine):
    """Run a simple long/cash equal-weighted vectorized backtest."""

    trading_days = 252

    def __init__(
        self,
        price_data: dict[str, pd.DataFrame],
        strategy: Strategy,
        transaction_cost_bps: float = 10.0,
    ) -> None:
        self.price_data = price_data
        self.strategy = strategy
        self.transaction_cost = transaction_cost_bps / 10_000

    def run(self) -> BacktestResult:
        """Run the strategy across all stocks and return portfolio metrics."""
        strategy_returns = {
            ticker: self._stock_returns(frame)
            for ticker, frame in self.price_data.items()
        }

        returns = pd.DataFrame(strategy_returns).fillna(0.0)
        portfolio_returns = returns.mean(axis=1)
        cumulative_returns = (1.0 + portfolio_returns).cumprod() - 1.0

        return BacktestResult(
            portfolio_returns=portfolio_returns,
            cumulative_returns=cumulative_returns,
            max_drawdown=self._max_drawdown(portfolio_returns),
            sharpe_ratio=self._sharpe_ratio(portfolio_returns),
        )

    def _stock_returns(self, prices: pd.DataFrame) -> pd.Series:
        """Apply strategy signals to one stock's daily returns."""
        close = prices["close"]
        daily_returns = close.pct_change().fillna(0.0)

        signals = pd.Series(self.strategy.generate_signals(prices), index=prices.index)
        signals = signals.reindex(prices.index).fillna(0.0)
        positions = signals.shift(1).fillna(0.0)
        costs = positions.diff().abs().fillna(0.0) * self.transaction_cost

        return daily_returns * positions - costs

    @staticmethod
    def _max_drawdown(portfolio_returns: pd.Series) -> float:
        """Return the maximum peak-to-trough portfolio drawdown."""
        equity_curve = (1.0 + portfolio_returns).cumprod()
        drawdowns = equity_curve / equity_curve.cummax() - 1.0
        return float(drawdowns.min())

    @classmethod
    def _sharpe_ratio(cls, portfolio_returns: pd.Series) -> float:
        """Return annualized Sharpe ratio using daily returns."""
        volatility = portfolio_returns.std()
        if volatility == 0:
            return 0.0

        return float(portfolio_returns.mean() / volatility * cls.trading_days**0.5)
