"""Simple vectorized backtesting engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeGuard, cast

import pandas as pd

from quant_trading.backtesting.base import BacktestEngine
from quant_trading.strategies.base import Strategy, UniverseStrategy

StrategyLike = Strategy | UniverseStrategy
StrategyWeight = tuple[StrategyLike, float]
StrategyInput = StrategyLike | list[StrategyLike] | list[StrategyWeight]


@dataclass(frozen=True)
class BacktestResult:
    """Container for core backtest outputs."""

    portfolio_returns: pd.Series
    cumulative_returns: pd.Series
    benchmark_returns: pd.Series
    benchmark_cumulative_returns: pd.Series
    max_drawdown: float
    sharpe_ratio: float
    benchmark_max_drawdown: float
    benchmark_sharpe_ratio: float


class VectorizedBacktestEngine(BacktestEngine):
    """Run a simple vectorized backtest."""

    trading_days = 252

    def __init__(
        self,
        price_data: dict[str, pd.DataFrame],
        strategy: StrategyInput,
        transaction_cost_bps: float = 10.0,
    ) -> None:
        self.price_data = price_data
        self.strategies = self._normalize_strategies(strategy)
        self.transaction_cost = transaction_cost_bps / 10_000

    def run(self) -> BacktestResult:
        """Run the strategy or strategies across all stocks."""
        universe_signals = self._generate_universe_signals()
        strategy_returns = {
            ticker: self._stock_returns(ticker, frame, universe_signals)
            for ticker, frame in self.price_data.items()
        }
        benchmark_returns = {
            ticker: self._buy_and_hold_returns(frame)
            for ticker, frame in self.price_data.items()
        }

        returns = pd.DataFrame(strategy_returns).fillna(0.0)
        portfolio_returns = returns.mean(axis=1)
        cumulative_returns = (1.0 + portfolio_returns).cumprod() - 1.0

        benchmark_returns_frame = pd.DataFrame(benchmark_returns).fillna(0.0)
        benchmark_portfolio_returns = benchmark_returns_frame.mean(axis=1)
        benchmark_cumulative_returns = (
            (1.0 + benchmark_portfolio_returns).cumprod() - 1.0
        )

        return BacktestResult(
            portfolio_returns=portfolio_returns,
            cumulative_returns=cumulative_returns,
            benchmark_returns=benchmark_portfolio_returns,
            benchmark_cumulative_returns=benchmark_cumulative_returns,
            max_drawdown=self._max_drawdown(portfolio_returns),
            sharpe_ratio=self._sharpe_ratio(portfolio_returns),
            benchmark_max_drawdown=self._max_drawdown(benchmark_portfolio_returns),
            benchmark_sharpe_ratio=self._sharpe_ratio(benchmark_portfolio_returns),
        )

    def _stock_returns(
        self,
        ticker: str,
        prices: pd.DataFrame,
        universe_signals: dict[int, dict[str, pd.Series]],
    ) -> pd.Series:
        """Apply averaged strategy signals to one stock's daily returns."""
        close = prices["close"]
        daily_returns = close.pct_change().fillna(0.0)

        signals = self._combined_signals(ticker, prices, universe_signals)
        positions = signals.shift(1).fillna(0.0)
        costs = positions.diff().abs().fillna(0.0) * self.transaction_cost

        return daily_returns * positions - costs

    def _combined_signals(
        self,
        ticker: str,
        prices: pd.DataFrame,
        universe_signals: dict[int, dict[str, pd.Series]],
    ) -> pd.Series:
        """Combine weighted signals from all strategies into one position series."""
        strategy_signals = []
        for strategy, weight in self.strategies:
            if isinstance(strategy, UniverseStrategy):
                raw_signals = universe_signals[id(strategy)].get(ticker)
            else:
                raw_signals = strategy.generate_signals(prices)

            if raw_signals is None:
                raw_signals = pd.Series(0.0, index=prices.index)

            signals = pd.Series(raw_signals, index=prices.index)
            strategy_signals.append(signals.reindex(prices.index).fillna(0.0) * weight)

        return pd.DataFrame(strategy_signals).T.sum(axis=1)

    def _generate_universe_signals(self) -> dict[int, dict[str, pd.Series]]:
        """Generate full-universe signals once for each universe strategy."""
        return {
            id(strategy): strategy.generate_signals(self.price_data)
            for strategy, _ in self.strategies
            if isinstance(strategy, UniverseStrategy)
        }

    @staticmethod
    def _normalize_strategies(strategy: StrategyInput) -> list[tuple[StrategyLike, float]]:
        """Return strategies as explicit strategy-weight pairs."""
        if isinstance(strategy, (Strategy, UniverseStrategy)):
            return [(strategy, 1.0)]

        if not strategy:
            raise ValueError("At least one strategy is required.")

        if VectorizedBacktestEngine._is_weighted_strategy_list(strategy):
            return [(strategy_item, float(weight)) for strategy_item, weight in strategy]

        unweighted_strategies = cast(list[StrategyLike], strategy)
        equal_weight = 1.0 / len(unweighted_strategies)
        return [
            (strategy_item, equal_weight)
            for strategy_item in unweighted_strategies
        ]

    @staticmethod
    def _is_weighted_strategy_list(
        strategies: list[StrategyLike] | list[StrategyWeight],
    ) -> TypeGuard[list[StrategyWeight]]:
        """Return whether a strategy collection contains explicit weights."""
        return all(isinstance(item, tuple) and len(item) == 2 for item in strategies)

    @staticmethod
    def _buy_and_hold_returns(prices: pd.DataFrame) -> pd.Series:
        """Return daily buy-and-hold returns for one stock."""
        return prices["close"].pct_change().fillna(0.0)

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
