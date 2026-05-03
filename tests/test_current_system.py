"""Simple unit tests for the current quant trading system."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_trading.backtesting import VectorizedBacktestEngine
from quant_trading.strategies import (
    CrossSectionalMomentumStrategy,
    MovingAverageStrategy,
    RSIMeanReversionStrategy,
)


def make_price_frame(days: int, start: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    """Create a simple daily close-price DataFrame."""
    index = pd.date_range("2020-01-01", periods=days)
    close = [start + step * day for day in range(days)]
    return pd.DataFrame({"close": close}, index=index)


class StrategyTests(unittest.TestCase):
    """Tests for strategy signal contracts."""

    def test_moving_average_strategy_returns_binary_signals(self) -> None:
        data = make_price_frame(260)

        signals = MovingAverageStrategy().generate_signals(data)

        self.assertTrue(set(signals.unique()).issubset({0, 1}))

    def test_rsi_mean_reversion_strategy_returns_binary_signals(self) -> None:
        data = make_price_frame(60, start=120.0, step=-0.5)

        signals = RSIMeanReversionStrategy().generate_signals(data)

        self.assertTrue(set(signals.unique()).issubset({0, 1}))

    def test_cross_sectional_momentum_returns_signals_for_all_tickers(self) -> None:
        price_data = {
            "AAA": make_price_frame(140, step=1.0),
            "BBB": make_price_frame(140, step=1.5),
            "CCC": make_price_frame(140, step=2.0),
            "DDD": make_price_frame(140, step=2.5),
            "EEE": make_price_frame(140, step=3.0),
        }

        signals = CrossSectionalMomentumStrategy().generate_signals(price_data)

        self.assertEqual(set(signals), set(price_data))
        for ticker, signal in signals.items():
            self.assertTrue(signal.index.equals(price_data[ticker].index))
            self.assertTrue(set(signal.unique()).issubset({0, 1}))


class BacktestEngineTests(unittest.TestCase):
    """Tests for the vectorized backtest engine."""

    def test_vectorized_backtest_engine_runs_on_fake_price_data(self) -> None:
        price_data = {
            "AAA": make_price_frame(260, step=1.0),
            "BBB": make_price_frame(260, step=0.5),
            "CCC": make_price_frame(260, step=1.5),
        }

        result = VectorizedBacktestEngine(price_data, MovingAverageStrategy()).run()

        self.assertIsInstance(result.portfolio_returns, pd.Series)
        self.assertIsInstance(result.cumulative_returns, pd.Series)
        self.assertEqual(len(result.portfolio_returns), 260)
        self.assertEqual(len(result.cumulative_returns), 260)
        self.assertIsInstance(result.max_drawdown, float)
        self.assertIsInstance(result.sharpe_ratio, float)


if __name__ == "__main__":
    unittest.main()
