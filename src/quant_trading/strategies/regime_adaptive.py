"""Regime-adaptive strategy blend."""

from __future__ import annotations

import pandas as pd

from quant_trading.strategies.base import UniverseStrategy
from quant_trading.strategies.cross_sectional_momentum import (
    CrossSectionalMomentumStrategy,
)
from quant_trading.strategies.moving_average import MovingAverageStrategy
from quant_trading.strategies.rsi_mean_reversion import RSIMeanReversionStrategy


class RegimeAdaptiveStrategy(UniverseStrategy):
    """Blend MA, RSI, and momentum signals using trend strength."""

    min_moving_average_weight = 0.2
    max_moving_average_weight = 0.6
    momentum_weight = 0.2

    def __init__(self) -> None:
        self.moving_average = MovingAverageStrategy()
        self.rsi_mean_reversion = RSIMeanReversionStrategy()
        self.cross_sectional_momentum = CrossSectionalMomentumStrategy()

    def generate_signals(
        self,
        price_data: dict[str, pd.DataFrame],
    ) -> dict[str, pd.Series]:
        """Return regime-weighted long-only signals by ticker."""
        trend_strength = self._trend_strength(price_data)
        momentum_signals = self.cross_sectional_momentum.generate_signals(price_data)

        signals = {}
        for ticker, frame in price_data.items():
            index = frame.index
            strength = trend_strength.reindex(index).fillna(0.0)
            moving_average_weight = self._moving_average_weight(strength)
            momentum_weight = pd.Series(self.momentum_weight, index=index)
            rsi_weight = 1.0 - moving_average_weight - momentum_weight

            moving_average_signal = self._series(
                self.moving_average.generate_signals(frame),
                index,
            )
            rsi_signal = self._series(
                self.rsi_mean_reversion.generate_signals(frame),
                index,
            )
            momentum_signal = self._series(
                momentum_signals.get(ticker, pd.Series(0.0, index=index)),
                index,
            )

            signals[ticker] = (
                moving_average_signal * moving_average_weight
                + rsi_signal * rsi_weight
                + momentum_signal * momentum_weight
            )

        return signals

    @staticmethod
    def _trend_strength(
        price_data: dict[str, pd.DataFrame],
    ) -> pd.Series:
        """Return lagged trend strength from an equal-weight universe index."""
        closes = pd.DataFrame(
            {ticker: frame["close"] for ticker, frame in price_data.items()}
        )
        universe_returns = closes.pct_change().fillna(0.0).mean(axis=1)
        universe_index = (1.0 + universe_returns).cumprod()

        moving_average_50 = universe_index.rolling(window=50).mean()
        moving_average_200 = universe_index.rolling(window=200).mean()
        moving_average_spread = (
            (moving_average_50 - moving_average_200) / moving_average_200
        )
        return_60_day = universe_index.pct_change(60)

        normalized_spread = (moving_average_spread / 0.10).clip(
            lower=0.0,
            upper=1.0,
        )
        normalized_return = (return_60_day / 0.10).clip(lower=0.0, upper=1.0)
        normalized_strength = (normalized_spread + normalized_return) / 2.0
        return normalized_strength.shift(1).fillna(0.0)

    @staticmethod
    def _series(values: pd.Series, index: pd.Index) -> pd.Series:
        """Return a numeric signal series aligned to an index."""
        return pd.Series(values, index=index).reindex(index).fillna(0.0)

    @classmethod
    def _moving_average_weight(cls, trend_strength: pd.Series) -> pd.Series:
        """Return MA weight that rises smoothly with trend strength."""
        weight_range = cls.max_moving_average_weight - cls.min_moving_average_weight
        return cls.min_moving_average_weight + trend_strength * weight_range
