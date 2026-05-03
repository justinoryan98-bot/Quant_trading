"""RSI mean reversion strategy."""

from __future__ import annotations

from typing import Any

import pandas as pd

from quant_trading.strategies.base import Strategy


class RSIMeanReversionStrategy(Strategy):
    """Generate long-only signals from a 14-day RSI."""

    rsi_window = 14
    entry_threshold = 30
    exit_threshold = 50

    def generate_signals(self, market_data: Any) -> pd.Series:
        """Return 1 below RSI 30, 0 above RSI 50, otherwise keep prior signal."""
        close = market_data["close"]
        rsi = self._rsi(close)

        signals = pd.Series(index=close.index, data=pd.NA, dtype="Float64")
        signals[rsi < self.entry_threshold] = 1
        signals[rsi > self.exit_threshold] = 0

        return signals.ffill().fillna(0).astype(int)

    def _rsi(self, close: pd.Series) -> pd.Series:
        """Calculate 14-day relative strength index."""
        delta = close.diff()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)

        average_gain = gains.rolling(window=self.rsi_window).mean()
        average_loss = losses.rolling(window=self.rsi_window).mean()
        relative_strength = average_gain / average_loss

        return 100 - (100 / (1 + relative_strength))
