"""Moving average crossover strategy."""

from __future__ import annotations

from typing import Any

from quant_trading.strategies.base import Strategy


class MovingAverageStrategy(Strategy):
    """Generate long-only signals from 100-day and 200-day moving averages."""

    short_window = 100
    long_window = 200

    def generate_signals(self, market_data: Any) -> Any:
        """Return 1 when the 100-day average is above the 200-day average.

        Expects `market_data` to be a pandas-like object with a `close` column.
        """
        close = market_data["close"]
        short_ma = close.rolling(window=self.short_window).mean()
        long_ma = close.rolling(window=self.long_window).mean()

        return (short_ma > long_ma).astype(int)
