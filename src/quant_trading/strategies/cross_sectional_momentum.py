"""Cross-sectional momentum strategy."""

from __future__ import annotations

from math import ceil

import pandas as pd

from quant_trading.strategies.base import UniverseStrategy


class CrossSectionalMomentumStrategy(UniverseStrategy):
    """Go long the top 20% of stocks by 6-month trailing return."""

    lookback_days = 126
    top_fraction = 0.20

    def generate_signals(
        self,
        price_data: dict[str, pd.DataFrame],
    ) -> dict[str, pd.Series]:
        """Return long-only cross-sectional momentum signals by ticker."""
        trailing_returns = pd.DataFrame(
            {
                ticker: frame["close"].pct_change(self.lookback_days)
                for ticker, frame in price_data.items()
            }
        )

        top_count = max(1, ceil(len(trailing_returns.columns) * self.top_fraction))
        ranks = trailing_returns.rank(axis=1, ascending=False, method="first")
        signals = (ranks <= top_count).astype(int)
        signals = signals.where(trailing_returns.notna(), 0)
        signals = signals.shift(1).fillna(0)

        return {
            ticker: signals[ticker].reindex(frame.index).fillna(0).astype(int)
            for ticker, frame in price_data.items()
        }
