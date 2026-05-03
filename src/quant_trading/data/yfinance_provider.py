"""Yahoo Finance market data provider."""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

import pandas as pd
import yfinance as yf

from quant_trading.data.base import DataProvider


class YFinanceDataProvider(DataProvider):
    """Download daily price data from Yahoo Finance."""

    def get_history(
        self,
        symbols: Sequence[str],
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Return daily close prices keyed by ticker symbol."""
        history: dict[str, pd.DataFrame] = {}

        for symbol in symbols:
            ticker = symbol.upper()
            raw_data = yf.download(
                ticker,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=False,
                progress=False,
            )

            close = self._get_close_series(raw_data, ticker)
            data = pd.DataFrame({"close": close})
            data.index = pd.to_datetime(data.index)

            history[ticker] = data

        return history

    @staticmethod
    def _get_close_series(raw_data: pd.DataFrame, ticker: str) -> pd.Series:
        """Return adjusted close prices when available, otherwise close prices."""
        if isinstance(raw_data.columns, pd.MultiIndex):
            price_column = (
                "Adj Close"
                if "Adj Close" in raw_data.columns.get_level_values(0)
                else "Close"
            )
            prices = raw_data[price_column]
            return prices[ticker] if ticker in prices.columns else prices.iloc[:, 0]

        price_column = "Adj Close" if "Adj Close" in raw_data.columns else "Close"
        return raw_data[price_column]
