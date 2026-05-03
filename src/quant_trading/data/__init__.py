"""Market data access interfaces and implementations."""

from quant_trading.data.base import DataProvider
from quant_trading.data.yfinance_provider import YFinanceDataProvider

__all__ = ["DataProvider", "YFinanceDataProvider"]
