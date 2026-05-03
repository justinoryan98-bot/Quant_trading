"""Base interfaces for trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class Strategy(ABC):
    """Interface for generating trading signals from market data."""

    @abstractmethod
    def generate_signals(self, market_data: Any) -> Any:
        """Return strategy signals for the supplied market data."""


class UniverseStrategy(ABC):
    """Interface for generating signals from a full stock universe."""

    @abstractmethod
    def generate_signals(
        self,
        price_data: dict[str, pd.DataFrame],
    ) -> dict[str, pd.Series]:
        """Return signal series keyed by ticker symbol."""
