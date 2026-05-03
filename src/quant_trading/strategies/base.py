"""Base interfaces for trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Strategy(ABC):
    """Interface for generating trading signals from market data."""

    @abstractmethod
    def generate_signals(self, market_data: Any) -> Any:
        """Return strategy signals for the supplied market data."""
