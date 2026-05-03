"""Base interfaces for market data access."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Sequence


class DataProvider(ABC):
    """Interface for loading market data from any source.

    Implementations may read from files, databases, broker APIs, or external
    data vendors. Returned data should be documented by the implementation.
    """

    @abstractmethod
    def get_history(
        self,
        symbols: Sequence[str],
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> Any:
        """Return historical market data for the requested symbols and dates."""
