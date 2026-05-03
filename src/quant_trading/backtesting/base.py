"""Base interfaces for backtesting engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BacktestEngine(ABC):
    """Interface for running historical simulations."""

    @abstractmethod
    def run(self) -> Any:
        """Run a backtest and return the results."""
