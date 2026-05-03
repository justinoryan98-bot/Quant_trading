"""Base interfaces for risk management."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class RiskManager(ABC):
    """Interface for applying risk checks before orders are executed."""

    @abstractmethod
    def validate_orders(self, orders: Any, portfolio_state: Any) -> Any:
        """Return approved, adjusted, or rejected orders based on risk rules."""
