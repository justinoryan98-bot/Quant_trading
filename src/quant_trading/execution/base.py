"""Base interfaces for order execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ExecutionHandler(ABC):
    """Interface for submitting and managing orders."""

    @abstractmethod
    def submit_order(self, order: Any) -> Any:
        """Submit an order and return the execution response."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        """Cancel an existing order by identifier."""
