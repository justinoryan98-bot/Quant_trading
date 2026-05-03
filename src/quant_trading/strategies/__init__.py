"""Trading strategy interfaces and implementations."""

from quant_trading.strategies.base import Strategy
from quant_trading.strategies.moving_average import MovingAverageStrategy

__all__ = ["MovingAverageStrategy", "Strategy"]
