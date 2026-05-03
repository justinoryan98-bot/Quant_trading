"""Backtest period configuration."""

from __future__ import annotations

from datetime import datetime

PERIODS = [
    ("2018-01-01 to 2020-01-01", datetime(2018, 1, 1), datetime(2020, 1, 1)),
    ("2020-01-01 to 2022-01-01", datetime(2020, 1, 1), datetime(2022, 1, 1)),
    ("2022-01-01 to 2024-01-01", datetime(2022, 1, 1), datetime(2024, 1, 1)),
    ("2024-01-01 to today", datetime(2024, 1, 1), datetime.today()),
]
