# Quant Trading

A modular Python project for researching, backtesting, and eventually running quantitative trading strategies.

This repository is intentionally scaffolded without trading logic yet. The goal is to keep responsibilities separated from the start so data handling, strategy research, risk controls, backtesting, reporting, and execution can evolve independently.

## Project Structure

```text
Quant_trading/
+-- config/
|   `-- Strategy, environment, and runtime configuration files.
+-- src/
|   `-- quant_trading/
|       +-- data/
|       |   `-- Market data loading, validation, cleaning, and storage adapters.
|       +-- features/
|       |   `-- Indicators, transformations, and reusable feature calculations.
|       +-- strategies/
|       |   `-- Strategy definitions and signal generation.
|       +-- backtesting/
|       |   `-- Historical simulation, fills, portfolio accounting, and performance evaluation.
|       +-- risk/
|       |   `-- Position sizing, exposure limits, drawdown controls, and risk checks.
|       +-- execution/
|       |   `-- Broker or exchange adapters for live and paper trading.
|       `-- reporting/
|           `-- Trade logs, metrics, charts, and research summaries.
`-- tests/
    `-- Unit and integration tests for the system.
```

## Design Principles

- Keep modules small and focused.
- Prefer configuration-driven runs over hard-coded parameters.
- Separate research, backtesting, risk, and execution concerns.
- Add trading logic only after the interfaces and tests are clear.

## Setup

Install the project in editable mode from the repository root:

```powershell
python -m pip install -e .
```

This makes the `quant_trading` package importable from scripts, tests, and notebooks.

## Core Interfaces

- `DataProvider`: `src/quant_trading/data/base.py`
- `Strategy`: `src/quant_trading/strategies/base.py`
- `RiskManager`: `src/quant_trading/risk/base.py`
- `ExecutionHandler`: `src/quant_trading/execution/base.py`
- `BacktestEngine`: `src/quant_trading/backtesting/base.py`

## Current Status

Initial project structure only. No trading logic has been implemented yet.
