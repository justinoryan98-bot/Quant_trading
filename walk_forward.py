"""Run simple walk-forward validation."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

import pandas as pd

from config.universe import TICKERS
from quant_trading.backtesting import BacktestResult, VectorizedBacktestEngine
from quant_trading.data import YFinanceDataProvider
from quant_trading.strategies import (
    CrossSectionalMomentumStrategy,
    MovingAverageStrategy,
    RegimeAdaptiveStrategy,
    RSIMeanReversionStrategy,
)


PriceData = dict[str, pd.DataFrame]
ResultRow = tuple[str, float, float, float]
Window = tuple[datetime, datetime, datetime]


def get_strategy_specs() -> list[tuple[str, Any]]:
    """Return strategies compared during walk-forward validation."""
    return [
        (
            "MA/RSI 100/0",
            [(MovingAverageStrategy(), 1.0), (RSIMeanReversionStrategy(), 0.0)],
        ),
        (
            "MA/RSI/Momentum 70/10/20",
            [
                (MovingAverageStrategy(), 0.7),
                (RSIMeanReversionStrategy(), 0.1),
                (CrossSectionalMomentumStrategy(), 0.2),
            ],
        ),
        ("Regime Adaptive", RegimeAdaptiveStrategy()),
    ]


def make_windows(start_year: int, today: datetime) -> list[Window]:
    """Create rolling 2-year train and 1-year test windows."""
    windows = []
    train_start_year = start_year

    while True:
        train_start = datetime(train_start_year, 1, 1)
        train_end = datetime(train_start_year + 2, 1, 1)
        test_end = min(datetime(train_start_year + 3, 1, 1), today)

        if train_end >= today:
            break

        windows.append((train_start, train_end, test_end))
        train_start_year += 1

    return windows


def slice_price_data(price_data: PriceData, start: datetime, end: datetime) -> PriceData:
    """Return price data clipped to a date window."""
    sliced = {}
    for ticker, frame in price_data.items():
        mask = (frame.index >= start) & (frame.index < end)
        sliced[ticker] = frame.loc[mask].copy()

    return sliced


def run_backtest(price_data: PriceData, strategy: Any) -> BacktestResult:
    """Run one strategy on one price-data window."""
    return VectorizedBacktestEngine(price_data, strategy).run()


def result_row(strategy_name: str, result: BacktestResult) -> ResultRow:
    """Return printable metrics for one strategy result."""
    return (
        strategy_name,
        result.cumulative_returns.iloc[-1],
        result.max_drawdown,
        result.sharpe_ratio,
    )


def benchmark_row(result: BacktestResult) -> ResultRow:
    """Return printable benchmark metrics from a backtest result."""
    return (
        "Benchmark",
        result.benchmark_cumulative_returns.iloc[-1],
        result.benchmark_max_drawdown,
        result.benchmark_sharpe_ratio,
    )


def print_table(title: str, rows: list[ResultRow]) -> None:
    """Print a simple metrics table."""
    print(title)
    print(f"{'Strategy':<28} {'Cumulative':>12} {'Max DD':>12} {'Sharpe':>10}")
    print("-" * 66)
    for strategy_name, cumulative_return, max_drawdown, sharpe_ratio in rows:
        print(
            f"{strategy_name:<28} "
            f"{cumulative_return:>12.2%} "
            f"{max_drawdown:>12.2%} "
            f"{sharpe_ratio:>10.2f}"
        )
    print()


def return_diagnostics(returns: pd.Series) -> tuple[float, float, float, float, float, float, float]:
    """Return daily diagnostics for one return series."""
    winning_returns = returns[returns > 0]
    losing_returns = returns[returns < 0]
    average_win = float(winning_returns.mean()) if not winning_returns.empty else 0.0
    average_loss = float(losing_returns.mean()) if not losing_returns.empty else 0.0
    payoff_ratio = average_win / abs(average_loss) if average_loss != 0 else 0.0

    return (
        float((returns > 0).mean()),
        average_win,
        average_loss,
        payoff_ratio,
        float(returns.mean()),
        sharpe_ratio(returns),
        max_drawdown(returns),
    )


def print_diagnostics(title: str, returns_by_name: dict[str, pd.Series]) -> None:
    """Print daily return diagnostics for strategies and benchmark."""
    print(title)
    print(
        f"{'Strategy':<28} {'Win Rate':>10} {'Avg Win':>10} "
        f"{'Avg Loss':>10} {'Payoff':>8} {'Exp Daily':>10} "
        f"{'Sharpe':>8} {'Max DD':>10}"
    )
    print("-" * 102)
    for strategy_name, returns in returns_by_name.items():
        (
            win_rate,
            average_win,
            average_loss,
            payoff_ratio,
            expected_daily_return,
            strategy_sharpe,
            strategy_drawdown,
        ) = return_diagnostics(returns)
        print(
            f"{strategy_name:<28} "
            f"{win_rate:>10.2%} "
            f"{average_win:>10.2%} "
            f"{average_loss:>10.2%} "
            f"{payoff_ratio:>8.2f} "
            f"{expected_daily_return:>10.3%} "
            f"{strategy_sharpe:>8.2f} "
            f"{strategy_drawdown:>10.2%}"
        )
    print()


def print_correlation_matrix(title: str, returns_by_name: dict[str, pd.Series]) -> None:
    """Print a correlation matrix for daily return series."""
    returns = pd.DataFrame(returns_by_name)
    print(title)
    print("-" * 102)
    print(returns.corr().to_string(float_format=lambda value: f"{value:0.2f}"))
    print()


def select_strategy(train_results: dict[str, BacktestResult]) -> str:
    """Select the best train strategy by Sharpe, then lower drawdown magnitude."""
    return max(
        train_results,
        key=lambda name: (
            train_results[name].sharpe_ratio,
            train_results[name].max_drawdown,
        ),
    )


def cumulative_return(returns: pd.Series) -> float:
    """Return cumulative return from a daily return series."""
    return float((1.0 + returns).cumprod().iloc[-1] - 1.0)


def max_drawdown(returns: pd.Series) -> float:
    """Return maximum drawdown from a daily return series."""
    equity_curve = (1.0 + returns).cumprod()
    drawdowns = equity_curve / equity_curve.cummax() - 1.0
    return float(drawdowns.min())


def sharpe_ratio(returns: pd.Series) -> float:
    """Return annualized Sharpe ratio from daily returns."""
    volatility = returns.std()
    if volatility == 0:
        return 0.0

    return float(returns.mean() / volatility * 252**0.5)


def print_aggregate_summary(
    selected_returns: list[pd.Series],
    fixed_returns: list[pd.Series],
    benchmark_returns: list[pd.Series],
    selected_sharpes: list[float],
    fixed_sharpes: list[float],
    benchmark_sharpes: list[float],
    selection_counts: Counter[str],
) -> None:
    """Print aggregate out-of-sample walk-forward metrics."""
    selected = pd.concat(selected_returns).sort_index()
    fixed = pd.concat(fixed_returns).sort_index()
    benchmark = pd.concat(benchmark_returns).sort_index()

    print("Walk-forward aggregate summary")
    print("-" * 66)
    print(f"{'Selected cumulative return':<36} {cumulative_return(selected):>12.2%}")
    print(f"{'Fixed 70/10/20 cumulative return':<36} {cumulative_return(fixed):>12.2%}")
    print(f"{'Benchmark cumulative return':<36} {cumulative_return(benchmark):>12.2%}")
    print(f"{'Selected full-period Sharpe':<36} {sharpe_ratio(selected):>12.2f}")
    print(f"{'Fixed 70/10/20 full-period Sharpe':<36} {sharpe_ratio(fixed):>12.2f}")
    print(f"{'Benchmark full-period Sharpe':<36} {sharpe_ratio(benchmark):>12.2f}")
    print(f"{'Average selected test Sharpe':<36} {pd.Series(selected_sharpes).mean():>12.2f}")
    print(f"{'Average fixed 70/10/20 Sharpe':<36} {pd.Series(fixed_sharpes).mean():>12.2f}")
    print(f"{'Average benchmark test Sharpe':<36} {pd.Series(benchmark_sharpes).mean():>12.2f}")
    print(f"{'Worst selected drawdown':<36} {max_drawdown(selected):>12.2%}")
    print(f"{'Worst fixed 70/10/20 drawdown':<36} {max_drawdown(fixed):>12.2%}")
    print(f"{'Worst benchmark drawdown':<36} {max_drawdown(benchmark):>12.2%}")
    print()
    print("Selections")
    print("-" * 66)
    for strategy_name, count in selection_counts.items():
        print(f"{strategy_name:<36} {count:>12}")


def main() -> None:
    """Download data and run walk-forward validation."""
    today = datetime.today()
    data_provider = YFinanceDataProvider()
    price_data = data_provider.get_history(TICKERS, start=datetime(2018, 1, 1), end=today)

    selected_returns = []
    fixed_returns = []
    benchmark_returns = []
    selected_sharpes = []
    fixed_sharpes = []
    benchmark_sharpes = []
    selection_counts: Counter[str] = Counter()

    for step, (train_start, train_end, test_end) in enumerate(
        make_windows(2018, today),
        start=1,
    ):
        test_start = train_end
        train_data = slice_price_data(price_data, train_start, train_end)
        test_data = slice_price_data(price_data, test_start, test_end)

        train_results = {
            strategy_name: run_backtest(train_data, strategy)
            for strategy_name, strategy in get_strategy_specs()
        }
        train_rows = [
            result_row(strategy_name, result)
            for strategy_name, result in train_results.items()
        ]
        train_rows.append(benchmark_row(next(iter(train_results.values()))))
        train_benchmark_result = next(iter(train_results.values()))
        train_returns_by_name = {
            strategy_name: result.portfolio_returns
            for strategy_name, result in train_results.items()
        }
        train_returns_by_name["Benchmark"] = train_benchmark_result.benchmark_returns

        selected_name = select_strategy(train_results)
        strategy_specs = dict(get_strategy_specs())
        selected_strategy = strategy_specs[selected_name]
        selected_test_result = run_backtest(test_data, selected_strategy)
        fixed_test_result = run_backtest(
            test_data,
            strategy_specs["MA/RSI/Momentum 70/10/20"],
        )

        test_rows = [
            result_row(selected_name, selected_test_result),
            benchmark_row(selected_test_result),
        ]

        print(f"Walk-forward step {step}")
        print(f"Train: {train_start.date()} to {train_end.date()}")
        print(f"Test:  {test_start.date()} to {test_end.date()}")
        print()
        print_table("Train results", train_rows)
        print_diagnostics("Train daily diagnostics", train_returns_by_name)
        print_correlation_matrix("Train daily return correlation", train_returns_by_name)
        print(f"Selected strategy: {selected_name}")
        print()
        print_table("Test results", test_rows)

        selected_returns.append(selected_test_result.portfolio_returns)
        fixed_returns.append(fixed_test_result.portfolio_returns)
        benchmark_returns.append(selected_test_result.benchmark_returns)
        selected_sharpes.append(selected_test_result.sharpe_ratio)
        fixed_sharpes.append(fixed_test_result.sharpe_ratio)
        benchmark_sharpes.append(selected_test_result.benchmark_sharpe_ratio)
        selection_counts[selected_name] += 1

    print_aggregate_summary(
        selected_returns,
        fixed_returns,
        benchmark_returns,
        selected_sharpes,
        fixed_sharpes,
        benchmark_sharpes,
        selection_counts,
    )


if __name__ == "__main__":
    main()
