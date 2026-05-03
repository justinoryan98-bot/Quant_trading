"""Run a simple strategy backtest."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from quant_trading.backtesting import VectorizedBacktestEngine
from quant_trading.data import YFinanceDataProvider
from quant_trading.strategies import (
    CrossSectionalMomentumStrategy,
    MovingAverageStrategy,
    RegimeAdaptiveStrategy,
    RSIMeanReversionStrategy,
)


def print_results(
    period_name: str,
    results: list[tuple[str, float, float, float]],
    benchmark_result,
) -> None:
    """Print strategy and benchmark metrics for one period."""
    print(period_name)
    print(f"{'Strategy':<22} {'Cumulative':>12} {'Max DD':>12} {'Sharpe':>10}")
    print("-" * 60)
    for strategy_name, cumulative_return, max_drawdown, sharpe_ratio in results:
        print(
            f"{strategy_name:<22} "
            f"{cumulative_return:>12.2%} "
            f"{max_drawdown:>12.2%} "
            f"{sharpe_ratio:>10.2f}"
        )

    print()
    print("Benchmark")
    print("-" * 60)
    print(
        f"{'Equal-weight buy & hold':<22} "
        f"{benchmark_result.benchmark_cumulative_returns.iloc[-1]:>12.2%} "
        f"{benchmark_result.benchmark_max_drawdown:>12.2%} "
        f"{benchmark_result.benchmark_sharpe_ratio:>10.2f}"
    )
    print()


def save_equity_curve(
    period_name: str,
    strategy_curves: dict[str, pd.Series],
    benchmark_curve: pd.Series,
    reports_dir: Path,
) -> None:
    """Save cumulative return curves for one period."""
    plt.figure(figsize=(10, 6))

    for strategy_name, cumulative_returns in strategy_curves.items():
        plt.plot(cumulative_returns.index, cumulative_returns, label=strategy_name)

    plt.plot(
        benchmark_curve.index,
        benchmark_curve,
        label="Benchmark",
        linestyle="--",
        color="black",
    )
    plt.title(period_name)
    plt.xlabel("Date")
    plt.ylabel("Cumulative return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    file_name = (
        period_name.lower()
        .replace(" ", "_")
        .replace("-", "")
        .replace("_to_", "_")
    )
    chart_path = reports_dir / f"{file_name}.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()


def main() -> None:
    """Download data, run strategies, and print summary metrics."""
    tickers = [
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "GOOGL",
        "META",
        "AVGO",
        "TSLA",
        "JPM",
        "V",
        "MA",
        "XOM",
        "CVX",
        "UNH",
        "JNJ",
        "LLY",
        "PG",
        "KO",
        "COST",
        "WMT",
        "HD",
        "MCD",
        "CAT",
        "GE",
        "NFLX",
        "PEP",
        "ABBV",
        "MRK",
        "TMO",
        "ADBE",
        "CRM",
        "ORCL",
        "INTC",
        "AMD",
        "QCOM",
        "TXN",
        "IBM",
        "AMAT",
        "LRCX",
        "GS",
        "MS",
        "BLK",
        "SCHW",
        "BK",
        "SPGI",
        "ADP",
        "NOW",
        "ISRG",
        "MDLZ",
        "CL",
    ]
    periods = [
        ("2018-01-01 to 2020-01-01", datetime(2018, 1, 1), datetime(2020, 1, 1)),
        ("2020-01-01 to 2022-01-01", datetime(2020, 1, 1), datetime(2022, 1, 1)),
        ("2022-01-01 to 2024-01-01", datetime(2022, 1, 1), datetime(2024, 1, 1)),
        ("2024-01-01 to today", datetime(2024, 1, 1), datetime.today()),
    ]

    data_provider = YFinanceDataProvider()
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    strategies = [
        ("MA/RSI 100/0", [(MovingAverageStrategy(), 1.0), (RSIMeanReversionStrategy(), 0.0)]),
        ("MA/RSI 80/20", [(MovingAverageStrategy(), 0.8), (RSIMeanReversionStrategy(), 0.2)]),
        ("MA/RSI 70/30", [(MovingAverageStrategy(), 0.7), (RSIMeanReversionStrategy(), 0.3)]),
        ("MA/RSI 60/40", [(MovingAverageStrategy(), 0.6), (RSIMeanReversionStrategy(), 0.4)]),
        ("MA/RSI 50/50", [(MovingAverageStrategy(), 0.5), (RSIMeanReversionStrategy(), 0.5)]),
        ("MA/RSI 30/70", [(MovingAverageStrategy(), 0.3), (RSIMeanReversionStrategy(), 0.7)]),
        ("MA/RSI 0/100", [(MovingAverageStrategy(), 0.0), (RSIMeanReversionStrategy(), 1.0)]),
        ("Cross-Sectional Momentum", CrossSectionalMomentumStrategy()),
        (
            "MA/RSI/Momentum 60/20/20",
            [
                (MovingAverageStrategy(), 0.6),
                (RSIMeanReversionStrategy(), 0.2),
                (CrossSectionalMomentumStrategy(), 0.2),
            ],
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

    for period_name, start_date, end_date in periods:
        price_data = data_provider.get_history(tickers, start=start_date, end=end_date)

        results = []
        strategy_curves: dict[str, pd.Series] = {}
        benchmark_result = None
        for strategy_name, strategy in strategies:
            result = VectorizedBacktestEngine(price_data, strategy).run()
            benchmark_result = result
            strategy_curves[strategy_name] = result.cumulative_returns
            results.append(
                (
                    strategy_name,
                    result.cumulative_returns.iloc[-1],
                    result.max_drawdown,
                    result.sharpe_ratio,
                )
            )

        if benchmark_result is not None:
            print_results(period_name, results, benchmark_result)
            save_equity_curve(
                period_name,
                strategy_curves,
                benchmark_result.benchmark_cumulative_returns,
                reports_dir,
            )


if __name__ == "__main__":
    main()
