from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from swing_research.portfolio_backtesting import (
    RelativeStrengthSpec,
    _target_weights,
    close_matrix,
    run_relative_strength_portfolio,
)
from swing_research.relative_strength_research import evaluate_benchmark_promotion


def _frames(periods: int = 420) -> dict[str, pd.DataFrame]:
    index = pd.bdate_range("2020-01-02", periods=periods)
    frames: dict[str, pd.DataFrame] = {}
    for offset, ticker in enumerate(("SPY", "QQQ", "SHY")):
        generator = np.random.default_rng(45 + offset)
        drift = 0.0005 if ticker != "SHY" else 0.00005
        close = 100 * np.exp(np.cumsum(generator.normal(drift, 0.008, periods)))
        open_price = close * (1 + generator.normal(0, 0.001, periods))
        frames[ticker] = pd.DataFrame(
            {
                "open": open_price,
                "high": np.maximum(open_price, close) * 1.003,
                "low": np.minimum(open_price, close) * 0.997,
                "close": close,
                "volume": 1_000_000,
            },
            index=index,
        )
    return frames


def _spec() -> RelativeStrengthSpec:
    return RelativeStrengthSpec(63, 126, 21, 5, 1, ("SPY", "QQQ"), "SHY")


def test_future_change_cannot_change_prior_target_weights() -> None:
    frames = _frames()
    original = close_matrix(frames)
    changed = original.copy()
    changed.iloc[-1, 0] *= 10
    assert _target_weights(original, 300, _spec()).equals(_target_weights(changed, 300, _spec()))


def test_portfolio_backtest_records_benchmark_comparison() -> None:
    frames = _frames()
    result = run_relative_strength_portfolio(
        frames,
        _spec(),
        datetime(2021, 1, 4),
        datetime(2021, 8, 1),
    )
    assert (result.equity > 0).all()
    assert result.daily_turnover.sum() > 0
    assert {"SPY", "QQQ"}.issubset(result.benchmark_metrics)
    assert "terminal_wealth_multiple" in result.benchmark_metrics["SPY"]


def test_promotion_requires_both_benchmarks_in_two_independent_periods() -> None:
    result = evaluate_benchmark_promotion(
        {
            "validation": {
                "benchmarks": {
                    "SPY": {"terminal_wealth_multiple": 1.7},
                    "QQQ": {"terminal_wealth_multiple": 1.6},
                }
            },
            "final_holdout": {
                "benchmarks": {
                    "SPY": {"terminal_wealth_multiple": 1.8},
                    "QQQ": {"terminal_wealth_multiple": 1.4},
                }
            },
        }
    )
    assert not result.passed
    assert result.observed_weakest_multiples == {"validation": 1.6, "final_holdout": 1.4}
    assert "final_holdout" in result.failures[0]


def test_defensive_sleeve_uses_strongest_positive_defensive_asset() -> None:
    frames = _frames()
    prices = close_matrix(frames)
    prices["SPY"] = np.linspace(120, 80, len(prices))
    prices["QQQ"] = np.linspace(120, 80, len(prices))
    prices["TLT"] = np.linspace(100, 130, len(prices))
    prices["GLD"] = np.linspace(110, 100, len(prices))
    prices["SHY"] = np.linspace(102, 100, len(prices))
    spec = RelativeStrengthSpec(
        63,
        126,
        21,
        5,
        1,
        ("SPY", "QQQ"),
        "SHY",
        ("SHY", "TLT", "GLD"),
    )
    weights = _target_weights(prices, 300, spec)
    assert weights["TLT"] == 1.0


def test_trend_pullback_selects_negative_short_return_inside_positive_trend() -> None:
    prices = close_matrix(_frames())
    prices["SPY"] = np.linspace(100, 260, len(prices))
    prices.loc[prices.index[295:301], "SPY"] = [250, 248, 246, 244, 242, 240]
    prices["QQQ"] = np.linspace(100, 260, len(prices))
    spec = RelativeStrengthSpec(
        63,
        126,
        21,
        5,
        1,
        ("SPY", "QQQ"),
        "SHY",
        selection_mode="trend_pullback",
        short_lookback_sessions=5,
    )
    weights = _target_weights(prices, 300, spec)
    assert weights["SPY"] == 1.0
