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
