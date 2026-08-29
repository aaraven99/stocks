from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from swing_research.stock_backtesting import (
    CrossSectionalMomentumSpec,
    _target_weights,
    run_point_in_time_cross_sectional_momentum,
)


def _frames(periods: int = 320) -> dict[str, pd.DataFrame]:
    index = pd.bdate_range("2020-01-02", periods=periods)
    frames: dict[str, pd.DataFrame] = {}
    for ticker, drift in {"AAA": 0.0012, "BBB": 0.0006, "SPY": 0.0005, "QQQ": 0.0007}.items():
        close = 100 * np.exp(np.cumsum(np.full(periods, drift)))
        frames[ticker] = pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000_000,
            },
            index=index,
        )
    return frames


def _membership(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"AAA": True, "BBB": True, "SPY": False, "QQQ": False}, index=index)


def _spec() -> CrossSectionalMomentumSpec:
    return CrossSectionalMomentumSpec(63, 126, 21, 5, 1)


def test_future_prices_cannot_change_prior_cross_sectional_selection() -> None:
    frames = _frames()
    closes = pd.concat({ticker: frame["close"] for ticker, frame in frames.items()}, axis=1)
    membership = _membership(closes.index).iloc[250]
    original = _target_weights(closes, membership, 250, _spec())
    closes.loc[closes.index[-1], "BBB"] *= 100
    changed = _target_weights(closes, membership, 250, _spec())
    assert original.equals(changed)


def test_cross_sectional_backtest_uses_prior_membership_and_benchmarks() -> None:
    frames = _frames()
    index = frames["AAA"].index
    membership = _membership(index)
    membership.loc[index[220]:, "AAA"] = False
    result = run_point_in_time_cross_sectional_momentum(
        frames,
        membership,
        _spec(),
        datetime(2020, 8, 3),
        datetime(2021, 3, 1),
    )
    assert (result.equity > 0).all()
    assert {"SPY", "QQQ"}.issubset(result.benchmark_metrics)
