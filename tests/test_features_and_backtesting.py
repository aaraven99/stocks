from __future__ import annotations

import numpy as np
import pandas as pd

from swing_research.backtesting import run_long_only_backtest
from swing_research.features import build_technical_features


def _bars(periods: int = 160) -> pd.DataFrame:
    index = pd.bdate_range("2023-01-03", periods=periods)
    close = np.linspace(100, 140, periods)
    return pd.DataFrame(
        {
            "open": close * 0.998,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000,
        },
        index=index,
    )


def test_future_bar_cannot_change_prior_features() -> None:
    original = _bars()
    changed = _bars()
    changed.iloc[-1, changed.columns.get_loc("close")] = 10_000.0
    changed.iloc[-1, changed.columns.get_loc("high")] = 10_100.0
    before = build_technical_features(original)
    after = build_technical_features(changed)
    pd.testing.assert_series_equal(before.iloc[-2], after.iloc[-2])


def test_signal_executes_at_next_open_not_same_close() -> None:
    bars = _bars(8)
    signal = pd.Series(False, index=bars.index)
    signal.iloc[0] = True
    result = run_long_only_backtest(bars, signal, holding_period_sessions=5)
    assert len(result.trades) == 1
    assert result.trades[0].entry_time == bars.index[1]
    assert result.trades[0].exit_time == bars.index[2]


def test_last_bar_signal_does_not_create_an_unfillable_trade() -> None:
    bars = _bars(8)
    signal = pd.Series(False, index=bars.index)
    signal.iloc[-1] = True
    result = run_long_only_backtest(bars, signal)
    assert result.trades == []
