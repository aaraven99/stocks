"""Backward-looking technical features only."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    difference = close.diff()
    gains = difference.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    losses = (
        (-difference.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    )
    relative_strength = gains / losses.replace(0, np.nan)
    return 100 - (100 / (1 + relative_strength))


def build_technical_features(
    ohlcv: pd.DataFrame, benchmark: pd.Series | None = None
) -> pd.DataFrame:
    """Compute rolling features; no centered windows, forward fills, or future observations."""
    frame = ohlcv.copy()
    close = frame["close"]
    returns = close.pct_change()
    frame["return_1d"] = returns
    for window in (3, 5, 10, 20, 60, 120):
        frame[f"return_{window}d"] = close.pct_change(window)
    frame["sma_20"] = close.rolling(20, min_periods=20).mean()
    frame["sma_50"] = close.rolling(50, min_periods=50).mean()
    frame["rsi_14"] = _rsi(close)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - close.shift()).abs(),
            (frame["low"] - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["atr_14"] = true_range.rolling(14, min_periods=14).mean()
    frame["realized_volatility_20d"] = returns.rolling(20, min_periods=20).std() * np.sqrt(252)
    frame["relative_volume_20d"] = (
        frame["volume"] / frame["volume"].rolling(20, min_periods=20).mean()
    )
    frame["distance_from_60d_high"] = close / close.rolling(60, min_periods=60).max() - 1
    if benchmark is not None:
        benchmark_return = benchmark.reindex(frame.index).pct_change(20)
        frame["relative_return_20d"] = frame["return_20d"] - benchmark_return
    else:
        frame["relative_return_20d"] = np.nan

    components = pd.DataFrame(
        {
            "momentum": ((frame["return_20d"] + 0.10) / 0.30).clip(0, 1),
            "relative_strength": ((frame["relative_return_20d"] + 0.08) / 0.20).clip(0, 1),
            "trend": ((close / frame["sma_50"] - 0.92) / 0.20).clip(0, 1),
            "rsi": (1 - (frame["rsi_14"] - 60).abs() / 40).clip(0, 1),
            "volume": ((frame["relative_volume_20d"] - 0.75) / 0.75).clip(0, 1),
            "pullback": (1 - frame["distance_from_60d_high"].abs() / 0.25).clip(0, 1),
        }
    )
    weights = np.array([0.28, 0.22, 0.20, 0.12, 0.10, 0.08])
    frame["technical_score"] = components.mul(weights, axis=1).sum(axis=1, min_count=6) * 100
    return frame
