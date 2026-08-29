"""Explainable market-state classification rather than an opaque prediction."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MarketRegime:
    name: str
    fit_multiplier: float
    evidence: dict[str, float]


def detect_regime(market: pd.DataFrame) -> MarketRegime:
    close = market["close"]
    if len(close) < 200:
        return MarketRegime("insufficient_history", 0.5, {})
    sma_50 = close.rolling(50).mean().iloc[-1]
    sma_200 = close.rolling(200).mean().iloc[-1]
    return_20d = close.pct_change(20).iloc[-1]
    volatility = close.pct_change().rolling(20).std().iloc[-1] * (252**0.5)
    evidence = {
        "close_vs_sma_50": float(close.iloc[-1] / sma_50 - 1),
        "sma_50_vs_sma_200": float(sma_50 / sma_200 - 1),
        "return_20d": float(return_20d),
        "annualized_volatility_20d": float(volatility),
    }
    if volatility > 0.38:
        return MarketRegime("volatility_shock", 0.45, evidence)
    if close.iloc[-1] > sma_50 > sma_200 and return_20d > 0:
        return MarketRegime("strong_bull", 1.0, evidence)
    if close.iloc[-1] > sma_200:
        return MarketRegime("weak_bull", 0.85, evidence)
    if close.iloc[-1] < sma_50 < sma_200:
        return MarketRegime("bear", 0.35, evidence)
    return MarketRegime("sideways_or_correction", 0.60, evidence)
