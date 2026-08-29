"""Transparent scoring and risk-adjusted ranking."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from .regime import MarketRegime
from .schemas import Candidate, SourceReference


def _score(value: float, lower: float, upper: float) -> float:
    return float(np.clip((value - lower) / (upper - lower) * 100, 0, 100))


def score_candidate(ticker: str, features: pd.DataFrame, regime: MarketRegime) -> Candidate | None:
    """Score the latest completed bar; absent data produces no candidate, not invented values."""
    row = features.iloc[-1]
    required = ("technical_score", "relative_return_20d", "realized_volatility_20d", "atr_14")
    if row.loc[list(required)].isna().any():
        return None
    relative_strength = _score(float(row["relative_return_20d"]), -0.08, 0.12)
    volatility = float(row["realized_volatility_20d"])
    risk_score = 100 - _score(volatility, 0.15, 0.70)
    regime_fit_score = regime.fit_multiplier * 100
    technical_score = float(row["technical_score"])
    composite = (
        0.58 * technical_score
        + 0.22 * relative_strength
        + 0.12 * risk_score
        + 0.08 * regime_fit_score
    )
    as_of_timestamp = pd.Timestamp(features.index[-1]).to_pydatetime().replace(tzinfo=UTC)
    source = SourceReference(
        source_type="market_data",
        url=f"provider://ohlcv/{ticker}",
        retrieved_at=datetime.now(UTC),
        available_at=as_of_timestamp,
        description="Completed daily OHLCV bar used by deterministic feature pipeline.",
    )
    values: dict[str, float | None] = {
        "close": float(row["close"]),
        "return_20d": float(row["return_20d"]),
        "relative_return_20d": float(row["relative_return_20d"]),
        "rsi_14": float(row["rsi_14"]),
        "atr_14": float(row["atr_14"]),
        "realized_volatility_20d": volatility,
        "relative_volume_20d": float(row["relative_volume_20d"]),
    }
    return Candidate(
        ticker=ticker,
        as_of=as_of_timestamp,
        composite_score=round(float(composite), 2),
        technical_score=round(technical_score, 2),
        relative_strength_score=round(relative_strength, 2),
        risk_score=round(float(risk_score), 2),
        regime_fit_score=round(float(regime_fit_score), 2),
        regime=regime.name,
        holding_period_sessions=10,
        source=source,
        feature_values=values,
        limitations=[
            "This scoring strategy has not passed the benchmark promotion gate; "
            "classification is WATCH only.",
            "Fundamental, earnings, and news adapters are not part of the starter composite.",
            "The configured universe is not point-in-time historical membership.",
        ],
    )


def rank_candidates(candidates: Iterable[Candidate], limit: int) -> list[Candidate]:
    return sorted(candidates, key=lambda candidate: candidate.composite_score, reverse=True)[:limit]
