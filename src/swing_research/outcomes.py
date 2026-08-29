"""Reconcile stored signals into next-open paper outcomes without revising predictions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pandas as pd

from .backtesting import CostModel
from .data import PriceProvider, validate_ohlcv
from .storage import PaperLedger, PredictionOutcome


@dataclass(frozen=True)
class EvaluatedSignal:
    entry_at: datetime
    entry_price: float
    evaluated_at: datetime
    evaluated_price: float
    gross_return: float
    net_return: float


def evaluate_signal_outcome(
    ohlcv: pd.DataFrame,
    predicted_at: datetime,
    holding_period_sessions: int,
    costs: CostModel | None = None,
) -> EvaluatedSignal | None:
    """Enter at the first later open and evaluate at the close of the configured session count."""
    if holding_period_sessions < 1:
        raise ValueError("holding_period_sessions must be positive")
    bars = validate_ohlcv(ohlcv)
    cutoff = pd.Timestamp(predicted_at).tz_localize(None).normalize()
    future = bars.loc[bars.index > cutoff]
    if len(future) < holding_period_sessions:
        return None
    effective_costs = costs or CostModel()
    entry_row = future.iloc[0]
    exit_row = future.iloc[holding_period_sessions - 1]
    entry_price = float(entry_row["open"] * (1 + effective_costs.one_way_fraction))
    exit_price = float(exit_row["close"] * (1 - effective_costs.one_way_fraction))
    gross_return = float(exit_row["close"] / entry_row["open"] - 1)
    return EvaluatedSignal(
        entry_at=pd.Timestamp(future.index[0]).to_pydatetime().replace(tzinfo=UTC),
        entry_price=entry_price,
        evaluated_at=pd.Timestamp(future.index[holding_period_sessions - 1])
        .to_pydatetime()
        .replace(tzinfo=UTC),
        evaluated_price=exit_price,
        gross_return=gross_return,
        net_return=float(exit_price / entry_price - 1),
    )


def reconcile_prediction_outcomes(
    ledger: PaperLedger,
    provider: PriceProvider,
    as_of: datetime,
    costs: CostModel | None = None,
) -> int:
    """Persist only fully observable outcomes; incomplete horizons remain pending."""
    reconciled = 0
    benchmark_cache: dict[str, pd.DataFrame] = {}
    for prediction in ledger.pending_predictions():
        start = prediction.predicted_at - timedelta(days=7)
        candidate_bars = provider.fetch_daily(prediction.ticker, start, as_of)
        evaluated = evaluate_signal_outcome(
            candidate_bars,
            prediction.predicted_at,
            prediction.holding_period_sessions,
            costs,
        )
        if evaluated is None:
            continue
        benchmark_returns: dict[str, float | None] = {"SPY": None, "QQQ": None}
        for benchmark in benchmark_returns:
            if benchmark not in benchmark_cache:
                benchmark_cache[benchmark] = provider.fetch_daily(benchmark, start, as_of)
            benchmark_evaluation = evaluate_signal_outcome(
                benchmark_cache[benchmark],
                prediction.predicted_at,
                prediction.holding_period_sessions,
                costs,
            )
            if benchmark_evaluation is not None:
                benchmark_returns[benchmark] = benchmark_evaluation.net_return
        ledger.record_prediction_outcome(
            PredictionOutcome(
                prediction_id=prediction.id,
                entry_at=evaluated.entry_at,
                entry_price=evaluated.entry_price,
                evaluated_at=evaluated.evaluated_at,
                evaluated_price=evaluated.evaluated_price,
                gross_return=evaluated.gross_return,
                net_return=evaluated.net_return,
                spy_return=benchmark_returns["SPY"],
                qqq_return=benchmark_returns["QQQ"],
            )
        )
        reconciled += 1
    return reconciled
