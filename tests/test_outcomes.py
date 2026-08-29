from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from swing_research.backtesting import CostModel
from swing_research.outcomes import evaluate_signal_outcome
from swing_research.schemas import PaperPrediction
from swing_research.storage import PaperLedger, PredictionOutcome


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0, 13.0],
            "high": [11.0, 12.0, 13.0, 14.0],
            "low": [9.0, 10.0, 11.0, 12.0],
            "close": [10.5, 11.5, 12.5, 13.5],
            "volume": [100, 100, 100, 100],
        },
        index=pd.date_range("2024-01-02", periods=4, freq="B"),
    )


def test_signal_outcome_enters_next_open_and_waits_for_complete_horizon() -> None:
    outcome = evaluate_signal_outcome(
        _bars(), datetime(2024, 1, 2, tzinfo=UTC), holding_period_sessions=2, costs=CostModel()
    )
    assert outcome is not None
    assert outcome.entry_at.date().isoformat() == "2024-01-03"
    assert outcome.evaluated_at.date().isoformat() == "2024-01-04"
    assert outcome.gross_return == 12.5 / 11 - 1
    assert evaluate_signal_outcome(
        _bars(), datetime(2024, 1, 2, tzinfo=UTC), holding_period_sessions=4
    ) is None


def test_ledger_persists_pending_prediction_and_one_outcome(tmp_path: Path) -> None:
    ledger = PaperLedger(tmp_path / "paper.sqlite3")
    prediction_id = ledger.record_prediction(
        PaperPrediction(
            ticker="ABC",
            predicted_at=datetime(2024, 1, 2, tzinfo=UTC),
            holding_period_sessions=2,
            composite_score=70,
            model_version="test-v1",
            feature_version="test-v1",
            payload={},
        )
    )
    assert [item.id for item in ledger.pending_predictions()] == [prediction_id]
    ledger.record_prediction_outcome(
        PredictionOutcome(
            prediction_id=prediction_id,
            entry_at=datetime(2024, 1, 3, tzinfo=UTC),
            entry_price=11,
            evaluated_at=datetime(2024, 1, 4, tzinfo=UTC),
            evaluated_price=12.5,
            gross_return=12.5 / 11 - 1,
            net_return=12.5 / 11 - 1,
            spy_return=0.01,
            qqq_return=0.02,
        )
    )
    assert ledger.pending_predictions() == []
    assert ledger.outcome_count() == 1
    ledger.close()
