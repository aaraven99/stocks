from __future__ import annotations

from datetime import UTC, datetime

from swing_research.market_calendar import should_start_daily_workflow
from swing_research.schemas import PaperPrediction
from swing_research.storage import PaperLedger


def test_prediction_ledger_is_idempotent(tmp_path: object) -> None:
    path = tmp_path / "ledger.sqlite3"  # type: ignore[operator]
    ledger = PaperLedger(path)
    prediction = PaperPrediction(
        ticker="MSFT",
        predicted_at=datetime(2026, 1, 5, tzinfo=UTC),
        holding_period_sessions=10,
        composite_score=70,
        model_version="v1",
        feature_version="f1",
        payload={"source": "test"},
    )
    ledger.record_prediction(prediction)
    ledger.record_prediction(prediction)
    assert ledger.prediction_count() == 1
    ledger.close()


def test_dst_schedule_gate_accepts_both_central_offsets() -> None:
    assert should_start_daily_workflow(datetime(2026, 1, 5, 11, tzinfo=UTC))
    assert should_start_daily_workflow(datetime(2026, 7, 6, 10, tzinfo=UTC))
    assert not should_start_daily_workflow(datetime(2026, 7, 6, 11, tzinfo=UTC))
