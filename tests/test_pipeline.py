from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from swing_research.data import YFinancePriceProvider
from swing_research.pipeline import run_daily_research, run_demo_research


def test_demo_pipeline_has_cited_candidates() -> None:
    candidates, regime, report = run_demo_research()
    assert candidates
    assert regime.name
    assert candidates[0].source.source_type == "market_data"
    assert "Morning swing-trading research" in report


def test_yfinance_private_research_cannot_run_in_github_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    with pytest.raises(RuntimeError, match="private local research"):
        run_daily_research(
            Path("config"),
            provider=YFinancePriceProvider(),
            now=datetime(2026, 8, 30, tzinfo=UTC),
        )
