from __future__ import annotations

from swing_research.pipeline import run_demo_research


def test_demo_pipeline_has_cited_candidates() -> None:
    candidates, regime, report = run_demo_research()
    assert candidates
    assert regime.name
    assert candidates[0].source.source_type == "market_data"
    assert "Morning swing-trading research" in report
