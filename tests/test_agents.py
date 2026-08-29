from __future__ import annotations

import swing_research.agents as agents
from swing_research.pipeline import run_demo_research


def test_narrative_synthesis_skips_without_a_key() -> None:
    candidates, _, _ = run_demo_research()
    result = agents.synthesize_narratives(candidates, enabled=False)
    assert result.narratives == {}
    assert result.status == "DISABLED by config"


def test_unpromoted_strategy_cannot_issue_long_classification() -> None:
    candidates, _, _ = run_demo_research()
    assessments = agents.deterministic_assessments(candidates[0])
    assert assessments[-1].classification == "WATCH"


def test_narrative_synthesis_rejects_unsourced_numbers(monkeypatch: object) -> None:
    candidates, _, _ = run_demo_research()

    class _NumericNarrativeClient:
        model = "openrouter/free"

        def __init__(self, model: str | None = None) -> None:
            pass

        def summarize(self, candidate: object, assessments: object) -> str:
            return "This claim adds a made-up number: 42."

    monkeypatch.setattr(agents, "OpenRouterNarrativeClient", _NumericNarrativeClient)
    result = agents.synthesize_narratives(candidates, enabled=True)
    assert result.narratives == {}
    assert "rejected" in result.status
