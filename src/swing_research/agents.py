"""Structured evidence agents; optional LLM prose is non-decisioning."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import requests

from .schemas import AgentAssessment, Candidate, Evidence, SourceReference


@dataclass(frozen=True)
class NarrativeSynthesis:
    narratives: dict[str, str]
    status: str


def deterministic_assessments(candidate: Candidate) -> list[AgentAssessment]:
    """Produce source-linked technical, risk, bull, bear, and final assessments."""
    technical = Evidence(
        agent="technical",
        claim="Technical score and momentum values are calculated from completed OHLCV bars.",
        values={
            "technical_score": candidate.technical_score,
            "return_20d": candidate.feature_values["return_20d"],
            "relative_return_20d": candidate.feature_values["relative_return_20d"],
            "rsi_14": candidate.feature_values["rsi_14"],
        },
        sources=[candidate.source],
    )
    risk = Evidence(
        agent="risk",
        claim=(
            "Risk score is based on realized price volatility; it is not a forecast "
            "of maximum loss."
        ),
        values={
            "risk_score": candidate.risk_score,
            "realized_volatility_20d": candidate.feature_values["realized_volatility_20d"],
            "atr_14": candidate.feature_values["atr_14"],
        },
        sources=[candidate.source],
    )
    bull = Evidence(
        agent="bull",
        claim=(
            "The bullish case is limited to the completed-bar technical and "
            "relative-strength evidence."
        ),
        values={
            "technical_score": candidate.technical_score,
            "relative_strength_score": candidate.relative_strength_score,
            "composite_score": candidate.composite_score,
        },
        sources=[candidate.source],
    )
    bear = Evidence(
        agent="bear",
        claim=(
            "The bearish case emphasizes realized volatility and evidence the starter system "
            "does not yet cover."
        ),
        values={
            "risk_score": candidate.risk_score,
            "realized_volatility_20d": candidate.feature_values["realized_volatility_20d"],
        },
        sources=[candidate.source],
    )
    regime_source = SourceReference(
        source_type="strategy_config",
        url="config://strategy.yaml",
        retrieved_at=datetime.now(UTC),
        available_at=candidate.as_of,
        description="Configured deterministic scoring weights and long-only policy.",
    )
    regime = Evidence(
        agent="regime",
        claim="Regime fit adjusts ranking exposure; it does not predict future market direction.",
        values={"regime": candidate.regime, "regime_fit_score": candidate.regime_fit_score},
        sources=[candidate.source, regime_source],
    )
    classification = "WATCH"
    if (
        candidate.composite_score >= 75
        and candidate.regime_fit_score >= 70
        and candidate.risk_score >= 45
    ):
        classification = "LONG CANDIDATE"
    if (
        candidate.composite_score >= 85
        and candidate.regime_fit_score >= 85
        and candidate.risk_score >= 60
    ):
        classification = "STRONG LONG CANDIDATE"
    final = Evidence(
        agent="final_decision",
        claim=(
            "Final classification is derived from fixed score and risk thresholds, "
            "not an LLM opinion."
        ),
        values={"classification": classification, "composite_score": candidate.composite_score},
        sources=[candidate.source, regime_source],
    )
    confidence = min(0.90, max(0.20, candidate.composite_score / 125))
    common_cautions = [
        "Research-only classification; no brokerage order is produced.",
        *candidate.limitations,
    ]
    return [
        AgentAssessment(
            ticker=candidate.ticker,
            classification="TECHNICAL",
            confidence=confidence,
            evidence_quality="MEDIUM",
            evidence=[technical],
        ),
        AgentAssessment(
            ticker=candidate.ticker,
            classification="RISK REVIEW",
            confidence=1 - confidence / 3,
            evidence_quality="MEDIUM",
            evidence=[risk],
            cautions=common_cautions,
        ),
        AgentAssessment(
            ticker=candidate.ticker,
            classification="BULL CASE",
            confidence=confidence,
            evidence_quality="MEDIUM",
            evidence=[bull],
        ),
        AgentAssessment(
            ticker=candidate.ticker,
            classification="BEAR CASE",
            confidence=1 - confidence / 3,
            evidence_quality="MEDIUM",
            evidence=[bear],
            cautions=common_cautions,
        ),
        AgentAssessment(
            ticker=candidate.ticker,
            classification="REGIME REVIEW",
            confidence=confidence,
            evidence_quality="MEDIUM",
            evidence=[regime],
        ),
        AgentAssessment(
            ticker=candidate.ticker,
            classification=classification,
            confidence=confidence,
            evidence_quality="MEDIUM",
            evidence=[technical, bull, bear, risk, regime, final],
            cautions=common_cautions,
        ),
    ]


class OpenRouterNarrativeClient:
    """Optional prose summarizer. Its output never enters scoring or trade generation."""

    endpoint = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        configured_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model or os.getenv("OPENROUTER_MODEL", "openrouter/free")
        if not configured_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is required only for optional narrative synthesis"
            )
        self.api_key: str = configured_key

    def summarize(self, candidate: Candidate, assessments: list[AgentAssessment]) -> str:
        evidence = [assessment.model_dump(mode="json") for assessment in assessments]
        prompt = (
            "Summarize only the supplied, sourced research evidence. Do not add values, price "
            "targets, or investment advice. State missing evidence plainly.\n\n"
            f"Candidate: {candidate.model_dump(mode='json')}\nEvidence: {evidence}"
        )
        response = requests.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            },
            timeout=45,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        try:
            return str(payload["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("OpenRouter returned an unexpected response shape") from exc


def synthesize_narratives(
    candidates: list[Candidate], enabled: bool, model: str | None = None
) -> NarrativeSynthesis:
    """Optionally produce non-numeric prose; failures never affect the quantitative run."""
    if not enabled:
        return NarrativeSynthesis({}, "DISABLED by config")
    try:
        client = OpenRouterNarrativeClient(model=model)
    except RuntimeError:
        return NarrativeSynthesis({}, "UNAVAILABLE: OPENROUTER_API_KEY is not configured")
    narratives: dict[str, str] = {}
    rejected = 0
    failed = 0
    for candidate in candidates:
        try:
            summary = client.summarize(candidate, deterministic_assessments(candidate)).strip()
        except (requests.RequestException, RuntimeError):
            failed += 1
            continue
        if not summary or re.search(r"\d", summary):
            rejected += 1
            continue
        narratives[candidate.ticker] = summary
    if failed:
        return NarrativeSynthesis(
            narratives, f"PARTIAL: {failed} OpenRouter narrative request(s) failed"
        )
    if rejected:
        return NarrativeSynthesis(
            narratives, f"PARTIAL: {rejected} narrative(s) rejected for containing numbers"
        )
    return NarrativeSynthesis(narratives, f"ACTIVE via {client.model}; prose cannot alter scores")
