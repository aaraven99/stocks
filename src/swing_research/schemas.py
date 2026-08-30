"""Validated, auditable records shared by data, agents, and reporting."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SourceReference(BaseModel):
    """Identifies the origin and availability time of an input."""

    model_config = ConfigDict(frozen=True)

    source_type: str
    url: str
    retrieved_at: datetime
    available_at: datetime
    description: str


class Evidence(BaseModel):
    """A claim whose numeric values must have source references."""

    agent: str
    claim: str
    values: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    sources: list[SourceReference] = Field(min_length=1)


class AgentAssessment(BaseModel):
    ticker: str
    classification: str
    confidence: float = Field(ge=0, le=1)
    evidence_quality: str
    evidence: list[Evidence] = Field(min_length=1)
    cautions: list[str] = Field(default_factory=list)


class Candidate(BaseModel):
    ticker: str
    as_of: datetime
    composite_score: float = Field(ge=0, le=100)
    technical_score: float = Field(ge=0, le=100)
    relative_strength_score: float = Field(ge=0, le=100)
    risk_score: float = Field(ge=0, le=100)
    regime_fit_score: float = Field(ge=0, le=100)
    regime: str
    holding_period_sessions: int = Field(ge=1)
    source: SourceReference
    feature_values: dict[str, float | None]
    limitations: list[str] = Field(default_factory=list)


class PaperPrediction(BaseModel):
    ticker: str
    predicted_at: datetime
    holding_period_sessions: int
    composite_score: float
    model_version: str
    feature_version: str
    payload: dict[str, Any]


class NewsArticle(BaseModel):
    """A timestamped external news record kept outside quantitative scoring."""

    article_id: int | str
    ticker: str
    published_at: datetime
    retrieved_at: datetime
    source: str
    headline: str
    summary: str
    url: str
