"""Auditable Markdown reports with explicit data limitations."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .agents import deterministic_assessments
from .regime import MarketRegime
from .schemas import Candidate


def _format_candidate(position: int, candidate: Candidate) -> str:
    assessments = deterministic_assessments(candidate)
    final = assessments[-1]
    values = candidate.feature_values
    citations = ", ".join(sorted({evidence.sources[0].url for evidence in final.evidence}))
    return "\n".join(
        [
            f"### {position}. {candidate.ticker} — {final.classification}",
            f"Composite swing score: **{candidate.composite_score:.1f}/100**  ",
            f"Confidence: **{final.confidence:.0%}** "
            f"(heuristic evidence quality: {final.evidence_quality})  ",
            f"Expected holding period: {candidate.holding_period_sessions} sessions",
            "",
            "| Component | Score |",
            "| --- | ---: |",
            f"| Technical | {candidate.technical_score:.1f} |",
            f"| Relative strength | {candidate.relative_strength_score:.1f} |",
            f"| Risk | {candidate.risk_score:.1f} |",
            f"| Regime fit | {candidate.regime_fit_score:.1f} |",
            "",
            f"Completed-bar 20-day return: {values['return_20d']:.2%}; relative 20-day return: "
            f"{values['relative_return_20d']:.2%}; RSI(14): {values['rsi_14']:.1f}; "
            f"20-day realized volatility: {values['realized_volatility_20d']:.1%}.",
            f"Source references: {citations}.",
            "",
            "Cautions: " + " ".join(f"{caution}" for caution in final.cautions),
        ]
    )


def render_morning_report(candidates: list[Candidate], regime: MarketRegime) -> str:
    timestamp = datetime.now(UTC).isoformat()
    candidates_section = "\n\n".join(
        _format_candidate(position, candidate)
        for position, candidate in enumerate(candidates, start=1)
    )
    if not candidates_section:
        candidates_section = "No candidate met history and data-quality requirements."
    return "\n".join(
        [
            "# Morning swing-trading research",
            "",
            f"Generated at: `{timestamp}`",
            f"Market regime: **{regime.name}**",
            "",
            "## Top long candidates",
            "",
            candidates_section,
            "",
            "## Data status",
            "",
            "- Market data: completed daily OHLCV required; "
            "the pipeline fails on stale critical data.",
            "- SEC/news/fundamental data: excluded from the starter composite "
            "until a validated adapter is enabled.",
            "- Execution: research ranking only. No brokerage orders are connected.",
            "",
            "## Research disclaimer",
            "",
            "This report is for research and paper simulation. It is not investment advice and "
            "does not predict or guarantee returns.",
        ]
    )


def write_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_equity_curve_svg(equity: pd.Series, path: Path) -> None:
    """Write a dependency-free equity curve chart for a reproducible backtest artifact."""
    width, height, padding = 900, 320, 42
    values = equity.astype(float)
    lower, upper = float(values.min()), float(values.max())
    spread = max(upper - lower, max(upper, 1.0) * 0.01)
    points = []
    for position, value in enumerate(values):
        x = padding + position / max(len(values) - 1, 1) * (width - 2 * padding)
        y = height - padding - (value - lower) / spread * (height - 2 * padding)
        points.append(f"{x:.1f},{y:.1f}")
    svg_header = (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" role="img">'
    )
    polyline = (
        '<polyline points="'
        f'{" ".join(points)}'
        '" fill="none" stroke="#36d399" stroke-width="2"/>'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    chart = "\n".join(
        [
            svg_header,
            '<rect width="100%" height="100%" fill="#0b1020"/>',
            f'<path d="M {padding} {padding} V {height - padding} H {width - padding}" '
            'stroke="#667085" fill="none"/>',
            polyline,
            f'<text x="{padding}" y="24" fill="#f8fafc" font-family="sans-serif" font-size="16">'
            "Strategy equity curve</text>",
            f'<text x="{padding}" y="{height - 12}" fill="#a5b4fc" font-family="sans-serif" '
            f'font-size="12">Start ${values.iloc[0]:,.0f}</text>',
            f'<text x="{width - 180}" y="{height - 12}" fill="#a5b4fc" font-family="sans-serif" '
            f'font-size="12">End ${values.iloc[-1]:,.0f}</text>',
            "</svg>",
        ]
    )
    path.write_text(chart, encoding="utf-8")
