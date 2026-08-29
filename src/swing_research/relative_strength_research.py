"""Predeclared parameter selection and untouched-period reporting for portfolio research."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from .backtesting import CostModel
from .portfolio_backtesting import RelativeStrengthSpec, run_relative_strength_portfolio


@dataclass(frozen=True)
class ResearchPeriod:
    name: str
    start: datetime
    end: datetime


@dataclass(frozen=True)
class SpecScore:
    spec: RelativeStrengthSpec
    selection_score: float
    metrics: dict[str, float | int | None]


@dataclass
class PredeclaredExperiment:
    selected: SpecScore
    validation: dict[str, Any]
    holdout: dict[str, Any]


@dataclass(frozen=True)
class RobustSpecScore:
    spec: RelativeStrengthSpec
    robustness_score: float
    period_scores: dict[str, float]
    worst_drawdown: float


@dataclass
class RobustExperiment:
    selected: RobustSpecScore
    validation: dict[str, Any]
    final_holdout: dict[str, Any]


def generate_specs(research: dict[str, Any]) -> list[RelativeStrengthSpec]:
    grid = research["parameter_grid"]
    return [
        RelativeStrengthSpec(
            momentum_lookback_sessions=momentum,
            trend_lookback_sessions=trend,
            volatility_lookback_sessions=volatility,
            rebalance_sessions=rebalance,
            top_n=top_n,
            risk_assets=tuple(research["risk_assets"]),
            defensive_asset=str(research["defensive_asset"]),
            defensive_assets=tuple(research.get("defensive_assets", [])),
        )
        for momentum, trend, volatility, rebalance, top_n in product(
            grid["momentum_lookback_sessions"],
            grid["trend_lookback_sessions"],
            grid["volatility_lookback_sessions"],
            grid["rebalance_sessions"],
            grid["top_n"],
        )
    ]


def _selection_score(metrics: dict[str, float | int | None]) -> float:
    """Favor risk-adjusted return and penalize material drawdown; never inspect future periods."""
    cagr = float(metrics["cagr"] or 0)
    max_drawdown = abs(float(metrics["maximum_drawdown"] or 0))
    sharpe = float(metrics["sharpe"] or -10)
    return cagr + 0.25 * sharpe - 0.50 * max_drawdown


def select_on_training(
    frames: dict[str, pd.DataFrame],
    specs: list[RelativeStrengthSpec],
    train: ResearchPeriod,
    costs: CostModel,
) -> SpecScore:
    scored: list[SpecScore] = []
    for spec in specs:
        result = run_relative_strength_portfolio(frames, spec, train.start, train.end, costs)
        scored.append(SpecScore(spec, _selection_score(result.metrics), result.metrics))
    if not scored:
        raise ValueError("No parameter specifications supplied")
    return max(scored, key=lambda result: result.selection_score)


def run_predeclared_experiment(
    frames: dict[str, pd.DataFrame],
    research: dict[str, Any],
    train: ResearchPeriod,
    validation: ResearchPeriod,
    holdout: ResearchPeriod,
    costs: CostModel,
) -> PredeclaredExperiment:
    """Choose once on train, then inspect validation and holdout without further tuning."""
    selected = select_on_training(frames, generate_specs(research), train, costs)
    validation_result = run_relative_strength_portfolio(
        frames, selected.spec, validation.start, validation.end, costs
    )
    holdout_result = run_relative_strength_portfolio(
        frames, selected.spec, holdout.start, holdout.end, costs
    )
    return PredeclaredExperiment(
        selected,
        {"metrics": validation_result.metrics, "benchmarks": validation_result.benchmark_metrics},
        {"metrics": holdout_result.metrics, "benchmarks": holdout_result.benchmark_metrics},
    )


def experiment_as_dict(experiment: PredeclaredExperiment) -> dict[str, Any]:
    return {
        "selected_on_training": {
            "spec": asdict(experiment.selected.spec),
            "selection_score": experiment.selected.selection_score,
            "metrics": experiment.selected.metrics,
        },
        "validation": experiment.validation,
        "holdout": experiment.holdout,
    }


def _benchmark_relative_score(benchmarks: dict[str, dict[str, float | int | None]]) -> float:
    """Use the weaker SPY/QQQ comparison to avoid selecting a single-benchmark winner."""
    multiples = [
        float(metrics["terminal_wealth_multiple"] or 0)
        for ticker, metrics in benchmarks.items()
        if ticker in {"SPY", "QQQ"}
    ]
    return min(multiples) if multiples else 0.0


def select_across_development_periods(
    frames: dict[str, pd.DataFrame],
    specs: list[RelativeStrengthSpec],
    periods: list[ResearchPeriod],
    costs: CostModel,
) -> RobustSpecScore:
    """Select by median relative terminal wealth across independent development periods."""
    if not periods:
        raise ValueError("At least one development period is required")
    scores: list[RobustSpecScore] = []
    for spec in specs:
        period_scores: dict[str, float] = {}
        drawdowns: list[float] = []
        for period in periods:
            result = run_relative_strength_portfolio(frames, spec, period.start, period.end, costs)
            period_scores[period.name] = _benchmark_relative_score(result.benchmark_metrics)
            drawdowns.append(abs(float(result.metrics["maximum_drawdown"] or 0)))
        median_relative = float(np.median(list(period_scores.values())))
        worst_drawdown = max(drawdowns)
        scores.append(
            RobustSpecScore(
                spec,
                median_relative - 0.20 * worst_drawdown,
                period_scores,
                worst_drawdown,
            )
        )
    return max(scores, key=lambda score: score.robustness_score)


def run_robust_experiment(
    frames: dict[str, pd.DataFrame],
    research: dict[str, Any],
    development_periods: list[ResearchPeriod],
    validation: ResearchPeriod,
    final_holdout: ResearchPeriod,
    costs: CostModel,
) -> RobustExperiment:
    """Select only on development folds, then leave validation and holdout unchanged."""
    selected = select_across_development_periods(
        frames,
        generate_specs(research),
        development_periods,
        costs,
    )
    validation_result = run_relative_strength_portfolio(
        frames, selected.spec, validation.start, validation.end, costs
    )
    final_result = run_relative_strength_portfolio(
        frames, selected.spec, final_holdout.start, final_holdout.end, costs
    )
    return RobustExperiment(
        selected,
        {"metrics": validation_result.metrics, "benchmarks": validation_result.benchmark_metrics},
        {"metrics": final_result.metrics, "benchmarks": final_result.benchmark_metrics},
    )


def robust_experiment_as_dict(experiment: RobustExperiment) -> dict[str, Any]:
    return {
        "selected_across_development_periods": {
            "spec": asdict(experiment.selected.spec),
            "robustness_score": experiment.selected.robustness_score,
            "period_scores": experiment.selected.period_scores,
            "worst_drawdown": experiment.selected.worst_drawdown,
        },
        "validation": experiment.validation,
        "final_holdout": experiment.final_holdout,
    }
