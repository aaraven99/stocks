"""Multi-fold selection for point-in-time cross-sectional stock research."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .backtesting import CostModel
from .relative_strength_research import ResearchPeriod
from .stock_backtesting import (
    CrossSectionalMomentumSpec,
    run_point_in_time_cross_sectional_momentum,
)


@dataclass(frozen=True)
class StockRobustSpecScore:
    spec: CrossSectionalMomentumSpec
    robustness_score: float
    period_scores: dict[str, float]
    worst_drawdown: float


@dataclass(frozen=True)
class StockRobustExperiment:
    selected: StockRobustSpecScore
    validation: dict[str, Any]
    final_holdout: dict[str, Any]


def generate_stock_specs(research: dict[str, Any]) -> list[CrossSectionalMomentumSpec]:
    grid = research["parameter_grid"]
    return [
        CrossSectionalMomentumSpec(momentum, trend, volatility, rebalance, top_n)
        for momentum in grid["momentum_lookback_sessions"]
        for trend in grid["trend_lookback_sessions"]
        for volatility in grid["volatility_lookback_sessions"]
        for rebalance in grid["rebalance_sessions"]
        for top_n in grid["top_n"]
    ]


def _weaker_benchmark_multiple(
    result_benchmarks: dict[str, dict[str, float | int | None]],
) -> float:
    multiples = [
        float(metrics["terminal_wealth_multiple"] or 0)
        for ticker, metrics in result_benchmarks.items()
        if ticker in {"SPY", "QQQ"}
    ]
    if len(multiples) != 2:
        raise ValueError("Both SPY and QQQ benchmark comparisons are required")
    return min(multiples)


def select_stock_spec_across_development_periods(
    frames: dict[str, pd.DataFrame],
    membership: pd.DataFrame,
    specs: list[CrossSectionalMomentumSpec],
    periods: list[ResearchPeriod],
    costs: CostModel,
) -> StockRobustSpecScore:
    """Select by median weaker-benchmark performance before later periods are observed."""
    if not specs or not periods:
        raise ValueError("At least one specification and development period are required")
    scored: list[StockRobustSpecScore] = []
    for spec in specs:
        period_scores: dict[str, float] = {}
        drawdowns: list[float] = []
        for period in periods:
            result = run_point_in_time_cross_sectional_momentum(
                frames, membership, spec, period.start, period.end, costs
            )
            period_scores[period.name] = _weaker_benchmark_multiple(result.benchmark_metrics)
            drawdowns.append(abs(float(result.metrics["maximum_drawdown"] or 0)))
        worst_drawdown = max(drawdowns)
        scored.append(
            StockRobustSpecScore(
                spec,
                float(np.median(list(period_scores.values()))) - 0.20 * worst_drawdown,
                period_scores,
                worst_drawdown,
            )
        )
    return max(scored, key=lambda score: score.robustness_score)


def run_robust_stock_experiment(
    frames: dict[str, pd.DataFrame],
    membership: pd.DataFrame,
    research: dict[str, Any],
    development_periods: list[ResearchPeriod],
    validation: ResearchPeriod,
    final_holdout: ResearchPeriod,
    costs: CostModel,
) -> StockRobustExperiment:
    """Select once on development data and leave validation/final stock periods untouched."""
    selected = select_stock_spec_across_development_periods(
        frames, membership, generate_stock_specs(research), development_periods, costs
    )
    validation_result = run_point_in_time_cross_sectional_momentum(
        frames, membership, selected.spec, validation.start, validation.end, costs
    )
    final_result = run_point_in_time_cross_sectional_momentum(
        frames, membership, selected.spec, final_holdout.start, final_holdout.end, costs
    )
    return StockRobustExperiment(
        selected,
        {"metrics": validation_result.metrics, "benchmarks": validation_result.benchmark_metrics},
        {"metrics": final_result.metrics, "benchmarks": final_result.benchmark_metrics},
    )


def stock_experiment_as_dict(experiment: StockRobustExperiment) -> dict[str, Any]:
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
