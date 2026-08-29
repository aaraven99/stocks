from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

import swing_research.stock_research as stock_research
from swing_research.backtesting import CostModel
from swing_research.relative_strength_research import ResearchPeriod
from swing_research.stock_backtesting import CrossSectionalMomentumSpec
from swing_research.stock_research import (
    StockRobustSpecScore,
    generate_stock_specs,
    run_robust_stock_experiment,
)


def test_stock_grid_expands_all_predeclared_parameters() -> None:
    research = {
        "parameter_grid": {
            "momentum_lookback_sessions": [63, 126],
            "trend_lookback_sessions": [200],
            "volatility_lookback_sessions": [21],
            "rebalance_sessions": [5, 20],
            "top_n": [10, 20],
        }
    }
    specs = generate_stock_specs(research)
    assert len(specs) == 8
    assert {spec.top_n for spec in specs} == {10, 20}


def test_cost_stress_reuses_the_development_selected_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = CrossSectionalMomentumSpec(126, 200, 63, 5, 10)
    selected = StockRobustSpecScore(spec, 1.0, {"development": 1.0}, 0.1)
    calls: list[CostModel] = []

    class _Result:
        metrics = {"maximum_drawdown": -0.1}
        benchmark_metrics = {
            "SPY": {"terminal_wealth_multiple": 1.1},
            "QQQ": {"terminal_wealth_multiple": 1.1},
        }

    def _fake_select(*args: object, **kwargs: object) -> StockRobustSpecScore:
        return selected

    def _fake_backtest(
        frames: object,
        membership: object,
        received_spec: CrossSectionalMomentumSpec,
        start: object,
        end: object,
        costs: CostModel,
    ) -> _Result:
        assert received_spec == spec
        calls.append(costs)
        return _Result()

    monkeypatch.setattr(
        stock_research, "select_stock_spec_across_development_periods", _fake_select
    )
    monkeypatch.setattr(
        stock_research, "run_point_in_time_cross_sectional_momentum", _fake_backtest
    )
    period = ResearchPeriod("period", datetime(2020, 1, 1), datetime(2020, 12, 31))
    experiment = run_robust_stock_experiment(
        {},
        pd.DataFrame(),
        {
            "parameter_grid": {
                "momentum_lookback_sessions": [126],
                "trend_lookback_sessions": [200],
                "volatility_lookback_sessions": [63],
                "rebalance_sessions": [5],
                "top_n": [10],
            }
        },
        [period],
        period,
        period,
        CostModel(half_spread_bps=2.0, slippage_bps=5.0),
    )
    assert set(experiment.cost_stress) == {"1x", "2x"}
    assert calls[-1].half_spread_bps == 4.0
    assert calls[-1].slippage_bps == 10.0
