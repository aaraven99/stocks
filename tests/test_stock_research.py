from __future__ import annotations

from swing_research.stock_research import generate_stock_specs


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
