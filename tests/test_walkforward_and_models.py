from __future__ import annotations

import pandas as pd

from swing_research.features import build_technical_features
from swing_research.models import add_forward_label, choose_challenger, evaluate_models
from swing_research.pipeline import make_demo_frames
from swing_research.walkforward import expanding_walk_forward_splits


def test_walk_forward_purges_label_overlap() -> None:
    index = pd.bdate_range("2020-01-02", periods=500)
    folds = expanding_walk_forward_splits(
        index,
        train_sessions=200,
        validation_sessions=60,
        test_sessions=60,
        label_horizon_sessions=5,
    )
    assert folds
    for fold in folds:
        assert fold.train_index[-1] < fold.validation_index[0]
        assert (fold.validation_index[0] - fold.train_index[-1]).days >= 5
        assert fold.validation_index[-1] < fold.test_index[0]


def test_model_evaluation_uses_untouched_folds() -> None:
    frames = make_demo_frames(periods=500)
    features = build_technical_features(frames["MSFT"], frames["SPY"]["close"])
    labeled = add_forward_label(features)
    folds = expanding_walk_forward_splits(
        labeled.index,
        train_sessions=220,
        validation_sessions=50,
        test_sessions=50,
        label_horizon_sessions=5,
    )
    results = evaluate_models(labeled, folds)
    assert results
    assert all(0 <= item.brier_score <= 1 for item in results)
    assert choose_challenger(results).decision in {"review_candidate", "retain_champion"}
