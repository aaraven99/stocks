"""Baselines and challenger selection that cannot promote models automatically."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from .walkforward import WalkForwardFold

FEATURE_COLUMNS = (
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "relative_return_20d",
    "rsi_14",
    "realized_volatility_20d",
    "relative_volume_20d",
    "distance_from_60d_high",
)


@dataclass(frozen=True)
class FoldModelResult:
    model: str
    fold: int
    brier_score: float
    auc: float | None
    observations: int


@dataclass(frozen=True)
class ChampionChallengerDecision:
    champion: str
    challenger: str
    decision: Literal["review_candidate", "retain_champion"]
    reason: str


def add_forward_label(frame: pd.DataFrame, horizon_sessions: int = 5) -> pd.DataFrame:
    """Create an outcome label that is permitted only for completed historical training rows."""
    labeled = frame.copy()
    labeled["forward_return"] = labeled["close"].shift(-horizon_sessions) / labeled["close"] - 1
    labeled["label_up"] = (labeled["forward_return"] > 0).astype(float)
    labeled.loc[labeled["forward_return"].isna(), "label_up"] = np.nan
    return labeled


def _estimator(name: str) -> LogisticRegression | RandomForestClassifier:
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1_000, C=0.5, class_weight="balanced", random_state=7)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=7,
        )
    raise ValueError(f"Unsupported model {name}")


def evaluate_models(
    labeled_features: pd.DataFrame,
    folds: list[WalkForwardFold],
    model_names: tuple[str, ...] = ("logistic_regression", "random_forest"),
) -> list[FoldModelResult]:
    """Fit each model only on a fold's purged train rows and score its untouched test rows."""
    required = list(FEATURE_COLUMNS) + ["label_up"]
    if any(column not in labeled_features for column in required):
        missing = [column for column in required if column not in labeled_features]
        raise ValueError(f"Missing model columns: {missing}")
    results: list[FoldModelResult] = []
    for fold_number, fold in enumerate(folds, start=1):
        train = labeled_features.reindex(fold.train_index).dropna(subset=required)
        test = labeled_features.reindex(fold.test_index).dropna(subset=required)
        if len(train) < 80 or len(test) < 15 or train["label_up"].nunique() < 2:
            continue
        x_train = train.loc[:, FEATURE_COLUMNS]
        y_train = train["label_up"].astype(int)
        x_test = test.loc[:, FEATURE_COLUMNS]
        y_test = test["label_up"].astype(int)
        for model_name in model_names:
            fitted = _estimator(model_name).fit(x_train, y_train)
            probabilities = fitted.predict_proba(x_test)[:, 1]
            auc = float(roc_auc_score(y_test, probabilities)) if y_test.nunique() > 1 else None
            results.append(
                FoldModelResult(
                    model=model_name,
                    fold=fold_number,
                    brier_score=float(brier_score_loss(y_test, probabilities)),
                    auc=auc,
                    observations=len(test),
                )
            )
    return results


def choose_challenger(results: list[FoldModelResult]) -> ChampionChallengerDecision:
    """Return a review decision only; humans promote models after full strategy validation."""
    grouped: dict[str, list[FoldModelResult]] = {}
    for result in results:
        grouped.setdefault(result.model, []).append(result)
    champion = "logistic_regression"
    challenger = "random_forest"
    champion_scores = grouped.get(champion, [])
    challenger_scores = grouped.get(challenger, [])
    if len(champion_scores) < 3 or len(challenger_scores) < 3:
        return ChampionChallengerDecision(
            champion,
            challenger,
            "retain_champion",
            "Fewer than three valid untouched folds; no promotion review is allowed.",
        )
    champion_brier = float(np.mean([item.brier_score for item in champion_scores]))
    challenger_brier = float(np.mean([item.brier_score for item in challenger_scores]))
    if challenger_brier <= champion_brier * 0.97:
        return ChampionChallengerDecision(
            champion,
            challenger,
            "review_candidate",
            "Challenger has at least a 3% lower mean OOS Brier score; require trading-cost "
            "and stability review.",
        )
    return ChampionChallengerDecision(
        champion,
        challenger,
        "retain_champion",
        "Challenger does not clear the minimum OOS calibration improvement threshold.",
    )


def write_model_manifest(
    destination: Path,
    model_name: str,
    training_start: str,
    training_end: str,
    dataset_version: str,
    code_commit: str,
    results: list[FoldModelResult],
) -> Path:
    """Write a machine-readable registry record; this does not mark a model as production."""
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": model_name,
        "status": "challenger",
        "training_period": {"start": training_start, "end": training_end},
        "features": list(FEATURE_COLUMNS),
        "dataset_version": dataset_version,
        "code_commit": code_commit,
        "created_at": datetime.now(UTC).isoformat(),
        "fold_results": [asdict(item) for item in results if item.model == model_name],
    }
    path = destination / f"{model_name}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
