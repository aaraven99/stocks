"""Chronological, label-purged train/validation/test folds for market research."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    train_index: pd.DatetimeIndex
    validation_index: pd.DatetimeIndex
    test_index: pd.DatetimeIndex


def expanding_walk_forward_splits(
    index: pd.DatetimeIndex,
    train_sessions: int = 252,
    validation_sessions: int = 63,
    test_sessions: int = 63,
    label_horizon_sessions: int = 5,
) -> list[WalkForwardFold]:
    """Create expanding folds and purge training rows whose labels overlap validation."""
    ordered = pd.DatetimeIndex(index).sort_values().unique()
    required = train_sessions + validation_sessions + test_sessions + label_horizon_sessions
    if len(ordered) < required:
        return []
    folds: list[WalkForwardFold] = []
    validation_start = train_sessions + label_horizon_sessions
    while validation_start + validation_sessions + test_sessions <= len(ordered):
        train_end = validation_start - label_horizon_sessions
        validation_end = validation_start + validation_sessions
        test_end = validation_end + test_sessions
        folds.append(
            WalkForwardFold(
                train_index=ordered[:train_end],
                validation_index=ordered[validation_start:validation_end],
                test_index=ordered[validation_end:test_end],
            )
        )
        validation_start += test_sessions
    return folds
