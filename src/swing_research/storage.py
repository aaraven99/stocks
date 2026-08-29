"""Durable, local audit trail for predictions and simulated positions."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .schemas import PaperPrediction


@dataclass(frozen=True)
class PendingPrediction:
    id: int
    ticker: str
    predicted_at: datetime
    holding_period_sessions: int
    model_version: str


@dataclass(frozen=True)
class PredictionOutcome:
    prediction_id: int
    entry_at: datetime
    entry_price: float
    evaluated_at: datetime
    evaluated_price: float
    gross_return: float
    net_return: float
    spy_return: float | None
    qqq_return: float | None


class PaperLedger:
    def __init__(self, database_path: Path) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(database_path)
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY,
                ticker TEXT NOT NULL,
                predicted_at TEXT NOT NULL,
                holding_period_sessions INTEGER NOT NULL,
                composite_score REAL NOT NULL,
                model_version TEXT NOT NULL,
                feature_version TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                UNIQUE(ticker, predicted_at, model_version)
            );
            CREATE TABLE IF NOT EXISTS paper_positions (
                id INTEGER PRIMARY KEY,
                ticker TEXT NOT NULL,
                entry_at TEXT NOT NULL,
                entry_price REAL NOT NULL,
                quantity REAL NOT NULL,
                stop_price REAL,
                target_price REAL,
                exit_at TEXT,
                exit_price REAL,
                exit_reason TEXT
            );
            CREATE TABLE IF NOT EXISTS prediction_outcomes (
                prediction_id INTEGER PRIMARY KEY REFERENCES predictions(id),
                entry_at TEXT NOT NULL,
                entry_price REAL NOT NULL,
                evaluated_at TEXT NOT NULL,
                evaluated_price REAL NOT NULL,
                gross_return REAL NOT NULL,
                net_return REAL NOT NULL,
                spy_return REAL,
                qqq_return REAL
            );
            """
        )

    def record_prediction(self, prediction: PaperPrediction) -> int:
        self.connection.execute(
            """
            INSERT OR IGNORE INTO predictions
            (ticker, predicted_at, holding_period_sessions, composite_score,
             model_version, feature_version, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prediction.ticker,
                prediction.predicted_at.isoformat(),
                prediction.holding_period_sessions,
                prediction.composite_score,
                prediction.model_version,
                prediction.feature_version,
                json.dumps(prediction.payload, sort_keys=True),
            ),
        )
        self.connection.commit()
        row = self.connection.execute(
            """
            SELECT id FROM predictions
            WHERE ticker = ? AND predicted_at = ? AND model_version = ?
            """,
            (prediction.ticker, prediction.predicted_at.isoformat(), prediction.model_version),
        ).fetchone()
        if row is None:  # pragma: no cover - database invariant
            raise RuntimeError("Prediction was not persisted")
        return int(row[0])

    def prediction_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])

    def pending_predictions(self) -> list[PendingPrediction]:
        rows = self.connection.execute(
            """
            SELECT p.id, p.ticker, p.predicted_at, p.holding_period_sessions, p.model_version
            FROM predictions p
            LEFT JOIN prediction_outcomes o ON o.prediction_id = p.id
            WHERE o.prediction_id IS NULL
            ORDER BY p.predicted_at, p.id
            """
        ).fetchall()
        return [
            PendingPrediction(
                id=int(row[0]),
                ticker=str(row[1]),
                predicted_at=datetime.fromisoformat(str(row[2])),
                holding_period_sessions=int(row[3]),
                model_version=str(row[4]),
            )
            for row in rows
        ]

    def record_prediction_outcome(self, outcome: PredictionOutcome) -> None:
        self.connection.execute(
            """
            INSERT OR IGNORE INTO prediction_outcomes
            (prediction_id, entry_at, entry_price, evaluated_at, evaluated_price,
             gross_return, net_return, spy_return, qqq_return)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                outcome.prediction_id,
                outcome.entry_at.isoformat(),
                outcome.entry_price,
                outcome.evaluated_at.isoformat(),
                outcome.evaluated_price,
                outcome.gross_return,
                outcome.net_return,
                outcome.spy_return,
                outcome.qqq_return,
            ),
        )
        self.connection.commit()

    def outcome_count(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) FROM prediction_outcomes").fetchone()
        return int(row[0])

    def close(self) -> None:
        self.connection.close()
