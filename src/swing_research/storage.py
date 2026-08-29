"""Durable, local audit trail for predictions and simulated positions."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .schemas import PaperPrediction


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
            """
        )

    def record_prediction(self, prediction: PaperPrediction) -> None:
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

    def prediction_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])

    def close(self) -> None:
        self.connection.close()
