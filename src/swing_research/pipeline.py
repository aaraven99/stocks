"""Daily research orchestration: fail closed on critical data, then rank and persist."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .agents import synthesize_narratives
from .config import load_config
from .data import PriceProvider, YFinancePriceProvider, configured_price_provider, validate_ohlcv
from .features import build_technical_features
from .market_calendar import latest_completed_nyse_session
from .regime import MarketRegime, detect_regime
from .reporting import render_morning_report
from .schemas import Candidate, PaperPrediction
from .scoring import rank_candidates, score_candidate
from .storage import PaperLedger


def _check_completed_session(frame: pd.DataFrame, expected_session: datetime) -> None:
    if frame.empty or frame.index[-1].date() != expected_session.date():
        received = "no bar" if frame.empty else str(frame.index[-1].date())
        raise RuntimeError(
            "Critical market data is stale: "
            f"expected completed session {expected_session.date()}, received {received}"
        )


def rank_from_frames(
    frames: dict[str, pd.DataFrame], benchmark_ticker: str, limit: int
) -> tuple[list[Candidate], MarketRegime]:
    if benchmark_ticker not in frames:
        raise ValueError(f"Benchmark {benchmark_ticker} is required")
    benchmark = validate_ohlcv(frames[benchmark_ticker])
    regime = detect_regime(benchmark)
    candidates: list[Candidate] = []
    for ticker, frame in frames.items():
        if ticker == benchmark_ticker:
            continue
        features = build_technical_features(validate_ohlcv(frame), benchmark["close"])
        candidate = score_candidate(ticker, features, regime)
        if candidate is not None:
            candidates.append(candidate)
    return rank_candidates(candidates, limit), regime


def run_daily_research(
    config_dir: Path,
    provider: PriceProvider | None = None,
    now: datetime | None = None,
    ledger: PaperLedger | None = None,
) -> tuple[list[Candidate], MarketRegime, str]:
    """Run the numerical pipeline using only the latest completed regular session."""
    effective_now = now or datetime.now(UTC)
    config = load_config(config_dir)
    universe = config["universe"]["universe"]
    strategy = config["strategy"]["strategy"]
    benchmark_ticker = str(config["universe"]["benchmarks"][0])
    expected_date = latest_completed_nyse_session(effective_now)
    expected_end = datetime.combine(expected_date, datetime.min.time(), tzinfo=UTC)
    start = expected_end - timedelta(days=500)
    market_provider = provider or configured_price_provider()
    if os.getenv("GITHUB_ACTIONS", "").lower() == "true" and isinstance(
        market_provider, YFinancePriceProvider
    ):
        raise RuntimeError(
            "YFinancePriceProvider is restricted to private local research and cannot run "
            "inside public GitHub Actions."
        )
    tickers = list(dict.fromkeys([benchmark_ticker, *universe["tickers"]]))
    frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        frame = market_provider.fetch_daily(ticker, start, expected_end)
        _check_completed_session(frame, expected_end)
        frames[ticker] = frame
    candidates, regime = rank_from_frames(
        frames, benchmark_ticker, int(strategy["thresholds"]["top_candidates"])
    )
    agent_config = config["agents"]["agents"]
    narrative = synthesize_narratives(
        candidates,
        enabled=bool(agent_config["use_openrouter_narrative"]),
        model=str(agent_config["openrouter_model"]),
    )
    report = render_morning_report(candidates, regime, narrative.narratives, narrative.status)
    if ledger is not None:
        for candidate in candidates:
            ledger.record_prediction(
                PaperPrediction(
                    ticker=candidate.ticker,
                    predicted_at=candidate.as_of,
                    holding_period_sessions=candidate.holding_period_sessions,
                    composite_score=candidate.composite_score,
                    model_version="deterministic-momentum-pullback-v1",
                    feature_version="technical-v1",
                    payload=candidate.model_dump(mode="json"),
                )
            )
    return candidates, regime, report


def make_demo_frames(periods: int = 320) -> dict[str, pd.DataFrame]:
    """Deterministic fixture for offline tests and a no-network CLI smoke run."""
    index = pd.bdate_range("2024-01-02", periods=periods)
    frames: dict[str, pd.DataFrame] = {}
    profiles = {
        "SPY": (0.00035, 0.008),
        "NVDA": (0.00070, 0.020),
        "MSFT": (0.00045, 0.012),
        "JPM": (0.00025, 0.011),
    }
    for offset, (ticker, (drift, noise)) in enumerate(profiles.items()):
        generator = np.random.default_rng(100 + offset)
        returns = generator.normal(drift, noise, periods)
        close = 100 * np.exp(np.cumsum(returns))
        open_price = close * (1 + generator.normal(0, noise / 5, periods))
        high = np.maximum(open_price, close) * (1 + 0.004)
        low = np.minimum(open_price, close) * (1 - 0.004)
        volume = generator.integers(2_000_000, 8_000_000, periods)
        frames[ticker] = pd.DataFrame(
            {"open": open_price, "high": high, "low": low, "close": close, "volume": volume},
            index=index,
        )
    return frames


def run_demo_research() -> tuple[list[Candidate], MarketRegime, str]:
    candidates, regime = rank_from_frames(make_demo_frames(), "SPY", limit=10)
    return candidates, regime, render_morning_report(candidates, regime)
