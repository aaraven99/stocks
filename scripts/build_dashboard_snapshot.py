from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Dict, List, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from breakout_system.demo import SimulatedSource, build_historical
from breakout_system.engine import Backtester, Bar, IndicatorEngine, RealTimeBreakoutEngine
from breakout_system.research import iterative_refinement


def is_market_open_now(now_utc: datetime | None = None) -> bool:
    now_utc = now_utc or datetime.now(tz=ZoneInfo("UTC"))
    et = now_utc.astimezone(ZoneInfo("America/New_York"))
    if et.weekday() >= 5:
        return False
    open_t = dtime(9, 30)
    close_t = dtime(16, 0)
    return open_t <= et.time() <= close_t


def fetch_recent_intraday(ticker: str, interval: str = "5m", period: str = "59d") -> Tuple[pd.DataFrame, float]:
    start = time.perf_counter()
    raw = yf.download(ticker, interval=interval, period=period, auto_adjust=False, progress=False)
    latency_ms = (time.perf_counter() - start) * 1000
    if raw.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]), latency_ms
    raw = raw.reset_index()
    raw = raw.rename(columns={
        raw.columns[0]: "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    return raw[["timestamp", "open", "high", "low", "close", "volume"]], latency_ms


def fetch_training_daily(ticker: str, years: int = 20) -> pd.DataFrame:
    end = pd.Timestamp.utcnow().tz_localize(None)
    start = end - pd.DateOffset(years=years)
    raw = yf.download(
        ticker,
        start=start.date(),
        end=end.date(),
        auto_adjust=False,
        progress=False,
        interval="1d",
    )
    if raw.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    raw = raw.reset_index().rename(
        columns={
            raw.reset_index().columns[0]: "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    return raw[["timestamp", "open", "high", "low", "close", "volume"]]


def maybe_refine_params(tickers: List[str], years: int, iterations: int) -> Dict[str, float]:
    if years <= 0 or iterations <= 0:
        return {"min_volume_ratio": 1.5, "learning_rate": 0.05, "n_estimators": 250, "max_depth": 2}
    results = iterative_refinement(tickers=tickers, years=years, max_iterations=iterations)
    if results.empty:
        return {"min_volume_ratio": 1.5, "learning_rate": 0.05, "n_estimators": 250, "max_depth": 2}
    best = results.sort_values(["backtest_sharpe", "wf_precision", "oos_precision"], ascending=False).iloc[0]["params"]
    return {
        "min_volume_ratio": float(best["min_volume_ratio"]),
        "learning_rate": float(best["learning_rate"]),
        "n_estimators": int(best["n_estimators"]),
        "max_depth": int(best["max_depth"]),
    }


def run_real_pipeline(tickers: List[str], bars: int, train_years: int, refine_years: int, refine_iterations: int) -> dict:
    training: Dict[str, pd.DataFrame] = {}
    recent: Dict[str, pd.DataFrame] = {}
    fetch_latency_ms: Dict[str, float] = {}
    stale_alerts: List[str] = []

    for t in tickers:
        recent_df, latency = fetch_recent_intraday(t)
        recent[t] = recent_df
        fetch_latency_ms[t] = round(latency, 2)
        training[t] = fetch_training_daily(t, years=train_years)

        if not recent_df.empty:
            last_ts = pd.Timestamp(recent_df["timestamp"].iloc[-1])
            age_minutes = (pd.Timestamp.utcnow().tz_localize(None) - last_ts.tz_localize(None) if last_ts.tzinfo else pd.Timestamp.utcnow().tz_localize(None) - last_ts).total_seconds() / 60
            if age_minutes > 20:
                stale_alerts.append(f"{t}: stale quote data ({age_minutes:.1f} min old)")

    ready = [t for t in tickers if not recent[t].empty and not training[t].empty]
    if not ready:
        raise RuntimeError("No tickers had both recent and training data from real feed")

    params = maybe_refine_params(ready, years=refine_years, iterations=refine_iterations)
    engine = RealTimeBreakoutEngine(
        timeframe="5m",
        min_volume_ratio=params["min_volume_ratio"],
        model_learning_rate=params["learning_rate"],
        model_estimators=params["n_estimators"],
        model_depth=params["max_depth"],
    )

    oos_precision = engine.fit({t: training[t] for t in ready})

    liquidity: Dict[str, float] = {}
    regime_score: Dict[str, float] = {}
    for t in ready:
        hist = recent[t].tail(bars)
        ind = IndicatorEngine.add_indicators(hist)
        liquidity[t] = float((hist["close"] * hist["volume"]).tail(20).mean())
        if len(ind) > 55 and pd.notna(ind["sma50"].iloc[-1]):
            regime_score[t] = 1.0 if ind["close"].iloc[-1] > ind["sma50"].iloc[-1] else 0.0
        else:
            regime_score[t] = 0.0

        for _, r in hist.iterrows():
            engine.on_bar(
                Bar(
                    ticker=t,
                    timestamp=pd.Timestamp(r["timestamp"]),
                    open=float(r["open"]),
                    high=float(r["high"]),
                    low=float(r["low"]),
                    close=float(r["close"]),
                    volume=float(r["volume"]),
                )
            )
            engine.invalidate_broken_structures()

    active_df = engine.active_setups_frame()
    if not active_df.empty:
        active_df["liquidity_20d_usd"] = active_df["ticker"].map(liquidity)
        active_df["regime_score"] = active_df["ticker"].map(regime_score)
        liq_max = max(active_df["liquidity_20d_usd"].max(), 1.0)
        active_df["opportunity_score"] = (
            active_df["ml_probability"] * 0.60
            + active_df["risk_reward"].clip(upper=4) / 4 * 0.20
            + (active_df["liquidity_20d_usd"] / liq_max) * 0.10
            + active_df["regime_score"] * 0.10
        )
        active_df = active_df.sort_values("opportunity_score", ascending=False)

    watch_df = engine.watchlist_frame()
    backtest = Backtester().run(engine.active_signals, {t: recent[t] for t in ready})

    safeguards = {
        "market_open_et": is_market_open_now(),
        "stale_alerts": stale_alerts,
        "fetch_latency_ms": fetch_latency_ms,
        "data_ready_tickers": ready,
        "failover_mode": False,
    }

    return {
        "data_mode": "real_yfinance",
        "tickers": ready,
        "bars": bars,
        "selected_params": params,
        "metrics": {
            "model_oos_precision": round(oos_precision, 4),
            "walk_forward_precision": round(engine.model.walk_forward_precision, 4),
            "walk_forward_sharpe": round(engine.model.walk_forward_sharpe, 4),
            "model_ready": bool(engine.model.trained),
            **{k: round(float(v), 4) for k, v in backtest.items()},
        },
        "operational_safeguards": safeguards,
        "top_ranked_opportunities": active_df.head(10).to_dict(orient="records") if not active_df.empty else [],
        "active_setups": active_df.to_dict(orient="records"),
        "watchlist": watch_df.to_dict(orient="records"),
    }


def run_simulated_fallback(tickers: List[str], bars: int, previous_snapshot: dict | None = None) -> dict:
    historical = build_historical(tickers, bars=max(1000, bars * 3))
    engine = RealTimeBreakoutEngine(timeframe="5m")
    oos_precision = engine.fit(historical)

    source = SimulatedSource(tickers=tickers, bars=bars)
    for bar in source.stream():
        engine.on_bar(bar)
        engine.invalidate_broken_structures()

    active_df = engine.active_setups_frame()
    watch_df = engine.watchlist_frame()
    backtest = Backtester().run(engine.active_signals, historical)

    payload = {
        "data_mode": "simulated_fallback",
        "tickers": tickers,
        "bars": bars,
        "selected_params": {"min_volume_ratio": 1.5, "learning_rate": 0.05, "n_estimators": 250, "max_depth": 2},
        "metrics": {
            "model_oos_precision": round(oos_precision, 4),
            "walk_forward_precision": round(engine.model.walk_forward_precision, 4),
            "walk_forward_sharpe": round(engine.model.walk_forward_sharpe, 4),
            "model_ready": bool(engine.model.trained),
            **{k: round(float(v), 4) for k, v in backtest.items()},
        },
        "operational_safeguards": {
            "market_open_et": is_market_open_now(),
            "stale_alerts": ["Real feed unavailable; using simulated fallback."],
            "fetch_latency_ms": {},
            "data_ready_tickers": tickers,
            "failover_mode": True,
        },
        "top_ranked_opportunities": active_df.head(10).to_dict(orient="records") if not active_df.empty else [],
        "active_setups": active_df.to_dict(orient="records"),
        "watchlist": watch_df.to_dict(orient="records"),
    }

    if previous_snapshot:
        payload["previous_snapshot_available"] = True
    return payload


def build_snapshot(
    tickers: List[str],
    bars: int,
    train_years: int,
    refine_years: int,
    refine_iterations: int,
    output_path: Path,
    allow_simulated_fallback: bool,
) -> dict:
    previous = None
    if output_path.exists():
        try:
            previous = json.loads(output_path.read_text())
        except Exception:
            previous = None

    try:
        return run_real_pipeline(
            tickers=tickers,
            bars=bars,
            train_years=train_years,
            refine_years=refine_years,
            refine_iterations=refine_iterations,
        )
    except Exception as exc:
        if not allow_simulated_fallback:
            raise RuntimeError(
                "Real-data snapshot build failed and simulated fallback is disabled. "
                f"Root cause: {exc}"
            ) from exc
        snapshot = run_simulated_fallback(tickers=tickers, bars=bars, previous_snapshot=previous)
        snapshot["fallback_reason"] = str(exc)
        return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Build static dashboard snapshot JSON for Vercel-hosted dashboard")
    parser.add_argument("--tickers", default="AAPL,MSFT,NVDA,AMZN")
    parser.add_argument("--bars", type=int, default=400)
    parser.add_argument("--train-years", type=int, default=20)
    parser.add_argument("--refine-years", type=int, default=5)
    parser.add_argument("--refine-iterations", type=int, default=1)
    parser.add_argument("--output", default="public/data/latest_snapshot.json")
    parser.add_argument("--allow-simulated-fallback", action="store_true", help="Allow simulated fallback if real feed fails")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    out = Path(args.output)
    snapshot = build_snapshot(
        tickers=tickers,
        bars=args.bars,
        train_years=args.train_years,
        refine_years=args.refine_years,
        refine_iterations=args.refine_iterations,
        output_path=out,
        allow_simulated_fallback=args.allow_simulated_fallback,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, indent=2, default=str))
    print(f"Wrote snapshot to {out} | mode={snapshot.get('data_mode')}")


if __name__ == "__main__":
    main()
