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

from breakout_system.engine import Backtester, Bar, IndicatorEngine, RealTimeBreakoutEngine
from breakout_system.research import iterative_refinement

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def is_market_open_now(now_utc: datetime | None = None) -> bool:
    now_utc = now_utc or datetime.now(tz=ZoneInfo("UTC"))
    et = now_utc.astimezone(ZoneInfo("America/New_York"))
    if et.weekday() >= 5:
        return False
    return dtime(9, 30) <= et.time() <= dtime(16, 0)


def _empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def _normalize_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return _empty_ohlcv()
    out = raw.reset_index().rename(
        columns={
            raw.reset_index().columns[0]: "timestamp",
            "Datetime": "timestamp",
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    for col in COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[COLUMNS].dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    return out


def _fetch_history(
    ticker: str,
    *,
    period: str,
    interval: str,
    attempts: int = 3,
) -> Tuple[pd.DataFrame, float, str | None]:
    total_start = time.perf_counter()
    last_error: str | None = None
    for i in range(attempts):
        try:
            raw = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
            df = _normalize_ohlcv(raw)
            if not df.empty:
                return df, (time.perf_counter() - total_start) * 1000, None
            last_error = "empty_data"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        if i < attempts - 1:
            time.sleep(1.2 * (i + 1))
    return _empty_ohlcv(), (time.perf_counter() - total_start) * 1000, last_error


def fetch_recent_intraday(ticker: str, interval: str = "5m", period: str = "59d") -> Tuple[pd.DataFrame, float, str | None]:
    return _fetch_history(ticker=ticker, period=period, interval=interval)


def fetch_training_daily(ticker: str, years: int = 20) -> Tuple[pd.DataFrame, str | None]:
    df, _, err = _fetch_history(ticker=ticker, period=f"{years}y", interval="1d")
    return df, err


def maybe_refine_params(tickers: List[str], years: int, iterations: int) -> Dict[str, float]:
    defaults = {"min_volume_ratio": 1.5, "learning_rate": 0.05, "n_estimators": 250, "max_depth": 2}
    if years <= 0 or iterations <= 0:
        return defaults
    results = iterative_refinement(tickers=tickers, years=years, max_iterations=iterations)
    if results.empty:
        return defaults
    best = results.sort_values(["backtest_sharpe", "wf_precision", "oos_precision"], ascending=False).iloc[0]["params"]
    return {
        "min_volume_ratio": float(best["min_volume_ratio"]),
        "learning_rate": float(best["learning_rate"]),
        "n_estimators": int(best["n_estimators"]),
        "max_depth": int(best["max_depth"]),
    }


def _age_minutes(ts: pd.Timestamp) -> float:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return float((pd.Timestamp.utcnow().tz_localize(None) - ts).total_seconds() / 60)


def validate_snapshot(snapshot: dict) -> None:
    required = {"data_mode", "metrics", "operational_safeguards", "active_setups", "watchlist", "top_ranked_opportunities"}
    missing = required - set(snapshot)
    if missing:
        raise RuntimeError(f"Snapshot missing required keys: {sorted(missing)}")


def run_real_pipeline(
    tickers: List[str],
    bars: int,
    train_years: int,
    refine_years: int,
    refine_iterations: int,
    min_real_tickers: int,
) -> dict:
    training: Dict[str, pd.DataFrame] = {}
    recent: Dict[str, pd.DataFrame] = {}
    fetch_latency_ms: Dict[str, float] = {}
    stale_alerts: List[str] = []
    dropped_tickers: Dict[str, str] = {}

    for t in tickers:
        recent_df, latency, err_recent = fetch_recent_intraday(t)
        train_df, err_train = fetch_training_daily(t, years=train_years)
        fetch_latency_ms[t] = round(latency, 2)

        if recent_df.empty or train_df.empty:
            dropped_tickers[t] = f"recent={err_recent or 'empty'}; train={err_train or 'empty'}"
            continue

        recent[t] = recent_df
        training[t] = train_df

        age = _age_minutes(recent_df["timestamp"].iloc[-1])
        if age > 20:
            stale_alerts.append(f"{t}: stale quote data ({age:.1f} min old)")

    ready = sorted(set(recent.keys()) & set(training.keys()))
    if len(ready) < min_real_tickers:
        raise RuntimeError(
            f"Only {len(ready)} real-data tickers ready (required {min_real_tickers}). Dropped: {dropped_tickers}"
        )

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
        regime_score[t] = 1.0 if len(ind) > 55 and pd.notna(ind["sma50"].iloc[-1]) and ind["close"].iloc[-1] > ind["sma50"].iloc[-1] else 0.0

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

    backtest = Backtester().run(engine.active_signals, {t: recent[t] for t in ready})
    snapshot = {
        "schema_version": "1.2",
        "generated_at_utc": datetime.now(tz=ZoneInfo("UTC")).isoformat(),
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
        "operational_safeguards": {
            "market_open_et": is_market_open_now(),
            "stale_alerts": stale_alerts,
            "fetch_latency_ms": fetch_latency_ms,
            "data_ready_tickers": ready,
            "dropped_tickers": dropped_tickers,
            "failover_mode": False,
        },
        "top_ranked_opportunities": active_df.head(10).to_dict(orient="records") if not active_df.empty else [],
        "active_setups": active_df.to_dict(orient="records"),
        "watchlist": engine.watchlist_frame().to_dict(orient="records"),
    }
    validate_snapshot(snapshot)
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Build static dashboard snapshot JSON for Vercel-hosted dashboard (real data only)")
    parser.add_argument("--tickers", default="AAPL,MSFT,NVDA,AMZN")
    parser.add_argument("--bars", type=int, default=400)
    parser.add_argument("--train-years", type=int, default=20)
    parser.add_argument("--refine-years", type=int, default=5)
    parser.add_argument("--refine-iterations", type=int, default=1)
    parser.add_argument("--min-real-tickers", type=int, default=2)
    parser.add_argument("--output", default="public/data/latest_snapshot.json")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    out = Path(args.output)

    snapshot = run_real_pipeline(
        tickers=tickers,
        bars=args.bars,
        train_years=args.train_years,
        refine_years=args.refine_years,
        refine_iterations=args.refine_iterations,
        min_real_tickers=max(1, args.min_real_tickers),
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, indent=2, default=str))
    print(
        f"Wrote snapshot to {out} | mode={snapshot.get('data_mode')} | "
        f"ready={len(snapshot.get('operational_safeguards', {}).get('data_ready_tickers', []))}"
    )


if __name__ == "__main__":
    main()
