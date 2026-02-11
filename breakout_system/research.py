from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from itertools import product
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .engine import Backtester, Bar, RealTimeBreakoutEngine

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional runtime dependency
    yf = None


@dataclass
class ChunkResult:
    chunk_start: pd.Timestamp
    chunk_end: pd.Timestamp
    params: Dict[str, float]
    oos_precision: float
    wf_precision: float
    wf_sharpe: float
    backtest_sharpe: float
    avg_r: float
    win_rate: float


def chunk_ranges(start: pd.Timestamp, end: pd.Timestamp, min_days: int = 14, max_days: int = 30) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Create contiguous chunks between 2 weeks and 1 month."""
    out: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = start
    toggle = True
    while cur < end:
        days = max_days if toggle else min_days
        nxt = min(cur + timedelta(days=days), end)
        out.append((cur, nxt))
        cur = nxt
        toggle = not toggle
    return out


def _download_daily(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is required for 20-year chunked research. Install with scripts/install_deps.sh")
    raw = yf.download(ticker, start=start.date(), end=end.date(), progress=False, auto_adjust=False, interval="1d")
    if raw.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    raw = raw.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}).reset_index()
    raw = raw.rename(columns={raw.columns[0]: "timestamp"})
    return raw[["timestamp", "open", "high", "low", "close", "volume"]]


def _simulate_signals(engine: RealTimeBreakoutEngine, ticker: str, df: pd.DataFrame) -> List:
    signals = []
    for _, r in df.iterrows():
        bar = Bar(
            ticker=ticker,
            timestamp=pd.Timestamp(r["timestamp"]),
            open=float(r["open"]),
            high=float(r["high"]),
            low=float(r["low"]),
            close=float(r["close"]),
            volume=float(r["volume"]),
        )
        s = engine.on_bar(bar)
        engine.invalidate_broken_structures()
        if s:
            signals.append(s)
    return signals


def iterative_refinement(
    tickers: List[str],
    years: int = 20,
    max_iterations: int = 3,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Walk across 2-week to 1-month chunks over ~20 years, tune parameters, retest, and iterate.
    """
    if end is None:
        end = pd.Timestamp.utcnow().tz_localize(None)
    if start is None:
        start = end - pd.DateOffset(years=years)

    ranges = chunk_ranges(start, end, min_days=14, max_days=30)
    param_grid = list(
        product(
            [1.4, 1.5, 1.7],
            [0.03, 0.05],
            [200, 250, 320],
            [2, 3],
        )
    )

    all_results: List[ChunkResult] = []
    best_global = None

    for i in range(max_iterations):
        for chunk_start, chunk_end in ranges:
            history: Dict[str, pd.DataFrame] = {}
            for t in tickers:
                history[t] = _download_daily(t, chunk_start, chunk_end)
            if any(df.empty for df in history.values()):
                continue

            best_chunk = None
            for vol_ratio, lr, n_est, depth in param_grid:
                engine = RealTimeBreakoutEngine(
                    timeframe="1d",
                    min_volume_ratio=vol_ratio,
                    model_learning_rate=lr,
                    model_estimators=n_est,
                    model_depth=depth,
                )
                oos = engine.fit(history)

                simulated = []
                for t in tickers:
                    simulated.extend(_simulate_signals(engine, t, history[t]))

                bt = Backtester().run(simulated, history, horizon=20)
                score = (oos * 0.5) + (engine.model.walk_forward_precision * 0.3) + (max(bt["sharpe"], 0) * 0.2)

                candidate = ChunkResult(
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    params={
                        "min_volume_ratio": vol_ratio,
                        "learning_rate": lr,
                        "n_estimators": float(n_est),
                        "max_depth": float(depth),
                        "iteration": float(i + 1),
                    },
                    oos_precision=oos,
                    wf_precision=engine.model.walk_forward_precision,
                    wf_sharpe=engine.model.walk_forward_sharpe,
                    backtest_sharpe=bt["sharpe"],
                    avg_r=bt["avg_r"],
                    win_rate=bt["win_rate"],
                )

                if best_chunk is None or score > (
                    (best_chunk.oos_precision * 0.5)
                    + (best_chunk.wf_precision * 0.3)
                    + (max(best_chunk.backtest_sharpe, 0) * 0.2)
                ):
                    best_chunk = candidate

            if best_chunk:
                all_results.append(best_chunk)

        if all_results:
            summary_df = pd.DataFrame(
                [
                    {
                        "chunk_start": r.chunk_start,
                        "chunk_end": r.chunk_end,
                        "params": r.params,
                        "oos_precision": r.oos_precision,
                        "wf_precision": r.wf_precision,
                        "wf_sharpe": r.wf_sharpe,
                        "backtest_sharpe": r.backtest_sharpe,
                        "avg_r": r.avg_r,
                        "win_rate": r.win_rate,
                    }
                    for r in all_results
                ]
            )
            best_idx = summary_df["backtest_sharpe"].idxmax()
            best_global = summary_df.loc[best_idx, "params"]

            # Narrow the grid around the best region for the next iteration.
            if isinstance(best_global, dict):
                vol = float(best_global["min_volume_ratio"])
                lr = float(best_global["learning_rate"])
                nest = int(best_global["n_estimators"])
                depth = int(best_global["max_depth"])
                param_grid = list(
                    product(
                        sorted(set([max(1.2, vol - 0.1), vol, vol + 0.1])),
                        sorted(set([max(0.01, lr - 0.01), lr, lr + 0.01])),
                        sorted(set([max(100, nest - 50), nest, nest + 50])),
                        sorted(set([max(1, depth - 1), depth, depth + 1])),
                    )
                )

    if not all_results:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "chunk_start": r.chunk_start,
                "chunk_end": r.chunk_end,
                "params": r.params,
                "oos_precision": r.oos_precision,
                "wf_precision": r.wf_precision,
                "wf_sharpe": r.wf_sharpe,
                "backtest_sharpe": r.backtest_sharpe,
                "avg_r": r.avg_r,
                "win_rate": r.win_rate,
            }
            for r in all_results
        ]
    ).sort_values(["chunk_start", "chunk_end"]).reset_index(drop=True)
