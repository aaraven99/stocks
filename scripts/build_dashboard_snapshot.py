from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from breakout_system.demo import SimulatedSource, build_historical
from breakout_system.engine import Backtester, RealTimeBreakoutEngine


def build_snapshot(tickers: List[str], bars: int) -> dict:
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

    metrics = {
        "model_oos_precision": round(oos_precision, 4),
        "walk_forward_precision": round(engine.model.walk_forward_precision, 4),
        "walk_forward_sharpe": round(engine.model.walk_forward_sharpe, 4),
        "model_ready": bool(engine.model.trained),
        **{k: round(float(v), 4) for k, v in backtest.items()},
    }

    return {
        "tickers": tickers,
        "bars": bars,
        "metrics": metrics,
        "active_setups": active_df.to_dict(orient="records"),
        "watchlist": watch_df.to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build static dashboard snapshot JSON for Vercel-hosted dashboard")
    parser.add_argument("--tickers", default="AAPL,MSFT,NVDA,AMZN")
    parser.add_argument("--bars", type=int, default=400)
    parser.add_argument("--output", default="public/data/latest_snapshot.json")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    snapshot = build_snapshot(tickers=tickers, bars=args.bars)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, indent=2, default=str))
    print(f"Wrote snapshot to {out}")


if __name__ == "__main__":
    main()
