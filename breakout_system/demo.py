from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Iterator, List

import numpy as np
import pandas as pd

from .engine import Backtester, Bar, RealTimeBreakoutEngine, StreamingOHLCVSource, run_realtime_loop


@dataclass
class SimulatedSource(StreamingOHLCVSource):
    tickers: List[str]
    bars: int = 500

    def _series(self, ticker: str) -> pd.DataFrame:
        rng = np.random.default_rng(abs(hash(ticker)) % 2**32)
        ts = pd.date_range("2024-01-01", periods=self.bars, freq="5min")
        base = 75 + np.cumsum(rng.normal(0.04, 0.8, size=self.bars))
        high = base + rng.uniform(0.1, 1.1, size=self.bars)
        low = base - rng.uniform(0.1, 1.1, size=self.bars)
        vol = rng.integers(15_000, 250_000, size=self.bars)
        return pd.DataFrame({"timestamp": ts, "open": base, "high": high, "low": low, "close": base, "volume": vol})

    def stream(self) -> Iterator[Bar]:
        tables = {t: self._series(t) for t in self.tickers}
        for i in range(self.bars):
            for ticker, df in tables.items():
                r = df.iloc[i]
                yield Bar(
                    ticker=ticker,
                    timestamp=pd.Timestamp(r["timestamp"]),
                    open=float(r["open"]),
                    high=float(r["high"]),
                    low=float(r["low"]),
                    close=float(r["close"]),
                    volume=float(r["volume"]),
                )


def build_historical(tickers: List[str], bars: int = 900) -> Dict[str, pd.DataFrame]:
    src = SimulatedSource(tickers=tickers, bars=bars)
    return {ticker: src._series(ticker) for ticker in tickers}


def main() -> None:
    parser = argparse.ArgumentParser(description="Reliable real-time ML-enhanced breakout system demo")
    parser.add_argument("--tickers", default="AAPL,MSFT,NVDA")
    parser.add_argument("--bars", type=int, default=350)
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    historical = build_historical(tickers, bars=max(args.bars * 3, 900))

    engine = RealTimeBreakoutEngine(timeframe="5m")
    precision = engine.fit(historical)
    print(
        "Model metrics | "
        f"oos_precision={precision:.3f} "
        f"wf_precision={engine.model.walk_forward_precision:.3f} "
        f"wf_sharpe={engine.model.walk_forward_sharpe:.3f} "
        f"trained={engine.model.trained}"
    )

    source = SimulatedSource(tickers=tickers, bars=args.bars)
    run_realtime_loop(source, engine)

    active = engine.display_state()
    print("\nActive setups:")
    print(active.tail(20).to_string(index=False))

    backtest = Backtester().run(engine.active_signals, historical)
    print("\nBacktest summary from emitted active signals:")
    print(pd.DataFrame([backtest]).to_string(index=False))


if __name__ == "__main__":
    main()
