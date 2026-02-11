# Reliable Real-Time ML-Enhanced Breakout System

A conservative, real-time breakout assistant focused on **capital protection**, **high precision**, and **strict filtering** over signal frequency.

## What this implementation includes

- Continuous multi-ticker OHLCV streaming abstraction (`StreamingOHLCVSource`) with rolling in-memory windows.
- Look-ahead-safe indicator/feature updates computed only from data available at each bar.
- Bullish-only, breakout-confirmed pattern detection for:
  - Ascending triangle
  - Double bottom
  - Bullish flag
  - Cup and handle
  - Falling wedge
- Mandatory breakout + volume expansion confirmation (no “forming” pattern entries).
- Feature set for ML:
  - RSI, MACD, EMA20, SMA50, ATR
  - Volume breakout ratio
  - Higher-timeframe trend alignment proxy
  - Distance from EMA20/SMA50
- Single-model ML layer (`GradientBoostingClassifier`) with:
  - Time-based split
  - Walk-forward validation
  - StandardScaler fit on train only (leakage-safe normalization)
  - Out-of-sample precision + walk-forward precision/sharpe stability gating
- Strict risk management:
  - Entry only on confirmed breakout close above resistance
  - Stop at logical pattern invalidation level
  - Minimum 2:1 R/R
  - Volatility-adjusted fixed-risk position sizing
  - Concurrent and portfolio exposure caps
  - Earnings proximity filter hook
  - Auto-invalidation + dynamic stop tightening
- Regime filter to avoid unfavorable/choppy conditions, with optional index confirmation.
- Backtester with slippage/fees and key metrics: win rate, avg R, max drawdown, Sharpe.
- Modern Streamlit dashboard for active setups + watchlist + metrics tables.
- 20-year chunked iterative research utility (2-week to 1-month chunks) for tune/retest loops.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
./scripts/install_deps.sh
python -m breakout_system.demo --tickers AAPL,MSFT,NVDA --bars 350
```

## Dashboard (modern table UI)

```bash
streamlit run breakout_system/dashboard.py
```

Shows:
- High-confidence active setups with columns: ticker, timeframe, pattern, entry, stop, target, risk/reward, quantity, ML probability, timestamp.
- Watchlist of forming setups.
- Model/backtest metrics cards.

## 20-year chunked iterative refinement

```bash
python - <<'PY'
from breakout_system.research import iterative_refinement

results = iterative_refinement(
    tickers=["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"],
    years=20,
    max_iterations=3,
)
print(results.tail(20).to_string(index=False))
PY
```

This performs:
1. Downloads historical data.
2. Splits into 14-to-30-day chunks.
3. Tunes parameters and retests chunk-by-chunk.
4. Narrows parameter grid and iterates.
5. Returns validation metrics for each chunk.

## If dependency install fails in restricted environments

Use the installer script:

```bash
./scripts/install_deps.sh
```

It attempts:

1. Offline install from `vendor/wheels`.
2. Online install through configured pip index/proxy.
3. If both fail, it prints exact fallback commands for internal mirror/proxy/offline bundle.

## Signal approval logic (strict)

A signal is emitted only if all are true:

1. Pattern is structurally complete.
2. Breakout close is above resistance.
3. Volume breakout ratio is confirmed.
4. Market regime is favorable.
5. ML probability >= 75%.
6. Risk/reward >= 2:1.
7. No structural invalidation and exposure constraints permit entry.

## Notes for production

- Plug real market feed/broker connector into `StreamingOHLCVSource`.
- Use corporate actions-adjusted data, split handling, and survivorship-safe universes.
- Persist state, decisions, and model versions for auditability.
- Run longer walk-forward tests across different market regimes before live deployment.
- Add broker-side hard risk limits and circuit breakers.
