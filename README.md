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
- Real-time terminal output split into high-confidence active setups and watchlist/forming state.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
./scripts/install_deps.sh
python -m breakout_system.demo --tickers AAPL,MSFT,NVDA --bars 350
```

### If dependency install fails in restricted environments

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
- Replace simulated data with clean historical + corporate events.
- Persist state, decisions, and model versions for auditability.
- Run longer walk-forward tests across different market regimes before live deployment.
