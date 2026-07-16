# Quant Research Swing Platform

An automated, SQLite-centered probabilistic swing-trading research system. It produces ranked watchlist research, not investment advice or guaranteed trades.

## Run locally

```bash
cd trading-ai-system
python -m pip install -r requirements.txt
python src/pipeline/run_daily.py --test-mode
python src/pipeline/run_backfill.py --max-tickers 10 --period 5y --max-samples 80 --mc-sims 1000
python src/pipeline/run_training.py
python src/pipeline/run_backtest.py --enforce-acceptance
python src/pipeline/run_validation.py
python src/pipeline/run_healthcheck.py
streamlit run dashboard/app.py
```

The runner uses provider data when available, records all degradation, and never replaces unavailable live data with synthetic prices. OHLCV ingestion tries yfinance, then the direct Yahoo Chart API, then Stooq, with provider provenance persisted on every raw bar. When Alpaca paper credentials are configured, current broker equity is the allocation capital source; the local fallback exists only for offline test mode. Set only the environment variables shown in `.env.example`; email sends only when credentials and recipient are supplied.

## Design

- SQLite is the system of record for raw/clean data, feature lineage, forecasts, simulations, ranking, governance, debate, paper execution, and reproducibility manifests.
- Data is causally ordered; feature construction uses only observations through the as-of bar. Training helpers provide purged walk-forward splits with embargo.
- Risk controls include data-confidence penalties, regime playbooks, stress penalties, allocation constraints, drift/calibration checks, drawdown derisking, CASH MODE, and kill switch history.
- GitHub Actions starts the daily run at approximately 02:00 America/Chicago (CST/CDT-aware UTC fallback schedules), leaves a multi-hour completion window, and sends the briefing at 07:00 America/Chicago. The workflow fails closed if required artifacts are missing or the scheduled delivery is more than five minutes late. Artifacts include the SQLite system of record, reports, and manifest.

## Operational notes

`UNIVERSE_MODE=full` is intended for a dynamically sourced broad US universe. The bundled CSV files are resilient offline fallbacks. Free providers have coverage and rate-limit constraints, so production results must be reviewed for universe-health/degradation flags before use.

The bundled fallback snapshots are ranked Nasdaq/NYSE screener exports (4,764 valid common-stock/ETF symbols in `universe_full.csv`, 2,000 in the backup, and 500 in the top snapshot). They are point-in-time snapshots and must be refreshed as part of regular data governance; the runtime parser also supports the live Nasdaq `data.table.rows` response shape.

Historical training requires real provider access. Configure `FINNHUB_API_KEY`, `POLYGON_API_KEY`, and `FRED_API_KEY` only when those licensed feeds are available; unavailable providers are excluded from decision features rather than replaced with deterministic proxy values. The weekly GitHub workflow performs a bounded real-history backfill before training and uploads the resulting SQLite database, champion artifact, calibration objects, and backfill manifest.

The backtest quality gate is deliberately strict: `--enforce-acceptance` exits non-zero unless the leakage-controlled out-of-sample test records at least a 65% closed-trade win rate and no more than two business days per trade. The production backtest uses an expanding XGBoost walk-forward estimator, a five-business-day purge/embargo matching the label horizon, and a fixed 0.78 probability floor. Equity and drawdown are marked over each five-business-day holding window with a capped notional allocation; overlapping labels are not compounded as same-day returns. A failed gate blocks weekly promotion; the floor and model choice must never be tuned against the held-out test period or by substituting synthetic data.
