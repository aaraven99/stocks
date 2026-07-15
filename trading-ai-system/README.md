# Quant Research Swing Platform

An automated, SQLite-centered probabilistic swing-trading research system. It produces ranked watchlist research, not investment advice or guaranteed trades.

## Run locally

```bash
cd trading-ai-system
python -m pip install -r requirements.txt
python src/pipeline/run_daily.py --test-mode
python src/pipeline/run_backfill.py --max-tickers 10 --period 5y --max-samples 80 --mc-sims 1000
python src/pipeline/run_training.py
streamlit run dashboard/app.py
```

The runner uses provider data when available, records all degradation, and never replaces unavailable live data with synthetic prices. When Alpaca paper credentials are configured, current broker equity is the allocation capital source; the local fallback exists only for offline test mode. Set only the environment variables shown in `.env.example`; email sends only when credentials and recipient are supplied.

## Design

- SQLite is the system of record for raw/clean data, feature lineage, forecasts, simulations, ranking, governance, debate, paper execution, and reproducibility manifests.
- Data is causally ordered; feature construction uses only observations through the as-of bar. Training helpers provide purged walk-forward splits with embargo.
- Risk controls include data-confidence penalties, regime playbooks, stress penalties, allocation constraints, drift/calibration checks, drawdown derisking, CASH MODE, and kill switch history.
- GitHub Actions runs research overnight and uses `zoneinfo` to target 7:00 AM America/Chicago (CST/CDT aware). Artifacts include the SQLite system of record, reports, and manifest.

## Operational notes

`UNIVERSE_MODE=full` is intended for a dynamically sourced broad US universe. The bundled CSV files are resilient offline fallbacks. Free providers have coverage and rate-limit constraints, so production results must be reviewed for universe-health/degradation flags before use.

Historical training requires real provider access. Configure `FINNHUB_API_KEY`, `POLYGON_API_KEY`, and `FRED_API_KEY` only when those licensed feeds are available; unavailable providers are excluded from decision features rather than replaced with deterministic proxy values. The weekly GitHub workflow performs a bounded real-history backfill before training and uploads the resulting SQLite database, champion artifact, calibration objects, and backfill manifest.
