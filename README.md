# Agentic Swing-Trading Research

A point-in-time US-equity **research and paper-simulation** platform for 2–15 session swing
ideas. It is long-only by default, has no brokerage execution, and does not provide investment
advice or guarantee returns.

```text
OHLCV / SEC -> validated data -> features -> regime / strategy -> risk-gated ranking
                                                               -> cited agent review
                                                               -> Markdown + SQLite ledger
```

## What is implemented

- Configurable liquid-US-equity universe and risk limits.
- Provider boundary, adjusted daily OHLCV adapter, official SEC submissions client, freshness and
  point-in-time checks.
- Technical factors, benchmark-relative strength, deterministic regime detection, and a
  momentum/pullback ranking strategy.
- Next-session-open event backtester with spread, slippage, commissions, full performance metrics,
  and a future-data leakage regression test.
- Structured evidence agents plus an optional OpenRouter (`openrouter/free`) narrative adapter.
  Narratives cannot change numeric ranks, create values, or execute trades.
- SQLite forward-prediction and paper-position ledger; CI and a DST-safe daily workflow gate.

## Quick start

```powershell
uv sync --extra dev
uv run pytest
uv run swing-research demo --output reports/daily/demo.md
```

To enable optional narrative synthesis, create the ignored file
`C:\Users\aarav\Documents\ChatGPT\stock-algo\.env` with `OPENROUTER_API_KEY=<rotated key>` and
set `agents.use_openrouter_narrative: true` in `config/agents.yaml`. The default pipeline works
without it.

For timestamped company-news evidence, add `FINNHUBKEY=<your Finnhub key>` to that same ignored
file. No separate news key is required for the Finnhub adapter. Official SEC EDGAR ingestion needs
no API key; set `SEC_USER_AGENT=Your Name contact@example.com` instead. News remains outside the
numerical ranking until it passes its own point-in-time validation.

For a licensed delisting-aware price export, set `MARKET_DATA_PROVIDER=local_csv` and
`LOCAL_OHLCV_DIRECTORY` in that same `.env` file, then run `universe-audit` before any
stock-universe study. The adapter expects one `<ticker>.csv` per symbol with date and OHLCV fields.

## Daily research flow

`swing-research daily` loads only completed market data, rejects stale critical data, derives
features and a market regime, ranks a small funnel, applies risk gates, stores predictions, and
writes a dated Markdown report. The GitHub workflow invokes it on both possible UTC schedule
hours and the Python gate runs only at 05:00 America/Chicago on NYSE sessions.

Run `swing-research reconcile-outcomes` after predicted holding horizons complete. It enters at
the next available open, applies the configured costs, records only fully observed outcomes, and
keeps SPY/QQQ comparisons alongside each paper result.

## Important limitations

- The starter ticker list is not historical index membership; broad historical tests therefore
  have survivorship bias until a point-in-time universe adapter is supplied. The included
  `universe-audit` command uses a pinned community constituent file and refuses a broad study
  when free-price coverage is below its configured 98% gate.
- Free data may be delayed, revised, incomplete, or licensed for limited use. Validate every
  provider before use.
- Results from backtests and paper positions do not predict future performance.
- The OpenRouter key is optional, must never be committed, and a chat-shared key should be rotated.

See [architecture](docs/architecture.md), [methodology](docs/methodology.md),
[backtesting](docs/backtesting.md), [data sources](docs/data-sources.md),
[model training](docs/model-training.md), [agents](docs/agent-system.md),
[roadmap](docs/roadmap.md), [licensing](docs/licensing.md), and the
[decision log](docs/decision-log.md). The current benchmark research status is recorded in
[the relative-strength ledger](reports/backtests/relative-strength-research.md).
