# Decision log

## 2026-08-29 — Core stack

- **Backtesting:** a small event-timestamped engine is the core, with adapter room for vectorbt
  research. It makes next-session execution and cost assumptions explicit instead of hiding them
  in a vectorized convenience call.
- **Market data:** provider protocol plus optional yfinance adapter. This is inexpensive for a
  starter system, but not a survivorship-bias-free institutional data feed.
- **Agents:** deterministic, evidence-carrying agents are first class. OpenRouter is narrative
  only and may not alter numeric scores or invent figures.
- **ML:** sklearn baseline candidates and champion/challenger metadata. Complex models are gated
  behind walk-forward and calibration checks.
- **Storage:** local SQLite for forward predictions and paper trades. It provides durable,
  inspectable records without paid infrastructure.

