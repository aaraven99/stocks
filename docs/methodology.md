# Methodology

The default research horizon is 2–15 regular trading sessions. Rankings combine independent
technical factors, benchmark-relative strength, and a deterministic risk gate. A higher score is
not investment advice and does not imply a probability unless calibration testing supports one.

The scoring engine must only learn weights from training windows. Validation, test, and paper
windows are chronological and mutually separated. Every prediction is stored before its outcome.

For benchmark-beating research, terminal wealth relative to both SPY and QQQ is reported alongside
risk metrics. A one-period win is insufficient: the research ledger defines the separate
multi-period promotion gate in `reports/backtests/relative-strength-research.md`.
