# Methodology

The default research horizon is 2–15 regular trading sessions. Rankings combine independent
technical factors, benchmark-relative strength, and a deterministic risk gate. A higher score is
not investment advice and does not imply a probability unless calibration testing supports one.

The scoring engine must only learn weights from training windows. Validation, test, and paper
windows are chronological and mutually separated. Every prediction is stored before its outcome.

For benchmark-beating research, terminal wealth relative to both SPY and QQQ is reported alongside
risk metrics. A one-period win is insufficient: the research ledger defines the separate
multi-period promotion gate in `reports/backtests/relative-strength-research.md`.

The stock cross-sectional engine consumes a membership matrix whose row for a signal date contains
only the securities known to be constituents at that completed close. It ranks those members, then
executes at the following regular-session open. Missing prices for a held security are fatal; the
engine does not silently forward-fill a delisting, rename, or provider gap.

After development-period selection, the same stock specification is evaluated with baseline and
doubled transaction costs on validation and final holdout. Cost stress never reselects a parameter
or uses later-period results to alter the selected model.
