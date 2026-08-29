# Backtesting protocol

1. Decide only after the prior completed session's close.
2. Execute no earlier than the next regular-session open.
3. Deduct configured half-spread, slippage, and commission costs.
4. Holdout windows are never used during feature selection or hyperparameter tuning.
5. Stress test costs, threshold, and ordering before interpreting results.

The test suite includes a regression test that changes a future bar and asserts that prior
features and orders are unchanged.

