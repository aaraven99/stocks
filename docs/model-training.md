# Model training and challenger policy

The baseline evaluates a regularized logistic regression and a constrained random forest against
the same purged, expanding walk-forward folds. Labels are forward returns solely for *historical
training/evaluation rows*; the last horizon rows have no label. Training rows that could use an
outcome overlapping the validation boundary are removed.

Model ranking begins with out-of-sample Brier score and AUC, not classification accuracy. A
challenger is merely submitted for review when it improves mean OOS calibration by at least 3% on
three or more folds. It is never promoted automatically: transaction-cost stress tests, parameter
sensitivity, feature ablation, strategy-level returns, and data provenance review remain required.

Every candidate manifest in `models/registry/` records the training window, feature set, code
commit, dataset version, fold results, creation time, and `challenger` status.

