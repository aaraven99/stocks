# Relative-strength research ledger

## Promotion standard

The objective is not met by a favorable backtest. A strategy can be called a research candidate
only after it beats **both SPY and QQQ by at least 1.5x terminal wealth** in two predeclared,
non-overlapping out-of-sample periods, after costs, and has no material failure in a separate final
holdout. A 2.0x result is an aspirational threshold, not a tuning target.

## Data and execution contract

- Instruments: nine long-lived US equity-sector/index ETFs plus TLT, GLD, and SHY.
- Provider: yfinance adjusted daily OHLCV convenience adapter; retrieved 2026-08-29.
- Signals: prior completed close; execution: next regular-session open.
- Costs: 2 bps half spread + 5 bps slippage per side; no commissions.
- Bias limitation: the ETF instrument set avoids constituent survivorship bias, but remains a
  limited asset universe and is not a stock-selection result.

## Experiment 1 — single training selection

`predeclared-etf-relative-strength.json` selected 189-session momentum, 200-session trend,
21-session volatility, 20-session rebalancing, and two risk assets using 2007–2015 only.

| Period | Outcome versus SPY | Outcome versus QQQ | Decision |
| --- | ---: | ---: | --- |
| Validation, 2016–2020 | 0.89x | 0.62x | Rejected |
| Later aggregate, 2021–2026 | 1.24x | 1.16x | Not promotable; validation failed |

## Experiment 2 — four development periods

`robust-etf-relative-strength.json` selected 189-session momentum, 200-session trend,
63-session volatility, 20-session rebalancing, and two risk assets by median weakest-benchmark
performance across 2007–2010, 2011–2014, 2015–2018, and 2019–2020. Its development multiples
were 1.04x, 0.90x, 0.77x, and 0.94x, respectively—already too inconsistent for promotion.

| Period | Outcome versus SPY | Outcome versus QQQ | Decision |
| --- | ---: | ---: | --- |
| Validation, 2021–2023 | 1.16x | 1.16x | Insufficient for target |
| Final period, 2024–2026 | 0.97x | 0.90x | Rejected |

## Current conclusion

No tested specification is a validated winner. The next family must be selected strictly from
earlier development data, stress-tested across multiple folds, and evaluated without looking at
its final period until its rules are frozen.

