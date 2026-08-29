# GitHub repository research

Live metadata snapshot: `2026-08-29T17:14:33+00:00`. Generated with `scripts/refresh_github_research.py`.
This discovery process reads public GitHub metadata only; no candidate code was executed or copied.

## Method and score

Score / 100 = popularity 20 + activity 15 + maintainability 10 + documentation 10 + CI/testing 10 + license compatibility 10 + US-equity swing usefulness 20 + integration ease 5.
Popularity uses capped log stars; activity uses days since GitHub `pushed_at`; documentation uses repository size as a conservative metadata proxy; CI is detected from `.github/workflows`. `Latest activity` is a push-time proxy, not a claim that a commit is semantically meaningful. The 30 deep reviews add manual purpose, strength, weakness, swing suitability, and conclusion.

Licenses marked GPL/AGPL receive no compatibility points and are not incorporated. `NOASSERTION` means GitHub did not identify an SPDX license, so it is not treated as reusable code.

## Candidate pool (60 available of 60 discovered)

| Rank | Repository | Stars | Forks | Contributors | Language | License | Latest activity | Latest release | Open issues | Docs proxy | CI | Score |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- | --- | ---: | --- | --- | ---: |
| 1 | [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn) | 67095 | 27332 | 3557 | Python | BSD-3-Clause | 2026-08-28 | 2026-06-02 | 2133 | Large | Yes | 99.0 |
| 2 | [lightgbm-org/LightGBM](https://github.com/lightgbm-org/LightGBM) | 18728 | 4060 | 351 | C++ | MIT | 2026-08-29 | 2026-07-18 | 515 | Large | Yes | 96.2 |
| 3 | [dmlc/xgboost](https://github.com/dmlc/xgboost) | 28709 | 8891 | 703 | C++ | Apache-2.0 | 2026-08-27 | 2026-08-15 | 428 | Large | Yes | 96.0 |
| 4 | [ranaroussi/yfinance](https://github.com/ranaroussi/yfinance) | 25104 | 3411 | 185 | Python | Apache-2.0 | 2026-08-27 | 2026-08-26 | 104 | Large | Yes | 92.9 |
| 5 | [JerBouma/FinanceToolkit](https://github.com/JerBouma/FinanceToolkit) | 5272 | 614 | 10 | Python | MIT | 2026-08-27 | 2026-08-18 | 5 | Large | Yes | 92.8 |
| 6 | [microsoft/qlib](https://github.com/microsoft/qlib) | 48044 | 7607 | 152 | Python | MIT | 2026-07-23 | 2025-08-15 | 470 | Large | Yes | 91.9 |
| 7 | [dgunning/edgartools](https://github.com/dgunning/edgartools) | 2632 | 470 | 60 | Python | MIT | 2026-08-26 | 2026-08-25 | 25 | Large | Yes | 91.6 |
| 8 | [QuantConnect/Lean](https://github.com/QuantConnect/Lean) | 21392 | 5202 | 236 | C# | Apache-2.0 | 2026-08-28 | 2017-08-08 | 258 | Large | Yes | 91.4 |
| 9 | [stefan-jansen/machine-learning-for-trading](https://github.com/stefan-jansen/machine-learning-for-trading) | 20715 | 5574 | 17 | Jupyter Notebook | MIT | 2026-08-29 | 2026-07-24 | 10 | Large | Yes | 91.4 |
| 10 | [dcajasn/Riskfolio-Lib](https://github.com/dcajasn/Riskfolio-Lib) | 4471 | 704 | 7 | C++ | BSD-3-Clause | 2026-08-18 | None | 6 | Large | Yes | 90.5 |
| 11 | [TA-Lib/ta-lib-python](https://github.com/TA-Lib/ta-lib-python) | 12218 | 2000 | 41 | Cython | BSD-2-Clause | 2026-07-29 | 2026-07-16 | 138 | Large | Yes | 90.1 |
| 12 | [skfolio/skfolio](https://github.com/skfolio/skfolio) | 2304 | 237 | 25 | Python | BSD-3-Clause | 2026-08-29 | 2026-08-29 | 26 | Large | Yes | 89.0 |
| 13 | [PyPortfolio/PyPortfolioOpt](https://github.com/PyPortfolio/PyPortfolioOpt) | 5994 | 1172 | 53 | Jupyter Notebook | MIT | 2026-07-07 | 2026-02-26 | 112 | Large | Yes | 87.7 |
| 14 | [OpenBB-finance/OpenBB](https://github.com/OpenBB-finance/OpenBB) | 72440 | 7469 | 269 | Python | NOASSERTION | 2026-07-30 | 2026-04-25 | 107 | Large | Yes | 86.0 |
| 15 | [AI4Finance-Foundation/FinRL](https://github.com/AI4Finance-Foundation/FinRL) | 16142 | 3485 | 130 | Jupyter Notebook | MIT | 2026-07-13 | 2026-03-20 | 310 | Large | Yes | 84.9 |
| 16 | [polakowo/vectorbt](https://github.com/polakowo/vectorbt) | 8902 | 1144 | 21 | Python | NOASSERTION | 2026-08-02 | 2026-07-05 | 138 | Large | Yes | 84.8 |
| 17 | [ranaroussi/quantstats](https://github.com/ranaroussi/quantstats) | 7598 | 1228 | 36 | Python | Apache-2.0 | 2026-07-20 | 2026-01-13 | 31 | Basic | Yes | 83.5 |
| 18 | [alpacahq/alpaca-py](https://github.com/alpacahq/alpaca-py) | 1475 | 395 | 58 | Python | Apache-2.0 | 2026-08-24 | 2026-08-11 | 87 | Basic | Yes | 82.9 |
| 19 | [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) | 101681 | 19534 | 20 | Python | Apache-2.0 | 2026-07-18 | 2026-07-05 | 390 | Basic | Yes | 80.6 |
| 20 | [rsheftel/pandas_market_calendars](https://github.com/rsheftel/pandas_market_calendars) | 994 | 199 | 74 | Python | MIT | 2026-07-12 | 2025-04-07 | 18 | Basic | Yes | 79.5 |
| 21 | [TheAlgorithms/Python](https://github.com/TheAlgorithms/Python) | 224102 | 50999 | 1336 | Python | MIT | 2026-08-29 | None | 1064 | Large | Yes | 78.4 |
| 22 | [hummingbot/hummingbot](https://github.com/hummingbot/hummingbot) | 19690 | 4876 | 322 | Python | Apache-2.0 | 2026-08-28 | 2026-07-29 | 152 | Large | Yes | 78.3 |
| 23 | [OpenByteInc/QuantDinger](https://github.com/OpenByteInc/QuantDinger) | 11197 | 2342 | 26 | Python | Apache-2.0 | 2026-08-29 | 2026-08-18 | 60 | Large | Yes | 77.2 |
| 24 | [sktime/sktime](https://github.com/sktime/sktime) | 9973 | 2312 | 659 | Python | BSD-3-Clause | 2026-08-29 | 2026-07-28 | 2403 | Large | Yes | 77.0 |
| 25 | [unit8co/darts](https://github.com/unit8co/darts) | 9511 | 1034 | 164 | Python | Apache-2.0 | 2026-08-21 | 2026-07-20 | 229 | Large | Yes | 76.9 |
| 26 | [catboost/catboost](https://github.com/catboost/catboost) | 9085 | 1329 | 1426 | C++ | Apache-2.0 | 2026-08-29 | 2026-02-21 | 715 | Large | Yes | 76.8 |
| 27 | [quantopian/alphalens](https://github.com/quantopian/alphalens) | 4436 | 1350 | 26 | Jupyter Notebook | Apache-2.0 | 2024-02-12 | 2020-04-30 | 50 | Large | Yes | 76.5 |
| 28 | [man-group/ArcticDB](https://github.com/man-group/ArcticDB) | 2497 | 214 | 50 | C++ | NOASSERTION | 2026-08-29 | 2026-08-17 | 329 | Large | Yes | 75.5 |
| 29 | [coding-kitties/investing-algorithm-framework](https://github.com/coding-kitties/investing-algorithm-framework) | 2013 | 287 | 16 | Python | Apache-2.0 | 2026-08-28 | 2026-08-28 | 70 | Large | Yes | 74.1 |
| 30 | [pmorissette/bt](https://github.com/pmorissette/bt) | 2971 | 495 | 39 | Python | MIT | 2026-08-07 | 2026-04-25 | 87 | Large | Yes | 73.9 |
| 31 | [AI4Finance-Foundation/FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) | 21178 | 3004 | 47 | Jupyter Notebook | MIT | 2026-08-02 | 2026-04-08 | 86 | Large | No | 73.8 |
| 32 | [LLMQuant/quant-mind](https://github.com/LLMQuant/quant-mind) | 2733 | 459 | 9 | Python | MIT | 2026-08-15 | None | 35 | Large | Yes | 72.7 |
| 33 | [zvtvz/zvt](https://github.com/zvtvz/zvt) | 4288 | 1013 | 68 | Python | MIT | 2026-07-01 | 2026-01-18 | 22 | Large | Yes | 72.5 |
| 34 | [AI4Finance-Foundation/FinRobot](https://github.com/AI4Finance-Foundation/FinRobot) | 7889 | 1340 | 7 | Jupyter Notebook | Apache-2.0 | 2026-08-23 | 2026-07-07 | 74 | Basic | No | 72.1 |
| 35 | [akfamily/akshare](https://github.com/akfamily/akshare) | 22300 | 3478 | 23 | Python | MIT | 2026-08-28 | 2026-08-21 | 1 | Basic | Yes | 71.7 |
| 36 | [edtechre/pybroker](https://github.com/edtechre/pybroker) | 3519 | 453 | 12 | Python | NOASSERTION | 2026-08-28 | 2026-08-28 | 0 | Basic | Yes | 71.3 |
| 37 | [UFund-Me/Qbot](https://github.com/UFund-Me/Qbot) | 18416 | 2592 | 3 | Jupyter Notebook | MIT | 2026-03-11 | 2024-06-16 | 74 | Large | Yes | 71.2 |
| 38 | [kernc/backtesting.py](https://github.com/kernc/backtesting.py) | 8908 | 1522 | 46 | Python | AGPL-3.0 | 2026-08-05 | None | 69 | Large | Yes | 70.6 |
| 39 | [Lumiwealth/lumibot](https://github.com/Lumiwealth/lumibot) | 2009 | 383 | 43 | Python | GPL-3.0 | 2026-08-29 | 2026-08-26 | 85 | Large | Yes | 69.1 |
| 40 | [freqtrade/freqtrade](https://github.com/freqtrade/freqtrade) | 53804 | 11174 | 400 | Python | GPL-3.0 | 2026-08-27 | 2026-07-31 | 35 | Large | Yes | 69.0 |
| 41 | [nautechsystems/nautilus_trader](https://github.com/nautechsystems/nautilus_trader) | 28084 | 3627 | 187 | Rust | LGPL-3.0 | 2026-08-29 | 2026-08-21 | 112 | Large | Yes | 68.9 |
| 42 | [OpenBB-finance/OpenBB](https://github.com/OpenBB-finance/OpenBB) | 72440 | 7469 | 269 | Python | NOASSERTION | 2026-07-30 | 2026-04-25 | 107 | Large | Yes | 67.0 |
| 43 | [google/tf-quant-finance](https://github.com/google/tf-quant-finance) | 5486 | 694 | 49 | Python | Apache-2.0 | 2026-08-06 | 2019-09-17 | 42 | Large | No | 66.8 |
| 44 | [lballabio/QuantLib](https://github.com/lballabio/QuantLib) | 7551 | 2311 | 299 | C++ | NOASSERTION | 2026-08-28 | 2026-07-14 | 49 | Large | Yes | 66.5 |
| 45 | [quantopian/zipline](https://github.com/quantopian/zipline) | 20072 | 5043 | 161 | Python | Apache-2.0 | 2024-02-13 | 2020-10-05 | 370 | Large | Yes | 66.3 |
| 46 | [The-Swarm-Corporation/AutoHedge](https://github.com/The-Swarm-Corporation/AutoHedge) | 4312 | 726 | 2 | Python | MIT | 2026-05-11 | None | 17 | Basic | Yes | 62.8 |
| 47 | [je-suis-tm/quant-trading](https://github.com/je-suis-tm/quant-trading) | 10643 | 1876 | 4 | Python | Apache-2.0 | 2026-06-20 | None | 4 | Large | No | 62.7 |
| 48 | [tradingstrategy-ai/trade-executor](https://github.com/tradingstrategy-ai/trade-executor) | 159 | 36 | 13 | Jupyter Notebook | NOASSERTION | 2026-08-28 | 2026-06-05 | 49 | Large | Yes | 59.4 |
| 49 | [OpenSourceRisk/Engine](https://github.com/OpenSourceRisk/Engine) | 779 | 303 | 178 | C++ | NOASSERTION | 2026-06-11 | 2026-05-21 | 84 | Large | Yes | 59.3 |
| 50 | [asavinov/intelligent-trading-bot](https://github.com/asavinov/intelligent-trading-bot) | 1860 | 404 | 5 | Python | MIT | 2026-08-11 | None | 47 | Basic | No | 56.4 |
| 51 | [bukosabino/ta](https://github.com/bukosabino/ta) | 5181 | 1154 | 34 | Jupyter Notebook | MIT | 2026-03-18 | None | 157 | Basic | No | 55.7 |
| 52 | [mementum/backtrader](https://github.com/mementum/backtrader) | 23014 | 5254 | 56 | Python | GPL-3.0 | 2024-08-19 | None | 63 | Large | No | 55.6 |
| 53 | [letianzj/QuantResearch](https://github.com/letianzj/QuantResearch) | 3009 | 574 | 4 | Jupyter Notebook | MIT | 2023-08-26 | None | 1 | Large | No | 54.8 |
| 54 | [bbfamily/abu](https://github.com/bbfamily/abu) | 18258 | 4675 | 3 | Python | GPL-3.0 | 2026-01-24 | 2019-06-20 | 6 | Large | No | 53.1 |
| 55 | [jugaad-py/jugaad-data](https://github.com/jugaad-py/jugaad-data) | 566 | 204 | 7 | Python | NOASSERTION | 2026-08-25 | None | 26 | Basic | Yes | 51.8 |
| 56 | [hugo2046/QuantsPlaybook](https://github.com/hugo2046/QuantsPlaybook) | 5914 | 1397 | 2 | Jupyter Notebook | NOASSERTION | 2026-05-08 | None | 9 | Large | No | 51.1 |
| 57 | [VivekPa/AIAlpha](https://github.com/VivekPa/AIAlpha) | 1956 | 449 | 2 | Python | MIT | 2020-06-18 | None | 14 | Large | No | 50.9 |
| 58 | [FinanceData/FinanceDataReader](https://github.com/FinanceData/FinanceDataReader) | 1536 | 407 | 15 | Python | MIT | 2026-05-13 | None | 50 | Basic | No | 50.2 |
| 59 | [hudson-and-thames/mlfinlab](https://github.com/hudson-and-thames/mlfinlab) | 4915 | 1286 | 3 | Python | NOASSERTION | 2023-10-02 | None | 49 | Basic | No | 36.0 |
| 60 | [virattt/ai-financial-agent](https://github.com/virattt/ai-financial-agent) | 2025 | 407 | 2 | TypeScript | NOASSERTION | 2025-08-19 | None | 5 | Basic | No | 34.9 |

## Deep evaluations (30)

### scikit-learn/scikit-learn — 99.0/100

- **Purpose:** general machine-learning toolkit
- **Strongest feature:** reproducible simple-model baselines and calibration
- **Weakness / overlap:** not finance-specific; data splitting remains this project's responsibility
- **US-equity swing-trading decision:** incorporated runtime dependency
- **Live metadata:** 67095 stars, 27332 forks, 3557 contributors, Python, license BSD-3-Clause, latest activity 2026-08-28, release 2026-06-02, CI Yes, open issues 2133.

### lightgbm-org/LightGBM — 96.2/100

- **Purpose:** gradient-boosted tree implementation
- **Strongest feature:** strong tabular challenger-model baseline
- **Weakness / overlap:** may overfit temporal data without strict walk-forward control
- **US-equity swing-trading decision:** future optional challenger dependency
- **Live metadata:** 18728 stars, 4060 forks, 351 contributors, C++, license MIT, latest activity 2026-08-29, release 2026-07-18, CI Yes, open issues 515.

### dmlc/xgboost — 96.0/100

- **Purpose:** gradient-boosted tree implementation
- **Strongest feature:** well-established nonlinear baseline
- **Weakness / overlap:** complexity is not proof of trading edge
- **US-equity swing-trading decision:** future optional challenger dependency
- **Live metadata:** 28709 stars, 8891 forks, 703 contributors, C++, license Apache-2.0, latest activity 2026-08-27, release 2026-08-15, CI Yes, open issues 428.

### ranaroussi/yfinance — 92.9/100

- **Purpose:** unofficial market-data convenience client
- **Strongest feature:** low-friction daily OHLCV retrieval
- **Weakness / overlap:** not an authoritative point-in-time or licensed institutional feed
- **US-equity swing-trading decision:** starter provider only
- **Live metadata:** 25104 stars, 3411 forks, 185 contributors, Python, license Apache-2.0, latest activity 2026-08-27, release 2026-08-26, CI Yes, open issues 104.

### JerBouma/FinanceToolkit — 92.8/100

- **Purpose:** transparent financial statement analysis toolkit
- **Strongest feature:** fundamental-ratio coverage
- **Weakness / overlap:** provider/data provenance needs checking per endpoint
- **US-equity swing-trading decision:** future fundamental adapter candidate
- **Live metadata:** 5272 stars, 614 forks, 10 contributors, Python, license MIT, latest activity 2026-08-27, release 2026-08-18, CI Yes, open issues 5.

### microsoft/qlib — 91.9/100

- **Purpose:** ML-oriented quantitative research platform
- **Strongest feature:** dataset, model, and workflow concepts
- **Weakness / overlap:** substantial data-format and orchestration commitment
- **US-equity swing-trading decision:** research and model-registry influence
- **Live metadata:** 48044 stars, 7607 forks, 152 contributors, Python, license MIT, latest activity 2026-07-23, release 2025-08-15, CI Yes, open issues 470.

### dgunning/edgartools — 91.6/100

- **Purpose:** typed SEC EDGAR filings toolkit
- **Strongest feature:** filing, XBRL, insider, and 13F coverage
- **Weakness / overlap:** filing availability timestamps still need pipeline validation
- **US-equity swing-trading decision:** future isolated SEC adapter candidate
- **Live metadata:** 2632 stars, 470 forks, 60 contributors, Python, license MIT, latest activity 2026-08-26, release 2026-08-25, CI Yes, open issues 25.

### QuantConnect/Lean — 91.4/100

- **Purpose:** full algorithmic trading and simulation engine
- **Strongest feature:** mature event-driven execution model
- **Weakness / overlap:** C#-centered infrastructure exceeds starter scope
- **US-equity swing-trading decision:** execution-assumption reference only
- **Live metadata:** 21392 stars, 5202 forks, 236 contributors, C#, license Apache-2.0, latest activity 2026-08-28, release 2017-08-08, CI Yes, open issues 258.

### stefan-jansen/machine-learning-for-trading — 91.4/100

- **Purpose:** ML-for-markets educational and research material
- **Strongest feature:** broad reproducible research examples
- **Weakness / overlap:** book companion, not a production dependency
- **US-equity swing-trading decision:** methodology influence only
- **Live metadata:** 20715 stars, 5574 forks, 17 contributors, Jupyter Notebook, license MIT, latest activity 2026-08-29, release 2026-07-24, CI Yes, open issues 10.

### dcajasn/Riskfolio-Lib — 90.5/100

- **Purpose:** risk-aware portfolio optimization
- **Strongest feature:** broad risk-measure support
- **Weakness / overlap:** advanced optimization adds dependency and model risk
- **US-equity swing-trading decision:** future constrained-allocation candidate
- **Live metadata:** 4471 stars, 704 forks, 7 contributors, C++, license BSD-3-Clause, latest activity 2026-08-18, release None, CI Yes, open issues 6.

### TA-Lib/ta-lib-python — 90.1/100

- **Purpose:** technical-indicator Python bindings
- **Strongest feature:** battle-tested indicator breadth
- **Weakness / overlap:** compiled dependency and indicator proliferation risk
- **US-equity swing-trading decision:** optional indicator adapter candidate
- **Live metadata:** 12218 stars, 2000 forks, 41 contributors, Cython, license BSD-2-Clause, latest activity 2026-07-29, release 2026-07-16, CI Yes, open issues 138.

### skfolio/skfolio — 89.0/100

- **Purpose:** scikit-learn-oriented portfolio optimization
- **Strongest feature:** estimator-style integration
- **Weakness / overlap:** portfolio layer belongs after signal validation
- **US-equity swing-trading decision:** future portfolio candidate
- **Live metadata:** 2304 stars, 237 forks, 25 contributors, Python, license BSD-3-Clause, latest activity 2026-08-29, release 2026-08-29, CI Yes, open issues 26.

### PyPortfolio/PyPortfolioOpt — 87.7/100

- **Purpose:** portfolio optimization library
- **Strongest feature:** well-known constrained allocation methods
- **Weakness / overlap:** estimation error can dominate optimized weights
- **US-equity swing-trading decision:** future portfolio optimizer candidate
- **Live metadata:** 5994 stars, 1172 forks, 53 contributors, Jupyter Notebook, license MIT, latest activity 2026-07-07, release 2026-02-26, CI Yes, open issues 112.

### OpenBB-finance/OpenBB — 86.0/100

- **Purpose:** modular financial-data and research platform
- **Strongest feature:** provider abstractions and broad market-data coverage
- **Weakness / overlap:** large platform; avoid importing its application layer
- **US-equity swing-trading decision:** architecture influence only
- **Live metadata:** 72440 stars, 7469 forks, 269 contributors, Python, license NOASSERTION, latest activity 2026-07-30, release 2026-04-25, CI Yes, open issues 107.

### AI4Finance-Foundation/FinRL — 84.9/100

- **Purpose:** reinforcement-learning finance research
- **Strongest feature:** formal environments and evaluation framing
- **Weakness / overlap:** RL is high variance and unsuitable as an unvalidated default
- **US-equity swing-trading decision:** future challenger-research input
- **Live metadata:** 16142 stars, 3485 forks, 130 contributors, Jupyter Notebook, license MIT, latest activity 2026-07-13, release 2026-03-20, CI Yes, open issues 310.

### polakowo/vectorbt — 84.8/100

- **Purpose:** vectorized strategy research and parameter exploration
- **Strongest feature:** fast wide parameter sweeps
- **Weakness / overlap:** vectorization can obscure execution-time semantics
- **US-equity swing-trading decision:** optional research adapter after event-engine validation
- **Live metadata:** 8902 stars, 1144 forks, 21 contributors, Python, license NOASSERTION, latest activity 2026-08-02, release 2026-07-05, CI Yes, open issues 138.

### ranaroussi/quantstats — 83.5/100

- **Purpose:** portfolio and strategy performance analytics
- **Strongest feature:** recognizable reporting metric conventions
- **Weakness / overlap:** metric definitions must be independently regression-tested
- **US-equity swing-trading decision:** metric naming and comparison influence
- **Live metadata:** 7598 stars, 1228 forks, 36 contributors, Python, license Apache-2.0, latest activity 2026-07-20, release 2026-01-13, CI Yes, open issues 31.

### alpacahq/alpaca-py — 82.9/100

- **Purpose:** official Alpaca Python SDK
- **Strongest feature:** paper-account and market-data integration path
- **Weakness / overlap:** broker integration is out of scope for this research-only phase
- **US-equity swing-trading decision:** future paper-broker adapter candidate
- **Live metadata:** 1475 stars, 395 forks, 58 contributors, Python, license Apache-2.0, latest activity 2026-08-24, release 2026-08-11, CI Yes, open issues 87.

### TauricResearch/TradingAgents — 80.6/100

- **Purpose:** multi-agent investment research and debate
- **Strongest feature:** bull/bear/risk committee decomposition
- **Weakness / overlap:** agent consensus is not predictive evidence
- **US-equity swing-trading decision:** agent taxonomy influence only
- **Live metadata:** 101681 stars, 19534 forks, 20 contributors, Python, license Apache-2.0, latest activity 2026-07-18, release 2026-07-05, CI Yes, open issues 390.

### rsheftel/pandas_market_calendars — 79.5/100

- **Purpose:** exchange trading calendars
- **Strongest feature:** NYSE session and holiday correctness
- **Weakness / overlap:** calendar package does not validate data availability
- **US-equity swing-trading decision:** incorporated runtime dependency
- **Live metadata:** 994 stars, 199 forks, 74 contributors, Python, license MIT, latest activity 2026-07-12, release 2025-04-07, CI Yes, open issues 18.

### quantopian/alphalens — 76.5/100

- **Purpose:** factor-return analysis
- **Strongest feature:** factor diagnostics and turnover framing
- **Weakness / overlap:** project activity and dependency fit need review
- **US-equity swing-trading decision:** factor-evaluation influence only
- **Live metadata:** 4436 stars, 1350 forks, 26 contributors, Jupyter Notebook, license Apache-2.0, latest activity 2024-02-12, release 2020-04-30, CI Yes, open issues 50.

### man-group/ArcticDB — 75.5/100

- **Purpose:** versioned dataframe database
- **Strongest feature:** research-data versioning ideas
- **Weakness / overlap:** operational complexity is excessive for the local starter
- **US-equity swing-trading decision:** future storage architecture reference
- **Live metadata:** 2497 stars, 214 forks, 50 contributors, C++, license NOASSERTION, latest activity 2026-08-29, release 2026-08-17, CI Yes, open issues 329.

### AI4Finance-Foundation/FinGPT — 73.8/100

- **Purpose:** financial language-model research
- **Strongest feature:** finance-specific language evaluation ideas
- **Weakness / overlap:** model weights and licensing/data scope require separate review
- **US-equity swing-trading decision:** agent-evaluation influence only
- **Live metadata:** 21178 stars, 3004 forks, 47 contributors, Jupyter Notebook, license MIT, latest activity 2026-08-02, release 2026-04-08, CI No, open issues 86.

### AI4Finance-Foundation/FinRobot — 72.1/100

- **Purpose:** financial-agent workflow framework
- **Strongest feature:** tool-oriented analyst roles
- **Weakness / overlap:** LLM output must remain non-decisioning here
- **US-equity swing-trading decision:** agent boundaries influence only
- **Live metadata:** 7889 stars, 1340 forks, 7 contributors, Jupyter Notebook, license Apache-2.0, latest activity 2026-08-23, release 2026-07-07, CI No, open issues 74.

### edtechre/pybroker — 71.3/100

- **Purpose:** Python algorithmic backtesting framework
- **Strongest feature:** ML-aware strategy interfaces
- **Weakness / overlap:** must compare timestamp and cost semantics carefully
- **US-equity swing-trading decision:** backtest API comparison input
- **Live metadata:** 3519 stars, 453 forks, 12 contributors, Python, license NOASSERTION, latest activity 2026-08-28, release 2026-08-28, CI Yes, open issues 0.

### kernc/backtesting.py — 70.6/100

- **Purpose:** compact strategy backtesting library
- **Strongest feature:** approachable strategy prototyping
- **Weakness / overlap:** AGPL-3.0 prevents direct dependency in this plan
- **US-equity swing-trading decision:** excluded; methodology comparison only
- **Live metadata:** 8908 stars, 1522 forks, 46 contributors, Python, license AGPL-3.0, latest activity 2026-08-05, release None, CI Yes, open issues 69.

### Lumiwealth/lumibot — 69.1/100

- **Purpose:** multi-asset backtesting and broker framework
- **Strongest feature:** paper-trading lifecycle concepts
- **Weakness / overlap:** GPL-3.0 and broker scope conflict with current phase
- **US-equity swing-trading decision:** excluded; lifecycle influence only
- **Live metadata:** 2009 stars, 383 forks, 43 contributors, Python, license GPL-3.0, latest activity 2026-08-29, release 2026-08-26, CI Yes, open issues 85.

### freqtrade/freqtrade — 69.0/100

- **Purpose:** full crypto trading bot
- **Strongest feature:** operational-risk and configuration patterns
- **Weakness / overlap:** crypto focus and GPL-3.0 are wrong for US-equity research
- **US-equity swing-trading decision:** excluded; operations reference only
- **Live metadata:** 53804 stars, 11174 forks, 400 contributors, Python, license GPL-3.0, latest activity 2026-08-27, release 2026-07-31, CI Yes, open issues 35.

### OpenBB-finance/OpenBB — 67.0/100

- **Purpose:** legacy terminal application
- **Strongest feature:** historical product breadth
- **Weakness / overlap:** superseded application; not selected as a dependency
- **US-equity swing-trading decision:** excluded; historical comparison only
- **Live metadata:** 72440 stars, 7469 forks, 269 contributors, Python, license NOASSERTION, latest activity 2026-07-30, release 2026-04-25, CI Yes, open issues 107.

### mementum/backtrader — 55.6/100

- **Purpose:** event-driven Python backtester
- **Strongest feature:** familiar strategy interface
- **Weakness / overlap:** GPL-3.0 is incompatible with this MIT integration plan
- **US-equity swing-trading decision:** excluded; methodology comparison only
- **Live metadata:** 23014 stars, 5254 forks, 56 contributors, Python, license GPL-3.0, latest activity 2024-08-19, release None, CI No, open issues 63.

## Selected coherent stack

The initial implementation uses original code plus pandas, NumPy, scikit-learn, Pydantic, PyYAML, requests, yfinance, and pandas-market-calendars. The selected projects influenced architecture, not copied source. The only incorporated package decisions at this stage are scikit-learn (baseline modelling) and pandas-market-calendars (NYSE timing); yfinance is a clearly labeled convenience provider. EDGAR, fundamental, portfolio, and vectorized research adapters remain separate future work until their exact versions, licenses, data terms, and tests are reviewed.

## Exclusions and next review

Crypto-first projects were not selected for the US-equity core. GPL/AGPL candidates are not integrated. Before adding any candidate, inspect the pinned release, dependency tree, license, data terms, API behavior, and point-in-time implications; then update `THIRD_PARTY_NOTICES.md`.
