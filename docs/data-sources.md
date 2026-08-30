# Data sources

`YFinancePriceProvider` is a convenience adapter for adjusted daily OHLCV. It validates basic
shape and timestamps but does not solve survivorship bias, licensing, or revision history. Use a
licensed point-in-time vendor and historical constituent files before making broad performance
claims. The yfinance project describes itself as an unaffiliated research/education library and
directs users to Yahoo's terms for data rights; configure it only for personal local research,
store its outputs in ignored paths, and do not publish downloaded data or derived performance
claims from it without obtaining the necessary rights. The local daily pipeline requests its small
starter universe in serial, bounded two-symbol batches, so one slow ticker cannot stall the entire
workflow or trigger an unbounded concurrent request burst.

`SecEdgarClient` uses the official `data.sec.gov` issuer-submissions endpoint and requires a
descriptive `SEC_USER_AGENT`; it does **not** need an SEC API key. It rate-limits requests and
returns raw source URLs for audit.

`FinnhubNewsClient` uses `FINNHUBKEY` and sends it in the `X-Finnhub-Token` request header rather
than exposing it in an application URL. It returns deduplicated, timestamped company-news evidence
only; no Finnhub headline can change a quantitative score until availability timing and historical
coverage have passed a separate point-in-time study. Finnhub documents one year of company-news
history on its free tier and requires an API token for GET requests. As this repository is public,
raw Finnhub data and any Finnhub-derived output are prohibited from being committed, published, or
shared unless Finnhub gives written redistribution approval. The client has no persistence or
reporting method; keep any permitted personal research local under the ignored `data/private/` and
`reports/private/` directories. `MARKET_DATA_PROVIDER=finnhub` is deliberately blocked: the
project does not assume that an account is licensed for Finnhub stock-candle access, and it will
not silently substitute a paid endpoint or attempt to bypass an entitlement restriction.

The proposed Hugging Face dataset
[`mito0o852/OHLCV-1m`](https://huggingface.co/datasets/mito0o852/OHLCV-1m) is recorded as a
candidate, not an admissible backtest provider. Its public card says it republishes minute data
originally from Finnhub, has files through March 2026, and does not declare a dataset license. It
also does not establish corporate-action treatment, point-in-time constituent membership, delisted
security coverage, or revision history. Do not download its 87+ GB archive or use it in the
stock-universe study unless both the publisher supplies a compatible license and Finnhub confirms
that the redistribution and intended use are authorized, and the required independent coverage
audit passes.

`Sp500HistoricalConstituentSource` pins an MIT-licensed community interval file from
[`fja05680/sp500`](https://github.com/fja05680/sp500) to commit
`c31ac3cc56f28cf9a02b4e694eff7ceab596a0ff`. It is useful for point-in-time membership but is
not official S&P data; its upstream notes identify possible early-history gaps and explain that
Yahoo does not supply complete delisted-security price history. The adapter therefore requires at
least 98% verified price coverage before a broad historical claim can proceed. It is not used by
the ETF research results committed so far.

Run `uv run swing-research universe-audit --as-of YYYY-MM-DD` before a historical stock-universe
experiment. The command writes an artifact even if its coverage gate fails, preserving the exact
unavailable tickers rather than silently dropping them.

For a licensed delisting-aware vendor, set `MARKET_DATA_PROVIDER=local_csv` and set
`LOCAL_OHLCV_DIRECTORY` to a directory of vendor exports named `<ticker>.csv`. Each export must
contain `date`, `open`, `high`, `low`, `close`, and `volume`; the adapter applies the same
point-in-time validation and the universe audit still must clear its coverage gate. Norgate Data
is a compatible Windows/Python candidate because its US Platinum/Diamond tiers advertise
historical constituents and delisted securities, but no subscription is assumed or bundled.

For the cross-sectional stock study, also provide `constituent_intervals.csv` in that directory
with `ticker,start_date,end_date` columns. The study uses only each prior-close membership row,
requires 98% aggregate active-member price coverage, and rejects missing prices for any holding.
