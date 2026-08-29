# Data sources

`YFinancePriceProvider` is a convenience adapter for adjusted daily OHLCV. It validates basic
shape and timestamps but does not solve survivorship bias, licensing, or revision history. Use a
licensed point-in-time vendor and historical constituent files before making broad performance
claims.

`SecEdgarClient` uses the official `data.sec.gov` issuer-submissions endpoint and requires a
descriptive `SEC_USER_AGENT`. It rate-limits requests and returns raw source URLs for audit.

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
