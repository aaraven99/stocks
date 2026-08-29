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
