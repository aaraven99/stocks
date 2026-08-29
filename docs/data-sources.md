# Data sources

`YFinancePriceProvider` is a convenience adapter for adjusted daily OHLCV. It validates basic
shape and timestamps but does not solve survivorship bias, licensing, or revision history. Use a
licensed point-in-time vendor and historical constituent files before making broad performance
claims.

`SecEdgarClient` uses the official `data.sec.gov` issuer-submissions endpoint and requires a
descriptive `SEC_USER_AGENT`. It rate-limits requests and returns raw source URLs for audit.

