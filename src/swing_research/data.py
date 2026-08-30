"""Market and SEC provider boundaries with timestamp-oriented validation."""

from __future__ import annotations

import os
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import requests

from .schemas import NewsArticle


class DataValidationError(ValueError):
    pass


class DataStaleError(DataValidationError):
    pass


class PriceProvider(Protocol):
    def fetch_daily(self, ticker: str, start: datetime, end: datetime) -> pd.DataFrame: ...


REQUIRED_OHLCV_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def validate_ohlcv(frame: pd.DataFrame, as_of: datetime | None = None) -> pd.DataFrame:
    """Return canonical OHLCV and reject malformed or future-visible bars."""
    if frame.empty:
        raise DataValidationError("No market data returned")
    normalized = frame.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = normalized.columns.get_level_values(0)
    normalized.columns = [str(column).lower().replace(" ", "_") for column in normalized.columns]
    missing = REQUIRED_OHLCV_COLUMNS.difference(normalized.columns)
    if missing:
        raise DataValidationError(f"OHLCV missing required columns: {sorted(missing)}")
    normalized.index = pd.to_datetime(normalized.index, utc=True).tz_convert(None).normalize()
    normalized = normalized[~normalized.index.duplicated(keep="last")].sort_index()
    normalized = normalized.loc[:, ["open", "high", "low", "close", "volume"]].astype(float)
    missing_price = normalized[["open", "high", "low", "close"]].isna().any(axis=1)
    if missing_price.any():
        first_missing_position = int(np.flatnonzero(missing_price.to_numpy())[0])
        if not missing_price.iloc[first_missing_position:].all():
            raise DataValidationError("Incomplete OHLCV bar occurs inside the requested history")
        normalized = normalized.iloc[:first_missing_position]
    if normalized.empty:
        raise DataValidationError("No completed OHLCV bars returned")
    if normalized["volume"].isna().any():
        raise DataValidationError("Volume is missing from a completed OHLCV bar")
    if (normalized[["open", "high", "low", "close"]] <= 0).any().any():
        raise DataValidationError("Non-positive price present")
    if (normalized["volume"] < 0).any():
        raise DataValidationError("Negative volume present")
    upper_reference = normalized[["open", "close", "low"]].max(axis=1)
    lower_reference = normalized[["open", "close", "high"]].min(axis=1)
    tolerance = normalized[["open", "high", "low", "close"]].abs().max(axis=1) * 1e-10
    if (normalized["high"] + tolerance < upper_reference).any():
        raise DataValidationError("High is below another bar price")
    if (normalized["low"] - tolerance > lower_reference).any():
        raise DataValidationError("Low is above another bar price")
    if as_of is not None:
        cutoff = pd.Timestamp(as_of).tz_localize(None).normalize()
        if (normalized.index > cutoff).any():
            raise DataValidationError("Provider returned a bar after the requested as-of date")
        normalized = normalized.loc[:cutoff]
    return normalized


def assert_fresh(last_bar: pd.Timestamp, now: datetime, max_age: timedelta) -> None:
    """Reject stale critical price data; calendar exceptions are handled before calling this."""
    observed = last_bar.to_pydatetime().replace(tzinfo=UTC)
    if now.astimezone(UTC) - observed > max_age:
        raise DataStaleError(
            f"Latest completed bar {last_bar.date()} exceeds freshness limit {max_age}"
        )


class YFinancePriceProvider:
    """Optional convenience adapter; not a survivorship-bias-free data source."""

    def fetch_daily(self, ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - dependency declaration covers normal use
            raise RuntimeError("Install the yfinance optional market-data dependency") from exc
        raw = yf.download(
            ticker,
            start=start.date().isoformat(),
            end=(end.date() + timedelta(days=1)).isoformat(),
            auto_adjust=True,
            progress=False,
            actions=False,
            threads=False,
        )
        return validate_ohlcv(raw, as_of=end)


class LocalCsvPriceProvider:
    """Validated daily OHLCV exports from a licensed point-in-time data vendor.

    Each file is named ``<ticker>.csv`` and contains a ``date`` column plus OHLCV columns.
    This intentionally accepts vendor exports rather than claiming that a free source has
    delisting-aware history.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).expanduser().resolve()
        if not self.directory.is_dir():
            raise ValueError(f"LOCAL_OHLCV_DIRECTORY does not exist: {self.directory}")

    def _path_for(self, ticker: str) -> Path:
        path = (self.directory / f"{ticker}.csv").resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError(f"Unsafe ticker filename: {ticker}")
        return path

    def fetch_daily(self, ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
        path = self._path_for(ticker)
        if not path.is_file():
            raise FileNotFoundError(f"No local OHLCV export for {ticker}: {path}")
        raw = pd.read_csv(path)
        date_columns = [column for column in raw.columns if str(column).lower() == "date"]
        if len(date_columns) != 1:
            raise DataValidationError(f"{path.name} must include exactly one date column")
        date_column = date_columns[0]
        raw.index = pd.to_datetime(raw.pop(date_column), errors="raise", utc=True)
        normalized = validate_ohlcv(raw, as_of=end)
        start_day = pd.Timestamp(start).tz_localize(None).normalize()
        return normalized.loc[start_day:]


def configured_price_provider() -> PriceProvider:
    """Choose an explicit local export or the convenience yfinance adapter from environment."""
    configured = os.getenv("MARKET_DATA_PROVIDER", "yfinance").strip().lower()
    if configured == "yfinance":
        return YFinancePriceProvider()
    if configured == "local_csv":
        directory = os.getenv("LOCAL_OHLCV_DIRECTORY")
        if not directory:
            raise ValueError(
                "LOCAL_OHLCV_DIRECTORY is required when MARKET_DATA_PROVIDER=local_csv"
            )
        return LocalCsvPriceProvider(directory)
    raise ValueError(f"Unsupported MARKET_DATA_PROVIDER: {configured}")


class SecEdgarClient:
    """Official SEC issuer submissions client with a polite rate limit."""

    base_url = "https://data.sec.gov/submissions"

    def __init__(self, user_agent: str | None = None, min_interval_seconds: float = 0.125) -> None:
        configured_user_agent = user_agent or os.getenv("SEC_USER_AGENT", "")
        if not configured_user_agent or "@" not in configured_user_agent:
            raise ValueError("SEC_USER_AGENT must include a descriptive contact email")
        self.user_agent: str = configured_user_agent
        self.min_interval_seconds = min_interval_seconds
        self._last_request_at = 0.0
        self.session = requests.Session()

    def issuer_submissions(self, cik: str) -> tuple[dict[str, object], str]:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)
        normalized_cik = str(cik).zfill(10)
        url = f"{self.base_url}/CIK{normalized_cik}.json"
        response = self.session.get(url, headers={"User-Agent": self.user_agent}, timeout=20)
        self._last_request_at = time.monotonic()
        response.raise_for_status()
        return response.json(), url


class FinnhubNewsClient:
    """Timestamped Finnhub company-news adapter for research evidence.

    The adapter deliberately returns source records only. It does not turn headlines into a
    trading score because the available-at timestamp and historic coverage need separate
    validation before a news feature can enter a backtest.
    """

    base_url = "https://finnhub.io/api/v1"

    def __init__(
        self,
        api_key: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        configured_api_key = api_key or os.getenv("FINNHUBKEY", "")
        if not configured_api_key:
            raise ValueError("FINNHUBKEY is required to request Finnhub news")
        self.api_key = configured_api_key
        self.session = session or requests.Session()

    def company_news(
        self,
        ticker: str,
        start: date,
        end: date,
        as_of: datetime | None = None,
    ) -> list[NewsArticle]:
        """Return unique North-American company articles published no later than ``as_of``.

        Finnhub's company-news endpoint is free-tier eligible for one year of history, but this
        method does not infer that every historical article was available at its publish time.
        Callers must retain the returned timestamps and source URL as evidence.
        """
        if end < start:
            raise ValueError("Finnhub news end date must not precede start date")
        normalized_ticker = ticker.strip().upper()
        if not normalized_ticker:
            raise ValueError("Finnhub news ticker is required")
        response = self.session.get(
            f"{self.base_url}/company-news",
            params={
                "symbol": normalized_ticker,
                "from": start.isoformat(),
                "to": end.isoformat(),
            },
            headers={"X-Finnhub-Token": self.api_key},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise DataValidationError("Finnhub company-news response must be a JSON list")
        observed_at = datetime.now(UTC)
        cutoff = as_of.astimezone(UTC) if as_of is not None else None
        articles: list[NewsArticle] = []
        seen_ids: set[int | str] = set()
        for item in payload:
            if not isinstance(item, dict):
                raise DataValidationError("Finnhub company-news item must be an object")
            article_id = item.get("id")
            published_timestamp = item.get("datetime")
            headline = item.get("headline")
            source = item.get("source")
            url = item.get("url")
            if (
                article_id is None
                or not isinstance(published_timestamp, int | float)
                or not isinstance(headline, str)
                or not isinstance(source, str)
                or not isinstance(url, str)
            ):
                raise DataValidationError("Finnhub company-news item is missing required evidence")
            published_at = datetime.fromtimestamp(published_timestamp, UTC)
            if cutoff is not None and published_at > cutoff:
                continue
            if article_id in seen_ids:
                continue
            seen_ids.add(article_id)
            summary = item.get("summary")
            articles.append(
                NewsArticle(
                    article_id=article_id,
                    ticker=normalized_ticker,
                    published_at=published_at,
                    retrieved_at=observed_at,
                    source=source,
                    headline=headline,
                    summary=summary if isinstance(summary, str) else "",
                    url=url,
                )
            )
        return sorted(articles, key=lambda article: (article.published_at, str(article.article_id)))


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]
