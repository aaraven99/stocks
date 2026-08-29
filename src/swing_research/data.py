"""Market and SEC provider boundaries with timestamp-oriented validation."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import requests


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
    if (normalized["high"] < normalized[["open", "close", "low"]].max(axis=1)).any():
        raise DataValidationError("High is below another bar price")
    if (normalized["low"] > normalized[["open", "close", "high"]].min(axis=1)).any():
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


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]
