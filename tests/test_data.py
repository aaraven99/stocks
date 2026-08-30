from __future__ import annotations

import sys
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from swing_research.data import (
    DataValidationError,
    FinnhubNewsClient,
    LocalCsvPriceProvider,
    YFinancePriceProvider,
    configured_price_provider,
    validate_ohlcv,
)


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [10.0, 11.0],
            "High": [11.0, 12.0],
            "Low": [9.0, 10.0],
            "Close": [10.5, 11.5],
            "Volume": [100, 200],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )


def test_ohlcv_rejects_future_bar_for_as_of() -> None:
    with pytest.raises(DataValidationError, match="after the requested"):
        validate_ohlcv(_bars(), datetime(2024, 1, 2, tzinfo=UTC))


def test_ohlcv_is_canonical() -> None:
    result = validate_ohlcv(_bars())
    assert list(result.columns) == ["open", "high", "low", "close", "volume"]
    assert result.index.is_monotonic_increasing


def test_ohlcv_drops_only_a_trailing_incomplete_bar() -> None:
    bars = _bars()
    bars.loc[pd.Timestamp("2024-01-04")] = [
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        10,
    ]
    result = validate_ohlcv(bars)
    assert result.index[-1] == pd.Timestamp("2024-01-03")


def test_local_csv_provider_reads_only_requested_completed_history(tmp_path: Path) -> None:
    export = _bars().reset_index(names="date")
    export.to_csv(tmp_path / "ABC.csv", index=False)
    result = LocalCsvPriceProvider(tmp_path).fetch_daily(
        "ABC", datetime(2024, 1, 3, tzinfo=UTC), datetime(2024, 1, 3, tzinfo=UTC)
    )
    assert result.index.tolist() == [pd.Timestamp("2024-01-03")]


def test_local_csv_provider_rejects_ticker_path_escape(tmp_path: Path) -> None:
    provider = LocalCsvPriceProvider(tmp_path)
    with pytest.raises(ValueError, match="Unsafe ticker filename"):
        provider.fetch_daily(
            "../escape", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)
        )


class _FakeResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self.payload


class _FakeSession:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.request: dict[str, object] | None = None

    def get(self, url: str, **kwargs: object) -> _FakeResponse:
        self.request = {"url": url, **kwargs}
        return _FakeResponse(self.payload)


def test_finnhub_news_is_deduplicated_timestamped_and_does_not_put_key_in_url() -> None:
    session = _FakeSession(
        [
            {
                "id": 7,
                "datetime": 1_704_067_200,
                "headline": "First",
                "source": "Example",
                "summary": "Summary",
                "url": "https://example.test/first",
            },
            {
                "id": 7,
                "datetime": 1_704_067_200,
                "headline": "Duplicate",
                "source": "Example",
                "summary": "Summary",
                "url": "https://example.test/duplicate",
            },
            {
                "id": 8,
                "datetime": 1_704_240_000,
                "headline": "Future",
                "source": "Example",
                "summary": "Summary",
                "url": "https://example.test/future",
            },
        ]
    )
    result = FinnhubNewsClient("secret", session).company_news(
        "aapl",
        date(2024, 1, 1),
        date(2024, 1, 3),
        as_of=datetime(2024, 1, 2, 12, tzinfo=UTC),
    )
    assert [article.article_id for article in result] == [7]
    assert result[0].ticker == "AAPL"
    assert session.request is not None
    assert "secret" not in str(session.request["url"])
    assert session.request["headers"] == {"X-Finnhub-Token": "secret"}


def test_finnhub_news_requires_key() -> None:
    with pytest.raises(ValueError, match="FINNHUBKEY"):
        FinnhubNewsClient("")


def test_finnhub_is_not_assumed_to_be_a_licensed_price_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MARKET_DATA_PROVIDER", "finnhub")
    with pytest.raises(ValueError, match="not enabled"):
        configured_price_provider()


def test_yfinance_provider_sets_a_finite_request_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    request: dict[str, object] = {}

    def download(ticker: str, **kwargs: object) -> pd.DataFrame:
        request.update({"ticker": ticker, **kwargs})
        return _bars()

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=download))
    result = YFinancePriceProvider(timeout_seconds=7.5).fetch_daily(
        "ABC", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 3, tzinfo=UTC)
    )
    assert request["timeout"] == 7.5
    assert not result.empty


def test_yfinance_timeout_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        YFinancePriceProvider(timeout_seconds=0)


def test_yfinance_batch_fetches_a_small_universe_in_one_bounded_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request: dict[str, object] = {}

    def download(tickers: list[str], **kwargs: object) -> pd.DataFrame:
        request.update({"tickers": tickers, **kwargs})
        return pd.concat({ticker: _bars() for ticker in tickers}, axis=1)

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=download))
    result = YFinancePriceProvider(timeout_seconds=7.5).fetch_daily_many(
        ["ABC", "DEF"],
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 3, tzinfo=UTC),
    )
    assert sorted(result) == ["ABC", "DEF"]
    assert request["tickers"] == ["ABC", "DEF"]
    assert request["group_by"] == "ticker"
    assert request["threads"] is False
    assert request["timeout"] == 7.5
