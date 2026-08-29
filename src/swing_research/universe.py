"""Point-in-time constituent universes with explicit provenance and coverage limits."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from io import StringIO

import pandas as pd
import requests

from .data import PriceProvider
from .schemas import SourceReference

FJA05680_SP500_COMMIT = "c31ac3cc56f28cf9a02b4e694eff7ceab596a0ff"
FJA05680_SP500_INTERVALS_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/"
    f"{FJA05680_SP500_COMMIT}/sp500_ticker_start_end.csv"
)


@dataclass(frozen=True)
class UniverseSnapshot:
    as_of: date
    tickers: tuple[str, ...]
    source: SourceReference
    minimum_price_coverage: float = 0.98


@dataclass(frozen=True)
class PriceCoverageAudit:
    snapshot: UniverseSnapshot
    covered_tickers: tuple[str, ...]
    unavailable_tickers: tuple[str, ...]
    checked_at: datetime

    @property
    def coverage(self) -> float:
        return len(self.covered_tickers) / len(self.snapshot.tickers)


def yahoo_symbol(ticker: str) -> str:
    """Map class-share notation to Yahoo's spelling without guessing successor links."""
    return ticker.replace(".", "-")


class Sp500HistoricalConstituentSource:
    """MIT-licensed community S&P 500 intervals pinned to an immutable Git commit.

    The source is not official S&P data. Its upstream README documents possible early-history gaps
    and says a delisted-security price vendor is needed for complete backtesting. Callers must
    retain the source reference and verify their price coverage before research is admissible.
    """

    def __init__(self, url: str = FJA05680_SP500_INTERVALS_URL) -> None:
        self.url = url

    def fetch_intervals(self) -> pd.DataFrame:
        response = requests.get(self.url, timeout=30)
        response.raise_for_status()
        return self.parse_intervals(response.text)

    @staticmethod
    def parse_intervals(csv_text: str) -> pd.DataFrame:
        intervals = pd.read_csv(StringIO(csv_text), dtype={"ticker": "string"})
        expected = {"ticker", "start_date", "end_date"}
        if not expected.issubset(intervals.columns):
            raise ValueError(f"Constituent source must include {sorted(expected)}")
        intervals = intervals.loc[:, ["ticker", "start_date", "end_date"]].copy()
        intervals["ticker"] = intervals["ticker"].str.strip().str.upper()
        intervals["start_date"] = pd.to_datetime(intervals["start_date"], errors="raise").dt.date
        intervals["end_date"] = pd.to_datetime(intervals["end_date"], errors="coerce").dt.date
        if intervals["ticker"].isna().any() or (intervals["ticker"] == "").any():
            raise ValueError("Constituent source contains a blank ticker")
        return intervals.sort_values(["ticker", "start_date"]).reset_index(drop=True)

    def snapshot(
        self, intervals: pd.DataFrame, as_of: date, minimum_price_coverage: float = 0.98
    ) -> UniverseSnapshot:
        if not 0 < minimum_price_coverage <= 1:
            raise ValueError("minimum_price_coverage must be in (0, 1]")
        active = intervals[
            (intervals["start_date"] <= as_of)
            & (intervals["end_date"].isna() | (intervals["end_date"] >= as_of))
        ]
        tickers = tuple(sorted({yahoo_symbol(str(ticker)) for ticker in active["ticker"]}))
        if len(tickers) < 400:
            raise ValueError(
                f"Constituent snapshot for {as_of.isoformat()} has only {len(tickers)} tickers; "
                "refusing it"
            )
        timestamp = datetime.now(UTC)
        return UniverseSnapshot(
            as_of=as_of,
            tickers=tickers,
            source=SourceReference(
                source_type="historical_constituents",
                url=self.url,
                retrieved_at=timestamp,
                available_at=datetime.combine(as_of, datetime.min.time(), tzinfo=UTC),
                description=(
                    "Community-maintained S&P 500 membership intervals; MIT licensed, "
                    "pinned to a Git commit, and not official index data. Price coverage "
                    "still requires verification."
                ),
            ),
            minimum_price_coverage=minimum_price_coverage,
        )


def audit_price_coverage(
    requested: UniverseSnapshot, provider: PriceProvider, lookback_days: int = 14
) -> PriceCoverageAudit:
    """Record provider availability around a historical membership date without masking failures."""
    if lookback_days < 1:
        raise ValueError("lookback_days must be positive")
    end = datetime.combine(requested.as_of, datetime.min.time(), tzinfo=UTC) + timedelta(days=1)
    start = end - timedelta(days=lookback_days)
    covered: list[str] = []
    unavailable: list[str] = []
    for ticker in requested.tickers:
        try:
            frame = provider.fetch_daily(ticker, start, end)
        except Exception:  # Provider/network failures are evidence of unavailable coverage.
            unavailable.append(ticker)
            continue
        if frame.empty:
            unavailable.append(ticker)
        else:
            covered.append(ticker)
    return PriceCoverageAudit(
        snapshot=requested,
        covered_tickers=tuple(covered),
        unavailable_tickers=tuple(unavailable),
        checked_at=datetime.now(UTC),
    )


def require_price_coverage(requested: UniverseSnapshot, returned_tickers: set[str]) -> None:
    """Reject claims when the price provider lacks a material part of a historical universe."""
    covered = len(set(requested.tickers).intersection(returned_tickers))
    ratio = covered / len(requested.tickers)
    if ratio < requested.minimum_price_coverage:
        raise ValueError(
            f"Price coverage {ratio:.1%} for {requested.as_of.isoformat()} is below the "
            f"{requested.minimum_price_coverage:.0%} research gate"
        )
