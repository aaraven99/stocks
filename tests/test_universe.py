from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from swing_research.universe import (
    Sp500HistoricalConstituentSource,
    audit_price_coverage,
    require_price_coverage,
)


def _interval_csv() -> str:
    rows = ["ticker,start_date,end_date"]
    rows.extend(f"T{number},2000-01-01," for number in range(401))
    rows.extend(["OLD,2000-01-01,2005-01-01", "BRK.B,2000-01-01,"])
    return "\n".join(rows)


def test_snapshot_is_point_in_time_and_maps_class_shares() -> None:
    source = Sp500HistoricalConstituentSource(url="https://example.invalid/intervals.csv")
    intervals = source.parse_intervals(_interval_csv())
    snapshot = source.snapshot(intervals, date(2004, 1, 1))
    assert "OLD" in snapshot.tickers
    assert "BRK-B" in snapshot.tickers
    later = source.snapshot(intervals, date(2006, 1, 1))
    assert "OLD" not in later.tickers


def test_coverage_gate_rejects_missing_historical_prices() -> None:
    source = Sp500HistoricalConstituentSource()
    snapshot = source.snapshot(source.parse_intervals(_interval_csv()), date(2004, 1, 1))
    with pytest.raises(ValueError, match="below the 98% research gate"):
        require_price_coverage(snapshot, {"T1", "T2", "BRK-B"})


class _PartialProvider:
    def fetch_daily(self, ticker: str, start: object, end: object) -> pd.DataFrame:
        if ticker == "OLD":
            raise RuntimeError("delisted price unavailable")
        return pd.DataFrame({"close": [100.0]})


def test_coverage_audit_records_provider_gaps() -> None:
    source = Sp500HistoricalConstituentSource()
    snapshot = source.snapshot(source.parse_intervals(_interval_csv()), date(2004, 1, 1))
    audit = audit_price_coverage(snapshot, _PartialProvider())
    assert audit.coverage < 1
    assert audit.unavailable_tickers == ("OLD",)
