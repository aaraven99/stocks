from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from swing_research.data import DataValidationError, LocalCsvPriceProvider, validate_ohlcv


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
