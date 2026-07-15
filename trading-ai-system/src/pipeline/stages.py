"""Small, deterministic contracts for the broad-universe daily pipeline stages."""

from collections.abc import Mapping


def select_fast_universe(tickers, limit=8000):
    """Deduplicate and cap the Stage 1 input while preserving source order."""
    values=list(dict.fromkeys(str(t).strip() for t in tickers if str(t).strip()))
    return values[:int(limit)]


def fast_filter(frames, scorer):
    """Score every successfully loaded frame for Stage 1."""
    if not isinstance(frames, Mapping):
        raise TypeError('frames must be a mapping of ticker to OHLCV frame')
    return sorted(
        ({'ticker': ticker, 'fast_score': float(scorer(frame))} for ticker, frame in frames.items()),
        key=lambda row: row['fast_score'], reverse=True,
    )


def select_deep_tickers(fast_rows, limit=250):
    """Select only the top Stage 1 rows for expensive feature analysis."""
    return [row['ticker'] for row in list(fast_rows)[:int(limit)]]


def validate_universe_contract(requested, loaded, deep, minimum_full=3000, deep_min=200, deep_max=500):
    """Return a machine-readable throughput result for CI and audit artifacts."""
    requested=int(requested); loaded=int(loaded); deep=int(deep)
    checks={
        'requested_at_least_3000': requested >= minimum_full,
        'loaded_at_least_3000': loaded >= minimum_full,
        'deep_stage_between_200_and_500': deep_min <= deep <= deep_max,
        'deep_smaller_than_loaded': deep < loaded,
    }
    return {'passed': all(checks.values()), 'checks': checks, 'requested': requested, 'loaded': loaded, 'deep': deep}
