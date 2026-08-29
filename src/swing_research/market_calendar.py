"""NYSE session and America/Chicago workflow timing helpers."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

CHICAGO = ZoneInfo("America/Chicago")


def is_nyse_session(session_date: date) -> bool:
    """Use exchange calendar data, including US market holidays, when installed."""
    try:
        import pandas_market_calendars as mcal
    except ImportError:  # pragma: no cover - retained for clear runtime error
        return session_date.weekday() < 5
    calendar = mcal.get_calendar("NYSE")
    schedule = calendar.schedule(start_date=session_date, end_date=session_date)
    return not schedule.empty


def should_start_daily_workflow(now: datetime) -> bool:
    local_now = now.astimezone(CHICAGO)
    return local_now.hour == 5 and is_nyse_session(local_now.date())


def latest_completed_nyse_session(now: datetime) -> date:
    """Return the latest session that had fully completed before a morning run."""
    local_now = now.astimezone(CHICAGO)
    candidate = local_now.date() - timedelta(days=1)
    while not is_nyse_session(candidate):
        candidate -= timedelta(days=1)
    return candidate
