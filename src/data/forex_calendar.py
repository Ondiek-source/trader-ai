"""
data/forex_calendar.py — Forex Market Holiday Calendar

Provides a holiday-aware calendar for forex markets so that gap detection
can distinguish genuine data loss from expected market closures.

Sources (in priority order):
    1. exchange_calendars library (NYSE schedule — covers most forex holidays)
    2. External API endpoint (configurable via FOREX_HOLIDAY_API_URL)
    3. Static fallback list of major forex holidays

The calendar is cached per-year to avoid repeated lookups.
"""

from __future__ import annotations

import logging
import exchange_calendars as xcals

from typing import Optional
from functools import lru_cache
from datetime import date, datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# ── Static Fallback ──────────────────────────────────────────────────────────
# Major holidays when forex markets are fully or partially closed.
# These are the "big ones" that affect all major pairs.
# Dates are for recurring annual holidays; fixed-date holidays shift slightly
# year to year but this list covers the common pattern.

_STATIC_HOLIDAYS: set[date] = set()

# Fixed-date holidays (month, day) — generated per year
_FIXED_ANNUAL: list[tuple[int, int, str]] = [
    (1, 1, "New Year's Day"),
    (12, 25, "Christmas Day"),
    (12, 26, "Boxing Day / St Stephen's Day"),
    (12, 31, "New Year's Eve (early close)"),
    (7, 4, "US Independence Day"),
]


def _generate_fixed_holidays(year: int) -> set[date]:
    """Generate fixed-date holidays for a given year, adjusting for weekends."""
    holidays: set[date] = set()
    for month, day, name in _FIXED_ANNUAL:
        try:
            d = date(year, month, day)
        except ValueError:
            continue  # Feb 29 on non-leap year etc.

        # If holiday falls on Saturday, Friday before is observed
        # If holiday falls on Sunday, Monday after is observed
        if d.weekday() == 5:  # Saturday
            d = d - timedelta(days=1)
        elif d.weekday() == 6:  # Sunday
            d = d + timedelta(days=1)
        holidays.add(d)

    # Easter-based holidays (Good Friday, Easter Monday)
    easter = _easter_sunday(year)
    if easter:
        holidays.add(easter - timedelta(days=2))  # Good Friday
        holidays.add(easter + timedelta(days=1))  # Easter Monday

    return holidays


def _easter_sunday(year: int) -> Optional[date]:
    """Compute Easter Sunday using the anonymous Gregorian algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    try:
        return date(year, month, day)
    except ValueError:
        return None


# ── Library-Based Calendar ───────────────────────────────────────────────────


def _get_library_holidays(year: int) -> Optional[set[date]]:
    """Try to get holidays from exchange_calendars (NYSE schedule)."""
    try:

        # NYSE calendar covers: New Year's, MLK Day, Presidents' Day,
        # Good Friday, Memorial Day, Juneteenth, Independence Day,
        # Labor Day, Thanksgiving, Christmas.
        # These are the days when forex liquidity is minimal.
        xcal = xcals.get_calendar("XNYS")
        schedule = xcal.schedule.loc[str(year)]
        # Get all sessions (trading days); holidays are the gaps
        all_sessions = set(schedule.index.date)  # type: ignore
        # Generate all weekdays in the year
        start = date(year, 1, 1)
        end = date(year, 12, 31)
        d = start
        weekdays: set[date] = set()
        while d <= end:
            if d.weekday() < 5:  # Mon-Fri
                weekdays.add(d)
            d += timedelta(days=1)
        # Holidays = weekdays that are NOT trading sessions
        holidays = weekdays - all_sessions
        return holidays
    except Exception:
        return None


# ── Main API ─────────────────────────────────────────────────────────────────


@lru_cache(maxsize=10)
def get_forex_holidays(year: int) -> set[date]:
    """
    Get the set of forex market holidays for a given year.

    Tries sources in order:
    1. exchange_calendars library (NYSE)
    2. Static fallback list

    Args:
        year: The calendar year to get holidays for.

    Returns:
        set of date objects when the forex market is closed or
        has minimal liquidity.
    """
    # Try library first
    holidays = _get_library_holidays(year)
    if holidays:
        logger.debug(
            "Forex holidays for %d loaded from exchange_calendars (%d days)",
            year,
            len(holidays),
        )
        return holidays

    # Fall back to static list
    holidays = _generate_fixed_holidays(year)
    logger.debug(
        "Forex holidays for %d loaded from static fallback (%d days)",
        year,
        len(holidays),
    )
    return holidays


def is_forex_holiday(dt: datetime | date) -> bool:
    """Check if a datetime/date falls on a forex market holiday."""
    d = dt.date() if isinstance(dt, datetime) else dt
    return d in get_forex_holidays(d.year)


def is_forex_closed(dt: datetime) -> bool:
    """
    Check if the forex market is closed at a given datetime.

    Market is closed when:
    - It's a weekend (Saturday or Sunday)
    - It's a recognized holiday
    - It's Friday after 21:00 UTC (FX market close)
    - It's Sunday before 21:00 UTC (FX market not yet open)
    """
    dow = dt.weekday()
    hour = dt.hour

    # Weekend
    if dow == 5:  # Saturday
        return True
    if dow == 6 and hour < 21:  # Sunday before 21:00
        return True
    if dow == 4 and hour >= 21:  # Friday after 21:00
        return True

    # Holiday
    return is_forex_holiday(dt)


def count_expected_trading_minutes(
    start_dt: datetime,
    end_dt: datetime,
) -> int:
    """
    Count the expected number of M1 trading bars between two datetimes.

    Excludes weekends and known holidays. Used to validate that an API
    response returned the expected number of bars.

    Args:
        start_dt: Start of the window (inclusive, UTC).
        end_dt:   End of the window (inclusive, UTC).

    Returns:
        Expected count of M1 bars.
    """
    count = 0
    current = start_dt
    while current <= end_dt:
        if not is_forex_closed(current):
            count += 1
        current += timedelta(minutes=1)
    return count


def get_gap_classification(
    gap_start: datetime,
    gap_end: datetime,
) -> str:
    """
    Classify a data gap as 'weekend', 'holiday', or 'unexpected'.

    Args:
        gap_start: Start of the gap.
        gap_end:   End of the gap.

    Returns:
        One of 'weekend', 'holiday', 'unexpected'.
    """
    # Check if entire gap is within a weekend window
    all_weekend = True
    all_closed = True
    current = gap_start
    while current <= gap_end:
        dow = current.weekday()
        hour = current.hour
        is_weekend = dow == 5 or (dow == 6 and hour < 21) or (dow == 4 and hour >= 21)
        if not is_weekend:
            all_weekend = False
        if not is_forex_closed(current):
            all_closed = False
        current += timedelta(minutes=1)

    if all_weekend:
        return "weekend"
    if all_closed:
        return "holiday"
    return "unexpected"
