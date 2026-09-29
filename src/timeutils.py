"""Timestamp and year-fraction conventions shared by pricing, data and events.

Every timestamp in the project is timezone-aware and normalised to UTC internally. Naive
timestamps are rejected -- an FX expiry at "10:00" means nothing without a timezone.

Year fractions use ACT/365 Fixed on *calendar time* measured to the second:

    T = (t_end - t_start) in seconds / (365 * 86400)

Using the same function for option maturities T and event times tau is what makes the
event-inclusion rule 0 < tau <= T identical in pricing and in the data layer.

Known simplification: calendar time treats weekends and holidays as carrying the same
variance as trading days. FX desks often use trading-day or event-weighted time instead;
for 1W options that span a weekend this matters and is noted as a limitation.
"""

from __future__ import annotations

import pandas as pd

SECONDS_PER_YEAR = 365.0 * 86400.0


def to_utc(ts: pd.Timestamp | str) -> pd.Timestamp:
    """Convert a tz-aware timestamp (or ISO string with offset) to UTC; reject naive input."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        raise ValueError(f"Naive timestamp {t!s}: a timezone is required")
    return t.tz_convert("UTC")


def localize(date: str, time: str, tz: str) -> pd.Timestamp:
    """Local wall-clock date + time in an IANA timezone -> UTC timestamp (handles DST)."""
    return pd.Timestamp(f"{date} {time}").tz_localize(tz).tz_convert("UTC")


def year_fraction(start: pd.Timestamp | str, end: pd.Timestamp | str) -> float:
    """ACT/365F calendar-time year fraction between two tz-aware timestamps."""
    return (to_utc(end) - to_utc(start)).total_seconds() / SECONDS_PER_YEAR
