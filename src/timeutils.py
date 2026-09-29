"""Timestamp and year-fraction conventions shared by pricing, data and events.

Every timestamp in the project is timezone-aware and normalised to UTC internally. Naive
timestamps are rejected -- an FX expiry at "10:00" means nothing without a timezone.

Two clocks, deliberately kept separate
-------------------------------------
1. Diffusion clock (T). Year fractions use ACT/365 Fixed on calendar time, to the second:

       T = (t_end - t_start) in seconds / (365 * 86400)

   T runs from the valuation time to the option's expiry CUT (e.g. 10:00 New York) -- not to
   the delivery date, which only affects the forward and discounting. This is the standard
   convention for quoting FX implied vols against calendar time, and it is what Heston and
   Bates see: ordinary diffusion variance (and Poisson-jump risk) accrues uniformly per unit
   of calendar time, including weekends and holidays.

2. Event clock (tau_i). Scheduled-event jumps are NOT spread over time. Each event adds its
   variance sigma_E^2 once, at its exact announcement timestamp, and only if
   0 < tau_i <= T. A weekend or holiday therefore contributes diffusion time but never
   scheduled-event variance -- event variance appears only when an event timestamp actually
   falls inside the option's life. Using the same year_fraction for T and tau makes that
   inclusion test exact.

Known limitation (for the write-up): calendar time gives a quiet weekend the same diffusion
variance as a trading day. FX desks often down-weight weekends ("business-time"). For 1W
options that span a weekend this biases the *diffusion* part of the fit, not the event part;
a weekend weighting on the diffusion clock is a possible extension and is testable by
comparing implied variance of expiries just before and after weekends.
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
