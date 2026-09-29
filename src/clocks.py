"""Diffusion clocks: how much ordinary (Heston/Bates) variance accrues between two timestamps.

The pricing models take a maturity T and event times tau measured on ONE clock. Scheduled events
are always placed at their true timestamps; only the diffusion clock changes.

CalendarClock (default; used for the FX/Bloomberg path)
    ACT/365F calendar time (src/timeutils.year_fraction). Variance accrues evenly over nights,
    weekends and holidays. Appropriate for 24h OTC FX.

TradingClock (equity/proxy path only)
    NYSE trading-day clock. Convention (fixed in advance, not tuned to any result):
      * each NYSE trading day -- full or early-close -- contributes 1/252 year of diffusion time;
      * a share (1 - overnight_share) of it accrues uniformly (in calendar time) over that day's
        Core Trading Session, 09:30-16:00 ET (13:00 on early-close days);
      * the remaining overnight_share accrues uniformly over the non-trading gap that PRECEDES the
        session (previous close -> this open). A weekend or holiday gap counts as ONE overnight,
        so weekends add no more variance than a weekday night;
      * default overnight_share = 0: all ordinary variance in trading hours, none overnight;
      * a tiny calendar component epsilon (1e-9 per calendar year) keeps the clock STRICTLY
        increasing, so time ordering -- and hence the event rule 0 < tau <= T -- is exactly the
        calendar ordering (a CPI print at 08:30 still comes after the previous day's 16:00 expiry).
    Sessions come from data/calendars/nyse_sessions.csv (official NYSE holiday/early-close table,
    2026-2028); timestamps outside it raise instead of being guessed.
"""

from __future__ import annotations

import bisect
import datetime as dt
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.timeutils import SECONDS_PER_YEAR, to_utc, year_fraction

NYSE_TABLE = Path(__file__).resolve().parents[1] / "data" / "calendars" / "nyse_sessions.csv"
NY = "America/New_York"


class CalendarClock:
    name = "calendar_act365"

    def year_fraction(self, start, end) -> float:
        return year_fraction(start, end)


@lru_cache(maxsize=None)
def nyse_sessions() -> tuple[tuple[pd.Timestamp, pd.Timestamp], ...]:
    """(open_utc, close_utc) for every NYSE trading day in the table's coverage years."""
    t = pd.read_csv(NYSE_TABLE, comment="#", dtype=str, keep_default_na=False)
    years = sorted({int(d[:4]) for d in t.date})
    hol = {pd.Timestamp(d).date() for d, k in zip(t.date, t.kind) if k == "holiday"}
    early = {pd.Timestamp(d).date(): c for d, k, c in zip(t.date, t.kind, t.close_time) if k == "early_close"}
    out = []
    d = dt.date(years[0], 1, 1)
    while d.year <= years[-1]:
        if d.weekday() < 5 and d not in hol:
            o = pd.Timestamp(f"{d} 09:30").tz_localize(NY).tz_convert("UTC")
            c = pd.Timestamp(f"{d} {early.get(d, '16:00')}").tz_localize(NY).tz_convert("UTC")
            out.append((o, c))
        d += dt.timedelta(days=1)
    return tuple(out)


@dataclass(frozen=True)
class TradingClock:
    overnight_share: float = 0.0
    days_per_year: float = 252.0
    epsilon: float = 1e-9
    name: str = field(default="nyse_trading_day", compare=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.overnight_share <= 1.0:
            raise ValueError("overnight_share must be in [0, 1]")

    def _cum(self, t: pd.Timestamp) -> float:
        """Trading time (years) from the first session's open to t, plus the epsilon term."""
        s = nyse_sessions()
        t = to_utc(t)
        first_open, last_close = s[0][0], s[-1][1]
        if not first_open <= t <= last_close:
            raise ValueError(f"{t} outside NYSE session table coverage {first_open} .. {last_close}")
        opens = [o for o, _ in s]
        i = bisect.bisect_right(opens, t) - 1          # last session whose open <= t
        day = 1.0 / self.days_per_year
        w_sess, w_night = (1 - self.overnight_share) * day, self.overnight_share * day
        o, c = s[i]
        total = i * day                                 # all completed trading days before session i
        # session i: before t it may be partly done (or fully, if t is after its close)
        total += w_sess * min(max((t - o) / (c - o), 0.0), 1.0)
        # the overnight belonging to session i (the gap before its open) is complete once o <= t,
        # and it was counted inside `i * day` only for sessions < i -> add session i's overnight
        total += w_night if i > 0 else 0.0
        # gap AFTER session i (belongs to session i+1): partial if t is past close
        if t > c and i + 1 < len(s):
            nxt = s[i + 1][0]
            total += w_night * (t - c) / (nxt - c)
        total += self.epsilon * (t - first_open).total_seconds() / SECONDS_PER_YEAR
        return total

    def year_fraction(self, start, end) -> float:
        return self._cum(end) - self._cum(start)
