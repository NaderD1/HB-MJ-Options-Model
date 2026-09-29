"""Scheduled macro-event calendar and THE event-inclusion rule.

One rule decides whether an event affects an option:

    0 < tau <= T       (tau, T = year fractions from valuation, same ACT/365F clock)

``select_events_before_expiry`` implements it and is the only place it exists. It is used by
  1. the HB-MJ pricing model (``ScheduledEventJumps.events_before``), and
  2. the data layer (``EventCalendar.tag``) when marking options as event-spanning,
so pricing and evaluation cannot disagree about which events an option spans.

tau <= 0: the announcement is already public at valuation time (its move is in spot).
tau == T: an event exactly at expiry counts (conservative; with a NY 10:00 cut and 08:30
data releases this never binds in practice, but the rule must be unambiguous).

The calendar file (data/events/macro_events.csv) stores official *local* times with an IANA
timezone; UTC is derived here, so daylight-saving changes are handled by the tz database.
Only events with status 'released' or 'scheduled' are used; 'canceled' rows (e.g. October
2025 CPI/NFP, not published during the 2025 appropriations lapse) are kept for audit only.

Limitation: the calendar records what actually happened. An option priced *before* a
reschedule or cancellation was priced on the old schedule. For those few dates an
as-known-at-valuation calendar vintage would be more accurate (see original_local_date).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.timeutils import localize, to_utc, year_fraction

EVENT_TYPES: tuple[str, ...] = ("FOMC", "ECB", "CPI", "NFP")
ACTIVE_STATUSES = ("released", "scheduled")
# All four official lists in the calendar file are complete from this instant on (the BLS
# lists start with the Feb-2025 releases). Update when older events are added.
COVERAGE_START = pd.Timestamp("2025-02-01T00:00:00Z")
DEFAULT_CALENDAR = Path(__file__).resolve().parents[1] / "data" / "events" / "macro_events.csv"


@dataclass(frozen=True)
class ScheduledEvent:
    tau: float         # year fraction from valuation to announcement (ACT/365F, see timeutils)
    kind: str          # one of EVENT_TYPES
    event_id: str = ""


def select_events_before_expiry(events: Iterable[ScheduledEvent], T: float) -> tuple[ScheduledEvent, ...]:
    """THE inclusion rule: events with 0 < tau <= T."""
    return tuple(e for e in events if 0.0 < e.tau <= T)


@dataclass(frozen=True)
class EventCalendar:
    table: pd.DataFrame  # one row per event, with timestamp_utc
    coverage_start: pd.Timestamp
    coverage_end: pd.Timestamp

    @classmethod
    def from_csv(cls, path: str | Path = DEFAULT_CALENDAR, coverage_start: pd.Timestamp = COVERAGE_START) -> "EventCalendar":
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        unknown = set(raw["event_type"]) - set(EVENT_TYPES)
        if unknown:
            raise ValueError(f"Unknown event types in calendar: {unknown}")
        if raw["event_id"].duplicated().any():
            raise ValueError("Duplicate event_id in calendar")
        raw["timestamp_utc"] = [localize(d, t, z) for d, t, z in zip(raw.local_date, raw.local_time, raw.local_tz)]
        raw = raw.sort_values("timestamp_utc").reset_index(drop=True)
        # Coverage ends at the earliest "last listed event" across types: beyond it at least one
        # type (usually CPI/NFP, whose next-year BLS schedule is published late) may be missing.
        end = raw.groupby("event_type")["timestamp_utc"].max().min()
        return cls(raw, to_utc(coverage_start), end)

    @property
    def active(self) -> pd.DataFrame:
        return self.table[self.table["status"].isin(ACTIVE_STATUSES)]

    def check_coverage(self, valuation_ts: pd.Timestamp, expiry_ts: pd.Timestamp) -> bool:
        """True if every event in (valuation, expiry] is known to be in the calendar.

        A valuation before coverage_start is an error (past events could be missing and the
        'next event' would be wrong). An expiry beyond coverage_end is allowed but flagged:
        the counts are then a lower bound, and event-focused tests should filter on it.
        """
        v, x = to_utc(valuation_ts), to_utc(expiry_ts)
        if v < self.coverage_start:
            raise ValueError(f"Valuation {v} precedes calendar coverage start {self.coverage_start}")
        return bool(x <= self.coverage_end)

    def scheduled_events(self, valuation_ts: pd.Timestamp, kinds: Iterable[str] = EVENT_TYPES) -> tuple[ScheduledEvent, ...]:
        """Active calendar events as (tau, kind) relative to a valuation time -- the input HB-MJ prices with."""
        kinds = set(kinds)
        a = self.active[self.active["event_type"].isin(kinds)]
        return tuple(
            ScheduledEvent(year_fraction(valuation_ts, ts), k, i)
            for ts, k, i in zip(a["timestamp_utc"], a["event_type"], a["event_id"])
        )

    def tag(self, valuation_ts: pd.Timestamp, expiry_ts: pd.Timestamp) -> dict:
        """Event tags for one option, via the same rule the pricing model uses."""
        complete = self.check_coverage(valuation_ts, expiry_ts)
        T = year_fraction(valuation_ts, expiry_ts)
        evs = self.scheduled_events(valuation_ts)
        inside = select_events_before_expiry(evs, T)
        future = [e for e in evs if e.tau > 0]
        nxt = min(future, key=lambda e: e.tau) if future else None
        return {
            "n_events": len(inside),
            "event_types": ";".join(e.kind for e in inside),
            "event_ids": ";".join(e.event_id for e in inside),
            "time_to_next_event": nxt.tau if nxt else np.nan,
            "next_event_type": nxt.kind if nxt else "",
            "events_complete": complete,
        }
