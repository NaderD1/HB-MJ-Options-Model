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

Two calendar vintages
  realized  what actually happened (released/scheduled rows). For ex-post analysis.
  as_known  what market participants knew at the valuation time. Built from the realized
            calendar plus data/events/schedule_changes.csv, which records for every
            rescheduled/canceled release: the original event, the revised event (if any),
            when its date became uncertain (e.g. a funding lapse began) and when the new
            schedule was announced. For a valuation time v:
              v < uncertain_from               -> the ORIGINAL date is used
              uncertain_from <= v < announced  -> the date was genuinely unknown: the event
                                                  is left out of pricing and the option is
                                                  flagged schedule_uncertain (exclude from
                                                  event tests)
              v >= announced                   -> the realized (revised/canceled) schedule
Ex-ante pricing and event tagging use as_known; realized tags are kept alongside for audit.
Known gap: for normally scheduled events we assume the date was known throughout our
coverage window (annual schedules are published months ahead), without per-event
publication dates.
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
ALL_STATUSES = ACTIVE_STATUSES + ("canceled", "superseded")
# All four official lists in the calendar file are complete from this instant on (the BLS
# lists start with the Feb-2025 releases). Update when older events are added.
COVERAGE_START = pd.Timestamp("2025-02-01T00:00:00Z")
DEFAULT_CALENDAR = Path(__file__).resolve().parents[1] / "data" / "events" / "macro_events.csv"
DEFAULT_CHANGES = Path(__file__).resolve().parents[1] / "data" / "events" / "schedule_changes.csv"
VINTAGES = ("as_known", "realized")


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
    table: pd.DataFrame    # one row per schedule entry (incl. superseded/canceled), with timestamp_utc
    changes: pd.DataFrame  # schedule-change log (see module docstring)
    coverage_start: pd.Timestamp
    coverage_end: pd.Timestamp

    @classmethod
    def from_csv(
        cls, path: str | Path = DEFAULT_CALENDAR, changes_path: str | Path | None = DEFAULT_CHANGES,
        coverage_start: pd.Timestamp = COVERAGE_START,
    ) -> "EventCalendar":
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        unknown = set(raw["event_type"]) - set(EVENT_TYPES)
        if unknown:
            raise ValueError(f"Unknown event types in calendar: {unknown}")
        if raw["event_id"].duplicated().any():
            raise ValueError("Duplicate event_id in calendar")
        if not set(raw["status"]) <= set(ALL_STATUSES):
            raise ValueError(f"Unknown status in calendar: {set(raw['status']) - set(ALL_STATUSES)}")
        raw["timestamp_utc"] = [localize(d, t, z) for d, t, z in zip(raw.local_date, raw.local_time, raw.local_tz)]
        raw = raw.sort_values("timestamp_utc").reset_index(drop=True)
        if changes_path is not None and Path(changes_path).exists():
            ch = pd.read_csv(changes_path, dtype=str, keep_default_na=False)
            for c in ("uncertain_from_utc", "announced_utc"):
                ch[c] = [to_utc(x) for x in ch[c]]
            ids = set(raw["event_id"])
            refs = list(ch.original_event_id) + [r for r in ch.revised_event_id if r]
            bad = [i for i in refs if i not in ids]
            if bad:
                raise ValueError(f"schedule_changes refers to unknown event ids: {bad}")
            if (ch["announced_utc"] < ch["uncertain_from_utc"]).any():
                raise ValueError("schedule_changes: announced_utc before uncertain_from_utc")
        else:
            ch = pd.DataFrame(columns=["original_event_id", "revised_event_id", "change_type",
                                       "uncertain_from_utc", "announced_utc"])
        # Coverage ends at the earliest "last listed event" across types: beyond it at least one
        # type (usually CPI/NFP, whose next-year BLS schedule is published late) may be missing.
        active = raw[raw.status.isin(ACTIVE_STATUSES)]
        end = active.groupby("event_type")["timestamp_utc"].max().min()
        return cls(raw, ch, to_utc(coverage_start), end)

    @property
    def active(self) -> pd.DataFrame:
        """The realized calendar (what actually happened / is scheduled now)."""
        return self.table[self.table["status"].isin(ACTIVE_STATUSES)]

    def view(self, valuation_ts: pd.Timestamp, vintage: str = "as_known") -> tuple[pd.DataFrame, pd.DataFrame]:
        """(events used for pricing/tagging, events whose date was unknown at valuation)."""
        if vintage not in VINTAGES:
            raise ValueError(f"vintage must be one of {VINTAGES}")
        by_id = self.table.set_index("event_id")
        events = self.active.set_index("event_id")
        uncertain_ids: list[str] = []
        if vintage == "as_known":
            v = to_utc(valuation_ts)
            drop, add = set(), []
            for _, c in self.changes.iterrows():
                if v >= c.announced_utc:
                    continue  # the realized schedule was already public
                if c.revised_event_id:
                    drop.add(c.revised_event_id)
                if v < c.uncertain_from_utc:
                    add.append(c.original_event_id)  # original date still believed
                else:
                    uncertain_ids += [c.original_event_id] + ([c.revised_event_id] if c.revised_event_id else [])
            events = pd.concat([events.drop(index=list(drop & set(events.index))), by_id.loc[add]])
        uncertain = by_id.loc[uncertain_ids].reset_index()
        return events.reset_index().sort_values("timestamp_utc").reset_index(drop=True), uncertain

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

    def scheduled_events(
        self, valuation_ts: pd.Timestamp, kinds: Iterable[str] = EVENT_TYPES, vintage: str = "as_known"
    ) -> tuple[ScheduledEvent, ...]:
        """Events as (tau, kind) relative to a valuation time -- the input HB-MJ prices with.

        Default vintage is as_known: a model priced at time v may only use what was known at v.
        """
        kinds = set(kinds)
        a, _ = self.view(valuation_ts, vintage)
        a = a[a["event_type"].isin(kinds)]
        return tuple(
            ScheduledEvent(year_fraction(valuation_ts, ts), k, i)
            for ts, k, i in zip(a["timestamp_utc"], a["event_type"], a["event_id"])
        )

    def tag(self, valuation_ts: pd.Timestamp, expiry_ts: pd.Timestamp, vintage: str = "as_known") -> dict:
        """Event tags for one option, via the same rule the pricing model uses."""
        complete = self.check_coverage(valuation_ts, expiry_ts)
        T = year_fraction(valuation_ts, expiry_ts)
        evs = self.scheduled_events(valuation_ts, vintage=vintage)
        inside = select_events_before_expiry(evs, T)
        future = [e for e in evs if e.tau > 0]
        nxt = min(future, key=lambda e: e.tau) if future else None
        _, unc = self.view(valuation_ts, vintage)
        unc_in = [i for i, ts in zip(unc.get("event_id", []), unc.get("timestamp_utc", []))
                  if 0.0 < year_fraction(valuation_ts, ts) <= T]
        return {
            "n_events": len(inside),
            "event_types": ";".join(e.kind for e in inside),
            "event_ids": ";".join(e.event_id for e in inside),
            "time_to_next_event": nxt.tau if nxt else np.nan,
            "next_event_type": nxt.kind if nxt else "",
            "events_complete": complete,
            "schedule_uncertain": bool(unc_in),
            "uncertain_event_ids": ";".join(unc_in),
        }

    def tag_both(self, valuation_ts: pd.Timestamp, expiry_ts: pd.Timestamp) -> dict:
        """as_known tags (used for pricing/evaluation) + realized ids kept for ex-post analysis."""
        t = self.tag(valuation_ts, expiry_ts, "as_known")
        r = self.tag(valuation_ts, expiry_ts, "realized")
        t.update({"event_vintage": "as_known", "n_events_realized": r["n_events"],
                  "event_ids_realized": r["event_ids"]})
        return t
