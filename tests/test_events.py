"""Event calendar: timestamps, the shared inclusion rule, coverage, and pricing/data agreement."""

import pandas as pd
import pytest

from src.events import EventCalendar, ScheduledEvent, select_events_before_expiry
from src.models.hbmj import ScheduledEventJumps
from src.timeutils import to_utc, year_fraction

CAL = EventCalendar.from_csv()
FOMC_OCT = pd.Timestamp("2026-10-28T18:00:00Z")  # 14:00 New York (EDT)
SEC = pd.Timedelta(seconds=1)


def test_official_local_times_convert_to_utc_across_dst():
    ts = CAL.table.set_index("event_id")["timestamp_utc"]
    assert ts["FOMC-2026-10-28"] == FOMC_OCT                               # NY still on EDT
    assert ts["ECB-2026-10-29"] == pd.Timestamp("2026-10-29T13:15:00Z")    # Frankfurt back on CET
    assert ts["CPI-2026-03-11"] == pd.Timestamp("2026-03-11T12:30:00Z")    # after US DST start
    assert ts["NFP-2026-12-04"] == pd.Timestamp("2026-12-04T13:30:00Z")    # EST


def test_canceled_releases_are_audited_but_never_used():
    assert set(CAL.table.loc[CAL.table.status == "canceled", "event_id"]) == {"CPI-2025-11-13", "NFP-2025-11-07"}
    tag = CAL.tag("2025-11-03T12:00:00Z", "2025-11-14T12:00:00Z")
    assert "CPI-2025-11-13" not in tag["event_ids"] and "NFP-2025-11-07" not in tag["event_ids"]


def test_rescheduled_release_uses_actual_time():
    tag = CAL.tag("2026-02-10T12:00:00Z", "2026-02-12T12:00:00Z")
    assert tag["event_ids"] == "NFP-2026-02-11"   # moved from 6 Feb by the 2026 lapse
    assert "CPI" not in tag["event_types"]        # CPI moved from 11 Feb to 13 Feb


@pytest.mark.parametrize("offset, included", [(-SEC, False), (pd.Timedelta(0), True), (SEC, True)])
def test_expiry_just_before_at_and_after_event(offset, included):
    valuation = pd.Timestamp("2026-10-27T12:00:00Z")
    tag = CAL.tag(valuation, FOMC_OCT + offset)
    assert ("FOMC-2026-10-28" in tag["event_ids"]) is included


def test_valuation_just_after_event_excludes_it():
    tag = CAL.tag(FOMC_OCT + SEC, FOMC_OCT + pd.Timedelta(days=2))
    assert "FOMC-2026-10-28" not in tag["event_ids"]
    tag = CAL.tag(FOMC_OCT, FOMC_OCT + pd.Timedelta(days=2))  # tau = 0: already public
    assert "FOMC-2026-10-28" not in tag["event_ids"]


def test_data_tags_equal_pricing_model_selection():
    """The same rule decides event inclusion for tagging and for HB-MJ pricing."""
    valuation = pd.Timestamp("2026-10-26T21:00:00Z")
    sigmas = {k: 0.005 for k in ("FOMC", "ECB", "CPI", "NFP")}
    model_events = ScheduledEventJumps(CAL.scheduled_events(valuation), sigmas)
    for days in (0.5, 1, 2, 3, 7, 10, 14, 20, 30):
        expiry = valuation + pd.Timedelta(days=days)
        tag = CAL.tag(valuation, expiry)
        chosen = model_events.events_before(year_fraction(valuation, expiry))
        assert tag["event_ids"] == ";".join(e.event_id for e in chosen)


def test_selection_rule_boundaries():
    evs = (ScheduledEvent(-0.01, "FOMC"), ScheduledEvent(0.0, "ECB"), ScheduledEvent(0.02, "CPI"), ScheduledEvent(0.05, "NFP"))
    assert [e.kind for e in select_events_before_expiry(evs, 0.02)] == ["CPI"]
    assert [e.kind for e in select_events_before_expiry(evs, 0.05)] == ["CPI", "NFP"]


def test_coverage_policy():
    with pytest.raises(ValueError, match="precedes"):
        CAL.tag("2025-01-15T12:00:00Z", "2025-02-15T12:00:00Z")
    assert CAL.tag("2026-10-26T21:00:00Z", "2026-11-30T15:00:00Z")["events_complete"] is True
    assert CAL.tag("2026-10-26T21:00:00Z", "2027-01-26T15:00:00Z")["events_complete"] is False


def test_naive_timestamps_rejected():
    with pytest.raises(ValueError, match="Naive"):
        to_utc("2026-10-28 14:00")


# ---------------------------------------------------------------- calendar vintages
@pytest.mark.parametrize(
    "valuation, as_known, uncertain, realized",
    [
        # Before the 2025 lapse: the original Sep-CPI date (15 Oct) was what everyone knew.
        ("2025-09-25T20:00:00Z", "NFP-2025-10-03;CPI-2025-10-15", "", "CPI-2025-10-24"),
        # During the lapse, before BLS announced 24 Oct: the CPI date was genuinely unknown.
        ("2025-10-06T20:00:00Z", "FOMC-2025-10-29;ECB-2025-10-30", "CPI-2025-10-15;CPI-2025-10-24",
         "CPI-2025-10-24;FOMC-2025-10-29;ECB-2025-10-30"),
        # After the 10 Oct announcement: the revised date is known.
        ("2025-10-14T20:00:00Z", "CPI-2025-10-24;FOMC-2025-10-29;ECB-2025-10-30", "NFP-2025-11-07;CPI-2025-11-13",
         "CPI-2025-10-24;FOMC-2025-10-29;ECB-2025-10-30"),
        # After all announcements: as-known equals realized.
        ("2025-11-24T20:00:00Z", "FOMC-2025-12-10;NFP-2025-12-16;ECB-2025-12-18;CPI-2025-12-18", "",
         "FOMC-2025-12-10;NFP-2025-12-16;ECB-2025-12-18;CPI-2025-12-18"),
    ],
)
def test_as_known_vs_realized_schedule(valuation, as_known, uncertain, realized):
    t = CAL.tag_both(valuation, pd.Timestamp(valuation) + pd.Timedelta(days=30))
    assert t["event_ids"] == as_known
    assert t["uncertain_event_ids"] == uncertain and t["schedule_uncertain"] == bool(uncertain)
    assert t["event_ids_realized"] == realized


def test_pricing_model_uses_as_known_schedule():
    v = pd.Timestamp("2025-09-25T20:00:00Z")
    ids = [e.event_id for e in CAL.scheduled_events(v) if 0 < e.tau <= 30 / 365]
    assert "CPI-2025-10-15" in ids and "CPI-2025-10-24" not in ids
    ids_realized = [e.event_id for e in CAL.scheduled_events(v, vintage="realized") if 0 < e.tau <= 30 / 365]
    assert "CPI-2025-10-24" in ids_realized and "CPI-2025-10-15" not in ids_realized


def test_superseded_and_canceled_rows_never_in_realized_calendar():
    assert not CAL.active.status.isin(["superseded", "canceled"]).any()
    changed = set(CAL.changes.original_event_id)
    assert changed == set(CAL.table.event_id[CAL.table.status.isin(["superseded", "canceled"])])
