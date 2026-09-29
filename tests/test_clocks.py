"""Diffusion clocks: calendar baseline vs NYSE trading-day clock (equity/proxy path)."""

import numpy as np
import pandas as pd
import pytest

from src.clocks import CalendarClock, TradingClock, nyse_sessions
from src.events import ScheduledEvent, select_events_before_expiry
from src.timeutils import year_fraction

NY = lambda s: pd.Timestamp(s).tz_localize("America/New_York")
DAY = 1 / 252
C0, C25 = TradingClock(), TradingClock(overnight_share=0.25)


def days(clock, a, b):
    return clock.year_fraction(NY(a), NY(b)) / DAY


def test_calendar_clock_is_the_unchanged_baseline():
    a, b = NY("2026-10-02 16:00"), NY("2026-10-05 16:00")
    assert CalendarClock().year_fraction(a, b) == year_fraction(a, b) == pytest.approx(3 / 365)


@pytest.mark.parametrize("clock", [C0, C25])
def test_friday_close_to_monday_close_is_one_trading_day(clock):
    assert days(clock, "2026-10-02 16:00", "2026-10-05 16:00") == pytest.approx(1.0, abs=1e-6)


def test_weekend_carries_no_variance_by_default_and_one_overnight_if_configured():
    assert days(C0, "2026-10-02 16:00", "2026-10-05 09:30") == pytest.approx(0.0, abs=1e-6)
    assert days(C25, "2026-10-02 16:00", "2026-10-05 09:30") == pytest.approx(0.25, abs=1e-6)
    # a weekend gap counts as ONE overnight, same as a weekday night
    assert days(C25, "2026-10-06 16:00", "2026-10-07 09:30") == pytest.approx(0.25, abs=1e-6)


def test_overnight_accrual_is_uniform_within_the_gap():
    # Tue 16:00 -> Wed 09:30 is 17.5h; halfway (00:45) holds half of the overnight share
    assert days(C25, "2026-10-06 16:00", "2026-10-07 00:45") == pytest.approx(0.125, abs=1e-6)


def test_intraday_and_early_close_sessions():
    assert days(C0, "2026-10-06 09:30", "2026-10-06 12:45") == pytest.approx(0.5, abs=1e-6)
    assert days(C0, "2026-11-25 16:00", "2026-11-27 13:00") == pytest.approx(1.0, abs=1e-6)  # Thanksgiving closed
    assert days(C0, "2026-11-27 09:30", "2026-11-27 11:15") == pytest.approx(0.5, abs=1e-6)  # early close 13:00


def test_clock_is_strictly_increasing_even_across_closed_periods():
    rng = np.random.default_rng(0)
    base = NY("2026-10-01 00:00")
    ts = sorted({base + pd.Timedelta(minutes=int(m)) for m in rng.integers(0, 60 * 24 * 60, 400)})
    vals = [C0._cum(t) for t in ts]
    assert all(b > a for a, b in zip(vals, vals[1:]))  # distinct times -> strictly increasing clock


def test_event_window_crossing_a_weekend_keeps_true_event_ordering():
    """NFP Friday 08:30 (pre-market). On the trading clock almost no diffusion time separates Thursday's
    16:00 close from Friday 08:30, yet the event rule must still put the event AFTER a Thursday
    expiry and BEFORE a Friday/Monday expiry."""
    v = NY("2026-09-29 15:09")
    nfp = ScheduledEvent(C0.year_fraction(v, NY("2026-10-02 08:30")), "NFP", "nfp")
    T_thu = C0.year_fraction(v, NY("2026-10-01 16:00"))
    T_fri = C0.year_fraction(v, NY("2026-10-02 16:00"))
    T_mon = C0.year_fraction(v, NY("2026-10-05 16:00"))
    assert select_events_before_expiry([nfp], T_thu) == ()
    assert select_events_before_expiry([nfp], T_fri) == (nfp,)
    assert select_events_before_expiry([nfp], T_mon) == (nfp,)
    assert nfp.tau - T_thu < 1e-9          # ~no diffusion variance between Thursday's close and the print
    assert (T_mon - T_fri) == pytest.approx(DAY, rel=1e-6)  # Friday->Monday adds exactly one trading day


def test_coverage_is_enforced():
    with pytest.raises(ValueError, match="coverage"):
        C0.year_fraction(NY("2025-12-30 10:00"), NY("2026-01-05 10:00"))
    assert len(nyse_sessions()) == 250 + 252 + 251  # 2026, 2027, 2028 trading days in the table
