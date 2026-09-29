"""FX date roller: holiday calendars, spot rule, tenor expiry/delivery.

These tests check the documented market RULES. Whether Bloomberg applies the same rules is a
separate empirical question, answered by docs/bloomberg_validation_table.csv.
"""

import datetime as dt

import pandas as pd
import pytest

from src.data_loaders import fx_trade_date
from src.fx_calendar import CalendarCoverageError, eur_holidays, is_business_day, option_dates, spot_date, usd_holidays

D = dt.date


# ---------------------------------------------------------------- sourced settlement calendars
# USD = Federal Reserve Banks (Fedwire) closures per Federal Reserve Board K.8.
# EUR = TARGET (T2) closing days per the ECB long-term calendar.

def test_usd_2026_closures_match_k8():
    assert usd_holidays(2026) == {
        D(2026, 1, 1), D(2026, 1, 19), D(2026, 2, 16), D(2026, 5, 25), D(2026, 6, 19),
        D(2026, 9, 7), D(2026, 10, 12), D(2026, 11, 11), D(2026, 11, 26), D(2026, 12, 25),
    }


@pytest.mark.parametrize(
    "day, usd_open, eur_open, why",
    [
        (D(2026, 7, 3), True, True, "4 Jul 2026 is a Saturday: K.8 says Reserve Banks are OPEN Fri 3 Jul"),
        (D(2026, 7, 6), True, True, "New York Fed open Mon 6 Jul 2026"),
        (D(2027, 7, 5), False, True, "4 Jul 2027 is a Sunday: closed Mon 5 Jul (K.8 **)"),
        (D(2026, 6, 19), False, True, "Juneteenth 2026 (Friday): closed"),
        (D(2027, 6, 18), True, True, "Juneteenth 2027 is a Saturday: open Fri 18 Jun"),
        (D(2026, 11, 26), False, True, "Thanksgiving 2026: USD closed, TARGET open"),
        (D(2026, 11, 27), True, True, "day after Thanksgiving: both open"),
        (D(2026, 12, 25), False, False, "Christmas: both closed"),
        (D(2027, 12, 24), True, True, "25 Dec 2027 is a Saturday: Reserve Banks open Fri 24 Dec; 24 Dec is not a T2 holiday"),
        (D(2027, 12, 31), True, True, "1 Jan 2028 is a Saturday: open Fri 31 Dec 2027 (K.8 corrected 2026-07-08)"),
        (D(2026, 12, 28), True, True, "26 Dec 2026 is a Saturday: T2 has no weekday substitute"),
        (D(2026, 4, 3), True, False, "Good Friday 2026: TARGET closed, USD open"),
        (D(2026, 4, 6), True, False, "Easter Monday 2026: TARGET closed, USD open"),
        (D(2026, 5, 1), True, False, "1 May: TARGET closed"),
    ],
)
def test_settlement_calendar_regressions(day, usd_open, eur_open, why):
    assert is_business_day(day, "USD") is usd_open, why
    assert is_business_day(day, "EUR") is eur_open, why
    assert is_business_day(day, "EURUSD") is (usd_open and eur_open), why


@pytest.mark.parametrize("year, easter_sunday", [(2025, D(2025, 4, 20)), (2026, D(2026, 4, 5)), (2027, D(2027, 3, 28)),
                                                 (2028, D(2028, 4, 16)), (2029, D(2029, 4, 1)), (2030, D(2030, 4, 21))])
def test_target2_good_friday_easter_monday_table(year, easter_sunday):
    h = eur_holidays(year)
    assert easter_sunday - dt.timedelta(days=2) in h and easter_sunday + dt.timedelta(days=1) in h and len(h) == 6


def test_dates_outside_sourced_tables_raise():
    with pytest.raises(CalendarCoverageError):
        is_business_day(D(2031, 3, 3), "USD")
    with pytest.raises(CalendarCoverageError):
        is_business_day(D(2024, 3, 4), "EUR")


def test_spot_over_july_3_2026():
    assert spot_date(D(2026, 7, 1)) == D(2026, 7, 3)   # Fri 3 Jul is a joint business day
    assert spot_date(D(2027, 7, 1)) == D(2027, 7, 6)   # Mon 5 Jul 2027 closed (Sunday holiday)


# ---------------------------------------------------------------- spot / tenor rules
@pytest.mark.parametrize(
    "trade, spot, why",
    [
        (D(2026, 10, 26), D(2026, 10, 28), "plain T+2"),
        (D(2026, 10, 29), D(2026, 11, 2), "over a weekend"),
        (D(2026, 11, 25), D(2026, 11, 27), "USD holiday on T+1 (Thanksgiving) does not delay EUR/USD spot"),
        (D(2026, 11, 24), D(2026, 11, 27), "USD holiday on T+2 pushes spot to the next joint day"),
        (D(2026, 4, 2), D(2026, 4, 8), "Good Friday + Easter Monday are TARGET2 holidays"),
        (D(2026, 12, 24), D(2026, 12, 29), "Christmas and 26 Dec (Saturday) on the way"),
    ],
)
def test_spot_date_rule(trade, spot, why):
    assert spot_date(trade) == spot, why


def test_week_tenors_expiry_plus_n_weeks_delivery_is_spot_of_expiry():
    d = option_dates(D(2026, 10, 26), "1W")
    assert d["expiry"] == D(2026, 11, 2) and d["delivery"] == spot_date(D(2026, 11, 2))


def test_month_tenor_modified_following_and_ambiguous_inverse():
    d = option_dates(D(2026, 10, 26), "1M")
    assert d["delivery"] == D(2026, 11, 30)          # 28 Nov is a Saturday -> Monday 30 Nov
    # No joint business day has spot date 30 Nov (Tue 24/Wed 25 -> Fri 27; Thu 26 is Thanksgiving;
    # Fri 27 -> Tue 1 Dec). The roller must NOT pick one before OVML validation.
    assert d["expiry"] is None and d["expiry_rule"] == "ambiguous"
    assert d["expiry_candidates"] == (D(2026, 11, 24), D(2026, 11, 25), D(2026, 11, 27))
    d2 = option_dates(D(2026, 10, 26), "2M")
    assert d2["delivery"] == D(2026, 12, 28) and d2["expiry"] == D(2026, 12, 23) and d2["expiry_rule"] == "exact"


def test_end_of_month_rule():
    trade = D(2026, 9, 28)                           # spot = Wed 30 Sep 2026, last business day of Sept
    assert spot_date(trade) == D(2026, 9, 30)
    assert option_dates(trade, "1M")["delivery"] == D(2026, 10, 30)  # last business day of Oct (31st is Sat)
    assert option_dates(trade, "2M")["delivery"] == D(2026, 11, 30)


@pytest.mark.parametrize("tenor", ["ON", "1W", "2W", "3W", "1M", "2M", "3M", "6M", "9M", "1Y"])
@pytest.mark.parametrize("trade", [D(2026, 1, 2), D(2026, 4, 1), D(2026, 6, 18), D(2026, 11, 20), D(2026, 12, 23)])
def test_rolled_dates_are_valid_business_days(tenor, trade):
    d = option_dates(trade, tenor)
    assert is_business_day(d["spot"], "EURUSD") and is_business_day(d["delivery"], "EURUSD")
    expiries = [d["expiry"]] if d["expiry"] else list(d["expiry_candidates"])
    assert expiries, "ambiguous expiry must list candidates"
    for e in expiries:
        assert is_business_day(e, "EURUSD") and trade < e


@pytest.mark.parametrize(
    "ts, trade",
    [
        ("2026-10-26T16:59:00-04:00", D(2026, 10, 26)),
        ("2026-10-26T17:00:00-04:00", D(2026, 10, 27)),   # 17:00 NY roll
        ("2026-10-30T18:00:00-04:00", D(2026, 11, 2)),    # Friday after the roll -> Monday
        ("2026-10-26T03:00:00+01:00", D(2026, 10, 25) + dt.timedelta(days=1)),  # Sunday 22:00 NY -> Monday
    ],
)
def test_fx_trade_date_roll(ts, trade):
    assert fx_trade_date(pd.Timestamp(ts)) == trade
