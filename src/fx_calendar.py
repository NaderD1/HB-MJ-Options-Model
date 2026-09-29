"""FX date roller for EUR/USD options: settlement calendars, spot date, expiry and delivery.

STATUS: implements the standard market rules as documented in Clark (2011), *Foreign
Exchange Option Pricing*, ch. 1, and Wystup (2017), *FX Options and Structured Products*.
It must be VALIDATED against Bloomberg OVML expiry/delivery dates (see
docs/bloomberg_validation_table.csv) before it is used for historical BDH data; until then
the loader only uses it when explicitly allowed, and flags every roller-generated date.

Settlement calendars (explicit, sourced tables -- no rule-based holiday generation)
-----------------------------------------------------------------------------------
USD  data/calendars/usd_fedwire_closed_days.csv
     The USD leg of an EUR/USD trade settles in Fedwire Funds, which operates on the days the
     Federal Reserve Banks are open. The table lists, per Federal Reserve Board K.8 "Holidays
     Observed by the Federal Reserve System", the weekday on which the Reserve Banks are
     CLOSED for each holiday. When a holiday falls on a Saturday, K.8 states the Reserve Banks
     are OPEN the preceding Friday (e.g. Fri 3 Jul 2026), so that holiday has no weekday
     closure; when it falls on a Sunday they are closed the following Monday (e.g. Mon 5 Jul
     2027). Business-day status comes from the table, not from an observation rule.
EUR  data/calendars/eur_target2_closing_days.csv
     TARGET (T2) closing days per the ECB long-term calendar: 1 Jan, Good Friday, Easter
     Monday, 1 May, 25 Dec, 26 Dec. The ECB states these are not settlement days for FX
     transactions involving the euro.
Both tables cover 2025-2030; a date outside the table raises CalendarCoverageError instead of
being guessed. Ad hoc closures go in data/calendars/extra_closures.csv. Weekends are closed in
both calendars.

Rules for EUR/USD
-----------------
  Spot date   T+2. The first day counted (T+1) only has to be a EUR business day (a USD-only
              holiday on T+1 does not delay spot); the spot date itself must be a business
              day in BOTH currencies.
  ON / nD / nW tenors
              expiry = trade date + n days (ON: +1 day), rolled forward to the next joint
              business day; delivery = spot date of the expiry date.
  nM / nY tenors
              delivery = spot date + n months, modified-following on the joint calendar,
              with the end-of-month rule (spot on the last business day of its month ->
              delivery on the last business day of the target month);
              expiry = the joint business day whose spot date is that delivery date.
              If holidays make that inverse impossible, NO expiry is chosen: the result is
              'ambiguous' with the candidate dates listed, pending validation against OVML.
Expiry days are required to be joint EUR/USD business days (to be confirmed against OVML).
"""

from __future__ import annotations

import datetime as dt
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

CAL_DIR = Path(__file__).resolve().parents[1] / "data" / "calendars"
USD_TABLE = CAL_DIR / "usd_fedwire_closed_days.csv"
EUR_TABLE = CAL_DIR / "eur_target2_closing_days.csv"
EXTRA_CLOSURES = CAL_DIR / "extra_closures.csv"


class CalendarCoverageError(ValueError):
    """A date lies outside the years covered by the sourced settlement-calendar tables."""


@lru_cache(maxsize=None)
def _usd_table() -> tuple[frozenset[int], frozenset[dt.date]]:
    t = pd.read_csv(USD_TABLE, dtype=str, comment="#", keep_default_na=False)
    closed = frozenset(pd.Timestamp(d).date() for d in t["reserve_banks_closed_on"] if d)
    return frozenset(int(y) for y in t["year"]), closed


@lru_cache(maxsize=None)
def _eur_table() -> tuple[frozenset[int], frozenset[dt.date]]:
    t = pd.read_csv(EUR_TABLE, dtype=str, comment="#", keep_default_na=False)
    return frozenset(int(y) for y in t["year"]), frozenset(pd.Timestamp(d).date() for d in t["date"])


def usd_holidays(year: int) -> frozenset[dt.date]:
    """Weekdays in `year` on which the Federal Reserve Banks (Fedwire) are closed."""
    years, closed = _usd_table()
    if year not in years:
        raise CalendarCoverageError(f"USD settlement calendar has no data for {year}; extend {USD_TABLE.name} from K.8")
    return frozenset(d for d in closed if d.year == year) | _extra("USD", year)


def eur_holidays(year: int) -> frozenset[dt.date]:
    """TARGET (T2) closing days in `year`."""
    years, closed = _eur_table()
    if year not in years:
        raise CalendarCoverageError(f"TARGET calendar has no data for {year}; extend {EUR_TABLE.name}")
    return frozenset(d for d in closed if d.year == year) | _extra("EUR", year)


@lru_cache(maxsize=None)
def _extra(ccy: str, year: int) -> frozenset[dt.date]:
    if not EXTRA_CLOSURES.exists():
        return frozenset()
    t = pd.read_csv(EXTRA_CLOSURES, dtype=str, comment="#")
    return frozenset(pd.Timestamp(d).date() for d, c in zip(t["date"], t["currency"]) if c == ccy and pd.Timestamp(d).year == year)


def is_business_day(d: dt.date, cal: str) -> bool:
    """cal: 'USD', 'EUR' or 'EURUSD' (joint)."""
    if d.weekday() >= 5:
        return False
    if cal in ("USD", "EURUSD") and d in usd_holidays(d.year):
        return False
    if cal in ("EUR", "EURUSD") and d in eur_holidays(d.year):
        return False
    return True


def next_business_day(d: dt.date, cal: str, include_self: bool = False) -> dt.date:
    d = d if include_self else d + dt.timedelta(days=1)
    while not is_business_day(d, cal):
        d += dt.timedelta(days=1)
    return d


def spot_date(trade: dt.date) -> dt.date:
    """EUR/USD spot: T+1 must be a EUR business day; spot must be a joint business day."""
    first = next_business_day(trade, "EUR")
    return next_business_day(first, "EURUSD")


def _add_months(d: dt.date, n: int) -> dt.date:
    y, m = divmod(d.month - 1 + n, 12)
    y, m = d.year + y, m + 1
    last = (dt.date(y + (m == 12), m % 12 + 1, 1) - dt.timedelta(days=1)).day
    return dt.date(y, m, min(d.day, last))


def _last_business_day_of_month(y: int, m: int, cal: str) -> dt.date:
    d = dt.date(y + (m == 12), m % 12 + 1, 1) - dt.timedelta(days=1)
    while not is_business_day(d, cal):
        d -= dt.timedelta(days=1)
    return d


def _modified_following(d: dt.date, cal: str) -> dt.date:
    f = next_business_day(d, cal, include_self=True)
    if f.month != d.month:
        f = d
        while not is_business_day(f, cal):
            f -= dt.timedelta(days=1)
    return f


def _expiry_for_delivery(delivery: dt.date) -> tuple[dt.date | None, str, tuple[dt.date, ...]]:
    """Joint business day whose spot date is `delivery` -> (expiry, 'exact', ()).

    If no such day exists (holidays break the inverse), returns (None, 'ambiguous', candidates):
    the joint business days just before delivery whose spot date precedes it. Which one the
    market uses is NOT decided here -- it must be validated against Bloomberg OVML.
    """
    candidates = [delivery - dt.timedelta(days=k) for k in range(1, 12)]
    valid = [e for e in candidates if is_business_day(e, "EURUSD")]
    exact = [e for e in valid if spot_date(e) == delivery]
    if exact:
        return max(exact), "exact", ()
    before = sorted((e for e in valid if spot_date(e) < delivery), reverse=True)[:2]
    after = sorted(e for e in valid if spot_date(e) > delivery)[:1]
    return None, "ambiguous", tuple(sorted(before + after))


_TENOR = re.compile(r"^(ON|(\d+)([DWMY]))$")


def option_dates(trade: dt.date, tenor: str) -> dict:
    """{'trade', 'spot', 'expiry', 'delivery', 'expiry_rule', 'expiry_candidates'} for an EUR/USD option.

    expiry is None when expiry_rule == 'ambiguous' (see _expiry_for_delivery).
    """
    m = _TENOR.match(tenor.upper())
    if not m:
        raise ValueError(f"Unrecognised tenor {tenor!r}")
    spot = spot_date(trade)
    if m.group(1) == "ON" or m.group(3) in ("D", "W"):
        days = 1 if m.group(1) == "ON" else int(m.group(2)) * (7 if m.group(3) == "W" else 1)
        expiry = next_business_day(trade + dt.timedelta(days=days), "EURUSD", include_self=True)
        delivery = spot_date(expiry)
        rule, candidates = "days_weeks", ()
    else:
        months = int(m.group(2)) * (12 if m.group(3) == "Y" else 1)
        if spot == _last_business_day_of_month(spot.year, spot.month, "EURUSD"):
            target = _add_months(spot, months)
            delivery = _last_business_day_of_month(target.year, target.month, "EURUSD")
        else:
            delivery = _modified_following(_add_months(spot, months), "EURUSD")
        expiry, rule, candidates = _expiry_for_delivery(delivery)
    return {"trade": trade, "spot": spot, "expiry": expiry, "delivery": delivery,
            "expiry_rule": rule, "expiry_candidates": candidates}
