"""Validation must pass clean data and catch each kind of corruption."""

from pathlib import Path

import pandas as pd
import pytest

from src.data_loaders import load_bloomberg_csv, normalize_fxe
from src.validation import run_validation
from tests.test_data_loaders import synthetic_fxe

EXAMPLE = Path(__file__).resolve().parents[1] / "data" / "examples" / "bloomberg_format_SYNTHETIC_eurusd.csv"


@pytest.fixture(scope="module")
def bbg():
    return load_bloomberg_csv(EXAMPLE)


@pytest.fixture(scope="module")
def fxe():
    return normalize_fxe(*synthetic_fxe()[:3])[0]


def _status(df):
    return run_validation(df).set_index("check")["passed"]


def test_clean_bloomberg_passes_every_check(bbg):
    assert _status(bbg).all()


def test_clean_fxe_passes_every_check(fxe):
    assert _status(fxe).all()


@pytest.mark.parametrize(
    "corrupt, check",
    [
        (lambda d: d.assign(forward=d.forward * 1.001), "forward_consistency"),
        (lambda d: d.assign(iv_mid=d.iv_mid.where(d.bucket != "25DC", d.iv_mid + 0.001)), "smile_reconstruction"),
        (lambda d: d.assign(strike=d.strike.where(d.bucket != "10DP", d.strike * 1.001)), "delta_round_trip"),
        (lambda d: d.assign(iv_bid=d.iv_ask + 0.01), "bid_ask"),
        (lambda d: d.assign(price_mid=d.price_mid.where(d.bucket != "ATM", 0.0)), "price_bounds"),
        (lambda d: d.assign(T=d["T"] * 1.0001), "time_consistency"),
        (lambda d: d.assign(n_events=d.n_events + 1), "event_tags"),
        (lambda d: d.assign(raw_fields=d.raw_fields.str.replace('"tenor": "1W"', '"tenor": "2W"')), "audit_trail"),
    ],
)
def test_corruption_is_detected(bbg, corrupt, check):
    bad = corrupt(bbg.copy())
    assert not _status(bad)[check]


def test_butterfly_and_calendar_arbitrage_detected(bbg):
    # Butterfly: make one wing call price exceed a nearer-the-money call (non-monotone/convex).
    d = bbg.copy()
    i = d.index[(d.tenor == "1M") & (d.bucket == "10DC")][0]
    d.loc[i, "price_mid"] *= 5
    assert not _status(d)["butterfly_arbitrage"]
    # Calendar: 2M ATM total variance below 1M.
    d = bbg.copy()
    j = d.index[(d.tenor == "2M") & (d.bucket == "ATM")][0]
    d.loc[j, "iv_mid"] = 0.03
    assert not _status(d)["calendar_arbitrage"]
