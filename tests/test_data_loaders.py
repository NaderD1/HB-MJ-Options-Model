"""Loaders: Bloomberg convention strictness, FXE normalization, and a single shared schema."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loaders import ConventionAmbiguityError, load_bloomberg_csv, normalize_fxe
from src.models.black_scholes import black76_price
from src.schema import COLUMNS, validate_schema
from src.timeutils import localize, year_fraction

EXAMPLE = Path(__file__).resolve().parents[1] / "data" / "examples" / "bloomberg_format_SYNTHETIC_eurusd.csv"


@pytest.fixture(scope="module")
def bbg():
    return load_bloomberg_csv(EXAMPLE)


def _write_variant(tmp_path, **changes):
    raw = pd.read_csv(EXAMPLE)
    for col, val in changes.items():
        if val is None:
            raw = raw.drop(columns=col)
        else:
            raw[col] = val
    p = tmp_path / "variant.csv"
    raw.to_csv(p, index=False)
    return p


def synthetic_fxe(F=105.30, D=0.99, vol=0.075, valuation="2026-10-26T20:00:00Z", expiry="2026-11-20"):
    """A European, noise-free FXE-like chain priced by Black-76 at known (F, D, vol)."""
    T = year_fraction(valuation, localize(expiry, "16:00", "America/New_York"))
    rows = []
    for K in np.arange(100.0, 111.0, 1.0):
        for cp in ("C", "P"):
            px = float(black76_price(F, K, T, vol, D, cp == "C"))
            rows.append({"contractSymbol": f"FXE{expiry}{cp}{K:g}", "strike": K, "lastPrice": px, "bid": px * 0.98,
                         "ask": px * 1.02, "volume": 5, "openInterest": 100, "impliedVolatility": vol,
                         "lastTradeDate": valuation, "expiration": expiry, "cp": cp})
    raw = pd.DataFrame(rows)
    meta = {"spot": 105.0, "spot_ts_utc": valuation, "raw_source_file": "synthetic_fxe_fixture", "raw_file_sha256": "fixture"}
    rates = pd.DataFrame({"series": ["X"], "tenor_years": [0.1], "obs_date": ["2026-10-23"],
                          "yield_pct": [200 * np.expm1(-np.log(D) / T / 2)]})  # BEY such that DF == D
    return raw, meta, rates, T


# ---------------------------------------------------------------- Bloomberg
def test_bloomberg_example_shape_and_labels(bbg):
    assert len(bbg) == 9 * 5 and list(bbg.columns) == COLUMNS
    assert (bbg.source == "bloomberg_format_synthetic").all() and bbg.is_dev_fallback.all()
    assert bbg.quality_flags.str.contains("non_research_origin_synthetic").all()
    assert validate_schema(bbg) == []


def test_genuine_bloomberg_origin_is_research_data(tmp_path):
    df = load_bloomberg_csv(_write_variant(tmp_path, data_origin="bloomberg"))
    assert (df.source == "bloomberg").all() and not df.is_dev_fallback.any()


@pytest.mark.parametrize(
    "changes, match",
    [
        (dict(bf_type=None), "missing"),
        (dict(delta_type=""), "missing"),
        (dict(delta_type="spot_pa"), "implies unadjusted"),         # USD premium + PA delta: contradictory
        (dict(premium_currency="EUR"), "implies premium-adjusted"),  # EUR premium + unadjusted: contradictory
        (dict(rr_sign="put_minus_call"), "rr_sign"),
        (dict(valuation_tz=None), "no timezone"),
        (dict(expiry_cut=None), "expiry cut"),
        (dict(fwd_points_scale=None), "fwd_points_scale"),
        (dict(vol_units="bp"), "vol_units"),
    ],
)
def test_bloomberg_loader_refuses_to_guess(tmp_path, changes, match):
    with pytest.raises(ConventionAmbiguityError, match=match):
        load_bloomberg_csv(_write_variant(tmp_path, **changes))


def test_bloomberg_timestamps_and_T(bbg):
    one_w = bbg[bbg.tenor == "1W"].iloc[0]
    assert one_w.valuation_ts_utc == pd.Timestamp("2026-10-26T20:00:00Z")  # 16:00 EDT
    assert one_w.expiry_ts_utc == pd.Timestamp("2026-11-02T15:00:00Z")      # NY 10:00 cut, EST after 1 Nov
    assert one_w["T"] == pytest.approx((6 * 86400 + 19 * 3600) / (365 * 86400), abs=1e-15)


def test_bloomberg_audit_trail(bbg):
    raw = pd.read_csv(EXAMPLE)
    for _, r in bbg.iterrows():
        fields = json.loads(r.raw_fields)
        src = raw.iloc[int(r.raw_row_id)]
        assert fields["tenor"] == src.tenor == r.tenor and fields["atm"] == src.atm


# ---------------------------------------------------------------- FXE fallback
def test_fxe_forward_from_parity_and_iv_recovered():
    raw, meta, rates, T = synthetic_fxe()
    df, report = normalize_fxe(raw, meta, rates)
    exp = report["expiries"]["2026-11-20"]
    assert exp["F"] == pytest.approx(105.30, abs=1e-9) and exp["D"] == pytest.approx(0.99, abs=1e-12)
    np.testing.assert_allclose(df.iv_mid, 0.075, atol=1e-9)
    assert ((df.is_call & (df.strike >= df.forward)) | (~df.is_call & (df.strike < df.forward))).all()  # OTM only
    assert df.is_dev_fallback.all() and df.quality_flags.str.contains("american_exercise").all()


def test_fxe_filters_drop_bad_quotes():
    raw, meta, rates, _ = synthetic_fxe()
    raw.loc[0, ["bid", "ask"]] = 0.0              # zero bid
    raw.loc[1, "openInterest"] = 0                # illiquid
    raw.loc[2, "ask"] = raw.loc[2, "bid"] * 3     # very wide
    _, report = normalize_fxe(raw, meta, rates)
    assert report["dropped"]["zero_bid_or_crossed"] == 1
    assert report["dropped"]["low_open_interest"] == 1
    assert report["dropped"]["wide_spread"] == 1


# ---------------------------------------------------------------- shared schema
def test_both_sources_produce_identical_schema(bbg):
    fxe, _ = normalize_fxe(*synthetic_fxe()[:3])
    assert list(fxe.columns) == list(bbg.columns) == COLUMNS
    assert validate_schema(fxe) == [] and validate_schema(bbg) == []
    for c in COLUMNS:
        assert (fxe[c].dtype.kind == bbg[c].dtype.kind) or (fxe[c].dtype == object and bbg[c].dtype == object), c


# ---------------------------------------------------------------- settlement timing
def test_discounting_runs_from_spot_date_to_delivery(bbg):
    raw = pd.read_csv(EXAMPLE)
    for _, r in bbg[bbg.bucket == "ATM"].iterrows():
        src = raw[raw.tenor == r.tenor].iloc[0]
        days = (pd.Timestamp(src.delivery_date) - pd.Timestamp(src.spot_date)).days
        assert r.df_dom == pytest.approx(1 / (1 + src.usd_depo_rate / 100 * days / 360), abs=1e-15)
        assert r.spot_date == src.spot_date and r.delivery_date == src.delivery_date and r.dates_source == "export"


def test_missing_dates_need_explicit_roller_permission(tmp_path):
    p = _write_variant(tmp_path, delivery_date=None, spot_date=None)
    with pytest.raises(ConventionAmbiguityError, match="allow_roller_dates"):
        load_bloomberg_csv(p)
    df = load_bloomberg_csv(p, allow_roller_dates=True)
    assert (df.dates_source == "fx_roller").all()
    assert df.quality_flags.str.contains("delivery_date_from_roller").all()


def test_roller_never_fills_an_ambiguous_expiry(tmp_path):
    p = _write_variant(tmp_path, expiry_date=None)
    with pytest.raises(ConventionAmbiguityError, match="cannot determine the 1M expiry"):
        load_bloomberg_csv(p, allow_roller_dates=True)


def test_exported_expiry_where_roller_is_ambiguous_is_flagged(bbg):
    one_m = bbg[bbg.tenor == "1M"]
    assert one_m.quality_flags.str.contains("roller_expiry_ambiguous:2026-11-24,2026-11-25,2026-11-27").all()


def test_exported_dates_that_disagree_with_roller_are_flagged(tmp_path):
    raw = pd.read_csv(EXAMPLE)
    raw.loc[raw.tenor == "1M", "delivery_date"] = "2026-12-01"
    p = tmp_path / "mismatch.csv"
    raw.to_csv(p, index=False)
    df = load_bloomberg_csv(p)
    one_m = df[df.tenor == "1M"]
    assert one_m.delivery_date.eq("2026-12-01").all()           # exported date wins
    assert one_m.quality_flags.str.contains("roller_mismatch_delivery:2026-11-30").all()


def test_discount_factor_must_state_its_start_date(tmp_path):
    p = _write_variant(tmp_path, usd_depo_rate=None, usd_df=0.999)
    with pytest.raises(ConventionAmbiguityError, match="usd_df_start"):
        load_bloomberg_csv(p)
    df = load_bloomberg_csv(_write_variant(tmp_path, usd_depo_rate=None, usd_df=0.999, usd_df_start="spot"))
    assert (df.df_dom == 0.999).all()
