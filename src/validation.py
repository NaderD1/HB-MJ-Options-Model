"""Pre-calibration data validation on the normalized schema.

``run_validation`` returns one row per check: how many quotes were checked, how many failed,
the worst value, the tolerance and the failing quote_ids. Nothing is dropped here -- checks
flag, the user decides. Checks:

  forward_consistency   F, D, D_f positive/plausible; Bloomberg forward rebuilt from raw fields
  price_vol_consistency price_mid == Black-76(iv_mid) (Bloomberg) / IV round trip (listed)
  smile_reconstruction  Bloomberg ATM/RR/BF re-derived from normalized points == raw quotes
  delta_round_trip      delta recomputed from (strike, vol, convention) == quoted bucket delta;
                        DNS ATM strike has zero straddle delta
  price_bounds          intrinsic <= price <= upper bound (no negative time value)
  bid_ask               iv_bid <= iv_mid <= iv_ask where quotes exist (no crossed markets)
  butterfly_arbitrage   call prices (via parity) decreasing and convex in strike, per expiry
  calendar_arbitrage    ATM total implied variance non-decreasing in maturity, per valuation
  time_consistency      T == year_fraction(valuation, expiry); expiry after valuation
  event_tags            tags == calendar re-tag == HB-MJ ScheduledEventJumps.events_before(T)
  schema                columns/dtypes/ids/audit fields valid
  audit_trail           raw_fields reproduce the row's key inputs
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.events import EventCalendar
from src.fx_conventions import FXConventions, SmilePoint, atm_strike, fx_delta, quotes_from_smile
from src.models.black_scholes import black76_price, implied_vol
from src.models.hbmj import ScheduledEventJumps
from src.schema import validate_schema
from src.timeutils import to_utc, year_fraction

VOL_TOL = 1e-9
REL_TOL = 1e-10


def _result(name, checked, failed_ids, worst, tol, detail=""):
    return {"check": name, "n_checked": int(checked), "n_failed": len(failed_ids), "passed": len(failed_ids) == 0,
            "worst": float(worst) if worst is not None and np.isfinite(worst) else np.nan, "tolerance": tol,
            "failed_ids": ";".join(map(str, failed_ids[:10])), "detail": detail}


def _conv(row) -> FXConventions:
    dt, at, bt, prem = row.delta_convention.split("/")
    return FXConventions(dt, at, bt, prem)


def check_forward_consistency(df):
    bad, worst = [], 0.0
    for _, r in df.iterrows():
        carry = np.log(r.forward / r.spot) / r["T"]
        ok = r.forward > 0 and 0 < r.df_dom <= 1 and r.df_for > 0 and abs(carry) < 0.25
        if r.bucket != "listed":
            raw = json.loads(r.raw_fields)
            if raw.get("forward") is not None:
                f_raw = float(raw["forward"])
            else:
                f_raw = float(raw["spot"]) + float(raw["fwd_points"]) / float(raw["fwd_points_scale"])
            err = abs(f_raw - r.forward) / r.forward
            worst = max(worst, err)
            ok &= err < REL_TOL
        if not ok:
            bad.append(r.quote_id)
    return _result("forward_consistency", len(df), bad, worst, REL_TOL, "carry |ln(F/S)/T| < 25%; Bloomberg F rebuilt from raw")


def check_cip_basis(df):
    """Information only: CIP gap between market forward and deposit-rate forward (Bloomberg, if EUR rates given)."""
    gaps = []
    for _, r in df[df.bucket == "ATM"].iterrows():
        raw = json.loads(r.raw_fields)
        if raw.get("eur_df") is not None:
            gaps.append(1e4 * (r.forward - r.spot * float(raw["eur_df"]) / r.df_dom) / r.forward)
    worst = max(np.abs(gaps)) if gaps else np.nan
    return _result("cip_basis_info", len(gaps), [], worst, np.nan, "bp of forward; non-zero = cross-currency basis (not an error)")


def check_price_vol_consistency(df):
    bad, worst = [], 0.0
    for _, r in df.iterrows():
        if r.bucket != "listed":
            err = abs(black76_price(r.forward, r.strike, r["T"], r.iv_mid, r.df_dom, r.is_call) - r.price_mid) / r.forward
        else:
            err = abs(implied_vol(r.price_mid, r.forward, r.strike, r["T"], r.df_dom, r.is_call).vol - r.iv_mid)
        worst = max(worst, err)
        if not err < 1e-8:
            bad.append(r.quote_id)
    return _result("price_vol_consistency", len(df), bad, worst, 1e-8)


def check_smile_reconstruction(df):
    b = df[df.bucket != "listed"]
    bad, worst, n = [], 0.0, 0
    for (_, rid), g in b.groupby(["raw_source_file", "raw_row_id"]):
        r0 = g.iloc[0]
        raw = json.loads(r0.raw_fields)
        sc = 0.01 if raw["vol_units"] == "pct" else 1.0
        conv = _conv(r0)
        pts = [SmilePoint(r.bucket, 0.0 if np.isnan(r.quoted_delta) else r.quoted_delta, r.is_call, r.iv_mid, r.strike) for _, r in g.iterrows()]
        back = quotes_from_smile(pts, r0.forward, r0.spot, r0["T"], r0.df_dom, r0.df_for, conv)
        for k, v in back.items():
            err = abs(v - float(raw[k]) * sc)
            worst = max(worst, err)
            if err > VOL_TOL:
                bad.append(f"{r0.tenor}:{k}")
        n += 1
    return _result("smile_reconstruction", n, bad, worst, VOL_TOL, "ATM, RR, BF (in the file's BF convention) vs raw quotes")


def check_delta_round_trip(df):
    b = df[df.bucket != "listed"]
    bad, worst = [], 0.0
    for _, r in b.iterrows():
        conv = _conv(r)
        if r.bucket == "ATM":
            if conv.atm_type != "dns":
                continue
            straddle = (fx_delta(r.forward, r.strike, r["T"], r.iv_mid, r.df_for, 1, conv.delta_type)
                        + fx_delta(r.forward, r.strike, r["T"], r.iv_mid, r.df_for, -1, conv.delta_type))
            err = abs(straddle)
        else:
            err = abs(fx_delta(r.forward, r.strike, r["T"], r.iv_mid, r.df_for, 1 if r.is_call else -1, conv.delta_type) - r.quoted_delta)
        worst = max(worst, err)
        if err > 1e-10:
            bad.append(r.quote_id)
    return _result("delta_round_trip", len(b), bad, worst, 1e-10, "wing delta == bucket; DNS straddle delta == 0")


def check_price_bounds(df):
    bad = []
    intrinsic = df.df_dom * np.where(df.is_call, np.maximum(df.forward - df.strike, 0), np.maximum(df.strike - df.forward, 0))
    upper = df.df_dom * np.where(df.is_call, df.forward, df.strike)
    tv = df.price_mid - intrinsic
    for qid, t, p, u in zip(df.quote_id, tv, df.price_mid, upper):
        if not (t > 0 and p < u):
            bad.append(qid)
    return _result("price_bounds", len(df), bad, float(np.min(tv)) if len(df) else np.nan, 0.0, "worst = smallest time value")


def check_bid_ask(df):
    q = df.dropna(subset=["iv_bid", "iv_ask"])
    bad = q.quote_id[(q.iv_bid > q.iv_mid + 1e-12) | (q.iv_ask < q.iv_mid - 1e-12) | (q.iv_bid > q.iv_ask)].tolist()
    return _result("bid_ask", len(q), bad, float((q.iv_bid - q.iv_ask).max()) if len(q) else np.nan, 0.0, "worst = max(bid - ask)")


def check_butterfly(df, tol=1e-12):
    bad, n = [], 0
    for key, g in df.groupby(["source", "valuation_ts_utc", "expiry_ts_utc"]):
        g = g.sort_values("strike")
        if len(g) < 3:
            continue
        calls = np.where(g.is_call, g.price_mid, g.price_mid + g.df_dom * (g.forward - g.strike))
        K = g.strike.to_numpy()
        slopes = np.diff(calls) / np.diff(K)
        n += 1
        # Call price must decrease in K (slope in [-D, 0]) and be convex (slopes non-decreasing).
        if (slopes > tol).any() or (slopes < -g.df_dom.iloc[0] - tol).any() or (np.diff(slopes) < -tol).any():
            bad.append(f"{key[0]}|{key[2]}")
    return _result("butterfly_arbitrage", n, bad, np.nan, tol, "per expiry: call prices decreasing & convex in K")


def check_calendar(df, tol=1e-10):
    bad, n = [], 0
    for key, g in df.groupby(["source", "valuation_ts_utc"]):
        atm = []
        for exp, h in g.groupby("expiry_ts_utc"):
            row = h.loc[h.bucket == "ATM"] if (h.bucket == "ATM").any() else h.iloc[[np.argmin(np.abs(np.log(h.strike / h.forward)))]]
            atm.append((row["T"].iloc[0], row.iv_mid.iloc[0] ** 2 * row["T"].iloc[0]))
        atm.sort()
        w = np.array([x[1] for x in atm])
        n += 1
        if (np.diff(w) < -tol).any():
            bad.append(str(key))
    return _result("calendar_arbitrage", n, bad, np.nan, tol, "ATM total variance non-decreasing in T")


def check_time(df):
    bad, worst = [], 0.0
    for _, r in df.iterrows():
        err = abs(year_fraction(r.valuation_ts_utc, r.expiry_ts_utc) - r["T"])
        worst = max(worst, err)
        if err > 1e-12 or r.expiry_ts_utc <= r.valuation_ts_utc or str(to_utc(r.valuation_ts_utc).tz) != "UTC":
            bad.append(r.quote_id)
    return _result("time_consistency", len(df), bad, worst, 1e-12)


def check_event_tags(df, calendar: EventCalendar):
    bad = []
    for (v, x), g in df.groupby(["valuation_ts_utc", "expiry_ts_utc"]):
        tag = calendar.tag_both(v, x)
        T = g["T"].iloc[0]
        model_ids = ";".join(e.event_id for e in ScheduledEventJumps(
            calendar.scheduled_events(v, vintage="as_known"), {k: 0.0 for k in ("FOMC", "ECB", "CPI", "NFP")}
        ).events_before(T))
        for _, r in g.iterrows():
            if (r.n_events != tag["n_events"] or r.event_ids != tag["event_ids"] or r.event_ids != model_ids
                    or r.event_ids_realized != tag["event_ids_realized"] or r.schedule_uncertain != tag["schedule_uncertain"]):
                bad.append(r.quote_id)
    return _result("event_tags", len(df), bad, np.nan, 0,
                   "as-known tags == calendar == HB-MJ events_before(T); realized ids kept")


def check_settlement_dates(df):
    """Bloomberg rows: spot < delivery, expiry <= delivery, spot/delivery are joint business days."""
    from src.fx_calendar import is_business_day
    b = df[df.bucket != "listed"]
    bad = []
    for _, r in b.iterrows():
        s_d, d_d = pd.Timestamp(r.spot_date).date(), pd.Timestamp(r.delivery_date).date()
        e_d = r.expiry_ts_utc.tz_convert("America/New_York").date()
        if not (s_d < d_d and e_d <= d_d and is_business_day(s_d, "EURUSD") and is_business_day(d_d, "EURUSD")):
            bad.append(r.quote_id)
    return _result("settlement_dates", len(b), bad, np.nan, 0, "spot < delivery, expiry <= delivery, joint business days")


def check_roller_agreement(df):
    """Information: exported Bloomberg dates vs the FX date roller (the roller validation study)."""
    b = df[(df.bucket == "ATM") & (df.dates_source == "export")]
    mism = b.quote_id[b.quality_flags.str.contains("roller_mismatch")].tolist()
    return _result("roller_agreement_info", len(b), [], float(len(mism)), np.nan,
                   f"{len(mism)} tenor rows where exported dates differ from the roller" + (f": {mism[:5]}" if mism else ""))


def check_schema(df):
    problems = validate_schema(df)
    return _result("schema", len(df), problems, np.nan, 0, "; ".join(problems))


def check_audit(df):
    bad = []
    for _, r in df.iterrows():
        raw = json.loads(r.raw_fields)
        if r.bucket != "listed":
            ok = str(raw["tenor"]) == r.tenor and abs(float(raw["spot"]) - r.spot) < 1e-12
        else:
            ok = abs(float(raw["strike"]) - r.strike) < 1e-12 and raw["cp"] == ("C" if r.is_call else "P")
        if not ok:
            bad.append(r.quote_id)
    return _result("audit_trail", len(df), bad, np.nan, 0, "raw_fields reproduce tenor/spot (Bloomberg) or strike/type (listed)")


def run_validation(df: pd.DataFrame, calendar: EventCalendar | None = None) -> pd.DataFrame:
    calendar = calendar or EventCalendar.from_csv()
    checks = [
        check_schema(df), check_audit(df), check_time(df), check_forward_consistency(df), check_cip_basis(df),
        check_price_vol_consistency(df), check_smile_reconstruction(df), check_delta_round_trip(df),
        check_price_bounds(df), check_bid_ask(df), check_butterfly(df), check_calendar(df),
        check_event_tags(df, calendar), check_settlement_dates(df), check_roller_agreement(df),
    ]
    return pd.DataFrame(checks)
