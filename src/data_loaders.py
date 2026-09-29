"""Data loaders: two separate tracks, one normalized schema (src/schema.py).

PRIMARY (research): Bloomberg EUR/USD OTC vol-surface CSV exports  -> load_bloomberg_csv
FALLBACK (development only): FXE ETF options via yfinance + FRED    -> fetch_fxe_snapshot,
                                                                       normalize_fxe
Downstream code sees only the normalized schema. FXE rows carry is_dev_fallback=True and
must not support final research conclusions when Bloomberg data exist.

Bloomberg CSV template: see docs/bloomberg_export_checklist.md. The loader never guesses a
market convention -- missing or contradictory convention fields raise ConventionAmbiguityError.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.events import EventCalendar
from src.fx_calendar import is_business_day, option_dates
from src.fx_conventions import BF_TYPES, DELTA_TYPES, ATM_TYPES, FXConventions, build_smile
from src.models.black_scholes import black76_price, implied_vol
from src.schema import conform, empty_frame
from src.timeutils import localize, to_utc, year_fraction


class ConventionAmbiguityError(ValueError):
    """Raised when a data file does not pin down a market convention. Never guessed."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _jsonable(row: pd.Series) -> str:
    return json.dumps({k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in row.items()}, default=str)


def _tags(calendar: EventCalendar, valuation: pd.Timestamp, expiry: pd.Timestamp) -> dict:
    """As-known-at-valuation event tags (used for pricing/evaluation) + realized ids for audit."""
    return calendar.tag_both(valuation, expiry)


# =========================================================================== #
# Bloomberg (primary)
# =========================================================================== #

EXPIRY_CUTS = {"NY10": ("10:00", "America/New_York"), "TOK15": ("15:00", "Asia/Tokyo")}
REQUIRED_BBG = ["valuation_ts", "pair", "spot", "tenor", "atm", "rr25", "bf25", "vol_units"]
# FX value dates roll at 17:00 New York: a quote stamped at or after 17:00 NY belongs to the
# next trade date.
FX_ROLL = ("17:00", "America/New_York")
CONVENTION_FIELDS = ["delta_type", "atm_type", "bf_type", "premium_currency", "rr_sign"]


def _bbg_conventions(row: pd.Series, where: str) -> FXConventions:
    missing = [c for c in CONVENTION_FIELDS if c not in row or pd.isna(row[c]) or str(row[c]).strip() == ""]
    if missing:
        raise ConventionAmbiguityError(
            f"{where}: convention field(s) {missing} missing. Record them from OVDV/OVML settings "
            f"(see docs/bloomberg_export_checklist.md); the loader will not guess."
        )
    dt, at, bt = str(row.delta_type).strip(), str(row.atm_type).strip(), str(row.bf_type).strip()
    prem, rr_sign = str(row.premium_currency).strip().upper(), str(row.rr_sign).strip()
    if dt not in DELTA_TYPES or at not in ATM_TYPES or bt not in BF_TYPES:
        raise ConventionAmbiguityError(f"{where}: unrecognised convention {dt}/{at}/{bt}")
    if rr_sign != "call_minus_put":
        raise ConventionAmbiguityError(f"{where}: rr_sign={rr_sign!r}; only 'call_minus_put' (EUR calls over) is supported")
    base, quote = str(row.pair)[:3].upper(), str(row.pair)[3:6].upper()
    if prem == quote and dt.endswith("_pa"):
        raise ConventionAmbiguityError(f"{where}: premium in {prem} (domestic) implies unadjusted delta, got {dt}")
    if prem == base and not dt.endswith("_pa"):
        raise ConventionAmbiguityError(f"{where}: premium in {prem} (foreign) implies premium-adjusted delta, got {dt}")
    if prem not in (base, quote):
        raise ConventionAmbiguityError(f"{where}: premium currency {prem} is neither {base} nor {quote}")
    return FXConventions(dt, at, bt, prem)


def _bbg_timestamp(row: pd.Series, where: str) -> pd.Timestamp:
    t = pd.Timestamp(row.valuation_ts)
    if t.tzinfo is None:
        tz = row.get("valuation_tz")
        if tz is None or pd.isna(tz) or str(tz).strip() == "":
            raise ConventionAmbiguityError(f"{where}: valuation_ts has no timezone and no valuation_tz column")
        t = t.tz_localize(str(tz))
    return to_utc(t)


def fx_trade_date(valuation: pd.Timestamp) -> dt.date:
    """FX trade date of a timestamp: New York date, rolled to the next weekday at/after 17:00 NY."""
    ny = to_utc(valuation).tz_convert(FX_ROLL[1])
    d = ny.date()
    if (ny.hour, ny.minute) >= (17, 0):
        d += dt.timedelta(days=1)
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d


def _present(row: pd.Series, col: str) -> bool:
    return col in row and not pd.isna(row.get(col)) and str(row.get(col)).strip() != ""


def _bbg_dates(row: pd.Series, valuation: pd.Timestamp, where: str, allow_roller: bool) -> tuple[dict, list[str]]:
    """Spot, expiry and delivery dates. Exported dates always win; the FX roller fills gaps only
    when explicitly allowed (i.e. after it has been validated against OVML), and every
    disagreement between an exported date and the roller is flagged for the validation study."""
    rolled = option_dates(fx_trade_date(valuation), str(row.tenor))
    flags, out = [], {}
    for key, col in (("spot", "spot_date"), ("expiry", "expiry_date"), ("delivery", "delivery_date")):
        if _present(row, col):
            out[key] = pd.Timestamp(str(row[col])).date()
            if rolled[key] is None:
                cands = ",".join(d.isoformat() for d in rolled["expiry_candidates"])
                flags.append(f"roller_expiry_ambiguous:{cands}")
            elif out[key] != rolled[key]:
                flags.append(f"roller_mismatch_{key}:{rolled[key].isoformat()}")
        elif allow_roller:
            if rolled[key] is None:
                raise ConventionAmbiguityError(
                    f"{where}: the FX roller cannot determine the {row.tenor} expiry (holidays break the "
                    f"delivery->expiry inverse; candidates {rolled['expiry_candidates']}). Export expiry_date from OVML."
                )
            out[key] = rolled[key]
            flags.append(f"{key}_date_from_roller")
        else:
            raise ConventionAmbiguityError(
                f"{where}: {col} missing. Export it from OVML, or pass allow_roller_dates=True once the "
                f"FX date roller has been validated against Bloomberg (docs/bloomberg_validation_table.csv)."
            )
    for key in ("spot", "delivery"):
        if not is_business_day(out[key], "EURUSD"):
            flags.append(f"{key}_date_not_joint_business_day")
    source = "export" if not any(f.endswith("_from_roller") for f in flags) else "fx_roller"
    return {**out, "source": source}, flags


def _bbg_expiry(row: pd.Series, expiry_date, where: str) -> pd.Timestamp:
    cut = row.get("expiry_cut")
    if cut is not None and not pd.isna(cut) and str(cut) in EXPIRY_CUTS:
        time, tz = EXPIRY_CUTS[str(cut)]
    elif all(c in row and not pd.isna(row[c]) for c in ("expiry_time", "expiry_tz")):
        time, tz = str(row.expiry_time), str(row.expiry_tz)
    else:
        raise ConventionAmbiguityError(f"{where}: expiry cut unknown (give expiry_cut=NY10/TOK15 or expiry_time+expiry_tz)")
    return localize(str(expiry_date), time, tz)


def _vol_scale(row: pd.Series, where: str) -> float:
    u = str(row.vol_units).strip().lower()
    if u not in ("pct", "decimal"):
        raise ConventionAmbiguityError(f"{where}: vol_units must be 'pct' or 'decimal', got {u!r}")
    return 0.01 if u == "pct" else 1.0


def _bbg_forward(row: pd.Series, where: str) -> float:
    if "forward" in row and not pd.isna(row.get("forward")):
        return float(row.forward)
    if "fwd_points" in row and not pd.isna(row.get("fwd_points")):
        scale = row.get("fwd_points_scale")
        if scale is None or pd.isna(scale):
            raise ConventionAmbiguityError(f"{where}: fwd_points given without fwd_points_scale (10000 for EURUSD pips)")
        return float(row.spot) + float(row.fwd_points) / float(scale)
    raise ConventionAmbiguityError(f"{where}: need 'forward' or 'fwd_points' (+ fwd_points_scale)")


def _simple_act360_df(rate_pct: float, start, end) -> float:
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    return 1.0 / (1.0 + rate_pct / 100.0 * days / 360.0)


def _bbg_df(row: pd.Series, ccy: str, spot_d, delivery_d, where: str, required: bool) -> float | None:
    """Discount factor from the SPOT date to the DELIVERY date.

    FX option premiums are paid on the spot date and the exercised FX deal settles on the
    delivery date, so the premium quoted by the market is  DF(spot -> delivery) * E[payoff at
    delivery], with the forward to the delivery date. Deposit rates are ACT/360 simple from
    spot to delivery. A directly exported discount factor must state its start date.
    """
    c = ccy.lower()
    if _present(row, f"{c}_df"):
        start = str(row.get(f"{c}_df_start", "")).strip().lower()
        if start != "spot":
            raise ConventionAmbiguityError(f"{where}: {c}_df given but {c}_df_start is {start!r}; only 'spot' (spot->delivery) is accepted")
        return float(row[f"{c}_df"])
    if _present(row, f"{c}_depo_rate"):
        return _simple_act360_df(float(row[f"{c}_depo_rate"]), spot_d, delivery_d)
    if required:
        raise ConventionAmbiguityError(f"{where}: need '{c}_df' (+ {c}_df_start=spot) or '{c}_depo_rate'")
    return None


def _wing_spreads(row: pd.Series, scale: float, tag: str) -> float | None:
    """Approximate wing vol spread: s_ATM + s_BF + s_RR/2 (spreads of the components add)."""
    try:
        s = lambda c: (float(row[f"{c}_ask"]) - float(row[f"{c}_bid"])) * scale
        return s("atm") + s(f"bf{tag}") + 0.5 * s(f"rr{tag}")
    except (KeyError, TypeError, ValueError):
        return None


def load_bloomberg_csv(path: str | Path, calendar: EventCalendar | None = None, allow_roller_dates: bool = False) -> pd.DataFrame:
    """Bloomberg EUR/USD vol-surface export (one row per valuation x tenor) -> normalized quotes.

    Timing: T runs from valuation to the expiry cut (diffusion clock); the forward is to the
    delivery date; df_dom discounts from the spot date (premium payment) to delivery.
    """
    path = Path(path)
    calendar = calendar or EventCalendar.from_csv()
    raw = pd.read_csv(path)
    missing = [c for c in REQUIRED_BBG if c not in raw.columns]
    if missing:
        raise ConventionAmbiguityError(f"{path.name}: required columns missing: {missing}")
    sha = _sha256(path)
    rows = []
    for i, r in raw.iterrows():
        where = f"{path.name} row {i}"
        conv = _bbg_conventions(r, where)
        valuation = _bbg_timestamp(r, where)
        dates, date_flags = _bbg_dates(r, valuation, where, allow_roller_dates)
        expiry = _bbg_expiry(r, dates["expiry"], where)
        T = year_fraction(valuation, expiry)
        if T <= 0:
            raise ValueError(f"{where}: expiry {expiry} is not after valuation {valuation}")
        if not dates["spot"] < dates["delivery"] or dates["delivery"] < dates["expiry"]:
            raise ValueError(f"{where}: inconsistent dates {dates}")
        S, F = float(r.spot), _bbg_forward(r, where)
        D = _bbg_df(r, "USD", dates["spot"], dates["delivery"], where, required=True)
        D_f = F * D / S  # EUR DF (spot->delivery) implied by the market forward (includes cross-currency basis)
        sc = _vol_scale(r, where)
        quotes = {0.25: (float(r.rr25) * sc, float(r.bf25) * sc)}
        if "rr10" in r and "bf10" in r and not pd.isna(r.rr10) and not pd.isna(r.bf10):
            quotes[0.10] = (float(r.rr10) * sc, float(r.bf10) * sc)
        points = build_smile(float(r.atm) * sc, quotes, F, S, T, D, D_f, conv)
        tags = _tags(calendar, valuation, expiry)
        # data_origin guards research integrity: only genuine terminal exports count as research data.
        origin = str(r.get("data_origin", "bloomberg")).strip().lower() if not pd.isna(r.get("data_origin", "bloomberg")) else "bloomberg"
        research = origin == "bloomberg"
        source = "bloomberg" if research else f"bloomberg_format_{origin}"
        conv_str = f"{conv.delta_type}/{conv.atm_type}/{conv.bf_type}/{conv.premium_currency}"
        for p in points:
            flags = ([] if research else [f"non_research_origin_{origin}"]) + date_flags
            if p.bucket == "ATM" and "atm_bid" in r and not pd.isna(r.get("atm_bid")):
                bid, ask = float(r.atm_bid) * sc, float(r.atm_ask) * sc
            else:
                spread = _wing_spreads(r, sc, p.bucket[:2]) if p.bucket != "ATM" else None
                if spread is not None:
                    bid, ask = p.vol - spread / 2, p.vol + spread / 2
                    flags.append("wing_spread_approx")
                else:
                    bid = ask = np.nan
            rows.append({
                "quote_id": f"{source}|{valuation.isoformat()}|{r.tenor}|{p.bucket}|{'C' if p.is_call else 'P'}",
                "source": source, "is_dev_fallback": not research,
                "raw_source_file": path.name, "raw_file_sha256": sha, "raw_row_id": str(i), "raw_fields": _jsonable(r),
                "pair": str(r.pair).upper(), "valuation_ts_utc": valuation, "expiry_ts_utc": expiry, "T": T,
                "tenor": str(r.tenor), "strike": p.strike, "is_call": p.is_call, "bucket": p.bucket,
                "quoted_delta": p.delta if p.bucket != "ATM" else np.nan, "delta_convention": conv_str,
                "spot_date": dates["spot"].isoformat(), "delivery_date": dates["delivery"].isoformat(),
                "dates_source": dates["source"],
                "spot": S, "forward": F, "df_dom": D, "df_for": D_f,
                "iv_mid": p.vol, "iv_bid": bid, "iv_ask": ask,
                "price_mid": float(black76_price(F, p.strike, T, p.vol, D, p.is_call)),
                **tags, "quality_flags": ";".join(flags),
            })
    return conform(pd.DataFrame(rows))


# =========================================================================== #
# FXE / yfinance / FRED (development fallback)
# =========================================================================== #

FRED_SERIES = {"DGS1MO": 1 / 12, "DGS3MO": 0.25, "DGS6MO": 0.5, "DGS1": 1.0}
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"


@dataclass(frozen=True)
class FXEFilters:
    min_open_interest: int = 10
    max_rel_spread: float = 0.5   # (ask - bid) / mid
    min_days: float = 2.0
    n_parity_strikes: int = 6     # strikes nearest spot used to infer the forward


def fetch_fred_treasury(series: dict[str, float] = FRED_SERIES) -> pd.DataFrame:
    """Latest Treasury constant-maturity yields from FRED (public). Records URL and retrieval time."""
    out = []
    retrieved = pd.Timestamp.now(tz="UTC")
    for sid, tenor in series.items():
        url = FRED_URL.format(sid=sid)
        s = pd.read_csv(url, na_values=".")
        s = s.dropna()
        last = s.iloc[-1]
        out.append({"series": sid, "tenor_years": tenor, "obs_date": str(last.iloc[0]), "yield_pct": float(last.iloc[1]),
                    "source_url": url, "retrieved_utc": retrieved.isoformat()})
    return pd.DataFrame(out)


def usd_discount_factor(rates: pd.DataFrame, T: float) -> float:
    """Treasury CMT yields (bond-equivalent, semiannual) -> continuous, linear in T, flat ends."""
    t = rates["tenor_years"].to_numpy(float)
    r_cc = 2.0 * np.log1p(rates["yield_pct"].to_numpy(float) / 200.0)
    order = np.argsort(t)
    return float(np.exp(-np.interp(T, t[order], r_cc[order]) * T))


def fetch_fxe_snapshot(max_expiries: int = 10) -> tuple[pd.DataFrame, dict]:
    """Raw FXE option chains via yfinance (network). Returns (raw_chain, meta)."""
    import yfinance as yf

    tk = yf.Ticker("FXE")
    retrieved = pd.Timestamp.now(tz="UTC")
    hist = tk.history(period="1d", interval="1m")
    if len(hist):
        spot, spot_ts, spot_kind = float(hist["Close"].iloc[-1]), to_utc(hist.index[-1]), "intraday_1m"
    else:  # before the open / holiday: last official close, stamped at 16:00 New York that day
        daily = tk.history(period="10d")
        spot = float(daily["Close"].iloc[-1])
        spot_ts = localize(str(daily.index[-1].date()), "16:00", "America/New_York")
        spot_kind = "previous_close"
    frames = []
    for exp in list(tk.options)[:max_expiries]:
        ch = tk.option_chain(exp)
        for kind, df in (("C", ch.calls), ("P", ch.puts)):
            df = df.copy()
            df["expiration"] = exp
            df["cp"] = kind
            frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    meta = {"underlying": "FXE", "spot": spot, "spot_ts_utc": spot_ts.isoformat(), "spot_kind": spot_kind,
            "retrieved_utc": retrieved.isoformat(),
            "source": "yfinance (Yahoo Finance; quotes may be delayed/stale)"}
    return raw, meta


def save_snapshot(raw: pd.DataFrame, meta: dict, rates: pd.DataFrame, directory: str | Path) -> Path:
    """Write raw chain CSV + rates CSV + meta JSON under one timestamped stem. Returns chain path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = "fxe_" + pd.Timestamp(meta["retrieved_utc"]).strftime("%Y%m%dT%H%M%SZ")
    raw.to_csv(directory / f"{stem}_chain.csv", index=False)
    rates.to_csv(directory / f"{stem}_rates.csv", index=False)
    (directory / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2))
    return directory / f"{stem}_chain.csv"


def load_snapshot(chain_path: str | Path) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    chain_path = Path(chain_path)
    stem = chain_path.name.replace("_chain.csv", "")
    meta = json.loads((chain_path.parent / f"{stem}_meta.json").read_text())
    meta["raw_source_file"] = chain_path.name
    meta["raw_file_sha256"] = _sha256(chain_path)
    return pd.read_csv(chain_path), meta, pd.read_csv(chain_path.parent / f"{stem}_rates.csv")


def infer_forward_from_parity(calls: pd.DataFrame, puts: pd.DataFrame, D: float, spot: float, n: int) -> tuple[float, float, int]:
    """F from put-call parity on the n strikes nearest spot: F_k = K + (C_k - P_k)/D, median.

    Parity is exact only for European options. FXE options are American, so this forward
    is biased by the (small, but non-zero) early-exercise premia; returns (F, residual std, n used).
    """
    m = calls[["strike", "mid"]].merge(puts[["strike", "mid"]], on="strike", suffixes=("_c", "_p"))
    if len(m) < 2:
        return np.nan, np.nan, len(m)
    m = m.iloc[np.argsort(np.abs(m["strike"].to_numpy() - spot))[:n]]
    f_k = m["strike"] + (m["mid_c"] - m["mid_p"]) / D
    F = float(np.median(f_k))
    return F, float(np.std(f_k - F, ddof=1)) if len(f_k) > 1 else np.nan, len(m)


def normalize_fxe(
    raw: pd.DataFrame, meta: dict, rates: pd.DataFrame, calendar: EventCalendar | None = None,
    filters: FXEFilters = FXEFilters(),
) -> tuple[pd.DataFrame, dict]:
    """FXE raw chain -> normalized OTM quotes (development fallback). Returns (frame, report)."""
    calendar = calendar or EventCalendar.from_csv()
    valuation = to_utc(meta["spot_ts_utc"])
    spot = float(meta["spot"])
    report = {"raw_rows": len(raw), "dropped": {}, "expiries": {}}
    rates_note = ";".join(f"{a}={b}@{c}" for a, b, c in zip(rates.series, rates.yield_pct, rates.obs_date))

    df = raw.copy()
    df["row_id"] = np.arange(len(df)).astype(str)
    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    ok = (df["bid"] > 0) & (df["ask"] >= df["bid"])
    report["dropped"]["zero_bid_or_crossed"] = int((~ok).sum())
    df = df[ok]
    wide = (df["ask"] - df["bid"]) / df["mid"] > filters.max_rel_spread
    report["dropped"]["wide_spread"] = int(wide.sum())
    df = df[~wide]
    low_oi = df["openInterest"].fillna(0) < filters.min_open_interest
    report["dropped"]["low_open_interest"] = int(low_oi.sum())
    df = df[~low_oi]

    rows = []
    et = valuation.tz_convert("America/New_York")
    stale = not (et.weekday() < 5 and (et.hour, et.minute) >= (9, 30) and et.hour < 16)
    for exp, g in df.groupby("expiration"):
        expiry = localize(str(exp), "16:00", "America/New_York")
        T = year_fraction(valuation, expiry)
        if T * 365 < filters.min_days:
            report["dropped"][f"too_short_{exp}"] = len(g)
            continue
        D = usd_discount_factor(rates, T)
        calls, puts = g[g.cp == "C"], g[g.cp == "P"]
        F, resid, n_used = infer_forward_from_parity(calls, puts, D, spot, filters.n_parity_strikes)
        report["expiries"][exp] = {"T": T, "D": D, "F": F, "parity_resid_std": resid, "n_parity": n_used}
        if not np.isfinite(F):
            report["dropped"][f"no_parity_{exp}"] = len(g)
            continue
        tags = _tags(calendar, valuation, expiry)
        otm = g[((g.cp == "C") & (g.strike >= F)) | ((g.cp == "P") & (g.strike < F))]
        for _, q in otm.iterrows():
            is_call = q.cp == "C"
            iv = implied_vol(float(q.mid), F, float(q.strike), T, D, is_call)
            if iv.status != "ok":
                report["dropped"].setdefault("iv_failed", 0)
                report["dropped"]["iv_failed"] += 1
                continue
            ivb = implied_vol(float(q.bid), F, float(q.strike), T, D, is_call).vol
            iva = implied_vol(float(q.ask), F, float(q.strike), T, D, is_call).vol
            flags = ["american_exercise", "forward_from_parity"] + (["quotes_possibly_stale"] if stale else [])
            rows.append({
                "quote_id": f"fxe_yfinance_dev|{valuation.isoformat()}|{exp}|{q.strike:g}|{q.cp}",
                "source": "fxe_yfinance_dev", "is_dev_fallback": True,
                "raw_source_file": meta.get("raw_source_file", "live_fetch"), "raw_file_sha256": meta.get("raw_file_sha256", "n/a"),
                "raw_row_id": q.row_id,
                "raw_fields": _jsonable(q.drop(labels=["row_id", "mid"])),
                "pair": "FXE", "valuation_ts_utc": valuation, "expiry_ts_utc": expiry, "T": T, "tenor": "listed",
                "strike": float(q.strike), "is_call": bool(is_call), "bucket": "listed", "quoted_delta": np.nan,
                "delta_convention": "n/a", "spot_date": "n/a", "delivery_date": "n/a", "dates_source": "listed",
                "spot": spot, "forward": F, "df_dom": D, "df_for": F * D / spot,
                "iv_mid": iv.vol, "iv_bid": ivb, "iv_ask": iva, "price_mid": float(q.mid),
                **tags, "quality_flags": ";".join(flags),
            })
    report["rates"] = rates_note
    report["kept"] = len(rows)
    return (conform(pd.DataFrame(rows)) if rows else empty_frame()), report
