"""The normalized option-quote schema shared by every data source.

Calibration and model comparison read ONLY these columns, so they cannot tell (or care)
whether a row came from a Bloomberg export or the FXE development fallback. The `source`
and `is_dev_fallback` columns exist so results can be filtered -- final research
conclusions must come from rows with is_dev_fallback == False.

Pricing inputs follow the pricing engine's convention: forward F and USD discount factor D
(price = D * E[payoff]); T = ACT/365F calendar-time years between UTC timestamps.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

# column -> (dtype kind, description)
SCHEMA: dict[str, tuple[str, str]] = {
    # identity & audit
    "quote_id": ("str", "Unique row id: <source>|<valuation>|<tenor or expiry>|<bucket or strike>|<C/P>"),
    "source": ("str", "'bloomberg' (research), 'bloomberg_format_<origin>' (e.g. synthetic examples), 'fxe_yfinance_dev'"),
    "is_dev_fallback": ("bool", "True for development-fallback sources (not for research conclusions)"),
    "raw_source_file": ("str", "File the raw quote came from"),
    "raw_file_sha256": ("str", "Hash of that file (detects silent edits)"),
    "raw_row_id": ("str", "Row identifier inside the raw file"),
    "raw_fields": ("str", "JSON of the original raw quote fields"),
    # contract
    "pair": ("str", "Currency pair / underlying, e.g. 'EURUSD' or 'FXE'"),
    "valuation_ts_utc": ("datetime", "Valuation timestamp, UTC"),
    "expiry_ts_utc": ("datetime", "Expiry timestamp (incl. cut time), UTC"),
    "T": ("float", "Year fraction ACT/365F from valuation to expiry"),
    "tenor": ("str", "Quoted tenor (e.g. '1W') or 'listed' for exchange options"),
    "strike": ("float", "Strike"),
    "is_call": ("bool", "Call (on EUR / on FXE) if True"),
    "bucket": ("str", "Original quote bucket: ATM, 25DC, 25DP, 10DC, 10DP, or 'listed'"),
    "quoted_delta": ("float", "Signed quoted delta for delta buckets (NaN for listed options)"),
    "delta_convention": ("str", "e.g. 'spot/dns/market/USD' or 'n/a'"),
    # market state
    "spot": ("float", "Spot at valuation"),
    "forward": ("float", "Outright forward to expiry/delivery"),
    "df_dom": ("float", "Domestic (USD) discount factor used for premiums"),
    "df_for": ("float", "Foreign discount factor implied by forward: F * df_dom / spot"),
    # quotes
    "iv_mid": ("float", "Market implied vol (decimal), Black-76 on (forward, df_dom)"),
    "iv_bid": ("float", "Bid vol if available, else NaN"),
    "iv_ask": ("float", "Ask vol if available, else NaN"),
    "price_mid": ("float", "Black-76 price from iv_mid (Bloomberg) or observed mid (listed)"),
    # events (same rule as the HB-MJ pricing model)
    "n_events": ("int", "Scheduled events with 0 < tau <= T"),
    "event_types": ("str", "';'-joined types of those events"),
    "event_ids": ("str", "';'-joined calendar ids of those events"),
    "time_to_next_event": ("float", "Years to the next scheduled event (any type)"),
    "next_event_type": ("str", "Type of the next event"),
    "events_complete": ("bool", "False if expiry lies beyond calendar coverage (counts are a lower bound)"),
    # quality
    "quality_flags": ("str", "';'-joined data-quality flags set by loaders/validation (empty = clean)"),
}
COLUMNS = list(SCHEMA)


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=_pandas_dtype(k)) for c, (k, _) in SCHEMA.items()})


def _pandas_dtype(kind: str):
    return {"str": object, "bool": bool, "float": float, "int": "int64", "datetime": "datetime64[ns, UTC]"}[kind]


def conform(df: pd.DataFrame) -> pd.DataFrame:
    """Order columns and cast dtypes; raises if a schema column is missing or extra ones exist."""
    missing, extra = set(COLUMNS) - set(df.columns), set(df.columns) - set(COLUMNS)
    if missing or extra:
        raise ValueError(f"Schema mismatch. Missing: {sorted(missing)}; extra: {sorted(extra)}")
    out = df[COLUMNS].copy()
    for c, (kind, _) in SCHEMA.items():
        if kind == "datetime":
            out[c] = pd.to_datetime(out[c], utc=True)
        else:
            out[c] = out[c].astype(_pandas_dtype(kind))
    return out.reset_index(drop=True)


def validate_schema(df: pd.DataFrame) -> list[str]:
    """Return a list of schema problems (empty = OK)."""
    problems = []
    if list(df.columns) != COLUMNS:
        problems.append("columns differ from schema order/content")
        return problems
    for c, (kind, _) in SCHEMA.items():
        dt = df[c].dtype
        if kind == "datetime":  # any resolution (pandas 3 defaults to us), but must be UTC-aware
            ok = isinstance(dt, pd.DatetimeTZDtype) and str(dt.tz) == "UTC"
        elif kind == "str":
            ok = dt == object or pd.api.types.is_string_dtype(dt)
        else:
            ok = str(dt) == str(pd.Series(dtype=_pandas_dtype(kind)).dtype)
        if not ok:
            problems.append(f"{c}: dtype {dt}, expected {kind}")
    if df["quote_id"].duplicated().any():
        problems.append("duplicate quote_id")
    for c in ("raw_row_id", "raw_fields", "raw_source_file", "raw_file_sha256"):
        if (df[c].astype(str).str.len() == 0).any():
            problems.append(f"{c} empty for some rows (audit trail broken)")
    try:
        df["raw_fields"].map(json.loads)
    except (TypeError, ValueError):
        problems.append("raw_fields not valid JSON")
    if not np.isfinite(df[["T", "strike", "spot", "forward", "df_dom", "df_for", "iv_mid"]].to_numpy(float)).all():
        problems.append("non-finite numeric inputs")
    return problems


def describe() -> pd.DataFrame:
    return pd.DataFrame([(c, k, d) for c, (k, d) in SCHEMA.items()], columns=["column", "type", "description"])
