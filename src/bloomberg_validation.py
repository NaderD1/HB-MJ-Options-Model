"""External validation against Bloomberg's OWN numbers (not an internal round trip).

Input: docs/bloomberg_validation_table.csv filled at the terminal. Each row is one tenor with
(a) the raw quotes and conventions exactly as exported, and (b) what Bloomberg itself reports
for that tenor in OVML/OVDV: the strike and vol of each delta bucket, the ATM premium, and the
spot/expiry/delivery dates.

``compare_with_bloomberg`` runs the normal loader on (a) and reports, per tenor and bucket, the
gap to (b), plus the FX date roller's dates vs Bloomberg's. Tolerances:
    strike 0.5 pip (5e-5), vol 0.01 vol pts, ATM premium 0.05 USD pips, dates exact.
Passing this is the condition for (1) trusting the convention handling on research data and
(2) enabling allow_roller_dates=True for historical BDH data.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loaders import fx_trade_date, load_bloomberg_csv
from src.fx_calendar import option_dates
from src.timeutils import to_utc

TEMPLATE = Path(__file__).resolve().parents[1] / "docs" / "bloomberg_validation_table.csv"
BUCKETS = ("ATM", "25DC", "25DP", "10DC", "10DP")
TOL = {"strike": 5e-5, "vol_pts": 0.01, "premium_pips": 0.05}


def compare_with_bloomberg(table_path: str | Path) -> pd.DataFrame:
    table = pd.read_csv(table_path)
    table = table.dropna(subset=["atm"])  # ignore unfilled template rows
    if table.empty:
        raise ValueError("Validation table has no filled rows")
    with tempfile.TemporaryDirectory() as tmp:  # load only the filled rows through the normal loader
        filled = Path(tmp) / Path(table_path).name
        table.to_csv(filled, index=False)
        norm = load_bloomberg_csv(filled)
    rows = []
    for _, t in table.iterrows():
        n = norm[norm.tenor == t.tenor]
        # dates: Bloomberg vs roller
        rolled = option_dates(fx_trade_date(to_utc(n.valuation_ts_utc.iloc[0])), str(t.tenor))
        for key, col in (("spot", "spot_date"), ("expiry", "expiry_date"), ("delivery", "delivery_date")):
            bbg = pd.Timestamp(str(t[col])).date()
            ours = rolled[key]
            label = ours.isoformat() if ours else "ambiguous: " + ",".join(d.isoformat() for d in rolled["expiry_candidates"])
            rows.append({"tenor": t.tenor, "item": f"{key}_date (roller)", "bloomberg": bbg.isoformat(),
                         "ours": label, "gap": 0.0 if bbg == ours else np.nan, "passed": bbg == ours})
        for b in BUCKETS:
            k_col, v_col = f"bbg_K_{b}", f"bbg_vol_{b}"
            if k_col not in t or pd.isna(t.get(k_col)):
                continue
            ours = n[n.bucket == b].iloc[0]
            gap_k = ours.strike - float(t[k_col])
            rows.append({"tenor": t.tenor, "item": f"{b} strike", "bloomberg": float(t[k_col]), "ours": ours.strike,
                         "gap": gap_k, "passed": abs(gap_k) <= TOL["strike"]})
            if not pd.isna(t.get(v_col)):
                gap_v = 100 * ours.iv_mid - float(t[v_col])
                rows.append({"tenor": t.tenor, "item": f"{b} vol (pts)", "bloomberg": float(t[v_col]),
                             "ours": 100 * ours.iv_mid, "gap": gap_v, "passed": abs(gap_v) <= TOL["vol_pts"]})
        if "bbg_premium_ATM_call_usd_pips" in t and not pd.isna(t.get("bbg_premium_ATM_call_usd_pips")):
            ours = 1e4 * n[n.bucket == "ATM"].price_mid.iloc[0]
            gap_p = ours - float(t.bbg_premium_ATM_call_usd_pips)
            rows.append({"tenor": t.tenor, "item": "ATM call premium (USD pips)", "bloomberg": float(t.bbg_premium_ATM_call_usd_pips),
                         "ours": ours, "gap": gap_p, "passed": abs(gap_p) <= TOL["premium_pips"]})
    return pd.DataFrame(rows)
