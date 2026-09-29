"""Generate data/examples/bloomberg_format_SYNTHETIC_eurusd.csv.

SYNTHETIC, ILLUSTRATIVE VALUES -- NOT BLOOMBERG DATA. The file uses the Bloomberg export
template (docs/bloomberg_export_checklist.md) so the pipeline can be exercised without a
terminal. data_origin=synthetic makes the loader tag every row as non-research.

Valuation: Mon 26 Oct 2026 17:00 New York, chosen so short tenors straddle the FOMC
(28 Oct), ECB (29 Oct), NFP (6 Nov) and CPI (10 Nov). Vol numbers are invented but shaped
like a typical EUR/USD surface, with an event bump in the 1W ATM vol.

Run: python -m scripts.make_synthetic_bloomberg_example
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "examples" / "bloomberg_format_SYNTHETIC_eurusd.csv"
VAL_DATE = pd.Timestamp("2026-10-26")
SPOT, USD_DEPO, EUR_DEPO, BASIS_PIPS_PER_YEAR = 1.1600, 3.80, 2.10, -3.0

#            tenor expiry        delivery      atm   rr25   bf25   rr10   bf10  atm_sprd rr_sprd bf_sprd
SURFACE = [("ON", "2026-10-27", "2026-10-29", 6.10, -0.10, 0.10, -0.19, 0.34, 0.60, 0.20, 0.15),
           ("1W", "2026-11-02", "2026-11-04", 7.60, -0.25, 0.14, -0.48, 0.48, 0.40, 0.20, 0.15),
           ("2W", "2026-11-09", "2026-11-12", 7.35, -0.28, 0.15, -0.53, 0.51, 0.35, 0.20, 0.15),
           ("3W", "2026-11-16", "2026-11-18", 7.20, -0.30, 0.16, -0.57, 0.54, 0.30, 0.18, 0.12),
           ("1M", "2026-11-25", "2026-11-30", 7.00, -0.32, 0.17, -0.61, 0.58, 0.25, 0.18, 0.12),
           ("2M", "2026-12-24", "2026-12-29", 6.95, -0.38, 0.19, -0.72, 0.65, 0.25, 0.15, 0.10),
           ("3M", "2027-01-26", "2027-01-28", 6.95, -0.42, 0.21, -0.80, 0.71, 0.20, 0.15, 0.10),
           ("6M", "2027-04-26", "2027-04-28", 7.05, -0.50, 0.24, -0.95, 0.82, 0.20, 0.15, 0.10),
           ("1Y", "2027-10-26", "2027-10-28", 7.20, -0.55, 0.27, -1.05, 0.92, 0.20, 0.15, 0.10)]


def main() -> None:
    rows = []
    for tenor, exp, dlv, atm, rr25, bf25, rr10, bf10, sa, sr, sb in SURFACE:
        days = (pd.Timestamp(dlv) - VAL_DATE).days
        usd_df = 1 / (1 + USD_DEPO / 100 * days / 360)
        eur_df = 1 / (1 + EUR_DEPO / 100 * days / 360)
        fwd_pts = (SPOT * eur_df / usd_df - SPOT) * 1e4 + BASIS_PIPS_PER_YEAR * days / 365
        rows.append(dict(
            data_origin="synthetic", valuation_ts="2026-10-26 17:00", valuation_tz="America/New_York",
            pair="EURUSD", spot=SPOT, tenor=tenor, expiry_date=exp, expiry_cut="NY10", delivery_date=dlv,
            fwd_points=round(fwd_pts, 2), fwd_points_scale=10000, usd_depo_rate=USD_DEPO, eur_df=eur_df,
            vol_units="pct", atm=atm, rr25=rr25, bf25=bf25, rr10=rr10, bf10=bf10,
            atm_bid=atm - sa / 2, atm_ask=atm + sa / 2, rr25_bid=rr25 - sr / 2, rr25_ask=rr25 + sr / 2,
            bf25_bid=bf25 - sb / 2, bf25_ask=bf25 + sb / 2, rr10_bid=rr10 - sr, rr10_ask=rr10 + sr,
            bf10_bid=bf10 - sb, bf10_ask=bf10 + sb,
            delta_type="spot", atm_type="dns", bf_type="market", premium_currency="USD", rr_sign="call_minus_put",
            bbg_pricing_source="SYNTHETIC", export_ts="n/a",
        ))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(10).to_csv(OUT, index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
