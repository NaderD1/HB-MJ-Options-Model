"""Calendar-time vs trading-time proxy results (reads saved outputs only; fits nothing).

Run:  python -m scripts.compare_clocks      (writes results/clock_comparison_*.csv, prints tables)
"""

from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
MODELS = ("Heston", "Bates", "HB-MJ")
DIRS = {"calendar": R / "proxy", "trading": R / "proxy_trading"}
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)



def read_oos(d):
    """OOS table: the original run's file if present, else the export from the checkpoint files."""
    f = d / "oos_leave_one_expiry.csv"
    o = pd.read_csv(f if f.exists() else d / "oos_from_checkpoints.csv")
    o["spans_event"] = o["spans_event"].astype(str).str.lower().eq("true")
    return o

def main():
    rows, steps, folds, evvar = [], [], [], []
    for clock, d in DIRS.items():
        mt = pd.read_csv(d / "model_table.csv")
        bk = pd.read_csv(d / "breakdown.csv")
        oos = read_oos(d)
        sq = pd.read_csv(d / "stable_quantities.csv")
        st = pd.read_csv(d / "event_steps.csv")
        for m in MODELS:
            r = mt[mt.model == m].iloc[0]
            b = lambda s: float(bk[(bk.subset == s) & (bk.model == m)].iv_rmse_volpts.iloc[0])
            o = oos[oos.model == m]
            agg = lambda g: float(np.sqrt(np.average(g.oos_iv_rmse_volpts**2, weights=g.n)))
            rows.append({"clock": clock, "model": m, "k": int(r.k), "iv_rmse": r.iv_rmse_volpts, "price_rmse_$": r["price_rmse_$"],
                         "aic": r.aic, "bic": r.bic, "event_spanning_iv_rmse": b("event_spanning"),
                         "non_event_iv_rmse": b("non_event"), "oos_event_spanning": agg(o[o.spans_event]),
                         "oos_non_event": agg(o[~o.spans_event]), "converged": r.converged, "bound_hits": r.bound_hits,
                         "warnings": r.identifiability_warnings})
        for _, f in oos.iterrows():
            folds.append({"clock": clock, "held_out": f.held_out_expiry, "model": f.model, "oos": f.oos_iv_rmse_volpts})
        for _, s in st.iterrows():
            steps.append({"clock": clock, "window": s.cluster, "market": s.step_market, "Heston": s.step_Heston,
                          "Bates": s.step_Bates, "HB-MJ": s["step_HB-MJ"], "hbmj_event_part": s.get("hbmj_event_contribution"),
                          "market_minus_bates": s.excess_market_step_over_bates})
        h = sq[sq.model == "HB-MJ"].iloc[0]
        for c in [c for c in sq.columns if c.startswith("event_variance")]:
            evvar.append({"clock": clock, "key": c[15:-1], "variance": h[c], "sigma_pct": 100 * np.sqrt(h[c])})
    out = {"summary": pd.DataFrame(rows), "steps": pd.DataFrame(steps), "folds": pd.DataFrame(folds),
           "event_variance": pd.DataFrame(evvar)}
    for k, v in out.items():
        v.to_csv(R / f"clock_comparison_{k}.csv", index=False)
    print(out["summary"].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(out["steps"].to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    print(out["folds"].pivot_table(index="held_out", columns=["clock", "model"], values="oos").round(3).to_string())
    print(out["event_variance"].to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
