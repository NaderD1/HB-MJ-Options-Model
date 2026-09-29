"""Aggregate results/pooled/*.csv into the pooled-vs-single-date comparison tables.

Run:  python -m scripts.pooled_report
"""

from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results" / "pooled"
pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 40)
EVENTS = ("FOMC", "ECB", "CPI", "NFP")
TRUE = dict(FOMC=0.006, ECB=0.005, CPI=0.004, NFP=0.0035)
fmt = lambda x: f"{x:.4g}"


def event_table(pooled: pd.DataFrame) -> pd.DataFrame:
    h = pooled[pooled.model == "HB-MJ"].copy()
    rows = []
    for _, r in h.iterrows():
        for col in [c for c in r.index if c.startswith("fit_sigma_") and c[10:] in EVENTS or c.startswith("fit_combo_")]:
            if pd.isna(r[col]):
                continue
            key = col[4:]
            t = r.get(f"true_{key}")
            rows.append({"experiment": r.experiment, "seed": r.seed, "key": key, "true_sigma": t, "fit_sigma": r[col],
                         "var_rel_err_%": 100 * (r[col] ** 2 / t**2 - 1), "se_sigma_at_0.1volpt": r.get(f"se_{key}", np.nan)})
    return pd.DataFrame(rows)


def main():
    pooled = pd.read_csv(R / "pooled.csv")
    single = pd.read_csv(R / "single.csv") if (R / "single.csv").exists() else pd.DataFrame()
    oos = pd.read_csv(R / "oos.csv") if (R / "oos.csv").exists() else pd.DataFrame()

    print("=== Pooled fits: overview")
    cols = ["experiment", "seed", "model", "k", "n", "iv_rmse_volpts", "bic", "success", "message", "wall_s"]
    print(pooled[cols].sort_values(["experiment", "seed", "model"]).to_string(index=False, float_format=fmt))

    ev = event_table(pooled)
    ev.to_csv(R / "report_events.csv", index=False)
    print("\n=== Pooled HB-MJ: shared event parameters")
    print(ev.to_string(index=False, float_format=fmt))

    print("\n=== Pooled HB-MJ: structural parameters and daily state")
    h = pooled[pooled.model == "HB-MJ"]
    sc = ["experiment", "seed"] + [c for c in h.columns if c.startswith(("fit_kappa", "fit_sigma", "fit_lam", "fit_mu_J", "fit_sigma_J"))
                                   and not c.startswith(("fit_sigma_FOMC", "fit_sigma_ECB", "fit_sigma_CPI", "fit_sigma_NFP"))]
    sc += [c for c in h.columns if c.startswith("daily_")] + ["max_abs_corr_events", "condition_number", "bound_hits"]
    print(h[[c for c in sc if c in h]].to_string(index=False, float_format=fmt))

    if not single.empty:
        print("\n=== Single-date HB-MJ fits on the same panels (event variance per date)")
        rows = []
        for (exp, nz), g in single.groupby(["experiment", "noise_volpts"]):
            for k in EVENTS:
                v = g[f"fit_sigma_{k}"].dropna().to_numpy()
                if len(v):
                    e = 100 * (v**2 / TRUE[k] ** 2 - 1)
                    rows.append({"experiment": exp, "noise": nz, "type": k, "n_dates_identifiable": len(v),
                                 "median_var_err_%": np.median(e), "p10_%": np.percentile(e, 10), "p90_%": np.percentile(e, 90),
                                 "share_within_20%": np.mean(np.abs(e) <= 20)})
            rows.append({"experiment": exp, "noise": nz, "type": "(fit quality)", "n_dates_identifiable": len(g),
                         "median_var_err_%": np.nan, "p10_%": np.nan, "p90_%": np.nan,
                         "share_within_20%": np.nan, "share_rmse_gt_1bp": float(np.mean(g.iv_rmse_bp > 1.0)),
                         "median_abs_v0_err_%": float(np.median(np.abs(g["v0_rel_err_%"])))})
        st = pd.DataFrame(rows)
        st.to_csv(R / "report_single.csv", index=False)
        print(st.to_string(index=False, float_format=fmt))

    if not oos.empty:
        print("\n=== Out of sample: leave the 2W tenor out, predict it")
        print(oos.to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    main()
