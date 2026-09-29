"""Aggregate the synthetic calibration / identifiability results into report tables.

Reads results/calibration/*.csv (from calibration_recovery, identifiability_study and
heston_noise_study) and writes results/calibration/report_*.csv, printing compact tables.

Run:  python -m scripts.identifiability_report
"""

from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results" / "calibration"
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

HESTON_P = ["v0", "kappa", "theta", "sigma", "rho"]
JUMP_P = ["lam", "mu_J", "sigma_J"]
JUMP_Q = ["jump_var_rate", "jump_c3_rate", "jump_c4_rate"]
EVENT_P = ["sigma_FOMC", "sigma_ECB", "sigma_CPI", "sigma_NFP"]


def add_quantities(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for k in EVENT_P:
        if k in df:
            df[f"var_{k[6:]}"] = df[k] ** 2
    if {"sigma_FOMC", "sigma_ECB"} <= set(df.columns):
        df["var_FOMC_plus_ECB"] = df["sigma_FOMC"] ** 2 + df["sigma_ECB"] ** 2
    if "v0" in df and "jump_var_rate" in df:
        df["short_var_rate"] = df["v0"] + df["jump_var_rate"]  # instantaneous total variance rate
    return df


def stability(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Across seeds (best fit per seed): mean relative error and dispersion vs the truth row."""
    rows = []
    for (case, noise), g in df[df.seed >= 0].groupby(["case", "noise_volpts"]):
        truth = df[(df.case == case) & (df.seed < 0)]
        if truth.empty:
            continue
        truth = truth.iloc[0]
        best = g[g.is_best]
        for c in cols:
            if c not in best or best[c].isna().all() or pd.isna(truth.get(c)):
                continue
            t = float(truth[c])
            v = best[c].to_numpy(float)
            scale = abs(t) if abs(t) > 1e-12 else np.nan
            rows.append({"case": case, "noise_volpts": noise, "quantity": c, "true": t, "mean_fit": v.mean(),
                         "min_fit": v.min(), "max_fit": v.max(),
                         "mean_abs_rel_err_%": 100 * np.mean(np.abs(v - t)) / scale,
                         "max_abs_rel_err_%": 100 * np.max(np.abs(v - t)) / scale, "n_seeds": len(v)})
    return pd.DataFrame(rows)


def start_sensitivity(df: pd.DataFrame, cols: list[str], tol_bp: float = 1.0) -> pd.DataFrame:
    """Within a seed: range of each quantity across starts ending within tol_bp (0.01 vol pt) of the best RMSE."""
    rows = []
    for (case, noise, seed), g in df[df.seed >= 0].groupby(["case", "noise_volpts", "seed"]):
        near = g[g.iv_rmse_bp <= g.iv_rmse_bp.min() + tol_bp]
        for c in cols:
            if c in near and not near[c].isna().all():
                ref = max(abs(near[c].mean()), 1e-12)
                rows.append({"case": case, "noise_volpts": noise, "seed": seed, "quantity": c,
                             "n_near_best_starts": len(near), "n_starts": len(g),
                             "range_rel_%": 100 * (near[c].max() - near[c].min()) / ref})
    out = pd.DataFrame(rows)
    return out.groupby(["case", "noise_volpts", "quantity"]).agg(
        starts_near_best=("n_near_best_starts", "mean"), starts=("n_starts", "mean"),
        worst_range_rel_pct=("range_rel_%", "max")).reset_index()


def main():
    parts = []
    for f in ("ident_noise.csv", "ident_noise_heston.csv"):
        if (R / f).exists():
            parts.append(add_quantities(pd.read_csv(R / f)))
    noise = pd.concat(parts, ignore_index=True)
    cols = HESTON_P + JUMP_P + JUMP_Q + EVENT_P + ["var_FOMC", "var_ECB", "var_CPI", "var_NFP", "var_FOMC_plus_ECB", "short_var_rate"]
    st = stability(noise, cols)
    ss = start_sensitivity(noise, cols)
    st.to_csv(R / "report_noise_stability.csv", index=False)
    ss.to_csv(R / "report_start_sensitivity.csv", index=False)
    show = st[~st.case.str.startswith("heston_stage")]
    print("\n=== Stability across seeds (best fit per seed)")
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\n=== Start sensitivity (worst relative range across starts within 0.01 vol pt of the best)")
    print(ss[~ss.case.str.startswith("heston_stage")].to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    if (R / "ident_placement.csv").exists():
        pl = pd.read_csv(R / "ident_placement.csv")
        print("\n=== Expiry placement (FOMC day 2.3, ECB day 3.3)")
        print(pl[["design", "noise_volpts", "expiries_days", "fitted_sigma_FOMC", "fitted_sigma_ECB", "fitted_combined_var",
                  "true_combined_var", "se_sigma_FOMC", "se_sigma_ECB", "corr_FOMC_ECB", "condition_number",
                  "splits_from_starts"]].to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
