"""Two final checks for the synthetic report.

1. D3 (no expiry before FOMC/ECB): is the 7.8 bp noise-free misfit a flat valley or an iteration
   budget problem? Refit from the reported optimum with a 5x larger budget, and from the truth.
2. Per-model example table at realistic noise (0.05 vol pt, seed 1): fitted vs true, IV and
   price RMSE, status, warnings.

Run:  python -m scripts.final_checks
"""

import ast

import numpy as np
import pandas as pd

from src.calibration import (
    BATES, HBMJ, HESTON, HESTON_HAND_STARTS, JUMP_STARTS, Objective, calibrate, calibrate_nested, fit, fit_metrics,
    identifiability, random_starts, synthetic_surface,
)
from scripts.calibration_recovery import EXP_PLAIN, HESTON_CASES
from scripts.identifiability_study import (
    ADJ, BASE, BATES_NOISE_CASES, DESIGNS, EVENTS, EXP_EVENTS, HBMJ_TRUE, JUMPS, SIGMAS,
)

OBJ = Objective()
R = "results/calibration/"


def d3_check():
    pl = pd.read_csv(R + "ident_placement.csv")
    row = pl[(pl.design == "D3_only_after_both") & (pl.noise_volpts == 0)].iloc[0]
    data = synthetic_surface(HBMJ.build(HBMJ_TRUE, ADJ), DESIGNS["D3_only_after_both"], ADJ)
    nc = calibrate_nested(data, OBJ, n_random=1, seed=7)
    best = nc.hbmj.best
    more = fit(HBMJ, data, OBJ, best.params, max_nfev=2000)
    truth = fit(HBMJ, data, OBJ, HBMJ_TRUE, max_nfev=50)
    idf = identifiability(best, data, OBJ)
    out = {"reported_rmse_bp": row.iv_rmse_bp, "refit_5x_budget_rmse_bp": 1e4 * more.iv_rmse,
           "refit_status": more.message, "from_truth_rmse_bp": 1e4 * truth.iv_rmse,
           "best_v0": best.params["v0"], "true_v0": BASE["v0"],
           "best_combined_var": best.params["sigma_FOMC"] ** 2 + best.params["sigma_ECB"] ** 2,
           "refit_combined_var": more.params["sigma_FOMC"] ** 2 + more.params["sigma_ECB"] ** 2,
           "refit_v0": more.params["v0"], "cond": idf["condition_number"], "flags": idf["flags"]}
    for k, v in out.items():
        print(f"D3 {k}: {v}")


def examples():
    rows = []
    cases = [("Heston", HESTON, HESTON_CASES["fx_typical"], (), EXP_PLAIN),
             ("Bates", BATES, BATES_NOISE_CASES["bates_negative_mean"], (), EXP_PLAIN[:0] + [2 / 365, 4 / 365] + EXP_PLAIN),
             ("HB-MJ", HBMJ, HBMJ_TRUE, EVENTS, EXP_EVENTS)]
    for name, spec, true, ev, exps in cases:
        for noise in (0.0, 0.0005):
            data = synthetic_surface(spec.build(true, ev), exps, ev, noise_vol=noise, seed=1)
            if name == "Heston":
                ms = calibrate(HESTON, data, OBJ, HESTON_HAND_STARTS + random_starts(HESTON, 3, 1))
            elif name == "Bates":
                h = calibrate(HESTON, data, OBJ, HESTON_HAND_STARTS)
                ms = calibrate(BATES, data, OBJ, [{**h.best.params, **j} for j in JUMP_STARTS])
            else:
                ms = calibrate_nested(data, OBJ, n_random=1, seed=1).hbmj
            f = ms.best
            m = fit_metrics(f, data, OBJ)
            idf = identifiability(f, data, OBJ)
            near = [g for g in ms.all if g.iv_rmse <= f.iv_rmse + 1e-4]
            spread = {k: (max(g.params[k] for g in near) - min(g.params[k] for g in near)) / max(abs(f.params[k]), 1e-9)
                      for k in f.free}
            rows.append({"model": name, "noise_volpts": 100 * noise,
                         "true": {k: true[k] for k in f.free}, "fitted": {k: round(f.params[k], 6) for k in f.free},
                         "rel_err_%": {k: round(100 * (f.params[k] - true[k]) / true[k], 2) for k in f.free if true[k]},
                         "iv_rmse_volpts": m["iv_rmse_volpts"], "price_rmse_pips": m["price_rmse_pips"],
                         "status": f"{f.status} {f.message[:40]}", "success": f.success, "bound_hits": f.bound_hits,
                         "starts_near_best": f"{len(near)}/{len(ms.all)}",
                         "max_start_spread_%": round(100 * max(spread.values()), 3),
                         "worst_start_param": max(spread, key=spread.get), "warnings": idf["flags"] or ["none"]})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(R + "report_examples.csv", index=False)


if __name__ == "__main__":
    d3_check()
    examples()
