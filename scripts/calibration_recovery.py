"""Synthetic-recovery experiments for the calibration engine (no market data).

Generates surfaces from KNOWN parameters and checks whether calibration recovers them, from
several deliberately different starting points, then reports identifiability diagnostics.

Run:  python -m scripts.calibration_recovery        (writes results/calibration/*.csv)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.calibration import (
    BATES, HBMJ, HESTON, HESTON_HAND_STARTS, JUMP_STARTS, CalibrationData, Objective, breakdown, calibrate,
    calibrate_nested, compare_models, equivalent_fits, identifiability, random_starts, synthetic_surface,
    variance_step, verify_fit,
)
from src.events import ScheduledEvent

OUT = Path(__file__).resolve().parents[1] / "results" / "calibration"
DAY = 1 / 365
OBJ = Objective("iv", "vega")
SEED = 20260930

HESTON_CASES = {
    "fx_typical": dict(v0=0.0049, kappa=2.5, theta=0.0081, sigma=0.45, rho=-0.25),
    "feller_violating": dict(v0=0.0100, kappa=1.0, theta=0.0064, sigma=0.90, rho=0.30),
}
BASE = HESTON_CASES["fx_typical"]
BATES_CASES = {
    "weak_jumps": dict(**BASE, lam=0.3, mu_J=-0.01, sigma_J=0.02),
    "strong_jumps": dict(**BASE, lam=6.0, mu_J=0.0, sigma_J=0.012),
    "negative_mean_jumps": dict(**BASE, lam=1.0, mu_J=-0.04, sigma_J=0.02),
    "lambda_zero": dict(**BASE, lam=0.0, mu_J=0.0, sigma_J=0.0),
}
JUMPS = dict(lam=1.0, mu_J=-0.02, sigma_J=0.015)
SIGMAS = dict(sigma_FOMC=0.006, sigma_ECB=0.005, sigma_CPI=0.004, sigma_NFP=0.0035)

EXP_PLAIN = [7 * DAY, 14 * DAY, 30 * DAY, 61 * DAY, 91 * DAY, 182 * DAY, 365 * DAY]
EVENTS = (ScheduledEvent(3.3 * DAY, "FOMC", "FOMC-1"), ScheduledEvent(10.2 * DAY, "ECB", "ECB-1"),
          ScheduledEvent(15.3 * DAY, "CPI", "CPI-1"), ScheduledEvent(24.3 * DAY, "NFP", "NFP-1"),
          ScheduledEvent(45.3 * DAY, "FOMC", "FOMC-2"))
# Expiries straddle each event (just before / just after) plus non-event and longer tenors.
EXP_EVENTS = [1 * DAY, 2.8 * DAY, 3.8 * DAY, 7 * DAY, 9.7 * DAY, 10.7 * DAY, 14.8 * DAY, 15.8 * DAY,
              23.8 * DAY, 24.8 * DAY, 35 * DAY, 61 * DAY, 91 * DAY, 182 * DAY]


def recovery_table(true: dict, fits, case: str) -> pd.DataFrame:
    rows = []
    for i, f in enumerate(fits):
        for k in f.free:
            t = true.get(k, 0.0)
            rows.append({"case": case, "start": i, "param": k, "true": t, "fitted": f.params[k],
                         "abs_err": f.params[k] - t, "rel_err": (f.params[k] - t) / t if t else np.nan,
                         "objective": f.cost, "iv_rmse_bp": 1e4 * f.iv_rmse, "n_eval": f.n_eval,
                         "status": f.status, "success": f.success, "bound_hits": ";".join(f.bound_hits),
                         "feller": f.feller_ratio})
    return pd.DataFrame(rows)


def ident_rows(case: str, fitres, data, obj=OBJ) -> list[dict]:
    idf = identifiability(fitres, data, obj)
    return [{"case": case, "model": fitres.model, "condition_number": idf["condition_number"],
             "se_at_0.1volpt": json.dumps({k: round(v, 6) for k, v in idf["se"].items()}),
             "flattest_direction": json.dumps({k: round(v, 3) for k, v in idf["flattest_direction"].items()}),
             "diagnostic_flags": " | ".join(idf["flags"]) or "none"}]


def run_heston():
    rec, ident, eq = [], [], []
    for case, true in HESTON_CASES.items():
        for noise in (0.0, 0.0005):
            data = synthetic_surface(HESTON.build(true, ()), EXP_PLAIN, noise_vol=noise, seed=SEED)
            starts = HESTON_HAND_STARTS + random_starts(HESTON, 3, SEED)
            ms = calibrate(HESTON, data, OBJ, starts)
            label = f"{case}{'_noise5bp' if noise else ''}"
            rec.append(recovery_table(true, ms.all, label))
            ident += ident_rows(label, ms.best, data)
            eq += [{"case": label, **e} for e in equivalent_fits(ms)]
            print(f"Heston {label}: best rmse {1e4 * ms.best.iv_rmse:.3g} bp, FFT check {verify_fit(ms.best, data):.1e}", flush=True)
    return pd.concat(rec), pd.DataFrame(ident), pd.DataFrame(eq)


def run_bates():
    rec, ident, eq = [], [], []
    for case, true in BATES_CASES.items():
        data = synthetic_surface(BATES.build(true, ()), EXP_PLAIN + [2 * DAY, 4 * DAY], seed=SEED)
        h = calibrate(HESTON, data, OBJ, HESTON_HAND_STARTS)
        ms = calibrate(BATES, data, OBJ, [{**h.best.params, **j} for j in JUMP_STARTS])
        rec.append(recovery_table(true, ms.all, case))
        ident += ident_rows(case, ms.best, data)
        eq += [{"case": case, **e} for e in equivalent_fits(ms)]
        print(f"Bates {case}: best rmse {1e4 * ms.best.iv_rmse:.3g} bp  lam={ms.best.params['lam']:.4g} "
              f"(true {true['lam']}), Heston-only rmse {1e4 * h.best.iv_rmse:.3g} bp", flush=True)
    return pd.concat(rec), pd.DataFrame(ident), pd.DataFrame(eq)


def hbmj_case(name, true_sigmas, events, expiries, jumps=JUMPS, noise=0.0):
    true = {**BASE, **jumps, **true_sigmas}
    model = HBMJ.build(true, events)
    data = synthetic_surface(model, expiries, events, noise_vol=noise, seed=SEED)
    nc = calibrate_nested(data, OBJ, n_random=1, seed=SEED)
    best = nc.best()
    rec = recovery_table(true, nc.hbmj.all, name)
    eq = [{"case": name, **e} for e in equivalent_fits(nc.hbmj)]
    bp = nc.bates.best.params
    comp = compare_models(best, data, OBJ).assign(case=name)
    brk = breakdown(best, data).assign(case=name)
    ident = ident_rows(name, nc.hbmj.best, data)
    info = {"case": name, "free_event_types": ";".join(nc.event_types_free),
            "unidentified_event_types": ";".join(nc.event_types_unidentified),
            "fft_check": verify_fit(nc.hbmj.best, data),
            # substitution check: what Bates's Poisson jumps do when the truth has scheduled events
            "bates_fit_lam": bp["lam"], "bates_fit_mu_J": bp["mu_J"], "bates_fit_sigma_J": bp["sigma_J"],
            "bates_fit_v0": bp["v0"], "true_lam": true["lam"], "true_sigma_J": true["sigma_J"],
            "equivalent_fits": len(eq)}
    for e in events:
        step = variance_step(data, data.iv, e)
        if step is not None:
            info[f"step_market_{e.event_id}"] = step
    print(f"HB-MJ {name}: free={nc.event_types_free} fitted="
          f"{ {k: round(nc.hbmj.best.params[k], 5) for k in true_sigmas} } rmse {1e4 * nc.hbmj.best.iv_rmse:.3g} bp", flush=True)
    return rec, comp, brk, ident, info, eq


def run_hbmj():
    recs, comps, brks, idents, infos, eqs = [], [], [], [], [], []

    def add(res):
        rec, comp, brk, ident, info, eq = res
        recs.append(rec); comps.append(comp); brks.append(brk); idents.extend(ident); infos.append(info); eqs.extend(eq)

    # C1: every event type spanned, straddling expiries -> recover all four sigmas.
    add(hbmj_case("all_events_spanned", SIGMAS, EVENTS, EXP_EVENTS))
    # C1n: same with 0.05 vol-pt IV noise.
    add(hbmj_case("all_events_spanned_noise5bp", SIGMAS, EVENTS, EXP_EVENTS, noise=0.0005))
    # C2a: ECB is in the calendar but no contract spans it -> must be reported unidentified.
    late_ecb = tuple(e if e.kind != "ECB" else ScheduledEvent(400 * DAY, "ECB", "ECB-late") for e in EVENTS)
    add(hbmj_case("ecb_not_spanned", SIGMAS, late_ecb, EXP_EVENTS))
    # C2b: ECB spanned but true sigma_ECB = 0 -> fitted ECB sigma should be ~0 (no false positive).
    add(hbmj_case("ecb_true_zero", {**SIGMAS, "sigma_ECB": 0.0}, EVENTS, EXP_EVENTS))
    # C3: all event sigmas 0 -> the surface is a Bates surface; event sigmas should be ~0.
    add(hbmj_case("all_sigmas_zero", {k: 0.0 for k in SIGMAS}, EVENTS, EXP_EVENTS))
    # C5: FOMC and ECB one day apart with no expiry in between -> only their combined variance is identified.
    adjacent = (ScheduledEvent(2.3 * DAY, "FOMC", "FOMC-adj"), ScheduledEvent(3.3 * DAY, "ECB", "ECB-adj"),
                ScheduledEvent(15.3 * DAY, "CPI", "CPI-1"), ScheduledEvent(24.3 * DAY, "NFP", "NFP-1"))
    add(hbmj_case("fomc_ecb_adjacent", SIGMAS, adjacent,
                  [1 * DAY, 7 * DAY, 14.8 * DAY, 15.8 * DAY, 23.8 * DAY, 24.8 * DAY, 35 * DAY, 61 * DAY, 91 * DAY, 182 * DAY]))
    return pd.concat(recs), pd.concat(comps), pd.concat(brks), pd.DataFrame(idents), pd.DataFrame(infos), pd.DataFrame(eqs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h_rec, h_id, h_eq = run_heston()
    b_rec, b_id, b_eq = run_bates()
    e_rec, e_comp, e_brk, e_id, e_info, e_eq = run_hbmj()
    for name, df in [("heston_recovery", h_rec), ("heston_ident", h_id), ("heston_equivalent", h_eq),
                     ("bates_recovery", b_rec), ("bates_ident", b_id), ("bates_equivalent", b_eq),
                     ("hbmj_recovery", e_rec), ("hbmj_comparison", e_comp), ("hbmj_breakdown", e_brk),
                     ("hbmj_ident", e_id), ("hbmj_info", e_info), ("hbmj_equivalent", e_eq)]:
        df.to_csv(OUT / f"{name}.csv", index=False)
    print("written to", OUT)


if __name__ == "__main__":
    main()
