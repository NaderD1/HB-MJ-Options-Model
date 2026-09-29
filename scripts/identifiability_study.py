"""Identifiability study on synthetic surfaces (no market data).

A. Noise robustness: repeated fits under realistic IV quote noise (several seeds) -- which
   individual parameters are stable, and which *combinations* are (jump cumulants, event variances).
B. Expiry placement: which expiry layouts separate adjacent FOMC/ECB events and which only
   identify their combined variance.
C. Can Bates mimic scheduled events? ATM total-variance steps across each event: truth vs
   Bates fit vs HB-MJ fit, and IV error by expiry.

Run:  python -m scripts.identifiability_study      (writes results/calibration/ident_*.csv)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.calibration import (
    BATES, HBMJ, HESTON, HESTON_HAND_STARTS, JUMP_STARTS, Objective, calibrate, calibrate_nested, evaluate,
    identifiability, synthetic_surface, variance_step,
)
from src.events import ScheduledEvent

OUT = Path(__file__).resolve().parents[1] / "results" / "calibration"
DAY = 1 / 365
OBJ = Objective("iv", "vega")
BASE = dict(v0=0.0049, kappa=2.5, theta=0.0081, sigma=0.45, rho=-0.25)
JUMPS = dict(lam=1.0, mu_J=-0.02, sigma_J=0.015)
SIGMAS = dict(sigma_FOMC=0.006, sigma_ECB=0.005, sigma_CPI=0.004, sigma_NFP=0.0035)
EXP_PLAIN = [2 * DAY, 4 * DAY, 7 * DAY, 14 * DAY, 30 * DAY, 61 * DAY, 91 * DAY, 182 * DAY, 365 * DAY]
EVENTS = (ScheduledEvent(3.3 * DAY, "FOMC", "FOMC-1"), ScheduledEvent(10.2 * DAY, "ECB", "ECB-1"),
          ScheduledEvent(15.3 * DAY, "CPI", "CPI-1"), ScheduledEvent(24.3 * DAY, "NFP", "NFP-1"),
          ScheduledEvent(45.3 * DAY, "FOMC", "FOMC-2"))
EXP_EVENTS = [1 * DAY, 2.8 * DAY, 3.8 * DAY, 7 * DAY, 9.7 * DAY, 10.7 * DAY, 14.8 * DAY, 15.8 * DAY,
              23.8 * DAY, 24.8 * DAY, 35 * DAY, 61 * DAY, 91 * DAY, 182 * DAY]
NOISE = (0.0005, 0.001)   # 0.05 and 0.10 vol pts (realistic mid-quote noise for liquid EUR/USD tenors)
SEEDS = (1, 2, 3, 4)


def jump_cumulants(p: dict) -> dict:
    """Per-year cumulants of the jump part -- what option prices actually see."""
    lam, m, s = p["lam"], p["mu_J"], p["sigma_J"]
    return {"jump_var_rate": lam * (m * m + s * s), "jump_c3_rate": lam * (m**3 + 3 * m * s * s),
            "jump_c4_rate": lam * (m**4 + 6 * m * m * s * s + 3 * s**4)}


# --------------------------------------------------------------------------- A
BATES_NOISE_CASES = {"bates_negative_mean": {**BASE, "lam": 1.0, "mu_J": -0.04, "sigma_J": 0.02},
                     "bates_strong": {**BASE, "lam": 6.0, "mu_J": 0.0, "sigma_J": 0.012}}
HBMJ_TRUE = {**BASE, **JUMPS, **SIGMAS}


def _bates_noise_task(case: str, noise: float, seed: int) -> list[dict]:
    true = BATES_NOISE_CASES[case]
    data = synthetic_surface(BATES.build(true, ()), EXP_PLAIN, noise_vol=noise, seed=seed)
    h = calibrate(HESTON, data, OBJ, HESTON_HAND_STARTS)
    ms = calibrate(BATES, data, OBJ, [{**h.best.params, **j} for j in JUMP_STARTS])
    rows = [{"case": f"heston_stage_{case}", "noise_volpts": 100 * noise, "seed": seed, "start": i,
             "is_best": f is h.best, "iv_rmse_bp": 1e4 * f.iv_rmse, "success": f.success, **f.params}
            for i, f in enumerate(h.all)]
    rows += [{"case": case, "noise_volpts": 100 * noise, "seed": seed, "start": i, "is_best": f is ms.best,
              "iv_rmse_bp": 1e4 * f.iv_rmse, "success": f.success, "status": f.status,
              **f.params, **jump_cumulants(f.params)} for i, f in enumerate(ms.all)]
    return rows


def _hbmj_noise_task(events_name: str, noise: float, seed: int) -> list[dict]:
    events, exps = (EVENTS, EXP_EVENTS) if events_name == "spread" else (ADJ, DESIGNS["D1_before_both_and_after_both"])
    data = synthetic_surface(HBMJ.build(HBMJ_TRUE, events), exps, events, noise_vol=noise, seed=seed)
    nc = calibrate_nested(data, OBJ, n_random=1, seed=seed)
    return [{"case": f"hbmj_{events_name}", "noise_volpts": 100 * noise, "seed": seed, "start": i,
             "is_best": f is nc.hbmj.best, "iv_rmse_bp": 1e4 * f.iv_rmse, "success": f.success, "status": f.status,
             **f.params, **jump_cumulants(f.params)} for i, f in enumerate(nc.hbmj.all)]


def noise_tasks() -> list[tuple]:
    t = [(_bates_noise_task, c, n, sd) for c in BATES_NOISE_CASES for n in NOISE for sd in SEEDS]
    t += [(_hbmj_noise_task, ev, n, sd) for ev in ("spread", "adjacent") for n in NOISE for sd in SEEDS]
    return t


def truth_rows() -> list[dict]:
    rows = [{"case": c, "noise_volpts": 0.0, "seed": -1, "start": -1, "is_best": True, **t, **jump_cumulants(t)}
            for c, t in BATES_NOISE_CASES.items()]
    rows += [{"case": f"hbmj_{e}", "noise_volpts": 0.0, "seed": -1, "start": -1, "is_best": True,
              **HBMJ_TRUE, **jump_cumulants(HBMJ_TRUE)} for e in ("spread", "adjacent")]
    return rows


# --------------------------------------------------------------------------- B
ADJ = (ScheduledEvent(2.3 * DAY, "FOMC", "FOMC-adj"), ScheduledEvent(3.3 * DAY, "ECB", "ECB-adj"),
       ScheduledEvent(15.3 * DAY, "CPI", "CPI-1"), ScheduledEvent(24.3 * DAY, "NFP", "NFP-1"))
LONG = [14.8 * DAY, 15.8 * DAY, 23.8 * DAY, 24.8 * DAY, 35 * DAY, 61 * DAY, 91 * DAY, 182 * DAY]
DESIGNS = {
    # FOMC at day 2.3, ECB at day 3.3 (like FOMC Wed 28 Oct / ECB Thu 29 Oct 2026)
    "D1_before_both_and_after_both": [1 * DAY, 7 * DAY] + LONG,
    "D2_plus_expiry_between": [1 * DAY, 2.8 * DAY, 7 * DAY] + LONG,
    "D3_only_after_both": [7 * DAY] + LONG,
    "D4_standard_tenors_only": [7 * DAY, 14 * DAY, 21 * DAY, 30 * DAY, 61 * DAY, 91 * DAY, 182 * DAY],
}


def _placement_task(name: str, noise: float) -> list[dict]:
    exps = DESIGNS[name]
    true = {**BASE, **JUMPS, **SIGMAS}
    data = synthetic_surface(HBMJ.build(true, ADJ), exps, ADJ, noise_vol=noise, seed=7)
    nc = calibrate_nested(data, OBJ, n_random=1, seed=7)
    idf = identifiability(nc.hbmj.best, data, OBJ)
    corr = idf["corr"]
    splits = [(round(f.params["sigma_FOMC"], 5), round(f.params["sigma_ECB"], 5), round(1e4 * f.iv_rmse, 3))
              for f in nc.hbmj.all]
    p = nc.hbmj.best.params
    return [{
        "design": name, "noise_volpts": 100 * noise, "expiries_days": ";".join(f"{365 * t:g}" for t in exps),
        "fitted_sigma_FOMC": p["sigma_FOMC"], "fitted_sigma_ECB": p["sigma_ECB"],
        "fitted_combined_var": p["sigma_FOMC"] ** 2 + p["sigma_ECB"] ** 2,
        "true_combined_var": SIGMAS["sigma_FOMC"] ** 2 + SIGMAS["sigma_ECB"] ** 2,
        "se_sigma_FOMC": idf["se"].get("sigma_FOMC"), "se_sigma_ECB": idf["se"].get("sigma_ECB"),
        "corr_FOMC_ECB": corr.loc["sigma_FOMC", "sigma_ECB"] if "sigma_ECB" in corr else np.nan,
        "condition_number": idf["condition_number"], "splits_from_starts": str(splits),
        "iv_rmse_bp": 1e4 * nc.hbmj.best.iv_rmse, "flags": " | ".join(idf["flags"]),
    }]


def placement_tasks() -> list[tuple]:
    return [(_placement_task, name, noise) for name in DESIGNS for noise in (0.0, 0.0005)]


# --------------------------------------------------------------------------- C
def mimic_study() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    true = {**BASE, **JUMPS, **SIGMAS}
    data = synthetic_surface(HBMJ.build(true, EVENTS), EXP_EVENTS, EVENTS)
    nc = calibrate_nested(data, OBJ, n_random=1, seed=3)
    fits = nc.best()
    ivs = {"true": data.iv}
    for name, f in fits.items():
        spec = {"Heston": HESTON, "Bates": BATES, "HB-MJ": HBMJ}[name]
        ivs[name] = evaluate(spec.build(f.params, EVENTS), data, OBJ).iv_model
    steps = []
    for e in EVENTS:
        row = {"event": e.event_id, "true_sigma2": true[f"sigma_{e.kind}"] ** 2}
        for k, iv in ivs.items():
            row[f"step_{k}"] = variance_step(data, iv, e)
        if row["step_true"] is not None:
            steps.append(row)
    c = data.contracts
    by_exp = []
    for t in np.unique(c.T):
        m = c.T == t
        r = {"expiry_days": 365 * t, "n_events_spanned": len(data.spanned_events[int(np.where(m)[0][0])])}
        for k in ("Heston", "Bates", "HB-MJ"):
            r[f"{k}_iv_rmse_volpts"] = 100 * float(np.sqrt(np.mean((ivs[k][m] - data.iv[m]) ** 2)))
        by_exp.append(r)
    b = fits["Bates"].params
    info = {"bates_params": b, "bates_jump_cumulants": jump_cumulants(b), "true_jump_cumulants": jump_cumulants(true),
            "true_v0": true["v0"]}
    return pd.DataFrame(steps), pd.DataFrame(by_exp), info


def _run(task: tuple) -> tuple[str, list[dict]]:
    fn, *args = task
    return fn.__name__, fn(*args)


def main(parts: tuple[str, ...] = ("mimic", "placement", "noise"), workers: int = 14):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    OUT.mkdir(parents=True, exist_ok=True)
    if "mimic" in parts:
        steps, by_exp, info = mimic_study()
        steps.to_csv(OUT / "ident_mimic_steps.csv", index=False)
        by_exp.to_csv(OUT / "ident_mimic_by_expiry.csv", index=False)
        pd.Series({k: str(v) for k, v in info.items()}).to_csv(OUT / "ident_mimic_info.csv")
    tasks = (placement_tasks() if "placement" in parts else []) + (noise_tasks() if "noise" in parts else [])
    placement, noise = [], truth_rows() if "noise" in parts else []
    with ProcessPoolExecutor(max_workers=workers) as pool:  # independent fits, fixed seeds -> order-free
        futs = [pool.submit(_run, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            name, rows = fut.result()
            (placement if name == "_placement_task" else noise).extend(rows)
            print(f"[{i}/{len(tasks)}] {name} done", flush=True)
    if placement:
        pd.DataFrame(placement).to_csv(OUT / "ident_placement.csv", index=False)
    if noise:
        pd.DataFrame(noise).to_csv(OUT / "ident_noise.csv", index=False)
    print("written to", OUT)


if __name__ == "__main__":
    import sys

    main(tuple(sys.argv[1:]) or ("mimic", "placement", "noise"))
