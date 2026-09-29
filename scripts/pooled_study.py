"""Pooled multi-date calibration on synthetic daily panels (no market data).

Panels: EUR/USD business days 19 Oct - 30 Nov 2026, 16:00 New York valuations, tenors
ON/1W/2W/3W/1M from the FX roller (ambiguous expiries dropped), 5 strikes, the real
as-known Oct-Dec 2026 event calendar. Truth: daily v0/theta/rho paths; kappa, sigma, jumps
constant; event variance shared by type (FOMC .006, ECB .005, CPI .004, NFP .0035).

Experiments
  P1  pooled Heston -> Bates -> HB-MJ, 4 seeds at 0.05 vol-pt noise (+ 1 noise-free)
  P2  single-date HB-MJ fits on every date of the same panels (noise-free and noisy)
  P3  nearby events: real calendar (FOMC/ECB separated by the 16:00 valuation) vs a catalog with
      ECB moved to 30 minutes after FOMC (nothing separates them -> combined parameter)
  P4  no ON tenor (1W-1M only): the single-date D3/D4 false-optimum situation, pooled vs single
  P5  all Heston/Bates parameters date-specific, without and with a smoothness penalty
  OOS leave-2W-out: fit without the 2W tenor, predict it (Heston vs Bates vs HB-MJ)

Run:  python -m scripts.pooled_study        (writes results/pooled/*.csv)
"""

from __future__ import annotations

import datetime as dt
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "results" / "pooled"
START, END = dt.date(2026, 10, 19), dt.date(2026, 11, 30)
SHARED_TRUE = dict(kappa=2.5, sigma=0.45, lam=1.0, mu_J=-0.02, sigma_J=0.015)
EVENT_SIGMA = dict(FOMC=0.006, ECB=0.005, CPI=0.004, NFP=0.0035)
NOISE = 0.0005
SEEDS = (1, 2, 3, 4)


def _setup(seed: int, noise: float, tenors=None, moved_ecb: bool = False):
    from src.panel_synthetic import PanelTruth, TENORS, business_days, catalog_from_calendar, make_panel, simulate_state
    from src.pooled import CatalogEvent

    dates = business_days(START, END)
    cat = catalog_from_calendar(pd.Timestamp("2026-10-19T00:00Z"), pd.Timestamp("2027-01-31T00:00Z"))
    if moved_ecb:  # ECB 30 min after FOMC on 28 Oct: no valuation or expiry can separate them
        cat = tuple(CatalogEvent(e.event_id, e.kind, pd.Timestamp("2026-10-28T18:30:00Z"))
                    if e.event_id == "ECB-2026-10-29" else e for e in cat)
    truth = PanelTruth(SHARED_TRUE, EVENT_SIGMA, simulate_state(len(dates), seed))
    panel, info = make_panel(dates, cat, truth, tenors=tenors or TENORS, noise_vol=noise, seed=seed)
    return panel, truth, info


def _summarize_pooled(tag: str, nc, panel, truth) -> list[dict]:
    from src.pooled import pooled_identifiability, pooled_metrics
    from src.pooled_study_utils import true_shared

    rows = []
    for name, f in nc.best().items():
        m = pooled_metrics(f, panel)
        ts = true_shared(truth, f.spec.event_keys)
        row = {"experiment": tag, "model": name, **m, "seconds": f.seconds, "n_eval": f.n_eval,
               "bound_hits": ";".join(f.bound_hits), "message": f.message[:60]}
        for k in f.spec.shared_all:
            row[f"fit_{k}"], row[f"true_{k}"] = f.shared[k], ts.get(k, np.nan)
        for k in f.spec.local:
            est = np.array([l[k] for l in f.local])
            tru = np.array([d.get(k, truth.shared.get(k)) for d in truth.daily])  # constants in truth
            row[f"daily_{k}_mean_abs_rel_err_%"] = 100 * float(np.mean(np.abs(est - tru) / np.abs(tru)))
        if name == "HB-MJ":
            try:
                idf = pooled_identifiability(f)
                for k in f.spec.event_keys:
                    row[f"se_{k}"] = idf["se"][k]
                ev = list(f.spec.event_keys)
                c = idf["corr"].loc[ev, ev].to_numpy()
                row["max_abs_corr_events"] = float(np.max(np.abs(c - np.eye(len(ev))))) if len(ev) > 1 else 0.0
                row["condition_number"] = idf["condition_number"]
            except Exception as exc:  # diagnostics must never hide a fit
                row["ident_error"] = repr(exc)
        rows.append(row)
    return rows


def task_pooled(tag: str, seed: int, noise: float, tenors=None, moved_ecb=False, local=None, smooth=0.0,
                threads: int = 1, timeout_s: float | None = None):
    from src.pooled import DEFAULT_LOCAL, calibrate_pooled_nested

    panel, truth, info = _setup(seed, noise, tenors, moved_ecb)
    t0 = time.perf_counter()
    log = OUT / f"progress_{tag}_seed{seed}.log"

    def progress(msg: str) -> None:  # append-only progress file, one flushed line per message
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

    nc = calibrate_pooled_nested(panel, local=local or DEFAULT_LOCAL, smooth=smooth, threads=threads,
                                 timeout_s=timeout_s, progress=progress)
    rows = _summarize_pooled(tag, nc, panel, truth)
    stages = {"Heston": nc.heston, "Bates": nc.bates, "HB-MJ": nc.hbmj}
    timed_out = ";".join(n for n, f in stages.items() if f.status == -99)
    for r in rows:
        r.update({"seed": seed, "noise_volpts": 100 * noise, "n_dates": len(panel.dates), "wall_s": time.perf_counter() - t0,
                  "clusters": " | ".join("+".join(e.event_id for e in c) for c in nc.clusters),
                  "local": ",".join(local or DEFAULT_LOCAL), "smooth": smooth,
                  "timeout_s": timeout_s, "timed_out_stages": timed_out})
    progress(f"DONE in {time.perf_counter() - t0:.0f}s; timed-out stages: {timed_out or 'none'}")
    return "pooled", rows


def task_single(tag: str, seed: int, noise: float, i: int, tenors=None):
    from src.calibration import calibrate_nested

    panel, truth, _ = _setup(seed, noise, tenors)
    pdt = panel.dates[i]
    nc = calibrate_nested(pdt.data, n_random=1, seed=seed)
    f = nc.hbmj.best
    row = {"experiment": tag, "seed": seed, "noise_volpts": 100 * noise, "date": pdt.label, "iv_rmse_bp": 1e4 * f.iv_rmse,
           "free_event_types": ";".join(nc.event_types_free), "success": f.success,
           "v0_rel_err_%": 100 * (f.params["v0"] / truth.daily[i]["v0"] - 1)}
    for k in EVENT_SIGMA:
        row[f"fit_sigma_{k}"] = f.params[f"sigma_{k}"] if k in nc.event_types_free else np.nan
    return "single", [row]


def task_oos(seed: int, noise: float):
    from src.panel_synthetic import split_heldout
    from src.pooled import calibrate_pooled_nested, holdout_error

    panel, truth, _ = _setup(seed, noise)
    train, held = split_heldout(panel, "2W")
    nc = calibrate_pooled_nested(train, threads=1)
    return "oos", [{**holdout_error(f, train, held), "seed": seed, "noise_volpts": 100 * noise} for f in nc.best().values()]


def tasks() -> list[tuple]:
    from src.panel_synthetic import business_days

    n_dates = len(business_days(START, END))
    t = [(task_pooled, "P1_main", s, NOISE) for s in SEEDS]
    t += [(task_pooled, "P1_noise_free", 1, 0.0)]
    t += [(task_pooled, "P3_ecb_moved_next_to_fomc", 1, NOISE, None, True)]
    t += [(task_pooled, "P4_no_ON_tenor", 1, NOISE, ("1W", "2W", "3W", "1M"))]
    t += [(task_pooled, "P4_no_ON_tenor_noise_free", 1, 0.0, ("1W", "2W", "3W", "1M"))]
    t += [(task_pooled, "P5_all_local", 1, NOISE, None, False, ("v0", "kappa", "theta", "sigma", "rho", "lam", "mu_J", "sigma_J"))]
    t += [(task_pooled, "P5_all_local_smooth", 1, NOISE, None, False, ("v0", "kappa", "theta", "sigma", "rho", "lam", "mu_J", "sigma_J"), 1e-4)]
    t += [(task_oos, 1, NOISE)]
    t += [(task_single, "P2_single_noisy", 1, NOISE, i) for i in range(n_dates)]
    t += [(task_single, "P2_single_noise_free", 1, 0.0, i) for i in range(n_dates)]
    t += [(task_single, "P4_single_no_ON_noise_free", 1, 0.0, i, ("1W", "2W", "3W", "1M")) for i in range(n_dates)]
    return t


def _run(task):
    fn, *args = task
    return fn(*args)


def main(workers: int = 14, only: str | None = None):
    """only: run just the tasks whose tag starts with this prefix, appending to existing CSVs."""
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"pooled": [], "single": [], "oos": []}
    ts = tasks()
    if only:
        ts = [t for t in ts if isinstance(t[1], str) and t[1].startswith(only)]
        for k in out:
            if (OUT / f"{k}.csv").exists():
                out[k] = pd.read_csv(OUT / f"{k}.csv").to_dict("records")
    # Heavy pooled tasks first so they are not left for the end.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_run, t) for t in ts]
        for i, fut in enumerate(as_completed(futs), 1):
            kind, rows = fut.result()
            out[kind].extend(rows)
            print(f"[{i}/{len(ts)}] {kind} done", flush=True)
            for k, v in out.items():  # checkpoint partial results
                if v:
                    pd.DataFrame(v).to_csv(OUT / f"{k}.csv", index=False)
    print("written to", OUT)


if __name__ == "__main__":
    import sys

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main(only=sys.argv[1] if len(sys.argv) > 1 else None)
