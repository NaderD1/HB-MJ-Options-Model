"""Rerun ONLY the missing P5_all_local case (all Heston/Bates parameters date-specific, no smoothing).

Everything else in results/pooled/ is kept: the new rows are appended to pooled.csv after any
previous P5_all_local rows are removed. 8 threads inside the fit; each individual least-squares
fit stops after TIMEOUT_S and then reports its best point so far as timed out (status -99),
never as converged. Progress: results/pooled/progress_P5_all_local_seed1.log

Run:  python -m scripts.rerun_p5_all_local
"""

import time

import pandas as pd

from scripts.pooled_study import NOISE, OUT, task_pooled

TIMEOUT_S = 12 * 60
ALL = ("v0", "kappa", "theta", "sigma", "rho", "lam", "mu_J", "sigma_J")

if __name__ == "__main__":
    t0 = time.perf_counter()
    _, rows = task_pooled("P5_all_local", 1, NOISE, None, False, ALL, 0.0, threads=8, timeout_s=TIMEOUT_S)
    path = OUT / "pooled.csv"
    old = pd.read_csv(path)
    old = old[old.experiment != "P5_all_local"]
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)
    print(f"appended {len(rows)} rows in {time.perf_counter() - t0:.0f}s")
