"""Heston-only noise robustness: several seeds x two noise levels, 6 starts each (parallel).

Run:  python -m scripts.heston_noise_study       (writes results/calibration/ident_noise_heston.csv)
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from src.calibration import HESTON, HESTON_HAND_STARTS, Objective, calibrate, random_starts, synthetic_surface
from scripts.calibration_recovery import EXP_PLAIN, HESTON_CASES

OUT = Path(__file__).resolve().parents[1] / "results" / "calibration"
SEEDS, NOISE = (1, 2, 3, 4), (0.0005, 0.001)


def task(case, noise, seed):
    true = HESTON_CASES[case]
    data = synthetic_surface(HESTON.build(true, ()), EXP_PLAIN, noise_vol=noise, seed=seed)
    ms = calibrate(HESTON, data, Objective(), HESTON_HAND_STARTS + random_starts(HESTON, 3, seed))
    return [{"case": f"heston_{case}", "noise_volpts": 100 * noise, "seed": seed, "start": i, "is_best": f is ms.best,
             "iv_rmse_bp": 1e4 * f.iv_rmse, "success": f.success, "status": f.status, "feller": f.feller_ratio,
             **f.params} for i, f in enumerate(ms.all)]


if __name__ == "__main__":
    jobs = [(c, n, s) for c in HESTON_CASES for n in NOISE for s in SEEDS]
    rows = [{"case": f"heston_{c}", "noise_volpts": 0.0, "seed": -1, "start": -1, "is_best": True, **t}
            for c, t in HESTON_CASES.items()]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for r in pool.map(task, *zip(*jobs)):
            rows += r
    pd.DataFrame(rows).to_csv(OUT / "ident_noise_heston.csv", index=False)
    print("written")
