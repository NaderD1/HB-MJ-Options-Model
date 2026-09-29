"""One command for the whole synthetic study (no market data needed).

    python -m scripts.run_synthetic            # run missing steps only (resume)
    python -m scripts.run_synthetic --force    # rerun every step

Steps (each skipped if its outputs exist, so an interrupted run resumes at the first missing step):
  1. Monte Carlo validation              scripts/mc_validation.py        -> results/mc_*.csv
  2. single-date synthetic recovery      scripts/calibration_recovery.py -> results/calibration/*recovery*.csv
  3. identifiability (mimic/placement/noise)  scripts/identifiability_study.py (+ heston_noise_study)
  4. pooled multi-date panels            scripts/pooled_study.py         -> results/pooled/*.csv
     (per-task checkpoint: rows are rewritten to CSV after every completed task)
  5. aggregation reports                 identifiability_report, pooled_report
Seeds are fixed inside each script (SEED / SEEDS constants); see README "Reproducibility".
Full runtime on 16 cores: roughly 2-3 hours, dominated by step 4.
"""

import argparse
import runpy
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
STEPS = [
    ("Monte Carlo validation", "scripts.mc_validation", [R / "mc_validation.csv", R / "mc_discretization.csv"]),
    ("Synthetic recovery", "scripts.calibration_recovery", [R / "calibration" / "hbmj_recovery.csv"]),
    ("Identifiability study", "scripts.identifiability_study", [R / "calibration" / "ident_placement.csv",
                                                                R / "calibration" / "ident_noise.csv"]),
    ("Heston noise study", "scripts.heston_noise_study", [R / "calibration" / "ident_noise_heston.csv"]),
    ("Pooled panel study", "scripts.pooled_study", [R / "pooled" / "pooled.csv", R / "pooled" / "single.csv"]),
    ("Identifiability report", "scripts.identifiability_report", [R / "calibration" / "report_noise_stability.csv"]),
    ("Pooled report", "scripts.pooled_report", [R / "pooled" / "report_events.csv"]),
]


def main(force: bool) -> None:
    for name, module, outputs in STEPS:
        if not force and all(p.exists() for p in outputs):
            print(f"[skip] {name}: outputs present")
            continue
        print(f"[run ] {name} ({module})", flush=True)
        saved = sys.argv
        sys.argv = [module]
        try:
            runpy.run_module(module, run_name="__main__")
        finally:
            sys.argv = saved


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
