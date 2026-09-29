"""Monte Carlo vs characteristic-function prices across models, maturities, strikes,
jump intensities and event configurations. Also a discretization-bias study for the
full-truncation Euler Heston scheme.

Run:  python -m scripts.mc_validation
Writes results/mc_validation.csv and results/mc_discretization.csv.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.bates import PoissonJumps
from src.models.black_scholes import black76_vega
from src.models.hbmj import ScheduledEvent, ScheduledEventJumps
from src.models.heston import HestonParams
from src.models.nested import NestedModel
from src.monte_carlo import mc_price
from src.pricing_engine import price_european

F, D = 1.085, 0.99
DAY = 1 / 365
SEED = 20260929
OUT = Path(__file__).resolve().parents[1] / "results"

HESTON = HestonParams(kappa=2.0, theta=0.0064, sigma=0.3, rho=-0.2, v0=0.0064)          # Feller 0.28
HESTON_HI_VOV = HestonParams(kappa=1.0, theta=0.0064, sigma=0.6, rho=-0.4, v0=0.0064)   # Feller 0.02
SIGMAS = {"FOMC": 0.006, "ECB": 0.005, "CPI": 0.004, "NFP": 0.0035}
EV_ONE_FOMC = (ScheduledEvent(3 * DAY, "FOMC"),)
EV_CLUSTER = (ScheduledEvent(2 * DAY, "CPI"), ScheduledEvent(4 * DAY, "FOMC"),
              ScheduledEvent(9 * DAY, "NFP"), ScheduledEvent(16 * DAY, "ECB"),
              ScheduledEvent(45 * DAY, "FOMC"), ScheduledEvent(200 * DAY, "ECB"))
EV_AFTER_1W = (ScheduledEvent(8 * DAY, "FOMC"), ScheduledEvent(40 * DAY, "ECB"))

CASES: dict[str, NestedModel] = {
    "Heston": NestedModel(HESTON),
    "Heston hi vol-of-vol": NestedModel(HESTON_HI_VOV),
    "Bates lam=0.5": NestedModel(HESTON, PoissonJumps(0.5, -0.01, 0.02)),
    "Bates lam=2": NestedModel(HESTON, PoissonJumps(2.0, -0.01, 0.02)),
    "Bates lam=8": NestedModel(HESTON, PoissonJumps(8.0, -0.01, 0.02)),
    "Bates rare big jumps": NestedModel(HESTON, PoissonJumps(0.5, -0.05, 0.05)),
    "HB-MJ one FOMC": NestedModel(HESTON, PoissonJumps(2.0, -0.01, 0.02), ScheduledEventJumps(EV_ONE_FOMC, SIGMAS)),
    "HB-MJ event cluster": NestedModel(HESTON, PoissonJumps(2.0, -0.01, 0.02), ScheduledEventJumps(EV_CLUSTER, SIGMAS)),
    "HB-MJ big FOMC 1%": NestedModel(HESTON, PoissonJumps(2.0, -0.01, 0.02), ScheduledEventJumps(EV_ONE_FOMC, {**SIGMAS, "FOMC": 0.01})),
    "HB-MJ events after 1W": NestedModel(HESTON, PoissonJumps(2.0, -0.01, 0.02), ScheduledEventJumps(EV_AFTER_1W, SIGMAS)),
}
MATURITIES = {"1W": 7 * DAY, "1M": 30 * DAY, "3M": 0.25, "1Y": 1.0}
N_SD = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])


def run_grid(n_pairs: int = 100_000, steps_per_year: int = 365 * 2) -> pd.DataFrame:
    rows = []
    for name, model in CASES.items():
        for tlab, T in MATURITIES.items():
            K = F * np.exp(N_SD * 0.08 * np.sqrt(T))  # same fixed contracts for every model
            is_call = K >= F
            t0 = time.perf_counter()
            mc = mc_price(model, F, K, T, D, is_call, n_pairs, steps_per_year, seed=SEED)
            cf = price_european(model, F, K, T, D, is_call)
            vega = black76_vega(F, K, T, 0.08, D)
            for i in range(len(K)):
                rows.append(dict(
                    case=name, tenor=tlab, T=T, n_sd=N_SD[i], K=K[i], is_call=bool(is_call[i]),
                    cf_price=cf[i], mc_price=mc.price[i], mc_se=mc.se[i],
                    z=(mc.price[i] - cf[i]) / mc.se[i],
                    err_iv_bp=1e4 * (mc.price[i] - cf[i]) / vega[i],  # approx IV error in vol bp
                    martingale=mc.martingale, martingale_z=mc.martingale / mc.martingale_se,
                    n_events=len(model.events.events_before(T)) if model.events else 0,
                    seconds=time.perf_counter() - t0,
                ))
            print(f"{name:24s} {tlab}: max|z|={max(abs(r['z']) for r in rows[-5:]):.2f}", flush=True)
    return pd.DataFrame(rows)


def discretization_study(n_pairs: int = 200_000) -> pd.DataFrame:
    """Euler bias vs step size at 1Y. Same seed across step sizes (common random numbers are
    not exactly aligned across grids, so SE is still reported per run)."""
    rows = []
    T = 1.0
    K = F * np.exp(N_SD * 0.08 * np.sqrt(T))
    is_call = K >= F
    for name in ("Heston", "Heston hi vol-of-vol"):
        model = CASES[name]
        cf = price_european(model, F, K, T, D, is_call)
        for spy in (12, 52, 365, 365 * 4):
            mc = mc_price(model, F, K, T, D, is_call, n_pairs, spy, seed=SEED)
            for i in range(len(K)):
                rows.append(dict(case=name, steps_per_year=spy, n_sd=N_SD[i], cf_price=cf[i],
                                 bias=mc.price[i] - cf[i], mc_se=mc.se[i],
                                 z=(mc.price[i] - cf[i]) / mc.se[i]))
            print(f"{name:24s} steps/yr={spy:5d}: max|z|={max(abs(r['z']) for r in rows[-5:]):.2f}", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    grid = run_grid()
    grid.to_csv(OUT / "mc_validation.csv", index=False)
    disc = discretization_study()
    disc.to_csv(OUT / "mc_discretization.csv", index=False)
