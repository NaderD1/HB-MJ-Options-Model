"""Figure: Bates vs HB-MJ around a scheduled event (synthetic truth = HB-MJ).

Left:  ATM total implied variance w(T) = IV^2 T for expiries from 0.5 to 8 days around the
       FOMC event at day 3.3 -- truth, the Bates fit and the HB-MJ fit (both fitted to the
       same synthetic HB-MJ surface by scripts/identifiability_study.py).
Right: IV RMSE by expiry of the Bates fit.

Run after the identifiability study:  python -m scripts.plot_event_step
"""

import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.calibration import BATES, HBMJ
from src.models.black_scholes import implied_vol_fast
from src.pricing_engine import price_european
from scripts.identifiability_study import BASE, EVENTS, JUMPS, SIGMAS

R = Path(__file__).resolve().parents[1] / "results"
DAY = 1 / 365
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5df"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # reference categorical slots 1 and 2


def atm_total_var(model, T):
    F = 1.16 * np.exp(0.017 * T)
    D = np.exp(-0.038 * T)
    p = price_european(model, F, F, T, D, True)
    iv, ok = implied_vol_fast(p, F, F, T, D, True, guess=np.full(T.shape, 0.08))
    assert ok.all()
    return iv**2 * T


def main():
    info = pd.read_csv(R / "calibration" / "ident_mimic_info.csv", index_col=0).iloc[:, 0]
    bates_p = ast.literal_eval(info["bates_params"])
    true = {**BASE, **JUMPS, **SIGMAS}
    T = np.linspace(0.5, 8.0, 301) * DAY
    w_true = atm_total_var(HBMJ.build(true, EVENTS), T)
    w_bates = atm_total_var(BATES.build(bates_p, EVENTS), T)
    by_exp = pd.read_csv(R / "calibration" / "ident_mimic_by_expiry.csv")

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    days = T / DAY
    a.plot(days, 1e4 * w_true, color=INK, lw=2, label="Truth (HB-MJ)")
    a.plot(days, 1e4 * w_true, color=ORANGE, lw=2, ls=(0, (1, 2.5)), label="HB-MJ fit (overlaps truth)")
    a.plot(days, 1e4 * w_bates, color=BLUE, lw=2, label="Bates fit")
    a.axvline(3.3, color=MUTED, lw=1, ls="--")
    a.text(3.4, a.get_ylim()[1] * 0.97, "FOMC", color=MUTED, fontsize=9, va="top")
    a.set_xlabel("Days to expiry", color=INK, fontsize=10)
    a.set_ylabel("ATM total implied variance  (×10⁻⁴)", color=INK, fontsize=10)
    a.set_title("Bates smooths the scheduled-event step away", color=INK, fontsize=11, loc="left")
    a.legend(frameon=False, fontsize=9, labelcolor=INK)
    x = np.arange(len(by_exp))
    b.bar(x, by_exp["Bates_iv_rmse_volpts"], color=BLUE, width=0.7)
    b.set_xticks(x, [f"{d:g}" for d in by_exp["expiry_days"]], rotation=60, fontsize=8)
    b.set_xlabel("Expiry (days)", color=INK, fontsize=10)
    b.set_ylabel("Bates IV RMSE (vol pts)", color=INK, fontsize=10)
    b.set_title("Bates error is concentrated at short expiries", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    out = R / "figures" / "bates_vs_hbmj_event_step.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print("saved", out)
    i0, i1 = np.searchsorted(days, 3.2), np.searchsorted(days, 3.4)
    print(f"step over 3.2d->3.4d: truth {w_true[i1] - w_true[i0]:.3e}, Bates {w_bates[i1] - w_bates[i0]:.3e}")


if __name__ == "__main__":
    main()
