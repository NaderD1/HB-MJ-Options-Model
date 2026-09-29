"""Final publication figures (results/figures/final/). Reads saved results; no calibration is rerun.

  1 model nesting GK -> Heston -> Bates -> HB-MJ          6 proxy model fit (trading clock)
  2 Heston vs Bates short-maturity smile                    7 proxy event-step comparison (both clocks)
  3 synthetic scheduled-event total-variance step            8 in-sample vs out-of-sample errors
  4 Bates attempting to mimic HB-MJ                          9 calendar-time vs trading-time proxy result
  5 pooled event-variance recovery vs single-date fits

Run:  python -m scripts.final_figures
"""

import ast
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = R / "figures" / "final"
INK, MUTED, GRID, FILL = "#1f1f1e", "#6b6a64", "#e6e5df", "#f4f3ee"
COL = {"Heston": "#2a78d6", "Bates": "#eb6834", "HB-MJ": "#1baf7a", "GK": "#6b6a64"}
MODELS = ("Heston", "Bates", "HB-MJ")
DAY = 1 / 365
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})


def style(ax, title=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, loc="left")


def save(fig, name, note=None):
    if note:
        fig.text(0.01, 0.005, note, fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03 if note else 0, 1, 1))
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)


# 1 ------------------------------------------------------------------ nesting
def fig1():
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.axis("off")
    boxes = [("Garman–Kohlhagen", "constant vol\n1 parameter", "GK"),
             ("Heston (1993)", "+ stochastic variance\nκ, θ, σ, ρ, v₀", "Heston"),
             ("Bates (1996)", "+ Poisson jumps\nλ, μ_J, σ_J", "Bates"),
             ("HB-MJ (this project)", "+ scheduled-time jumps,\nstochastic size\nσ_FOMC, σ_ECB, σ_CPI, σ_NFP", "HB-MJ")]
    for i, (t, sub, key) in enumerate(boxes):
        x = 0.02 + i * 0.25
        ax.add_patch(FancyBboxPatch((x, 0.2), 0.21, 0.6, boxstyle="round,pad=0.01", fc=FILL, ec=COL[key], lw=2,
                                    transform=ax.transAxes))
        ax.text(x + 0.105, 0.66, t, ha="center", va="center", fontsize=10, color=INK, weight="bold", transform=ax.transAxes)
        ax.text(x + 0.105, 0.40, sub, ha="center", va="center", fontsize=8.5, color=INK, transform=ax.transAxes)
        if i < 3:
            ax.annotate("", xy=(x + 0.25, 0.5), xytext=(x + 0.215, 0.5), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.5))
    ax.text(0.5, 0.05, "φ_HB-MJ(u;T) = φ_Heston(u;T) · φ_Poisson(u;T) · ∏_{0<τᵢ≤T} exp(−½σ²_{E,i} u(u+i))   "
            "— each factor compensated (φ(−i)=1); switching one off recovers the smaller model exactly",
            ha="center", fontsize=8.5, color=MUTED, transform=ax.transAxes)
    save(fig, "fig1_model_nesting.png")


# 2 ------------------------------------------------------------------ Heston vs Bates smiles
def fig2():
    from src.models.bates import bates
    from src.models.black_scholes import implied_vol_array
    from src.models.heston import HestonParams
    from src.pricing_engine import price_european

    H = HestonParams(2.0, 0.0064, 0.3, -0.2, 0.0064)
    B = bates(H, 2.0, -0.01, 0.02)
    F, D = 1.085, 0.99
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (T, lab) in zip(axes, [(7 * DAY, "1 week"), (1.0, "1 year")]):
        k = np.linspace(-2.5, 2.5, 41) * 0.08 * np.sqrt(T)
        K = F * np.exp(k)
        for name, m in (("Heston", H), ("Bates", B)):
            iv, _ = implied_vol_array(price_european(m, F, K, T, D, K >= F), F, K, T, D, K >= F)
            ax.plot(k / (0.08 * np.sqrt(T)), 100 * iv, color=COL[name], lw=2, label=name)
        style(ax, f"{lab} smile, same Heston parameters")
        ax.set_xlabel("standardised moneyness ln(K/F) / (8% √T)")
    axes[0].set_ylabel("implied vol, %")
    axes[0].legend(frameon=False)
    save(fig, "fig2_heston_vs_bates_smile.png",
         "Simulation: κ=2, θ=v₀=0.0064, σ=0.3, ρ=−0.2; Bates adds λ=2/yr, μ_J=−1%, σ_J=2%. Jumps matter most at short maturity.")


# 3 ------------------------------------------------------------------ synthetic event step
def fig3():
    from src.calibration import HBMJ, BATES
    from src.models.black_scholes import implied_vol_fast
    from src.pricing_engine import price_european
    from scripts.identifiability_study import BASE, EVENTS, JUMPS, SIGMAS

    T = np.linspace(0.5, 30, 600) * DAY
    F, D = 1.16 * np.exp(0.017 * T), np.exp(-0.038 * T)

    def w(model):
        p = price_european(model, F, F, T, D, True)
        iv, _ = implied_vol_fast(p, F, F, T, D, True, guess=np.full(T.shape, 0.08))
        return iv**2 * T

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(T / DAY, 1e4 * w(BATES.build({**BASE, **JUMPS}, EVENTS)), color=COL["Bates"], lw=2, label="Heston–Bates (no events)")
    ax.plot(T / DAY, 1e4 * w(HBMJ.build({**BASE, **JUMPS, **SIGMAS}, EVENTS)), color=COL["HB-MJ"], lw=2, label="HB-MJ")
    for e in EVENTS:
        if e.tau < 30 * DAY:
            ax.axvline(e.tau / DAY, color=MUTED, ls="--", lw=0.8)
            ax.text(e.tau / DAY + 0.2, ax.get_ylim()[1] * 0.95, e.kind, fontsize=7, color=MUTED, va="top")
    style(ax, "Scheduled events add discrete steps to total implied variance")
    ax.set_xlabel("days to expiry")
    ax.set_ylabel("ATM total implied variance (×10⁻⁴)")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig3_synthetic_event_step.png",
         "Simulation: σ_FOMC=0.6%, σ_ECB=0.5%, σ_CPI=0.4%, σ_NFP=0.35% at known times; normal jumps with stochastic size.")


# 4 ------------------------------------------------------------------ Bates mimic
def fig4():
    import runpy
    runpy.run_module("scripts.plot_event_step", run_name="__main__")
    src = R / "figures" / "bates_vs_hbmj_event_step.png"
    (OUT / "fig4_bates_mimics_hbmj.png").write_bytes(src.read_bytes())


# 5 ------------------------------------------------------------------ pooled recovery
def fig5():
    ev = pd.read_csv(R / "pooled" / "report_events.csv")
    single = pd.read_csv(R / "pooled" / "single.csv")
    TRUE = dict(FOMC=0.006, ECB=0.005, CPI=0.004, NFP=0.0035)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    keys = ["FOMC", "ECB", "CPI", "NFP"]
    for i, k in enumerate(keys):
        s = single[single.experiment == "P2_single_noisy"][f"fit_sigma_{k}"].dropna()
        e = 100 * (s**2 / TRUE[k] ** 2 - 1)
        ax.boxplot(e, positions=[i - 0.18], widths=0.28, patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor=FILL, color=MUTED), medianprops=dict(color=INK), whiskerprops=dict(color=MUTED),
                   capprops=dict(color=MUTED))
        p = ev[(ev.experiment == "P1_main") & (ev.key == f"sigma_{k}")]
        ax.scatter(np.full(len(p), i + 0.18), p["var_rel_err_%"], color=COL["HB-MJ"], s=36, zorder=3,
                   label="pooled (4 seeds)" if i == 0 else None)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(4), keys)
    ax.set_ylabel("event-variance error, %")
    style(ax, "Pooled estimation recovers shared event variances; single-date fits cannot")
    ax.plot([], [], color=MUTED, lw=6, label="single-date fits (29 dates)")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "fig5_pooled_event_variance_recovery.png",
         "Simulation: daily synthetic panels 19 Oct–30 Nov 2026, ON/1W/2W/3W/1M, real event calendar, 0.05 vol-pt quote noise.")


# 6-9 ---------------------------------------------------------------- proxy
PROXY_NOTE = "PROXY: SPY listed options, one snapshot 29 Sep 2026 15:09 ET — not OTC EUR/USD evidence."



def read_oos(d):
    """OOS table: the original run's file if present, else the export from the checkpoint files."""
    f = d / "oos_leave_one_expiry.csv"
    o = pd.read_csv(f if f.exists() else d / "oos_from_checkpoints.csv")
    o["spans_event"] = o["spans_event"].astype(str).str.lower().eq("true")
    return o

def load_proxy(d):
    res = pd.read_csv(R / d / "contract_results.csv", keep_default_na=False, na_values=[""])
    res["event_types"] = res["event_types"].fillna("")
    return (res, pd.read_csv(R / d / "model_table.csv"), pd.read_csv(R / d / "breakdown.csv"),
            pd.read_csv(R / d / "event_steps.csv"), read_oos(R / d))


def fig6():
    res = load_proxy("proxy_trading")[0]
    exps = sorted(res.expiry.unique())
    pick = [exps[0], exps[1], exps[2], exps[9]]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, e in zip(axes, pick):
        g = res[res.expiry == e].sort_values("k")
        ax.errorbar(100 * g.k, 100 * g.iv_mkt, yerr=[100 * (g.iv_mkt - g.iv_bid), 100 * (g.iv_ask - g.iv_mkt)], fmt="o",
                    color=INK, ms=3.5, capsize=2, lw=1, label="market bid/mid/ask")
        for m in MODELS:
            ax.plot(100 * g.k, 100 * g[f"iv_{m}"], color=COL[m], lw=2, label=m)
        style(ax, f"{e} ({g.event_types.iloc[0] or 'no event'})")
        ax.set_xlabel("ln(K/F), %")
    axes[0].set_ylabel("implied vol, % (trading-time)")
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, "fig6_proxy_model_fit.png", PROXY_NOTE + " Trading-day clock.")


def fig7():
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, (d, lab) in zip(axes, [("proxy", "calendar-time clock"), ("proxy_trading", "trading-day clock")]):
        st = load_proxy(d)[3]
        x = np.arange(len(st))
        ax.bar(x - 0.3, 1e4 * st.step_market, 0.2, color=INK, label="market")
        for j, m in enumerate(MODELS):
            ax.bar(x - 0.1 + 0.2 * j, 1e4 * st[f"step_{m}"], 0.2, color=COL[m], label=m)
        ax.set_xticks(x, [c.replace("-2026-", " ").replace("+", "+\n") for c in st.cluster], fontsize=7)
        style(ax, f"Total-variance step across each event window ({lab})")
    axes[0].set_ylabel("Δ ATM total implied variance (×10⁻⁴)")
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, "fig7_proxy_event_steps.png", PROXY_NOTE + " Market steps are clock-invariant; model steps are not.")


def _oos_agg(oos, flag):
    o = oos[oos.spans_event == flag]
    return {m: float(np.sqrt(np.average(o[o.model == m].oos_iv_rmse_volpts ** 2, weights=o[o.model == m].n))) for m in MODELS}


def fig8():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (d, lab) in zip(axes, [("proxy", "calendar clock"), ("proxy_trading", "trading-day clock")]):
        res, mt, bk, st, oos = load_proxy(d)
        ins = {m: float(bk[(bk.subset == "event_spanning") & (bk.model == m)].iv_rmse_volpts.iloc[0]) for m in MODELS}
        out = _oos_agg(oos, True)
        x = np.arange(3)
        ax.bar(x - 0.18, [ins[m] for m in MODELS], 0.34, color=[COL[m] for m in MODELS], alpha=0.45, label="in-sample")
        ax.bar(x + 0.18, [out[m] for m in MODELS], 0.34, color=[COL[m] for m in MODELS], label="out-of-sample")
        ax.set_xticks(x, MODELS)
        style(ax, f"Event-spanning expiries, {lab}")
        ax.set_ylabel("IV RMSE (vol pts, own clock)")
    axes[0].legend(frameon=False)
    save(fig, "fig8_in_vs_out_of_sample.png", PROXY_NOTE + " Out of sample = leave-one-expiry-out (12 event-spanning folds).")


def fig9():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(3)
    for j, (d, lab, a) in enumerate([("proxy", "calendar", 0.45), ("proxy_trading", "trading", 1.0)]):
        res, mt, bk, st, oos = load_proxy(d)
        pr = [float(mt[mt.model == m]["price_rmse_$"].iloc[0]) for m in MODELS]
        axes[0].bar(x + (j - 0.5) * 0.36, pr, 0.34, color=[COL[m] for m in MODELS], alpha=a, label=f"{lab} clock")
        o = _oos_agg(oos, True)
        axes[1].bar(x + (j - 0.5) * 0.36, [o[m] / o["Heston"] for m in MODELS], 0.34, color=[COL[m] for m in MODELS],
                    alpha=a, label=f"{lab} clock")
    axes[0].set_xticks(x, MODELS)
    axes[0].set_ylabel("in-sample price RMSE ($, clock-invariant)")
    style(axes[0], "In-sample pricing error")
    axes[1].set_xticks(x, MODELS)
    axes[1].axhline(1, color=INK, lw=0.8)
    axes[1].set_ylabel("OOS IV RMSE relative to Heston")
    style(axes[1], "Out-of-sample, event-spanning folds")
    axes[0].legend(frameon=False)
    save(fig, "fig9_calendar_vs_trading_time.png", PROXY_NOTE + " Pale = calendar clock, solid = trading-day clock.")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9):
        f()
        print(f"{f.__name__} done", flush=True)


if __name__ == "__main__":
    main()
