"""Figures for the PROXY (SPY) study. Reads results/proxy/*.csv, writes results/figures/proxy_*.png.

Every figure is labelled as proxy evidence (SPY listed options), not OTC EUR/USD.
Run:  python -m scripts.proxy_figures
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R, FIG = ROOT / "results" / "proxy", ROOT / "results" / "figures"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5df"
COL = {"Heston": "#2a78d6", "Bates": "#eb6834", "HB-MJ": "#1baf7a"}  # reference categorical slots 1-3
MODELS = ("Heston", "Bates", "HB-MJ")
TAG = "PROXY: SPY listed options, 29 Sep 2026 15:09 ET (not OTC EUR/USD)"


def style(ax, title):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, color=INK, fontsize=10, loc="left")


def save(fig, name):
    fig.text(0.01, 0.005, TAG, fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / name, dpi=160)
    plt.close(fig)


def smiles(res, expiries, name, title):
    fig, axes = plt.subplots(1, len(expiries), figsize=(3.6 * len(expiries), 3.6), sharey=False)
    for ax, e in zip(np.atleast_1d(axes), expiries):
        g = res[res.expiry == e].sort_values("k")
        ax.errorbar(100 * g.k, 100 * g.iv_mkt, yerr=[100 * (g.iv_mkt - g.iv_bid), 100 * (g.iv_ask - g.iv_mkt)],
                    fmt="o", color=INK, ms=4, capsize=2, lw=1, label="Market (bid/mid/ask)")
        for m in MODELS:
            ax.plot(100 * g.k, 100 * g[f"iv_{m}"], color=COL[m], lw=2, label=m)
        style(ax, f"{e}  ({g.T_days.iloc[0]:.1f}d, events: {g.event_types.iloc[0] or 'none'})")
        ax.set_xlabel("log-moneyness ln(K/F), %", fontsize=8, color=INK)
    np.atleast_1d(axes)[0].set_ylabel("implied vol, %", fontsize=8, color=INK)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=7)
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    save(fig, name)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    res = pd.read_csv(R / "contract_results.csv", keep_default_na=False, na_values=[""])
    res["event_types"] = res["event_types"].fillna("")
    bk = pd.read_csv(R / "breakdown.csv")
    steps = pd.read_csv(R / "event_steps.csv")
    oos = pd.read_csv(R / "oos_leave_one_expiry.csv")
    se = pd.read_csv(R / "hbmj_shared_se.csv") if (R / "hbmj_shared_se.csv").exists() else None

    # 1. fits on representative expiries
    exps = sorted(res.expiry.unique())
    smiles(res, [exps[0], exps[1], exps[2], exps[9]], "proxy_1_fit.png", "Heston vs Bates vs HB-MJ on the same contracts")

    # 2. error by subset
    order = [s for s in ["full", "event_spanning", "non_event", "NFP", "CPI", "FOMC", "ECB", "overlapping_events",
                         "short (<= 14d)", "long (> 14d)"] if s in set(bk.subset)]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(order))
    for i, m in enumerate(MODELS):
        v = [bk[(bk.subset == s) & (bk.model == m)].iv_rmse_volpts.iloc[0] for s in order]
        ax.bar(x + (i - 1) * 0.27, v, width=0.25, color=COL[m], label=m)
    ax.set_xticks(x, [s.replace("_", " ") for s in order], rotation=25, fontsize=8)
    ax.set_ylabel("IV RMSE (vol pts)", fontsize=9, color=INK)
    style(ax, "In-sample IV error by contract subset")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "proxy_2_error_by_subset.png")

    # 3. ATM total implied variance around events
    atm = res.loc[res.groupby("expiry").k.apply(lambda s: s.abs().idxmin())].sort_values("T_days")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(atm.T_days, 1e4 * atm.iv_mkt**2 * atm.T_days / 365, "o", color=INK, ms=5, label="Market (ATM)", zorder=5)
    for m in MODELS:
        ax.plot(atm.T_days, 1e4 * atm[f"iv_{m}"] ** 2 * atm.T_days / 365, color=COL[m], lw=2, label=m)
    ev = pd.read_csv(R / "event_steps.csv")
    val = pd.Timestamp("2026-09-29T19:06:00Z")
    import json
    cl = json.load(open(R / "event_clusters.json"))["clusters"]
    from src.events import EventCalendar
    tsmap = EventCalendar.from_csv().table.set_index("event_id").timestamp_utc
    for c in cl:
        for eid in c:
            d = (tsmap[eid] - val).total_seconds() / 86400
            ax.axvline(d, color=MUTED, lw=0.8, ls="--")
            ax.text(d, ax.get_ylim()[1] * 0.98, eid.split("-")[0], rotation=90, fontsize=7, color=MUTED, va="top", ha="right")
    ax.set_xlabel("days to expiry", fontsize=9, color=INK)
    ax.set_ylabel("ATM total implied variance (×10⁻⁴)", fontsize=9, color=INK)
    style(ax, "Total implied variance across scheduled events")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "proxy_3_total_variance.png")

    # 4. event variance by key (HB-MJ) with SE
    if se is not None:
        s = se[se.key.str.startswith(("sigma_", "combo_")) & ~se.key.isin(["sigma", "sigma_J"])].copy()
        s["var"] = s.value**2
        s["var_se"] = 2 * s.value * s["se_at_0.1volpt"]
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.bar(range(len(s)), 1e4 * s["var"], yerr=1e4 * s.var_se, color=COL["HB-MJ"], capsize=4, width=0.6)
        ax.set_xticks(range(len(s)), [k.replace("sigma_", "").replace("combo_", "") for k in s.key], fontsize=8)
        ax.set_ylabel("event variance σ_E² (×10⁻⁴)", fontsize=9, color=INK)
        style(ax, "HB-MJ event variance by event key (± SE at 0.1 vol-pt quote noise)")
        save(fig, "proxy_4_event_variance.png")

    # 5. in-sample vs out-of-sample
    ins = res.copy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (label, flag) in zip(axes, [("expiries spanning an event", True), ("expiries spanning no event", False)]):
        heldset = oos[oos.spans_event == flag].held_out_expiry.unique()
        m = ins.expiry.isin(heldset)
        x = np.arange(len(MODELS))
        is_v = [100 * np.sqrt(np.mean((ins[m][f"iv_{k}"] - ins[m].iv_mkt) ** 2)) for k in MODELS]
        o = oos[oos.spans_event == flag]
        oos_v = [np.sqrt(np.average(o[o.model == k].oos_iv_rmse_volpts ** 2, weights=o[o.model == k].n)) for k in MODELS]
        ax.bar(x - 0.18, is_v, width=0.34, color=[COL[k] for k in MODELS], alpha=0.45, label="in-sample")
        ax.bar(x + 0.18, oos_v, width=0.34, color=[COL[k] for k in MODELS], label="out-of-sample")
        ax.set_xticks(x, MODELS, fontsize=9)
        style(ax, f"{label} (n={len(heldset)})")
    axes[0].set_ylabel("IV RMSE (vol pts)", fontsize=9, color=INK)
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, "proxy_5_in_vs_out_of_sample.png")

    # 6. example where models differ most
    diff = res.groupby("expiry").apply(lambda g: np.sqrt(np.mean((g["iv_HB-MJ"] - g["iv_Bates"]) ** 2)))
    pick = diff.idxmax()
    i = exps.index(pick)
    pair = [exps[max(i - 1, 0)], pick] if i > 0 else [pick, exps[i + 1]]
    smiles(res, pair, "proxy_6_example.png", f"Where the models differ most: {pick} (HB-MJ vs Bates RMS gap {100 * diff[pick]:.2f} vol pts)")
    print("figures written")


if __name__ == "__main__":
    main()
