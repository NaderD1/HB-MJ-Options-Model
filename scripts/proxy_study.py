"""PROXY empirical study: Heston vs Bates vs HB-MJ on public listed-option data.

PROXY EVIDENCE ONLY. SPY options (and FXE, which fails the quality filters) are not OTC EUR/USD
options; nothing here is a claim about the EUR/USD volatility market.

Data: one intraday snapshot (29 Sep 2026, ~15:09 New York) saved by scripts/snapshot_proxy.py,
normalized by src.data_loaders.normalize_listed with the standard quality filters. The as-known
event calendar supplies FOMC/ECB/CPI/NFP. Only expiries whose event set is complete
(events_complete) and that are not schedule-uncertain are used.

Contracts: per expiry, the OTM quote nearest to each target log-moneyness
k = z * ATMvol * sqrt(T), z in {-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2}. The same contracts are used
for all three models.

Calibration: the committed pooled framework (src/pooled.py) with the smoothed default specification
(date-specific v0, theta, rho; smooth=1e-4). With a single snapshot the panel has ONE valuation date,
so the smoothing penalty is inactive and the pooled fit reduces to a joint fit of that date.
Objective: vega-per-expiry weighted IV errors (as in the synthetic study).

Out of sample: leave-one-expiry-out -- each calibration expiry is removed, all three models are
refit on the rest, and the removed expiry is predicted.

Run:  python -m scripts.proxy_study           (results/proxy/*.csv, results/figures/proxy_*.png)
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "proxy"
SNAP = ROOT / "data" / "snapshots" / "spy_20260929T190930Z_chain.csv"
FXE_SNAP = ROOT / "data" / "snapshots" / "fxe_20260929T190930Z_chain.csv"
Z_TARGETS = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
SHORT_DAYS = 14
SMOOTH = 1e-4
TIMEOUT_S = 600


# --------------------------------------------------------------------------- data
def load_normalized(chain: Path):
    from src.data_loaders import load_snapshot, normalize_listed

    raw, meta, rates = load_snapshot(chain)
    df, rep = normalize_listed(raw, meta, rates)
    return df, rep, meta


def select_contracts(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Same contracts for all models: complete event coverage, not schedule-uncertain, 9 moneyness targets."""
    info = {"normalized": len(df)}
    d = df[df.events_complete & ~df.schedule_uncertain].copy()
    info["after_event_coverage_filter"] = len(d)
    out = []
    for exp, g in d.groupby("expiry_ts_utc"):
        k = np.log(g.strike / g.forward)
        atm = g.iloc[np.argmin(np.abs(k.to_numpy()))].iv_mid
        T = g["T"].iloc[0]
        picked = set()
        for z in Z_TARGETS:
            target = z * atm * np.sqrt(T)
            i = int(np.argmin(np.abs(k.to_numpy() - target)))
            if abs(k.iloc[i] - target) <= 0.25 * atm * np.sqrt(T):  # only if a strike is actually near the target
                picked.add(g.index[i])
        out.append(g.loc[sorted(picked)])
    sel = pd.concat(out)
    info["selected"] = len(sel)
    info["expiries"] = int(sel.expiry_ts_utc.nunique())
    return sel, info


def build_panel(sel: pd.DataFrame):
    from src.calibration import CalibrationData
    from src.events import EventCalendar
    from src.metrics import ContractSet
    from src.pooled import CatalogEvent, Panel, PanelDate

    cal = EventCalendar.from_csv()
    v = sel.valuation_ts_utc.iloc[0]
    horizon = sel.expiry_ts_utc.max()
    view, _ = cal.view(v, "as_known")
    view = view[(view.timestamp_utc > v) & (view.timestamp_utc <= horizon)]
    catalog = tuple(CatalogEvent(i, k, t) for i, k, t in zip(view.event_id, view.event_type, view.timestamp_utc))
    events = tuple(e for e in cal.scheduled_events(v, vintage="as_known") if e.event_id in {c.event_id for c in catalog})
    c = ContractSet.from_arrays(sel.forward, sel.strike, sel["T"], sel.df_dom, sel.is_call)
    labels = tuple(sel.expiry_ts_utc.dt.strftime("%Y-%m-%d"))
    data = CalibrationData(c, sel.iv_mid.to_numpy(), sel.price_mid.to_numpy(), events,
                           sel.iv_bid.to_numpy(), sel.iv_ask.to_numpy(), labels)
    return Panel((PanelDate(v, data, str(v.date())),), catalog)


# --------------------------------------------------------------------------- fitting
def fit_all(panel, tag: str):
    from src.pooled import DEFAULT_LOCAL, calibrate_pooled_nested

    log = OUT / f"progress_{tag}.log"

    def progress(msg):
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

    return calibrate_pooled_nested(panel, local=DEFAULT_LOCAL, smooth=SMOOTH, threads=4,
                                   timeout_s=TIMEOUT_S, progress=progress)


def contract_results(nc, panel) -> pd.DataFrame:
    """Per contract: market vs each model's IV and price, plus tags."""
    from src.calibration import Objective, evaluate
    from src.pooled import build_date_model

    pdt = panel.dates[0]
    d = pdt.data
    c = d.contracts
    rows = pd.DataFrame({"expiry": d.labels, "T_days": 365 * c.T, "K": c.K, "F": c.F, "is_call": c.is_call,
                         "k": np.log(c.K / c.F), "iv_mkt": d.iv, "iv_bid": d.iv_bid, "iv_ask": d.iv_ask,
                         "price_mkt": d.price})
    spans = d.spanned_events
    mapping = nc.hbmj.mapping
    rows["event_types"] = [";".join(e.kind for e in s) for s in spans]
    rows["n_events"] = [len(s) for s in spans]
    rows["overlapping"] = [any(mapping.get(e.event_id, "x") in ("",) or str(mapping.get(e.event_id, "")).startswith("combo_")
                               for e in s) for s in spans]
    for name, f in nc.best().items():
        m = build_date_model(f.spec, f.shared, f.local[0], d.events, f.mapping)
        ev = evaluate(m, d, Objective("iv", "equal"))
        rows[f"iv_{name}"], rows[f"price_{name}"] = ev.iv_model, ev.price_model
    return rows


SUBSETS = {
    "full": lambda r: np.ones(len(r), bool),
    "event_spanning": lambda r: r.n_events > 0,
    "non_event": lambda r: r.n_events == 0,
    "FOMC": lambda r: r.event_types.str.contains("FOMC"),
    "ECB": lambda r: r.event_types.str.contains("ECB"),
    "CPI": lambda r: r.event_types.str.contains("CPI"),
    "NFP": lambda r: r.event_types.str.contains("NFP"),
    "overlapping_events": lambda r: r.overlapping,
    f"short (<= {SHORT_DAYS}d)": lambda r: r.T_days <= SHORT_DAYS,
    f"long (> {SHORT_DAYS}d)": lambda r: r.T_days > SHORT_DAYS,
}


def breakdown(res: pd.DataFrame, models=("Heston", "Bates", "HB-MJ")) -> pd.DataFrame:
    out = []
    for sub, fn in SUBSETS.items():
        m = np.asarray(fn(res), bool)
        if not m.any():
            continue
        for name in models:
            e = 100 * (res[f"iv_{name}"][m] - res.iv_mkt[m])
            out.append({"subset": sub, "model": name, "n": int(m.sum()), "iv_rmse_volpts": float(np.sqrt(np.mean(e**2))),
                        "iv_mae_volpts": float(np.mean(np.abs(e)))})
    return pd.DataFrame(out)


def model_table(nc, panel, res) -> pd.DataFrame:
    from src.pooled import pooled_identifiability, pooled_metrics

    rows = []
    for name, f in nc.best().items():
        m = pooled_metrics(f, panel)
        e = 100 * (res[f"iv_{name}"] - res.iv_mkt)
        warn = []
        try:
            idf = pooled_identifiability(f)
            for k, se in idf["se"].items():
                if se > abs(f.shared[k]):
                    warn.append(f"{k}: SE {se:.3g} > |value| {abs(f.shared[k]):.3g}")
            corr = idf["corr"]
            for i, a in enumerate(corr.index):
                for b in corr.columns[i + 1:]:
                    if abs(corr.loc[a, b]) > 0.95:
                        warn.append(f"{a}~{b} corr {corr.loc[a, b]:+.2f}")
            cond = idf["condition_number"]
        except Exception as exc:
            cond, warn = np.nan, [f"diagnostics unavailable: {exc!r}"]
        rows.append({"model": name, "n": m["n"], "k": m["k"], "iv_rmse_volpts": float(np.sqrt(np.mean(e**2))),
                     "iv_mae_volpts": float(np.mean(np.abs(e))),
                     "price_rmse_$": float(np.sqrt(np.mean((res[f"price_{name}"] - res.price_mkt) ** 2))),
                     "weighted_obj": m["weighted_obj"], "aic": m["aic"], "bic": m["bic"],
                     "converged": f.success, "status": f.status, "message": f.message[:50],
                     "bound_hits": ";".join(f.bound_hits) or "none", "condition_number": cond,
                     "identifiability_warnings": " | ".join(warn) or "none",
                     **{f"p_{k}": v for k, v in f.shared.items()}, **{f"p_{k}": v for k, v in f.local[0].items()}})
    return pd.DataFrame(rows)


def stable_quantities(nc) -> pd.DataFrame:
    rows = []
    for name, f in nc.best().items():
        x = {**f.shared, **f.local[0]}
        q = {"model": name, "short_dated_variance_v0": x["v0"], "long_run_variance_theta": x["theta"]}
        if "lam" in x:
            q["jump_variance_rate"] = x["lam"] * (x["mu_J"] ** 2 + x["sigma_J"] ** 2)
            q["v0_plus_jump_variance"] = x["v0"] + q["jump_variance_rate"]
        for k in f.spec.event_keys:
            q[f"event_variance[{k[6:] if k.startswith('sigma_') else k}]"] = x[k] ** 2
        rows.append(q)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- event steps
def event_steps(nc, panel, res) -> pd.DataFrame:
    """Model-consistent event-step test for each event cluster bracketed by two expiries.

    Delta w = ATM total implied variance (IV^2 T at the contract nearest the forward) of the first
    expiry after the cluster minus the last expiry before it -- computed identically for the market
    and for each fitted model on the SAME contracts. For HB-MJ the event contribution is isolated as
    Delta w(HB-MJ) - Delta w(HB-MJ with that cluster's event variance set to 0); this is the
    model-consistent measure (it is not assumed to equal sigma_E^2 exactly).
    """
    from src.calibration import Objective, evaluate
    from src.pooled import build_date_model, event_clusters

    d = panel.dates[0].data
    c = d.contracts
    Ts = np.unique(c.T)
    atm_idx = {t: np.where(c.T == t)[0][np.argmin(np.abs(np.log(c.K[c.T == t] / c.F[c.T == t])))] for t in Ts}
    f = nc.hbmj
    out = []
    v = panel.dates[0].valuation_ts
    for cl in event_clusters(panel):
        t_first = (cl[0].ts - v).total_seconds() / (365 * 86400)
        t_last = (cl[-1].ts - v).total_seconds() / (365 * 86400)
        before, after = Ts[Ts < t_first], Ts[Ts >= t_last]
        if not len(before) or not len(after):
            continue
        tb, ta = before.max(), after.min()
        other = [e for e in panel.catalog if t_first > (e.ts - v).total_seconds() / (365 * 86400) > tb
                 or ta >= (e.ts - v).total_seconds() / (365 * 86400) > t_last]
        ib, ia = atm_idx[tb], atm_idx[ta]
        w = lambda iv: iv[ia] ** 2 * ta - iv[ib] ** 2 * tb
        row = {"cluster": "+".join(e.event_id for e in cl), "key": f.mapping.get(cl[0].event_id, ""),
               "expiry_before": d.labels[ib], "expiry_after": d.labels[ia], "gap_days": 365 * (ta - tb),
               "other_events_in_gap": ";".join(e.event_id for e in other), "step_market": w(d.iv)}
        for name in ("Heston", "Bates", "HB-MJ"):
            row[f"step_{name}"] = w(res[f"iv_{name}"].to_numpy())
        key = row["key"]
        if key:
            no_ev = dict(f.shared)
            no_ev[key] = 0.0
            m0 = build_date_model(f.spec, no_ev, f.local[0], d.events, f.mapping)
            iv0 = evaluate(m0, d, Objective("iv", "equal")).iv_model
            row["hbmj_event_contribution"] = row["step_HB-MJ"] - w(iv0)
            row["hbmj_event_variance"] = f.shared[key] ** 2
        row["excess_market_step_over_bates"] = row["step_market"] - row["step_Bates"]
        out.append(row)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- checkpointing
CKPT_DIR = OUT / "fit_results"          # one append-only JSONL file per fold: <fold_id>.jsonl
MODELS = ("Heston", "Bates", "HB-MJ")
SEED = 0                                 # the pooled fits use fixed deterministic starts (no RNG)
DONE_STATUSES = ("converged", "timed_out", "migrated")  # finished work: never re-optimised on resume


def _fold_ids(expiries) -> list[str]:
    return ["full_sample"] + [f"loeo_{e}" for e in expiries]


def read_checkpoints() -> dict[tuple[str, str], dict]:
    """(fold_id, model) -> last saved row. Corrupt trailing lines (e.g. a killed writer) are skipped."""
    out = {}
    if not CKPT_DIR.exists():
        return out
    for f in sorted(CKPT_DIR.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[(row["fold_id"], row["model"])] = row
    return out


def append_checkpoint(row: dict) -> None:
    """Append one row and fsync, so a stopped run never loses a finished fit."""
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CKPT_DIR / f"{row['fold_id']}.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=float) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _status(f) -> str:
    return "timed_out" if f.status == -99 else ("converged" if f.success else "not_converged")


def _row(fold_id, name, f, panel, test=None) -> dict:
    from src.calibration import Objective, evaluate
    from src.pooled import build_date_model, pooled_metrics

    m = pooled_metrics(f, panel)
    row = {"fold_id": fold_id, "model": name, "seed": SEED, "status": _status(f), "status_code": f.status,
           "success": f.success, "message": f.message, "objective": f.cost, "iv_rmse_volpts": 100 * f.iv_rmse,
           "n": m["n"], "k": m["k"], "aic": m["aic"], "bic": m["bic"], "bound_hits": list(f.bound_hits),
           "seconds": f.seconds, "params_shared": f.shared, "params_local": f.local, "source": "run",
           "written_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    if test is not None:
        mdl = build_date_model(f.spec, f.shared, f.local[0], test.events, f.mapping)
        e = evaluate(mdl, test, Objective("iv", "equal")).residuals
        row.update({"oos_n": len(e), "oos_iv_rmse_volpts": 100 * float(np.sqrt(np.mean(e**2))),
                    "spans_event": bool(test.event_spanning.any()),
                    "event_types": ";".join(sorted({x.kind for s in test.spanned_events for x in s}))})
    return row


def load_selection() -> pd.DataFrame:
    """Reuse the saved contract selection (no re-normalisation on resume)."""
    sel = pd.read_csv(OUT / "selected_contracts.csv")
    for c in ("valuation_ts_utc", "expiry_ts_utc"):
        sel[c] = pd.to_datetime(sel[c], utc=True)
    for c in ("is_call", "events_complete", "schedule_uncertain", "is_dev_fallback"):
        sel[c] = sel[c].astype(str).str.lower().eq("true")
    return sel


def run_fold(fold_id: str) -> dict:
    """Fit Heston -> Bates -> HB-MJ for one fold, checkpointing after EACH model; skip finished stages."""
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    from src.pooled import Panel, PanelDate, calibrate_pooled_nested, DEFAULT_LOCAL

    t0 = time.perf_counter()
    panel = build_panel(load_selection())
    test = None
    if fold_id != "full_sample":
        held = fold_id[len("loeo_"):]
        pdt = panel.dates[0]
        lab = np.asarray(pdt.data.labels)
        test = pdt.data.subset(lab == held)
        panel = Panel((PanelDate(pdt.valuation_ts, pdt.data.subset(lab != held), pdt.label),), panel.catalog)
    ck = read_checkpoints()
    saved = {m: ck[(fold_id, m)] for m in MODELS if (fold_id, m) in ck and ck[(fold_id, m)]["status"] in DONE_STATUSES}
    if len(saved) == len(MODELS):
        return {"fold_id": fold_id, "skipped": True, "seconds": 0.0}
    if any(r.get("params_shared") is None for r in saved.values()):  # migrated rows without parameters
        return {"fold_id": fold_id, "skipped": True, "seconds": 0.0, "note": "completed earlier; parameters unavailable"}
    done = {m: {"shared": r["params_shared"], "local": r["params_local"], "status": r["status_code"],
                "success": r["success"], "message": r["message"]} for m, r in saved.items()}
    log = OUT / f"progress_{fold_id}.log"

    def progress(msg):
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

    calibrate_pooled_nested(panel, local=DEFAULT_LOCAL, smooth=SMOOTH, threads=4, timeout_s=TIMEOUT_S,
                            progress=progress, done=done,
                            on_stage=lambda name, f: append_checkpoint(_row(fold_id, name, f, panel, test)))
    return {"fold_id": fold_id, "skipped": False, "seconds": time.perf_counter() - t0}


def write_progress_summary(total_folds: int, t_start: float, fold_times: list[float], workers: int) -> str:
    ck = read_checkpoints()
    done_fits = sum(1 for r in ck.values() if r["status"] in DONE_STATUSES)
    total_fits = total_folds * len(MODELS)
    folds_done = len({f for (f, _m), r in ck.items()
                      if all((f, m) in ck and ck[(f, m)]["status"] in DONE_STATUSES for m in MODELS)})
    remaining = total_folds - folds_done
    avg = np.mean(fold_times) if fold_times else np.nan
    eta = avg * np.ceil(remaining / workers) if fold_times else np.nan
    fmt = lambda x: "n/a" if not np.isfinite(x) else f"{x:.0f}s"
    txt = (f"{time.strftime('%Y-%m-%d %H:%M:%S')}  model fits done {done_fits}/{total_fits}  "
           f"folds done {folds_done}/{total_folds}  elapsed {time.perf_counter() - t_start:.0f}s  "
           f"avg fold {fmt(avg)}  est. remaining {'0s' if remaining == 0 else fmt(eta)}")
    with open(OUT / "progress_summary.txt", "a", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    return txt


def migrate_completed_run() -> int:
    """Convert outputs of the run that finished before checkpointing existed into checkpoint rows.

    Full sample: parameters from model_table.csv. Leave-one-expiry-out: only held-out metrics survived
    (parameters were never saved) -> params_shared=None, source='migrated'. Idempotent."""
    ck = read_checkpoints()
    n = 0
    if (OUT / "model_table.csv").exists():
        mt = pd.read_csv(OUT / "model_table.csv")
        for _, r in mt.iterrows():
            if ("full_sample", r.model) in ck:
                continue
            p = {c[2:]: float(r[c]) for c in mt.columns if c.startswith("p_") and pd.notna(r[c])}
            local = {k: p.pop(k) for k in ("v0", "theta", "rho")}
            append_checkpoint({"fold_id": "full_sample", "model": r.model, "seed": SEED,
                               "status": "converged" if r.converged else "not_converged", "status_code": int(r.status),
                               "success": bool(r.converged), "message": r.message, "objective": float(r.weighted_obj),
                               "iv_rmse_volpts": float(r.iv_rmse_volpts), "n": int(r.n), "k": int(r.k),
                               "aic": float(r.aic), "bic": float(r.bic), "bound_hits": [] if r.bound_hits == "none" else [r.bound_hits],
                               "seconds": None, "params_shared": p, "params_local": [local], "source": "migrated",
                               "written_utc": pd.Timestamp.now(tz="UTC").isoformat()})
            n += 1
    if (OUT / "oos_leave_one_expiry.csv").exists():
        oo = pd.read_csv(OUT / "oos_leave_one_expiry.csv", keep_default_na=False)
        for _, r in oo.iterrows():
            fid = f"loeo_{r.held_out_expiry}"
            if (fid, r.model) in ck:
                continue
            append_checkpoint({"fold_id": fid, "model": r.model, "seed": SEED, "status": "migrated",
                               "status_code": None, "success": str(r.train_converged) == "True", "message": "migrated",
                               "objective": None, "iv_rmse_volpts": None, "n": None, "k": None, "aic": None, "bic": None,
                               "bound_hits": None, "seconds": None, "params_shared": None, "params_local": None,
                               "oos_n": int(r.n), "oos_iv_rmse_volpts": float(r.oos_iv_rmse_volpts),
                               "spans_event": str(r.spans_event) == "True", "event_types": r.event_types,
                               "source": "migrated", "written_utc": pd.Timestamp.now(tz="UTC").isoformat()})
            n += 1
    return n


def export_oos_table() -> None:
    rows = [r for (f, _m), r in read_checkpoints().items() if f.startswith("loeo_")]
    if rows:
        pd.DataFrame([{"held_out_expiry": r["fold_id"][5:], "model": r["model"], "n": r.get("oos_n"),
                       "spans_event": r.get("spans_event"), "event_types": r.get("event_types"),
                       "oos_iv_rmse_volpts": r.get("oos_iv_rmse_volpts"), "status": r["status"],
                       "train_converged": r["success"], "source": r["source"]} for r in rows]
                     ).sort_values(["held_out_expiry", "model"]).to_csv(OUT / "oos_from_checkpoints.csv", index=False)  # never overwrites study outputs


def full_sample_outputs() -> None:
    """Contract-level results, breakdowns, event steps from the full-sample checkpoint (restored fits)."""
    from src.pooled import calibrate_pooled_nested

    ck = read_checkpoints()
    done = {m: {"shared": ck[("full_sample", m)]["params_shared"], "local": ck[("full_sample", m)]["params_local"],
                "status": ck[("full_sample", m)]["status_code"], "success": ck[("full_sample", m)]["success"],
                "message": ck[("full_sample", m)]["message"]} for m in MODELS}
    panel = build_panel(load_selection())
    nc = calibrate_pooled_nested(panel, local=("v0", "theta", "rho"), smooth=SMOOTH, done=done)
    res = contract_results(nc, panel)
    res.to_csv(OUT / "contract_results.csv", index=False)
    breakdown(res).to_csv(OUT / "breakdown.csv", index=False)
    stable_quantities(nc).to_csv(OUT / "stable_quantities.csv", index=False)
    event_steps(nc, panel, res).to_csv(OUT / "event_steps.csv", index=False)


# --------------------------------------------------------------------------- main
def prepare_data() -> None:
    """Normalise + select once; later launches reuse results/proxy/selected_contracts.csv."""
    from src.validation import run_validation

    fxe, fxe_rep, fxe_meta = load_normalized(FXE_SNAP)
    json.dump({"report": fxe_rep, "meta": fxe_meta, "kept_rows": len(fxe),
               "expiries_kept": sorted(map(str, fxe.expiry_ts_utc.dt.date.unique()))}, open(OUT / "fxe_quality.json", "w"),
              indent=2, default=str)
    df, rep, meta = load_normalized(SNAP)
    sel, info = select_contracts(df)
    json.dump({"normalize_report": rep, "selection": info, "meta": meta}, open(OUT / "spy_dataset.json", "w"), indent=2, default=str)
    run_validation(df).to_csv(OUT / "validation_all_normalized.csv", index=False)
    run_validation(sel).to_csv(OUT / "validation_selected.csv", index=False)
    sel.to_csv(OUT / "selected_contracts.csv", index=False)


def main(workers: int = 4):
    from concurrent.futures import as_completed

    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / "selected_contracts.csv").exists():
        prepare_data()
    migrated = migrate_completed_run()
    if migrated:
        print(f"migrated {migrated} rows from the previously completed run into {CKPT_DIR}", flush=True)
    expiries = sorted(load_selection().expiry_ts_utc.dt.strftime("%Y-%m-%d").unique())
    folds = _fold_ids(expiries)
    ck = read_checkpoints()
    todo = [f for f in folds if not all((f, m) in ck and ck[(f, m)]["status"] in DONE_STATUSES for m in MODELS)]
    t_start, times = time.perf_counter(), []
    print(write_progress_summary(len(folds), t_start, times, workers), flush=True)
    if todo:
        with ProcessPoolExecutor(max_workers=min(workers, len(todo))) as pool:  # workers x 4 threads, BLAS capped at 1
            futs = {pool.submit(run_fold, f): f for f in todo}
            for fut in as_completed(futs):
                r = fut.result()
                if not r["skipped"]:
                    times.append(r["seconds"])
                print(write_progress_summary(len(folds), t_start, times, workers), flush=True)
    export_oos_table()
    if not (OUT / "contract_results.csv").exists():
        full_sample_outputs()
    print("all folds complete", flush=True)


if __name__ == "__main__":
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    main()
