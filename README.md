# HB-MJ: a Heston–Bates model with scheduled macro-event jumps for currency options

**Status: prototype complete.** The mathematics is implemented and tested, validated in simulation,
and tested on public proxy data. Validation on Bloomberg OTC EUR/USD data is future work; nothing
in this repository requires Bloomberg to run or to reproduce the reported results.
Numerical results: **[RESULTS.md](RESULTS.md)**. Final figures: `results/figures/final/`.

## Research question

Currency moves are driven by *surprises* in rates, inflation and activity, and much of that
information arrives on known dates: FOMC and ECB decisions, US CPI and Nonfarm Payrolls. Do
option prices concentrate risk-neutral variance on those dates, and does a model that places
jumps at those scheduled times price short-dated options better, after penalising its extra
parameters and out of sample, than the Heston (1993) and Bates (1996) models it extends?

The hypothesis is about the **maturity term structure**. Total implied variance w(T) = σ²·T
should step up as expiry crosses an event. It does not require extra smile curvature (see
"What is original").

## Models: a nested sequence

Every model is a characteristic function (CF) of X_T = ln(S_T/F) plugged into one pricing
engine. Each richer model multiplies the previous CF by one independent, compensated factor
(φ(−i) = 1), so switching a factor off recovers the smaller model exactly (`src/models/nested.py`).

| Model | Adds | Parameters |
|---|---|---|
| **Garman–Kohlhagen (1983)** — baseline | constant vol (Black-76 on the forward) | σ |
| **Heston (1993)** | mean-reverting stochastic variance correlated with spot | κ, θ, σ, ρ, v₀ |
| **Bates (1996)** | Poisson jumps with lognormal size, compensated | λ, μ_J, σ_J |
| **HB-MJ** (this project) | a jump at each **scheduled** event time τᵢ before expiry, with **stochastic size** Yᵢ ~ N(−½σ²_E, σ²_E), one σ_E per event type | σ_FOMC, σ_ECB, σ_CPI, σ_NFP |

φ_HB-MJ(u;T) = φ_Heston(u;T) · φ_Poisson(u;T) · ∏_{0<τᵢ≤T} exp(−½σ²_{E,i}·u(u+i)).

Event *times* are known in advance; event *sizes* are random and only their distribution is known.
The mean −½σ²_E makes E[e^Y] = 1, so each event leaves the forward unchanged.

## What is original about HB-MJ, and what is not

The scheduled-jump idea follows Dubinsky, Johannes, Kaeck & Seeger (2019, RFS) on equity
earnings announcements, and Piazzesi (2005) on FOMC dates in bond yields. FX desks already add
"event variance" to short-dated vol term structures. The contribution here is:

- a formal nested **Heston–Bates model with scheduled-time, stochastic-size jumps** for FX macro
  events, with one exact rule (0 < τ ≤ T) shared by pricing and data tagging;
- a test design that separates event variance from diffusion variance, including pooled
  multi-date estimation and handling of events that no contract can tell apart;
- an honest identifiability analysis.

A property shown in the tests: with normal event jumps, a single European expiry sees each event as
an independent normal shock. That adds variance but **no excess kurtosis**, so for European pricing the
component behaves like a burst of variance on the event date. The evidence therefore lives in the
term structure, and in hedging across events, not in smile shape.

## Pricing, simulation, calibration

- **Pricing** (`src/pricing_engine.py`): Lewis single-integral characteristic-function pricing with
  adaptive Gauss–Legendre panels (production); Carr–Madan FFT (independent cross-check; it refuses to
  silently coarsen its grid); Gil-Pelaez adaptive quadrature (reference). The Heston CF uses the
  "Little Heston Trap" form of Albrecher et al. (2007), with cancellation-free algebra.
- **Monte Carlo** (`src/monte_carlo.py`): full-truncation Euler for the variance, exact Poisson and
  event jumps, antithetic variates. Used for validation only.
- **Single-date calibration** (`src/calibration.py`): least squares on vega-weighted implied-vol errors
  (vega normalised within each expiry), parameters in transformed space, Heston → Bates → HB-MJ
  each started from the previous stage, multi-start, AIC/BIC, identifiability diagnostics.
- **Pooled calibration** (`src/pooled.py`): date-specific state (v₀, θ, ρ) with shared structural and
  event parameters. Events that no valuation time or expiry separates get one combined parameter.
  An optional random-walk smoothness penalty on the daily state (required if every parameter is
  date-specific). Per-fit timeouts are reported as "timed out", never "converged"; checkpointed and resumable.
- **Clocks** (`src/timeutils.py`, `src/clocks.py`): ACT/365 calendar time is the default and is used for
  FX. A NYSE **trading-day clock** is available for the equity proxy only: 1/252 year per trading day,
  accrued during the regular session, weekends/holidays adding nothing, configurable overnight
  share. Scheduled events always stay at their true timestamps.

## Data layer

- **Research source (future):** Bloomberg EUR/USD OTC surfaces (`load_bloomberg_csv`). Conventions
  (delta type, ATM type, smile vs market strangle, premium currency, RR sign, cut, timezone) must be
  stated in the export, and the loader refuses to guess. It covers premium discounting from the spot
  date to delivery and an FX date roller built on sourced USD (Federal Reserve K.8 / Fedwire) and
  EUR (TARGET2) calendars. Export and validation guide: `docs/bloomberg_export_checklist.md`,
  `docs/bloomberg_validation_table.csv`.
- **Proxy source (used here):** listed options from saved Yahoo Finance snapshots (`normalize_listed`),
  with every row `is_dev_fallback=True` and flagged `proxy_not_otc_eurusd`.
- **Event calendar:** official FOMC/ECB/CPI/NFP times in UTC (`data/events/`). It includes the
  2025/2026 BLS funding-lapse cancellations and reschedules, and an **as-known-at-valuation**
  vintage (`schedule_changes.csv`). Rows whose event date was unknown at the time are flagged
  `schedule_uncertain`.

## Evidence, kept separate

| Kind | Where | What it can support |
|---|---|---|
| Mathematical implementation | `tests/` (429 tests) | correctness of pricing, nesting, conventions, data handling |
| Simulation evidence | `results/calibration`, `results/pooled` | identifiability, estimator behaviour, what data design is needed |
| Proxy evidence (SPY, one snapshot) | `results/proxy`, `results/proxy_trading` | whether the pipeline works on real quotes; weak evidence about equity event variance |
| OTC EUR/USD claims | — | **none yet**, pending Bloomberg data |

## Reproducibility

```
pip install -r requirements.txt
python -m pytest                        # full test suite
python -m scripts.run_synthetic         # synthetic study (resumes; --force to redo)
python -m scripts.run_proxy             # proxy study from the committed snapshots (resumes)
python -m scripts.final_figures         # figures in results/figures/final/
```

- **Seeds** are fixed constants in each script: `SEED` (calibration_recovery 20260930;
  mc_validation 20260929; identifiability_study per-case seeds 1–4 and 7), `SEEDS = (1, 2, 3, 4)`
  (pooled_study, heston_noise_study). Pooled and proxy fits use deterministic starts (no RNG).
- **Resume:** the proxy study checkpoints every model fit (append-only JSONL under
  `results/proxy*/fit_results/`) and restores finished fits on relaunch. The synthetic runner skips
  steps whose outputs exist, and the pooled study rewrites its CSVs after every task.
- **Data:** the proxy snapshots are committed (`data/snapshots/spy_20260929T190930Z_*`,
  `fxe_20260929T190930Z_*`). New snapshots: `python -m scripts.snapshot_proxy SPY` (network needed).
- Environment used: Python 3.14, numpy 2.5, scipy 1.18, pandas 3.0, Windows 11.

## Repository layout

```
src/            pricing engine, models/, calibration, pooled, clocks, events, data loaders, validation
scripts/        studies, figure scripts, snapshot tools
tests/          pytest suite
data/           event calendar, settlement/NYSE calendars, example & proxy snapshots
docs/           Bloomberg export checklist and validation table
results/        study outputs (csv/json/logs) and figures
```

## Limitations and future Bloomberg validation

See RESULTS.md §8. In short: the proxy data are one SPY snapshot, not a panel, and not FX; FXE failed
the quality filters. Individual jump parameters (λ, μ_J, σ_J) and κ are weakly identified, so stable
combinations are reported instead. Nearby events are identified only jointly unless an expiry or a
valuation time separates them. The normal event-jump specification adds no kurtosis by construction.
Bloomberg OTC EUR/USD data (a daily panel of ON–1M quotes with bid/ask and exact dates) is the planned
real-market validation. `docs/bloomberg_export_checklist.md` states exactly what to export.

**Model freeze:** Heston, Bates and the current HB-MJ specification are the final prototype models.
No further jump distributions, mixtures or parameters are planned for this prototype.
