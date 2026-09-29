# HB-MJ: A Heston–Bates Model with Scheduled Macro-Event Jumps for Currency Options

> **Status:** model mathematics implemented and validated (Garman–Kohlhagen, Heston, Bates,
> HB-MJ; characteristic-function integration, Carr–Madan FFT, Monte Carlo cross-checks).
> Market-data calibration and empirical results are not yet done. This README will grow into
> the full write-up; sections marked *TBD* are placeholders.

## Model nesting

All models share one pricing engine and differ only in their characteristic function (CF) of
the log-return to expiry relative to the forward, X_T = ln(S_T / F):

| Model | CF | Parameters |
|---|---|---|
| Garman–Kohlhagen (1983) | normal | vol |
| Heston (1993) | φ_H | κ, θ, σ, ρ, v₀ |
| Bates (1996) | φ_H · φ_J | + λ, μ_J, σ_J |
| **HB-MJ** (this project) | φ_H · φ_J · φ_E | + σ_FOMC, σ_ECB, σ_CPI, σ_NFP |

Each factor is independently compensated (φ(−i) = 1), so switching a component off recovers the
smaller model exactly. Code: `src/models/nested.py`.

**HB-MJ event component.** Each scheduled announcement *i* (FOMC, ECB, US CPI, NFP) happens at
a known time τᵢ, but the move it causes is random: a log-jump Yᵢ ~ N(−½σ_E², σ_E²) with one σ_E
per event type. Options expiring at T are affected by events with 0 < τᵢ ≤ T:

  φ_E(u; T) = ∏_{0 < τᵢ ≤ T} exp(−½ σ²_{E,i} · u(u + i))

## Research hypothesis

**Baseline HB-MJ hypothesis (primary specification: normal event jumps).**
Part of the risk-neutral variance priced into EUR/USD options is *concentrated on scheduled
macro-event dates* rather than spread evenly through time. This should show up in the
**maturity term structure**: total implied variance w(T) = σ_imp(T)² · T should rise in discrete
steps as expiry crosses an FOMC, ECB, CPI or NFP date, beyond what a smooth
Heston/Bates term structure produces.

What the hypothesis does **not** claim: that event jumps add smile curvature. With normal
event jumps, a single European expiry sees each event as an independent normal shock, which
adds variance but no excess kurtosis; on fixed contracts it *lowers* implied-vol curvature
(verified in `tests/test_hbmj.py`). For European pricing the event component is therefore
equivalent to a deterministic burst of Gaussian variance on the event date — close to the
"event-weighted variance" FX desks already use. The contribution tested here is a formal,
nested, statistically evaluated version of that idea inside Heston–Bates. Fatter-tailed event
distributions are deferred to *Next steps*.

## Main HB-MJ diagnostics (for the empirical stage)

1. **Term-structure step.** Change in total implied variance, Δw = w(T_after) − w(T_before),
   between expiries just before and just after each scheduled event, compared with the model's
   σ_E² and with the smooth Bates term structure.
2. **Event vs non-event errors.** Pricing/calibration error (IV RMSE, price MAE) for
   event-spanning options versus non-event-spanning options, always on identical contract
   coordinates for every model (`src/metrics.py`).
3. **Penalized out-of-sample gain over Bates.** Incremental out-of-sample fit of HB-MJ versus
   Bates after penalizing the extra event parameters (information criteria and held-out
   strikes/expiries/days).
4. **Event variance by type.** Estimated σ²_FOMC, σ²_ECB, σ²_CPI, σ²_NFP with uncertainty.
5. **Stability over time.** Whether those event-variance estimates are stable across
   calibration dates.

A negative result on any of these (e.g. no out-of-sample gain over Bates after penalties) is a
valid finding and will be reported as such.

## Validation (done)

**Analytic engines.** Heston matches Fang & Oosterlee (2008) benchmarks and QuantLib to ~1e-10;
Lewis integration, Gil-Pelaez quadrature and Carr–Madan FFT agree to ≤0.2 bp of implied vol
for every model from 1 day to 10 years. Integration is the calibration engine; the FFT is a
cross-check (it is 8–20× slower for sparse FX strike sets).

**Monte Carlo** (`src/monte_carlo.py`, `scripts/mc_validation.py`) is an independent engine
used only for validation: full-truncation Euler for Heston variance, explicit Poisson jumps,
explicit scheduled-event jumps on grid nodes at each event time, antithetic variates.
10 model configurations × 4 maturities (1W–1Y) × 5 fixed strikes, 200,000 paths, 730 steps/yr:

| | Contracts | \|z\| > 2 | \|z\| > 3 | Mean z |
|---|---|---|---|---|
| Heston, Bates (λ = 0.5, 2, 8, rare big jumps), HB-MJ (1 FOMC, event cluster, 1% FOMC, events after expiry) | 180 | 6 (3.3%) | 0 | −0.23 |
| Heston, high vol-of-vol (Feller ratio 0.02) | 20 | 11 | 3 | **+1.65** |

z = (MC − CF)/SE. For a correct CF and an unbiased simulator, z should look like N(0,1)
(≈4.6% beyond ±2); the first row does. The second row shows **Euler discretization bias**,
not a CF error: halving the step size removes it (1Y, ATM, bias in bp of forward):

| Steps per year | Heston (Feller 0.28) | Heston high vol-of-vol (Feller 0.02) |
|---|---|---|
| 12 | 10.8 (z = 15.9) | 86.9 (z = 107.7) |
| 52 | 2.2 (z = 3.3) | 21.5 (z = 31.5) |
| 365 | 0.09 (z = 0.1) | 2.3 (z = 3.7) |
| 1460 | 0.02 (z = 0.0) | 0.5 (z = 0.8) |

Bias shrinks roughly in proportion to the step size and is worst when variance often hits 0
(Feller badly violated). The Monte Carlo used for hedging experiments must therefore use
fine steps (≥ 4 per day) when calibrated parameters strongly violate Feller.

## Data layer (built; no calibration yet)

Two tracks, one normalized schema (`src/schema.py`); calibration code reads only the schema.

| Track | Loader | Role |
|---|---|---|
| Bloomberg EUR/USD OTC surface (OVDV/OVML exports, ATM/RR/BF by tenor) | `load_bloomberg_csv` | **Research dataset** |
| FXE ETF options (yfinance) + FRED Treasury yields | `fetch_fxe_snapshot`, `normalize_fxe` | Development fallback only (`is_dev_fallback=True`) |

- Bloomberg conventions (delta type, ATM type, smile vs market butterfly, premium currency,
  RR sign, vol units, timezone, expiry cut) must be stated in the file; the loader raises
  `ConventionAmbiguityError` rather than guessing. Export guide: `docs/bloomberg_export_checklist.md`.
- FXE differs from OTC EUR/USD options: American exercise, ETF (fees, share-price units),
  listed monthly/quarterly expiries only (no 1W–3W), thin liquidity, delayed/stale Yahoo quotes,
  and a forward inferred from put-call parity (biased by early exercise). It is used for
  pipeline development, never for research conclusions.
- Event calendar (`data/events/macro_events.csv`): official FOMC, ECB, CPI and NFP times,
  stored in local time and converted to UTC; includes 2025/2026 appropriations-lapse
  cancellations and reschedules. One inclusion rule (`src/events.py`) serves both pricing and
  tagging.
- **Time conventions.** Diffusion clock: T = ACT/365F calendar time from valuation to the expiry
  cut (weekends carry diffusion variance). Event clock: each scheduled event adds σ_E² once, at
  its timestamp, only if it falls inside the option's life; weekends add no event variance.
- **Settlement.** Bloomberg premiums are priced as DF_USD(spot date → delivery) × Black-76 on the
  forward to the delivery date, with T to the expiry cut. Spot/expiry/delivery dates come from
  the export; the FX date roller (`src/fx_calendar.py`) fills them only when explicitly allowed,
  after validation against OVML. Settlement calendars are explicit sourced tables, 2025–2030:
  USD = days the Federal Reserve Banks (Fedwire Funds) are closed, per Federal Reserve Board K.8
  (Saturday holidays: Banks open the preceding Friday, e.g. 3 Jul 2026); EUR = TARGET (T2) closing
  days per the ECB. Where holidays make the month-tenor delivery→expiry inverse impossible
  (e.g. 1M from 26 Oct 2026 across Thanksgiving), the roller returns *ambiguous* with candidates
  rather than choosing one. Roller-vs-roller agreement on synthetic data is not an external validation.
- **Event vintages.** Pricing and tagging use the schedule *as known at the valuation time*
  (`data/events/schedule_changes.csv` records when each reschedule/cancellation was announced);
  options whose event dates were genuinely unknown at valuation (e.g. during the 2025 and 2026
  funding lapses) are flagged `schedule_uncertain`. The realized calendar is kept for ex-post
  analysis.
- Pre-calibration checks (`src/validation.py`): forward consistency, price/vol consistency,
  Bloomberg smile reconstruction, delta round trip, price bounds, bid/ask, butterfly and
  calendar arbitrage, timestamp consistency, event tags, schema, audit trail.

## Calibration framework (synthetic validation only; no market data fitted yet)

**Single-date engine** (`src/calibration.py`): least squares on implied-vol errors (vega weights
normalised per expiry, so short-dated contracts keep their weight), parameters optimised in
transformed space (log / atanh / scaled linear), Heston → Bates → HB-MJ, each stage started
from the previous one, multi-start, identifiability diagnostics. Synthetic findings
(`results/calibration/`): Heston parameters recover to a few percent (κ weakest); Bates' λ, μ_J,
σ_J are individually unstable but the jump variance rate and short-dated total variance are
stable; event variance is identified only when expiries bracket the event; adjacent events with
no separating expiry identify only their combined variance; with no pre-event expiry a
single-date fit can land in a false optimum where diffusion variance absorbs event variance.
Bates fitted to HB-MJ data matches the average vol level but cannot produce the discrete
event-time step (`results/figures/bates_vs_hbmj_event_step.png`).

**Pooled multi-date engine** (`src/pooled.py`): Heston/Bates state parameters are date-specific
(default v0, θ, ρ; κ, σ and jump parameters shared), event variances are shared by event type
across dates. Events that no contract in the panel can separate — no valuation time and no
expiry between them — get one combined parameter (e.g. `combo_CPI+FOMC`). A 16:00 New York
valuation between an FOMC (14:00 ET) and a next-day ECB separates them. Sharing by type is an
assumption. One joint sparse least-squares problem; per-fit timeouts report "timed out", never
"converged".

Synthetic daily panels (19 Oct – 30 Nov 2026, ON/1W/2W/3W/1M, real event calendar, 0.05 vol-pt
noise, 4 seeds; `results/pooled/`):

| | Pooled HB-MJ | Single-date HB-MJ, same dates |
|---|---|---|
| Event-variance error (FOMC, ECB, CPI, NFP, CPI+FOMC combo) | ≤ 2.3% in every seed; exact noise-free | median FOMC −26%, p10–p90 of CPI/NFP up to +65–70% (nearby events only identified as sums on a single date) |
| No ON tenor (single-date D3/D4 false-optimum case) | noise-free: exact; 0.05 vol pt: FOMC/ECB ±11–13%, θ poorly identified | 7% of dates in false optima |
| Out of sample (leave 2W out) | HB-MJ 0.050 vol pts vs Bates 0.381, Heston 0.379 | — |

**Daily-parameter specification.** Making every Heston/Bates parameter date-specific
(8 per date) is weakly identified and numerically ill-conditioned: each date's κ and jump
parameters rest on ~25 quotes, and the unregularised problem has near-flat directions.
A temporal smoothness penalty (random-walk prior on the transformed daily parameters,
`smooth=1e-4`) regularises the daily paths and makes pooled estimation practical; the default
specification keeps only v0, θ, ρ date-specific. Diagnostic
(`results/pooled/p5_all_local_diagnostic.csv`): without smoothing, the Heston and Bates stages
took 8–9 minutes each; two Bates starts ended in different local optima (objectives 5% apart);
and the HB-MJ stage had not converged when stopped (objective 23× the smoothed fit's). The
smoothed all-local HB-MJ fit is already at the noise floor (IV RMSE 0.045 vol pts against 0.05
vol-pt quote noise), so there is no material pricing gain left for the unsmoothed specification.
Its lower Heston/Bates objectives (4–15%) come from daily parameters absorbing event variance
that those models cannot represent — overfitting, not better pricing.

## Literature, results, market context, limitations

*TBD.*

## Next steps (not in the baseline model)

- Fatter-tailed event jumps (e.g. two-point hawkish/dovish mixture) or variance-dependent event
  jump sizes, so events can also affect smile shape.
- Surprise-size-dependent jumps; double Heston; rough volatility.
