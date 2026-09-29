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

## Literature, data, calibration, results, market context, limitations

*TBD.*

## Next steps (not in the baseline model)

- Fatter-tailed event jumps (e.g. two-point hawkish/dovish mixture) or variance-dependent event
  jump sizes, so events can also affect smile shape.
- Surprise-size-dependent jumps; double Heston; rough volatility.
