# HB-MJ prototype: results summary

Evidence is labelled throughout: **[math]** implementation tests, **[sim]** simulation, **[proxy]**
public SPY option data (not OTC EUR/USD). No result here is a claim about the OTC EUR/USD market.
Figures: `results/figures/final/fig1…fig9`. Raw outputs: `results/`.

## 1. Synthetic recovery [sim]

Single-date calibration of surfaces generated from known parameters (`results/calibration/`).
Noise-free: every model recovers its true parameters from several different starts, to ~1e-10.
Mean error across 4 seeds at 0.05 / 0.10 vol-point IV noise:

| Quantity | Heston | Bates | HB-MJ (events spread out) |
|---|---|---|---|
| v₀, θ, σ, ρ | 0.4–2% | 1–5% | 1–7% |
| κ | 4% / 7% (≤14%) | 5–14% | 14% / 27% (≤55%) |
| λ, μ_J, σ_J individually | — | 17–45% (≤74%) | 20–41% |
| jump variance rate λ(μ_J²+σ_J²) | — | 3–13% | 12–21% |
| short-dated total variance v₀ + jump variance | — | 0.3–0.8% | 0.6–1.2% |
| event variance σ²_E | — | — | FOMC 1–2%, ECB 4–7%, CPI 10–20%, NFP 21–40% |

Starting values: Heston starts agree to 0.01%. At 0.10 vol-pt noise, Bates starts that fit equally well
differ by 71% in λ but by only 11% in jump variance and 1% in short-dated variance.

## 2. Can Bates mimic scheduled events? [sim]

Truth = HB-MJ; fitted with Bates and HB-MJ (fig 4). Bates fits λ = 7.1 (true 1.0) and a jump
variance rate 2.9× the truth. It matches the average vol level at long maturities (IV error
0.1–0.15 vol pts at 2M–6M), but it **cannot produce the discrete event step**:

| Event | true σ²_E | Δw truth | Δw Bates | Δw HB-MJ |
|---|---|---|---|---|
| FOMC (day 3.3) | 3.6e-5 | 5.1e-5 | 1.8e-5 | 5.1e-5 |
| ECB (day 10.2) | 2.5e-5 | 4.0e-5 | 1.8e-5 | 4.0e-5 |
| CPI (day 15.3) | 1.6e-5 | 3.1e-5 | 1.7e-5 | 3.1e-5 |
| NFP (day 24.3) | 1.2e-5 | 2.7e-5 | 1.7e-5 | 2.7e-5 |

Across the FOMC (day 3.2 → 3.4) the truth jumps by 3.99e-5 and Bates moves by 3.6e-6. Bates IV error
is about 1 vol pt at 1–4 day expiries, including expiries that span no event.

## 3. Pooled multi-date calibration [sim]

Daily panels 19 Oct–30 Nov 2026 (ON/1W/2W/3W/1M from the FX roller, real event calendar, 0.05
vol-pt noise, 4 seeds; `results/pooled/`, fig 5):

| | Pooled HB-MJ | Single-date HB-MJ, same dates |
|---|---|---|
| event-variance error | ≤ 2.3% in every seed, every event key | median FOMC −26%; CPI/NFP 10th–90th percentile up to +65–70% |
| no ON tenor | exact without noise; ±11–13% (FOMC/ECB) with noise | 7% of dates in false optima |
| combined FOMC+ECB (no separating contract) | recovered within 1.1% | — |
| out of sample (leave 2W out) | HB-MJ 0.050 vs Bates 0.381, Heston 0.379 vol pts | — |

Making all 8 Heston/Bates parameters date-specific without smoothing is ill-conditioned (Bates
starts land in different optima; HB-MJ did not converge). A random-walk smoothness penalty
makes it practical.

## 4. Proxy study, calendar-time clock [proxy]

One SPY snapshot, 29 Sep 2026 15:09 ET: 117 contracts, 9 per expiry on 13 expiries (1 Oct–30 Nov).
Event parameters: σ_NFP, σ_CPI, and FOMC+ECB combined (no expiry between 28 and 29 Oct). FXE failed the
filters (3 of 72 quotes usable).

| | Heston | Bates | HB-MJ |
|---|---|---|---|
| IV RMSE / MAE (vol pts) | 0.913 / 0.680 | 0.739 / 0.575 | 0.716 / 0.537 |
| price RMSE ($) | 0.221 | 0.196 | 0.176 |
| AIC / BIC | −1119 / −1105 | −1134 / **−1111** | **−1139** / −1109 |
| event-spanning / non-event IV RMSE | 0.807 / 1.743 | 0.678 / **1.259** | **0.614** / 1.466 |
| OOS event-spanning / non-event | 0.923 / 2.006 | **0.794** / **1.508** | 0.819 / 4.372 |

**Conclusion then:** inconclusive. HB-MJ was slightly better in sample, but BIC and out-of-sample
results favoured Bates. All models mispriced 2–6 day options by 1–3 vol pts because calendar time
puts variance on the weekend, confounding the event-step test.

## 5. Proxy study, trading-day clock [proxy]

Same snapshot, contracts, filters, event keys, parameter counts, folds, budgets and metrics. Only the
diffusion clock changes (`src/clocks.TradingClock`: 1/252 year per NYSE trading day accrued during the
session, weekends and holidays adding nothing; overnight share 0, chosen in advance). Run exactly once.
IV units differ between clocks, so **price RMSE ($) is the clock-invariant comparison**.

| | Heston | Bates | HB-MJ |
|---|---|---|---|
| IV RMSE (vol pts, trading time) | 0.501 | 0.259 | **0.228** |
| price RMSE ($): calendar → trading | 0.221 → 0.143 | 0.196 → 0.108 | 0.176 → **0.075** |
| AIC / BIC | −1313 / −1300 | −1416 / −1394 | **−1465 / −1434** |
| event-spanning / non-event IV RMSE | 0.481 / 0.699 | 0.267 / **0.113** | **0.234** / 0.139 |
| OOS event-spanning (12 folds) | 0.527 | 0.297 | **0.249** |
| OOS non-event (1 fold, 1 Oct) | 0.707 | **0.131** | 0.344 |

Out-of-sample folds, trading clock (vol pts): HB-MJ beats Bates on **9 of 13** held-out expiries
(2, 5, 7, 8, 9, 16, 30 Oct; 6, 30 Nov); Bates is better on 1, 6, 23 Oct and 20 Nov. Under the calendar
clock HB-MJ won 5 of 13. Full table: `results/clock_comparison_folds.csv`.

Event steps (ΔATM total implied variance ×10⁻⁴; the market column is clock-invariant; fig 7):

| Window | Market | Bates (cal → trade) | HB-MJ (cal → trade) |
|---|---|---|---|
| NFP 2 Oct | 0.62 | 0.45 → 0.54 | 0.57 → **0.64** |
| CPI 14 Oct | 3.35 | 3.16 → 3.05 | 3.43 → **3.28** |
| FOMC+ECB 28–29 Oct | 4.24 | 3.35 → 3.55 | 4.97 → **4.44** |
| NFP 6 Nov | 4.46 | 3.76 → **4.09** | 3.27 → 3.82 |
| CPI 10 Nov | 8.02 | 7.86 → 8.98 | 7.03 → **8.49** |

With the trading clock, HB-MJ's step is closer to the market's than Bates' in 4 of 5 windows (all except
NFP 6 Nov).

Fitted event variances (HB-MJ): FOMC+ECB 1.84e-4 → **1.06e-4** (σ ≈ 1.0%); CPI 5.5e-5 → **4.1e-5**
(0.64%); NFP 1.3e-5 → **1.2e-5** (0.34%). The calendar-clock values were inflated by absorbing
weekend variance. Standard errors assume 0.1 vol-pt noise, so they are understated wherever fit errors
are larger.

**Does the conclusion change? Yes, for this proxy sample.** Once weekends stop receiving diffusion
variance, all three models fit much better. HB-MJ then has the lowest pricing error (30% below Bates),
is preferred by both AIC and BIC, and beats Bates out of sample on event-spanning expiries (0.249 vs
0.297 vol pts, 9 of 13 folds). It is worse on the single non-event expiry. This supports the
term-structure hypothesis *in SPY on one afternoon*; it is not evidence about EUR/USD.

## 6. In-sample vs out-of-sample [sim + proxy]

- Simulation: the pooled HB-MJ out-of-sample error sits at the noise floor (0.050 vol pts).
- Proxy, trading clock: in-sample → out-of-sample on event-spanning expiries: Heston 0.481 → 0.527,
  Bates 0.267 → 0.297, HB-MJ 0.234 → 0.249 (fig 8).
- Proxy, both clocks: when the only pre-event expiry (1 Oct) is held out, HB-MJ's NFP parameter loses
  its anchor and HB-MJ does worst on that fold. The simulation predicted this identification failure.

## 7. Identifiability findings [sim + proxy]

- Reliably identified: v₀, θ, σ, ρ in synthetic surfaces; short-dated total variance; per-event variance
  when expiries (or valuation times) bracket the event; combined variance when they do not.
- Not individually identified: λ, μ_J, σ_J (report the jump variance rate). κ is weak. In the proxy fits
  θ hits its upper bound (0.25) in the trading-clock Bates and HB-MJ fits, and λ hits 50 under the
  calendar clock. The long-run variance is not identified with a 2-month maximum maturity. (These bound
  hits were missed by the original tolerance, 1e-6 of the range, and are now caught at 1e-3.)
- Nearby events with no separating expiry or valuation time: only the sum is identified
  (correlation −1, condition number ~4e8). A single-date surface with no pre-event expiry can land in a
  false optimum; pooling across dates removes it.

## 8. Limitations

1. The proxy data are **one snapshot of SPY**: not FX, not a panel. FXE, the FX proxy, failed the quality filters.
2. The trading clock uses overnight share 0, chosen before the rerun. Other values were not explored,
   by design. Equity overnight variance is not zero, so this is a modelling convention, not a measurement.
3. FOMC and ECB could not be separated in the proxy; ECB relevance for SPY is weak.
4. Normal event jumps add variance but not kurtosis. Event effects on smile shape are outside the model
   by construction.
5. Event variance is assumed shared by type across meetings and independent of the variance state.
6. Standard errors are local (Jacobian) and assume 0.1 vol-pt quote noise.
7. The FX date roller is unvalidated against Bloomberg OVML. Month tenors across holidays are reported
   as ambiguous rather than guessed.
8. No hedging experiment was run on real data.

**Next step (future work):** Bloomberg OTC EUR/USD daily panels (ON–1M with bid/ask, OVML dates and
conventions), estimated with the pooled calendar-clock specification. See README.
