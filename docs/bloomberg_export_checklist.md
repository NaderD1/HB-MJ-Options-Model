# Bloomberg checklist: EUR/USD vol surface for HB-MJ

Two jobs at the terminal, in this order:

- **A. Validation table (≈20 minutes, do this first):** fill in `docs/bloomberg_validation_table.csv`
  so we can check that our loader reproduces **Bloomberg's own** strikes, vols, premium and dates.
- **B. Research export:** today's surface plus BDH history, in the export template.

The loader refuses to guess: a missing or contradictory convention stops the import with a
`ConventionAmbiguityError`. Only rows with `data_origin = bloomberg` count as research data;
synthetic and FXE rows are always development-only.

---

## 0. Conventions: record once per session (OVDV settings, confirm in OVML)

| Column | Record | Typical for EUR/USD ≤ 1Y (**confirm, don't assume**) |
|---|---|---|
| `delta_type` | spot or forward; premium-adjusted (`_pa`) or not | `spot` |
| `atm_type` | `dns` (delta-neutral straddle), `atmf`, or `spot` | `dns` |
| `bf_type` | `smile` or `market` (broker) strangle | **check** (see below) |
| `premium_currency` | premium currency | `USD` |
| `rr_sign` | RR = EUR-call vol − EUR-put vol | `call_minus_put` |
| `vol_units` | `pct` (7.25) or `decimal` | `pct` |
| `bbg_pricing_source` | BGN, CMPN, BVOL, a dealer… | record exactly |

**How to check `bf_type`:** in OVML, price a 25D strangle at ATM + BF vol. If both legs use that
one vol at its own 25D strikes, it's a `market` strangle; if each leg uses the smile vol, it's `smile`.
Screenshot the OVDV settings panel and save it next to the CSV.

## A. Validation table: `docs/bloomberg_validation_table.csv`

One row per tenor: **ON, 1W, 2W, 1M, 2M, 3M, 6M, 1Y**. All rows must come from the same moment
and pricing source.

| Group | Columns | Where |
|---|---|---|
| Time | `valuation_ts` (with offset) or `valuation_tz` | Screen time. Note it: FX value dates roll at **17:00 New York** |
| Market | `spot`, `fwd_points` (`fwd_points_scale` = 10000) | OVML / FRD |
| Rates | `usd_depo_rate`, `eur_depo_rate` | OVML rates tab (simple ACT/360, spot → delivery) |
| Quotes | `atm`, `rr25`, `bf25`, `rr10`, `bf10` | OVDV |
| Conventions | section 0 columns | OVDV settings |
| Dates | `spot_date`, `expiry_date`, `delivery_date`, `expiry_cut` (`NY10`) or `expiry_time` + `expiry_tz` | OVML for each tenor ("Expiry", "Delivery", "Premium/Settle date") |
| **Bloomberg's own outputs** | `bbg_K_ATM`, `bbg_vol_ATM`, `bbg_K_25DC`, `bbg_vol_25DC`, `bbg_K_25DP`, `bbg_vol_25DP`, `bbg_K_10DC`, `bbg_vol_10DC`, `bbg_K_10DP`, `bbg_vol_10DP` | OVML: enter the strike as `25D` call / `25D` put / `ATM`, then read off Bloomberg's solved strike and vol (or the OVDV strike view) |
| | `bbg_premium_ATM_call_usd_pips` | OVML premium for the ATM call, in USD pips (premium currency USD) |

Afterwards, run:

```
python -c "from src.bloomberg_validation import compare_with_bloomberg; print(compare_with_bloomberg('docs/bloomberg_validation_table.csv'))"
```

Passing means strikes within 0.5 pip, vols within 0.01 vol pts, ATM premium within 0.05 pips, and
dates exact. A failure tells us which convention or date rule to fix. Only after the date columns
pass is `allow_roller_dates=True` enabled for historical data.

## B. Research export (template: `data/examples/bloomberg_format_SYNTHETIC_eurusd.csv`)

Save to `data/raw/bloomberg/`, one row per valuation × tenor, and never edit it by hand (the
loader stores a SHA-256 hash).

| Column | Required? | Notes |
|---|---|---|
| `data_origin` | REQUIRED | `bloomberg` |
| `valuation_ts`, `valuation_tz` | REQUIRED | Same time as all quotes |
| `pair`, `spot`, `tenor` | REQUIRED | Tenors ON, 1W, 2W, 3W, 1M, 2M, 3M, 6M, 9M, 1Y |
| `spot_date`, `expiry_date`, `delivery_date` | REQUIRED for snapshots | From OVML. For BDH history, the validated roller may supply them |
| `expiry_cut` | REQUIRED | `NY10` unless OVML shows otherwise |
| `forward` or `fwd_points` + `fwd_points_scale` | REQUIRED | Points to the **delivery** date |
| `usd_depo_rate` or `usd_df` + `usd_df_start=spot` | REQUIRED | Discounting spot → delivery |
| `eur_depo_rate` or `eur_df` + `eur_df_start=spot` | optional | Cross-currency basis diagnostic |
| `atm`, `rr25`, `bf25` | REQUIRED | |
| `rr10`, `bf10` | strongly recommended | Wings identify the jump parameters |
| `*_bid`, `*_ask` for each quote | recommended | Error bands and filters |
| section 0 conventions | REQUIRED | Every row |
| `bbg_pricing_source`, `export_ts` | recommended | Audit |

**History (the big win with limited terminal time).** Use Excel BDH on the ATM/RR/BF tickers for
every tenor (e.g. `EURUSDV1W`, `EURUSD25R1W`, `EURUSD25B1W`, `EURUSD10R1W`, `EURUSD10B1W`
`BGN Curncy`; verify the names with SECF), plus spot and forward-points tickers. Pull as many years as
allowed, `PX_LAST` plus `PX_BID`/`PX_ASK`. Also record the BDH close time for the pricing source
(it sets `valuation_ts` and whether the 17:00 NY roll applies).

**Event context (optional):**
- **ECO:** CPI/NFP consensus vs actual (surprise sizes).
- **WIRP:** implied FOMC/ECB move probabilities.
- **FOMC** screen: cross-check `data/events/macro_events.csv`.

## Before leaving the terminal

- [ ] The validation table is filled for ≥ 5 tenors, including one month-end or holiday-affected tenor.
- [ ] Spot, forwards and vols come from the same source and time; the time is noted relative to 17:00 NY.
- [ ] The conventions row is filled in, and the OVDV settings screenshot is saved.
- [ ] Every snapshot row has spot, expiry and delivery dates plus the cut.
- [ ] `load_bloomberg_csv(...)` runs without a `ConventionAmbiguityError`.

**Note on `bf_type`.** For a typical 3M smile, smile vs market strangle changes the 25D wing vols by
about 0.003 vol pts and the 10D wings by about 0.02. It's small but *systematic*, and it sits in the wings
where jump parameters are identified.
