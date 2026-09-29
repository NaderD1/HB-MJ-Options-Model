# Bloomberg export checklist: EUR/USD vol surface for HB-MJ

Use this at the terminal. The loader (`src/data_loaders.py::load_bloomberg_csv`) **refuses
to guess** conventions: anything marked **REQUIRED** that is missing stops the import.

Save exports as CSV in `data/raw/bloomberg/` with one row per *valuation time × tenor*,
using the column names below (see `data/examples/bloomberg_format_SYNTHETIC_eurusd.csv`).
Never edit an exported file by hand; the loader stores its SHA-256 hash for auditing.

## 1. Before exporting: write down the conventions (once per session)

Open **OVDV** for EURUSD and check its settings/legend. Confirm with **OVML** if unsure.

| Column | What to record | Expected for EUR/USD ≤ 1Y (confirm, don't assume) |
|---|---|---|
| `delta_type` | Spot or forward delta; premium-adjusted or not | `spot` (premium in USD → unadjusted) |
| `atm_type` | ATM definition: delta-neutral straddle, ATM-forward, or ATM-spot | `dns` |
| `bf_type` | Butterfly: **smile** strangle or **market/broker** strangle | Check OVDV. This changes the wing vols (see note) |
| `premium_currency` | Premium currency | `USD` |
| `rr_sign` | RR = EUR call vol − EUR put vol? | `call_minus_put` |
| `vol_units` | Vols in percent (7.25) or decimals (0.0725) | `pct` |
| `bbg_pricing_source` | Pricing source shown (BGN, CMPN, BVOL, a dealer…) | Record exactly |

**How to check `bf_type`.** Price a 25-delta strangle in OVML at the surface's ATM + BF vol.
If OVML uses **one** vol for both legs at that vol's own 25D strikes, it's a market strangle.
If each leg is priced off the smile, it's a smile strangle. When in doubt, screenshot the
OVDV settings panel and save it next to the CSV.

For tenors above 1Y, EUR/USD usually switches to **forward** delta. Record `delta_type`
per row if you export >1Y.

## 2. Per-row fields

| Column | REQUIRED? | Source / how | Notes |
|---|---|---|---|
| `data_origin` | REQUIRED | Type `bloomberg` | Anything else is treated as non-research data |
| `valuation_ts` | REQUIRED | Export time, or the close time of the pricing source | Include a UTC offset, or fill `valuation_tz` |
| `valuation_tz` | if no offset | e.g. `America/New_York`, `Europe/London` | IANA name |
| `pair` | REQUIRED | `EURUSD` | |
| `spot` | REQUIRED | `EURUSD BGN Curncy` (same source and time as the vols) | Mid |
| `tenor` | REQUIRED | ON, 1W, 2W, 3W, 1M, 2M, 3M, 6M, 9M, 1Y | Short tenors matter most |
| `expiry_date` | REQUIRED | **OVDV / OVML** expiry date for that tenor | Not computed from the tenor (holiday rules) |
| `expiry_cut` | REQUIRED | `NY10` (New York 10:00) unless OVML shows another cut | Or give `expiry_time` + `expiry_tz` |
| `delivery_date` | recommended | OVML delivery/settlement date | Needed if you give a deposit rate |
| `forward` **or** `fwd_points` | REQUIRED | Outright, or forward points (**FRD**, or the `EUR1W Curncy`-style points tickers; verify with SECF) | With points, give `fwd_points_scale` = 10000 |
| `usd_df` **or** `usd_depo_rate` | REQUIRED | USD discount factor, or the USD deposit/OIS rate used by OVML | Rate = simple ACT/360 to `delivery_date` |
| `eur_df` or `eur_depo_rate` | optional | EUR discounting | Only used to measure the cross-currency basis |
| `atm`, `rr25`, `bf25` | REQUIRED | OVDV grid, or tickers such as `EURUSDV1W`, `EURUSD25R1W`, `EURUSD25B1W` `BGN Curncy` (verify names with SECF) | Mid |
| `rr10`, `bf10` | strongly recommended | e.g. `EURUSD10R1W`, `EURUSD10B1W` | Pins down the wings / jump tails |
| `atm_bid/_ask`, `rr25_bid/_ask`, `bf25_bid/_ask`, `rr10_bid/_ask`, `bf10_bid/_ask` | recommended | Bid/ask fields (`PX_BID`, `PX_ASK`) | Used for error bands and filtering |
| `export_ts` | recommended | When you pressed export | Audit |

## 3. What to export

1. **Today's surface snapshot:** all tenors ON–1Y, all fields above. It's best to export
   **on a day just before a cluster of events** (e.g. the week of an FOMC or ECB meeting).
2. **History (the big win with limited terminal time).** Use Excel **BDH** on the vol tickers
   (`PX_LAST`, plus `PX_BID`/`PX_ASK` if available) for ON, 1W, 2W, 3W, 1M, 2M, 3M, 6M, 1Y,
   and spot and forward points, for as many years as allowed. This gives daily surfaces for
   the parameter-stability and event-kink studies in a single session.
   *Caveat:* historical rows need expiry dates per day. Either export them, or we build an FX
   date roller with USD/EUR holiday calendars (planned; tell me which you prefer).
3. **Event context (optional but useful for the write-up):**
   - **ECO:** consensus and actual values for CPI and NFP (surprise sizes, for a later
     surprise-dependent extension).
   - **WIRP:** market-implied probability of a rate move at each FOMC/ECB meeting.
   - **FOMC** screen: meeting dates, to cross-check `data/events/macro_events.csv`.

## 4. Quick sanity checks before you leave the terminal

- [ ] The spot, forwards and vols all come from the **same pricing source and time**.
- [ ] Every row has an `expiry_date` and a known cut.
- [ ] The conventions row (section 1) is filled in, and the OVDV settings screenshot is saved.
- [ ] The file opens and `python -c "from src.data_loaders import load_bloomberg_csv; load_bloomberg_csv('data/raw/bloomberg/<file>.csv')"` runs without a `ConventionAmbiguityError`.

**Note on `bf_type`.** For the same quoted butterflies, smile-strangle and market-strangle
readings give different wing vols. For a typical 3M EUR/USD smile the gap is small (about
0.003 vol points at 25D and 0.02 at 10D in our tests), but it grows with smile steepness and
maturity. More importantly, it is a *systematic* error in the wings, which is where jump
parameters are identified, so the convention must be recorded rather than assumed.
