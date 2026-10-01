# Review: p-values, uncertainty bands and histograms after the MIROC multi-model change

**Branch:** `feature/pvalues-uncertainty-histograms` (off `feature/add-miroc-inputs`)
**Status:** review complete; all three tracks decided and implemented (31 commits, listed at
the end). Production cache regenerated under v1.5.0 on 2026-09-21 and verified (see
"Regenerating everything").

## Why this review

`feature/add-miroc-inputs` made MIROC-ES2H a second training model. That changed how the
regional maps, the sea-ice sigmoids and the SAI deltas aggregate across models — but the
grid-cell significance test, the dashboard uncertainty bands and the PDF histograms were
inherited unchanged from the single-model (CESM-only) design. This document records what
each of those does today, what is wrong or questionable about it, what was decided, what
changed, and the measured before/after effect on **real CESM + MIROC output regenerated from
the R2 stores by this branch's pipeline**.

Every numeric claim here is either measured on that output or explicitly marked unverified.

## How the models combine today

| Stage | Script | Granularity | Where models combine | Weighting |
|---|---|---|---|---|
| No-SAI projection map | `scripts/fit_map.py` | Per model — own emulators, regressed on the model's **own** global-tas anomaly | Prediction time (`backend/baseline.py`, `.mean(dim=['model','emulator'])`) | 50/50 |
| Sea-ice sigmoids | `scripts/fit_icefrac_arctic.py` | Per model, stacked on a `model` dim | Prediction time (`backend/icefrac.py`, mean of sigmoids) | 50/50 |
| SAI response slope | `scripts/fit_delta.py` | **Pooled**: one through-origin least squares over all anchors of all models; output has **no** `model` dim | Fit time | `wᵢ = xᵢ² / Σⱼxⱼ²` per anchor — cooling magnitude squared |
| Internal variability | `scripts/variability.py` | **Was CESM only** (`compute_variability: False` for MIROC; hard-selected CESM). **Now per model**, from each model's regridded projection-baseline members, combined in the band as `sqrt(Σσ²_m)/M` and in the test as pooled `sqrt(mean σ²_m)`. | Request time | see Uncertainty bands |
| Historical reference | `backend/historical.py` | Per model, then mean — **was CESM-only for icefrac** (X1); now every variable is the mean of both models, and the per-model records are kept for the bands. | Load time | 50/50 |

The pooled slope is `β = Σᵢ xᵢyᵢ / Σⱼ xⱼ²`, i.e. a weighted average of the per-anchor slopes
`yᵢ/xᵢ` with weights `xᵢ²/Σxⱼ²`. "Influence" below means that weight: how much one
simulation moves the final β. The `fit_delta.py` docstring puts the model split at CESM 60.4%
/ MIROC 39.6%; **that figure is unverified until the harness recomputes it from the real
anchors** (see Phase 2).

Consequence worth holding onto: the SAI delta aggregates models at a different stage and with
a different weighting than everything else, and cannot be isolated per model without a refit.

## Data provenance — verified 2026-09-21

These were checked directly against the R2 bucket before any processing ran.

### Two MIROC prefixes on the bucket, and which one is right

| Prefix | Experiments present | Members | Monthly variables | Role |
|---|---|---|---|---|
| `simulator-inputs/MIROC-ES2H_*.zarr` — what `version.ZARR_STORES_URI` points at | baseline, ssp245, historical, G6-1.5K-SAI, G6-0.5K-SAI | 3 (`r01–r03`) | `TREFHT, PRECT, QFLX, ICEFRAC` (CESM-style names) | Consumer-ready; **what the pipeline reads. Correct as-is.** |
| `MIROC-ES2H/MIROC-ES2H_*.zarr` | G6-1.5K-SAI, baseline only | 10 (`r01–r10`) | `SurfT, P, SLP, AOD550nm` | Raw JAMSTEC output as converted to zarr; producer side |

`scripts/data_sanity_checks.py` passes for all 10 consumer stores (5 CESM + 5 MIROC).

**Are the 3 consumer members the same simulations as raw `r01–r03`?** Yes. On
`G6-1.5K-SAI`, consumer `TREFHT` equals raw `SurfT` member-for-member to float32 precision —
max |Δ| = 0.00098 / 0.00095 / 0.00104 K for `r01` / `r02` / `r03` over the first 24 months at
20 random cells, identical means to 3 decimals. Raw `r04–r10` are seven additional
realizations of the same experiment that were never carried into the consumer stores.

**Decision (Ali):** 3 members everywhere, and this is the correct set rather than a
compromise: only `r01`–`r03` have the daily output the daily-count and exposure variables
need, so the seven additional raw members could never feed the full pipeline. Not a
follow-up.

### MIROC's raw variable `SurfT` is 2 m air temperature, misnamed

The check above shows that for the MIROC GeoMIP-family stores (`baseline`, `G6-*`), consumer
`TREFHT` equals raw `SurfT`. During this review that was read as a physics mismatch (skin
temperature vs CESM's 2 m reference-height temperature), and several readings below were
first written under that assumption. **Clarification (Ali, 2026-09-21): the name is the
mistake.** Whoever ran the model labelled the output `SurfT`, but the field is the
near-surface air temperature, the same quantity as CESM's `TREFHT`. There is no
cross-model mismatch in what "tas" means; the earlier attributions of MIROC's warm tails and
level differences to skin temperature are withdrawn, and those differences stand as genuine
inter-model differences. The `fit_map.py` docstring is corrected accordingly.

### Environment note

esmpy takes its version from the git checkout it is installed from and refuses to load
against an ESMF library of another version. The machine's library is 8.7.0 while the source
tree HEAD is v8.9.0b16; the fix used here is a worktree at tag `v8.7.0`
(`/Users/ali/Documents/GitHub/util/esmf-8.7.0`) — see README §"Install ESMF Library".
Only `scripts/process_monthly_gauss.py` needs xesmf; nothing in the three audit tracks does.

## Uncertainty bands

### State before this work

| # | Finding | Where |
|---|---|---|
| U1 | The only inter-model band (icefrac, global) is `2·std(ddof=1)` over **2** models — for two values that is exactly `1.41·|a−b|`, not a 95% interval. | `get_outputs.py:470-500` |
| U2 | That band is built from unsmoothed series while the central line is Butterworth-smoothed (`apply_constraints`). | same |
| U3 | Two different quantities are both drawn as "uncertainty": inter-model spread (icefrac global) vs internal variability (everything else), unlabelled. | launchers |
| U4 | `model_internal_variability` is **CESM-only** (`compute_variability: False` for MIROC; `variability.py` hard-selects CESM) while the plotted field is a 2-model mean. Measured: the scalar is 0.119 K for tas in *all three* configs. | `process_monthly_gauss.py`, `variability.py:339` |
| U5 | `variability.py` labelled the icefrac grid σ `million km^2`; it is a std of ice **fraction**. | `variability.py:428` |
| U6 | Bands exist only in global mode for icefrac; every other variable and all regional views use the CESM scalar. | `get_outputs.py` |
| U7 | **p-e's variability artifacts were evaporation-only.** `var2esm_var["p-e"] = "QFLX"` and both variability functions ran on `esm_var` *before* the P−E subtraction, so P−E's band and its t-test σ described evaporation. | `process_monthly_gauss.py` |
| U8 | **The icefrac inter-model band measured the wrong disagreement.** It evaluated each model's sigmoid at the *shared* combined temperature (CESM 3.86 vs MIROC 3.78 million km² at 2100 with SAI, band ±0.11); run end-to-end on their own temperature patterns the models give **5.83 vs 2.58** (gap 3.25). The disagreement lives in the Arctic temperature patterns, and the construction was blind to it by ~30×. | `_load_variable_fields`, `predict_icefrac(per_model=True)` |

### Decisions (Ali, 2026-09-21)

- **Per-model internal variability for both models**, computed by one operator from the
  **regridded per-member** no-SAI ensemble (the same regridding the tested fields went
  through). MIROC's members come from its **projection baseline** (`ssp245`, fallback GeoMIP
  baseline — the `fit_map` rule). 3 members per model, the only members with daily output.
- **Every variable writes `*-all-members.nc`** for its no-SAI runs; variability is a
  **standalone step** (`scripts/variability.py`) so future changes never need reprocessing.
- **P−E variability from per-member P−E** (`pr − QFLX`, member- and time-aligned), fixing U7.
- **Explicit model name on every artifact**: `model_internal_variability_<model>.npy`,
  `grid_level_model_internal_variability_<model>.nc`, geojson column
  `variability_<var>_<model>`. Legacy unsuffixed CESM files stay until readers move.
- **σ_model = end-to-end per-model outcomes** (each model's own emulator/sigmoid on its own
  temperature pattern and own historical reference), fixing U8. For icefrac this also moves
  the **central line and map** to the mean of end-to-end outcomes (3.82 → ~4.20 million km²
  at 2100 with SAI in the default request); for linear variables the centre is unchanged.
- **One interval, the 2σ prediction interval**: `2·sqrt(σ²_model + σ²_internal(mean))`
  with `σ²_internal(mean) = (1/M²)·Σσ²_int,m`. A confidence interval on the mean
  (`2·σ_model/√M`) was built and drawn nested inside it first; it was removed on 2026-09-22
  (Ali) because with M=2 it is either zero (global tas, where both models are anchored to the
  same FaIR temperature) or nearly as wide as the prediction interval, so the nested shading
  read as two shades of one colour and explained nothing. **ddof=0 while M<3** (half-width = half the
  two-model gap), ddof=1 at M≥3; legends state M.
- **Revised 2026-09-28 (Ali): the band is the internal variability of the multi-model mean
  only**, `2·σ_internal(mean)`, constant in time; the model spread is no longer in it. The
  prediction interval above (spread + internal) is preserved, in all three repos, on the
  branch `aa-feature/multi-model-mean-uncertainty-both-internal-and-modelspread`. `σ_model`
  is still computed (same ddof rule) and emitted as the diagnostic `sigma_model_<seg>`; the
  drawn key is `band_internal_<seg>` (was `band_prediction_<seg>`). The front end shows the
  mean line and this band only, no per-model lines. See "Revision: internal variability only"
  below for the measured effect.
- **Smooth each per-model series first** (same `apply_constraints` filter as the line), then
  take mean and spread. **Same construction on the historical segment.**
- **All variables, global and regional.** The legacy `model_internal_variability` key keeps
  its meaning; the new bands are new keys (front-end adoption is out of scope).

### Phase 3a result — per-model σ (commits `be03341`, `a1625f4`, geojsons)

Global-series internal variability σ (std of per-member linearly detrended residuals,
2025–2099), from the regenerated members. "Legacy" is the pre-existing CESM-only,
native-grid, in-process value.

| Variable | Unit | CESM2-WACCM | MIROC-ES2H | Legacy CESM | Regridded vs legacy |
|---|---|---|---|---|---|
| tas | K | 0.1190 | 0.1812 | 0.1190 | +0.01%, grid diff 0 |
| tasmax | K | 0.1606 | 0.2617 | 0.1606 | 0.00%, grid diff 0 |
| tasmin | K | 0.1839 | 0.2022 | 0.1839 | +0.02%, grid diff 0 |
| pr | mm/day | 0.0133 | 0.0158 | 0.0133 | 0.00%, grid diff 0 |
| p-e | mm/day | **0.0015** | **0.0200** | 0.0129 (evaporation only) | **−88%; grid diff up to 5× legacy max** |
| tas_above_35 | days/yr | 0.3125 | 0.6903 | 0.3126 | −0.01%, grid diff 0 |
| tas_above_40 | days/yr | 0.0860 | 0.3429 | 0.0860 | −0.01%, grid diff 0 |
| tas_below_0 | days/yr | 0.5802 | 0.5412 | 0.5802 | 0.00%, grid diff 0 |
| pr_above_10 | days/yr | 0.1862 | 0.2437 | 0.1862 | −0.01%, grid diff 0 |
| pr_above_20 | days/yr | 0.1595 | 0.1554 | 0.1596 | −0.01%, grid diff 0 |
| icefrac (extent, hindcast residual) | million km² | 0.5899 | 0.3753 | 0.5899 | identical |

Readings:

- **The regrid to the CESM grid is an identity for CESM**: every CESM σ reproduces the legacy
  value (grid max|diff| = 0.0). The one-operator design costs nothing.
- **MIROC's internal variability is larger for most variables** — tas +52%, tasmax +63%,
  tas_above_40 4×, tas_above_35 2.2× — and smaller for sea-ice extent (0.38 vs 0.59).
- **p-e (U7 quantified).** The global-mean P−E series barely varies (0.0015 mm/day) because
  global P ≈ global E by water balance, so the old band (0.0129, evaporation's σ) was ~9×
  too wide; per cell, P−E in the tropics varies far *more* than evaporation, so the old
  t-test σ was too small there → too many "significant" cells. Both directions were wrong.
- **MIROC p-e σ is inflated (0.020 vs 0.0015).** MIROC's `ssp245` QFLX is filled from a
  *different realization* (the GeoMIP baseline's hfls, README caveat), so P and E do not
  co-vary and Var(P−E) loses the −2Cov(P,E) cancellation. A real data limitation to carry
  into the band's interpretation for p-e, not a code bug.

Regional: `variability_<var>_<model>` columns exist in all five region geojsons for the ten
field variables (icefrac regional aggregates extent by sum; a region-mean of per-cell σ is
not on that scale, so it has no column — as before).

### Phase 3b result — combined bands (commits `8dd66ee`, `aeb8549`, `22e27f8`, `856663f`, `d3e67c9`)

**What changed in the code.**

- `backend/uncertainty.py` — `combine_series(per_model, sigma_internal)`: prediction half-width
  `2·sqrt(σ²_model + σ²_internal(mean))`,
  `σ_internal(mean) = sqrt(Σσ²_m)/M`, `ddof = 0` while `M < 3` else `1`. Unit-tested.
- `backend/baseline.py`, `backend/historical.py` — `get_regional_map` returns the per-model
  maps too (mean over emulators, each rebased on its own history; cached as
  `<ssp>_regional_per_model_<var>.nc`); `get_historical_model_per_model` stacks the models'
  historical records (cached as `historical_per_model_<var>.nc`). X1 applied: icefrac's
  historical is the mean of both models.
- `backend/get_outputs.py` — the series construction is one helper (`_aggregate_series`) run
  for the combined fields (the lines) and for each model's own fields; per-model series are
  smoothed first, then combined. icefrac's per-model fields are each model's sigmoid on its
  **own** temperature (`_predict_icefrac_end_to_end`), and its combined map/line is their mean.
  Per-model σ_internal comes from `model_internal_variability_<model>.npy` (global) or the
  `variability_<var>_<model>` geojson column (regional), population-scaled for exposure.
  New `mean_over_time` keys: `band_prediction_<seg>`,
  `sigma_model_<seg>`, `band_mean_<seg>` (seg ∈ historical / no_sai / with_sai),
  `band_models`, `band_n_models`, `band_ddof`, `sigma_internal_per_model`,
  `sigma_internal_mean`, `band_labels`, `per_model_series`. The icefrac-only
  `model_spread_2sigma_*` keys are gone; `model_internal_variability` keeps its legacy value.
- Both launchers shade the prediction band and name it (with M) in the legend.

**Invariants verified on real data.** Combined maps unchanged (max |Δ| ≤ 2.5e-12 vs the
baseline dumps); each per-model map slice equals the corresponding single-model harness
config exactly (0.0); the plotted lines are unchanged to ≤ 1e-14 for every variable except
icefrac (by decision: historical max |Δ| 0.48, no-SAI 0.14, with-SAI 0.39 million km²).

**Before / after, global, SSP2-4.5 / +1.0 °C / start 2035, no-SAI at 2100.**
Figures: `figs/fix-3b/<var>/` (`render_matrix --configs combined --shas baseline,3b-bands
--step fix-3b`).

| Variable | Legacy band ±2σ (CESM scalar) | New PI ± | New CI ± | σ_model | σ_int CESM | σ_int MIROC |
|---|---|---|---|---|---|---|
| tas (K) | 0.238 | 0.217 | ~0 | ~0 | 0.119 | 0.181 |
| tasmin (K) | 0.368 | 0.305 | 0.095 | 0.067 | 0.184 | 0.202 |
| tasmax (K) | 0.321 | 0.447 | 0.230 | 0.163 | 0.161 | 0.262 |
| p-e (mm/day, land mean) | 0.0258 | 0.0811 | 0.0556 | 0.039 | 0.0015 | 0.020 |
| exposure_above_35 (M person-days) | 3,894 | **80,330** | 56,710 | 40,100 | 1,947 | 4,301 |
| exposure_above_40 | 1,071 | **62,940** | 44,480 | 31,450 | 536 | 2,136 |
| exposure_below_0 | 7,229 | 8,515 | 4,903 | 3,467 | 3,615 | 3,372 |
| exposure_above_10 | 2,320 | 13,600 | 9,520 | 6,732 | 1,160 | 1,518 |
| exposure_above_20 | 1,988 | 7,526 | 5,231 | 3,699 | 994 | 968 |
| icefrac (M km²) | 1.18 | 0.847 | 0.338 | 0.239 | 0.590 | 0.375 |

icefrac with SAI at 2100: per-model **5.83 / 2.58**, centre **3.82 → 4.21**, σ_model 1.625,
**PI ±3.33, CI ±2.30** (the legacy inter-model band here was ±0.11 — U8).

Readings:

- **tas: σ_model ≈ 0 for the projections by construction** — `fit_map` pins every model's
  global mean to FaIR, so the models cannot disagree about global-mean tas; they do disagree
  historically (σ_model 0.05 K at 1990) and regionally (Africa: 0.19 K). The tas PI is the
  combined internal σ alone, slightly narrower than the old CESM-only scalar.
- **Every derived or absolute variable is dominated by model spread**, not internal
  variability: tasmax σ_model ≈ σ_int; exposure σ_model is 10–20× σ_int. The legacy band
  understated hot-day-exposure uncertainty by ~20×.
- **icefrac's projection band shrinks under no-SAI** (0.85 vs 1.18 — both models agree ice is
  nearly gone and the combined σ_int is smaller than CESM's alone) **and widens hugely under
  SAI** (±3.33), where the models' Arctic patterns disagree.
- **Regional mode works off the new geojson columns** (Africa, IPCC-WGII): tas PI ±0.80 K
  (legacy ±0.88), exposure_above_35 PI 24,450 (legacy 9,330); icefrac regional has no σ
  column and no ice there, so its band is 0.

### Revision: internal variability only (2026-09-28)

**Why.** With the spread in the band, the band's shape over time was set almost entirely by
the gap between the two models' series, and part of that gap is not model disagreement the
dashboard should display: (1) the with-SAI series of every linear variable inherit the no-SAI
gap unchanged, because the SAI delta is one pooled fit added equally to both models (tasmax
half-gap 0.164 K with SAI vs 0.163 K without at 2100, although SAI holds warming near 1 K);
(2) absolute variables carry each model's level bias (U9: MIROC counts about twice CESM's
>35 °C person-days already in 1900); (3) the historical-to-projection seam changes the gap's
nature in 2015 (own historical runs vs a shared FaIR temperature); (4) under SAI, sea ice
differs between the models partly because the pooled delta cools both Arctics by the same
4.6 K although MIROC's own SAI runs cool its Arctic about 3.1× its global cooling and CESM's
about 2.35×. The band now shows one well-defined quantity: how much a two-model mean varies
from year to year.

**What changed.** `backend/uncertainty.py`: `BandResult.internal = Z·σ_internal(mean)`
replaces `prediction`; `as_dict` emits `band_internal_<seg>` and `sigma_model_<seg>`;
`band_label` reads "±2σ internal variability of the multi-model mean, M=2". Launchers and the
harness renderer draw `band_internal_*` (the renderer still reads `band_prediction_*` from
older dumps). API: forwards `band_internal` per series and no longer the per-model series.
FE: mean line + this band; per-model lines and their legend entries removed.

**Measured, SSP2-4.5 / +1.0 °C / default deployment, 2100 (half-widths).**

| Variable | Spread + internal (frozen branch), no SAI / with SAI | Internal only (now) | Model spread σ_model, no SAI / with SAI |
|---|---|---|---|
| tas (K) | 0.217 / 0.217 | 0.217 | ~0 / ~0 |
| tasmax (K) | 0.447 / 0.449 | 0.307 | 0.163 / 0.164 |
| p-e (mm/day) | 0.081 / 0.081 | 0.020 | 0.039 / 0.039 |
| exposure_above_35 (person-days) | 80,300 / 79,400 | 4,720 | 40,100 / 39,600 |
| icefrac (M km²) | 0.903 / 3.274 | 0.676 | 0.300 / 1.600 |

For tas nothing changes: its model spread is zero by construction. For every other variable
the band narrows, most for the exposure counts (about 17×), whose spread was mostly level
bias. The spread remains in the output for anyone who wants it.

### U9 — absolute variables carry each model's mean bias into σ_model

For un-rebased variables the per-model series differ in *level*, not only in projected
change. At year 2000 (historical): exposure_above_35 **CESM 59,279 vs MIROC 121,814**
million person-days (2.05×); icefrac 5.47 vs 6.23 million km²; whereas rebased tasmax shows
a genuine change disagreement (anomaly 0.60 vs 0.37 K). The exposure gap is a genuine
inter-model difference in how many hot days each model produces (an earlier reading that
attributed it to a skin-temperature variable was wrong; see Data provenance), and it means
the exposure PI mostly expresses "the models count hot days very differently", which the
previous CESM-only band hid entirely. Whether absolute variables should be
combined as per-model anomalies (bias-removed) rather than absolute levels is a design
question raised here, not decided.

## P-values

### State before this work (`backend/p_values.py`, `get_outputs.py:_compute_sai_p_values`, `scripts/cache.py`)

| # | Finding |
|---|---|
| P1 | Two-sample t with `n = (2039-2025+1)·3 = 45` per group, `df = 88`, `SE = σ·√(2/45)`: `n` described the *variability sample*, not either compared group (a 51-year pre-industrial mean vs a 10-year decadal mean of a deterministic, ensemble-mean-trained emulator). |
| P2 | An **annual** per-cell σ applied to **multi-year means**, without autocorrelation. |
| P3 | σ was **CESM-only** while the tested field became a 2-model mean. |
| P4 | No multiple-comparisons control over ~18,000 land cells × 6 decades. |
| P5 | `σ = 0` cells: p = 1 if unchanged, p = 0 if changed. |
| P6 | Cached with-SAI p-values keyed by (scenario, target, variable) only, computed for **start 2035 / ramp-up 0** (icefrac 2090 / 10) and served for every deployment. |
| P7 | The IPCC-WGII land mask was rasterized from geojson on every clip — six times per p-value call. |
| P8 | No unit tests. |
| U7 (→P) | p-e's σ was **evaporation-only** (`QFLX`), so P−E's stipple used the wrong noise. |

Baseline measurement (combined, SSP2-4.5 / +1.0 °C / 2035 / 10, with SAI, 2100): tas 98.4% of
land cells significant, tasmin 88.1%, tasmax 52.3%, p-e 78.7%, exposure_above_35 37.9%,
icefrac 74.4%; no-SAI tas 100% in every decade. Per-cell lag-1 autocorrelation of the detrended
residuals: median 0.0–0.25 by variable, 90th percentile up to 0.5.

### What the hatching should test — the σ what-if (`scripts/compare_runs.py sigma_options`)

Before choosing, the effect of three candidate test σ's was mapped
(`figs/p3-sigma-options/<var>_sigma_options.png`), all with the detectability statistic:

| Variable | today | A: σ_pooled | B: σ of ensemble mean | C: internal + model spread |
|---|---|---|---|---|
| tas | 98.4% | 96.6% | 98.0% | 61.4% |
| tasmax | 52.3% | 33.6% | 49.0% | 10.0% |
| p-e | 78.7% | 31.6% | 43.7% | 19.8% |
| exposure_above_35 | 37.9% | 32.7% | 35.1% | 3.3% |
| exposure_above_10 | 18.0% | 8.3% | 17.5% | 1.8% |
| icefrac | 74.4% | 67.7% | 78.1% | 16.9% |

C would blank the exposure maps because there σ_model ≈ 2.4× σ_pooled at the median — and by
U9 that is mostly level bias. The what-if also surfaced **X6**: the icefrac σ file carries
`area.nc`'s latitudes, the maps `correct_lat.npy`'s; they differ in the last digits, so
xarray arithmetic between such artifacts silently inner-joins to the 125 latitudes that match
exactly. The backend avoids it by working in `.values`; the harness pins coordinates.

### Decisions (Ali, 2026-09-21)

- **Test:** *detectability of a decadal mean.* `Δ = window mean − 1850-1900 reference`,
  `SE = σ·√(c_ρ/k)`, `c_ρ = (1+ρ)/(1−ρ)`, `k` = years in the window, `z = Δ/SE`, two-sided
  normal p (σ is known from 45 member-years, not from the 10 tested years). Reference mean
  treated as a constant (decade noise only).
- **σ = A, pooled internal** `√(mean_m σ²_m)` — the noise one realization shows; model
  disagreement is the band's job, not the hatching's.
- **ρ:** per-cell lag-1 autocorrelation of the detrended residuals, mean over models — a new
  artifact `grid_level_lag1_autocorr_<model>.nc` from `scripts/variability.py`.
- **No multiple-comparisons correction** (after BH was explained: it would remove the ~5%
  chance speckle; measured effect on the baseline: tas unchanged, exposure_above_10 with-SAI
  2050 13.4% → 2.5%). `benjamini_hochberg` stays available and tested, off by default.
- **σ = 0 cells keep the legacy rule** (p = 1 unchanged / p = 0 changed): treating them as
  NaN made ~11,000 zero-exposure land cells lose their "not significant" hatching, which the
  front-end renders exactly like significance.
- **Raw p-values, threshold 0.05**, now a single constant `SIGNIFICANCE_THRESHOLD`
  (`backend/constants.py`) used by the band-2 mask and both launchers. Verified: the
  front-end (`LeafletMap.js`) reads band 2 as a finished mask and applies no threshold of its
  own; the API passes GeoTIFFs through.
- **Cache:** with-SAI p-values precomputed for the dashboard's default deployment only
  (`DEFAULT_SAI_START_YEAR = 2035`, `DEFAULT_RAMP_UP = 10`); any other start year, ramp-up or
  custom injection is computed live (~1 s per variable after the mask cache).

### Phase 4 result (commits `dec6c38`, `9004185`, `860ec4d`)

`backend/p_values.py` rewritten (`detectability_pvalues`, `load_pooled_sigma`,
`load_mean_rho`, `benjamini_hochberg`, `get_grid_level_p_values` with the same signature);
`backend/utils.py` caches the rasterized land mask per grid (P7; all ten variables' p-values
recompute in 12 s); `_compute_sai_p_values` gained `start_year`/`ramp_up` and the
default-deployment rule; `scripts/cache.py` computes the cached file for that deployment;
14 synthetic unit tests. Figures: `figs/fix-4-pvalues/<var>/`.

**Share of valid cells significant (p < 0.05), combined config, before → after:**

| Variable | valid cells | no-SAI 2100 | with-SAI 2050 | with-SAI 2100 | cells changed at 2100 (+/−) |
|---|---|---|---|---|---|
| tas | 18,378 | 100.0 → 100.0% | 99.9 → 99.0% | 98.4 → 96.7% | +0 / −325 |
| tasmin | 18,378 | 100.0 → 99.9% | 96.8 → 71.2% | 88.1 → 66.5% | +0 / −3,978 |
| tasmax | 18,378 | 100.0 → 99.7% | 75.3 → 44.3% | 52.3 → 33.6% | +0 / −3,435 |
| p-e | 18,378 | 80.7 → 43.7% | 72.6 → 27.7% | 78.7 → 31.6% | +1 / −8,653 |
| exposure_above_35 | 18,378 | 39.5 → 38.5% | 36.4 → 28.5% | 37.9 → 33.6% | +1 / −786 |
| exposure_above_40 | 18,378 | 29.9 → 29.2% | 29.7 → 22.8% | 30.3 → 27.3% | +0 / −539 |
| exposure_below_0 | 18,378 | 40.8 → 39.7% | 39.3 → 35.8% | 36.8 → 30.9% | +0 / −1,085 |
| exposure_above_10 | 18,378 | 38.9 → 28.8% | 13.4 → 4.0% | 18.0 → 8.3% | +1 / −1,785 |
| exposure_above_20 | 18,378 | 35.4 → 22.9% | 11.2 → 3.2% | 17.5 → 7.6% | +1 / −1,827 |
| icefrac | 5,983 | 89.7 → 85.5% | 71.0 → 58.2% | 74.4 → 67.9% | +346 / −740 |

Per decade the test moves as it should — e.g. tasmax with SAI: 75.3/67.4/63.3/56.9/54.1/52.3%
→ 44.3/38.1/36.6/35.2/34.3/33.6% for 2050…2100 — because Δ is each window's own mean while σ
and ρ describe the noise and are fixed.

Readings:

- The new SE is ~1.5× the old one for uncorrelated cells (`√(c_ρ/10)` vs `√(2/45)`), more
  where ρ > 0, so every variable loses stipple; the direction is the same everywhere except
  where the σ changed meaning.
- **p-e is the largest change (−8,653 cells)** and the most defensible: its σ is now P−E's,
  not evaporation's (U7). The old stipple claimed significant water-availability change over
  79% of land; the corrected test finds it over 32%.
- **tas is barely affected** (98.4 → 96.7%): +1 °C above pre-industrial is detectable in a
  decade almost everywhere, under any reasonable σ.
- Exposure variables move modestly; their maps are mostly zero-population/zero-exposure
  cells at p = 1 either way.
- icefrac is the only variable where cells also *gain* significance (+346): its σ is now
  the pooled CESM/MIROC hindcast residual, smaller than CESM's alone over much of the Arctic.

### P9 — a decade before the SAI start (found by the production-path check; fixed `644ceae`)

A request whose decade ends before SAI starts (e.g. 2041–2050 with a 2060 start — reachable
in the dashboard) had a with-SAI field with no years in the window. The old code returned an
all-NaN p-value map (no hatching), an empty with-SAI histogram (NaN counts; an `IndexError`
for icefrac) and a NaN decade-mean delta map; the new p-value code dropped the decade and
raised. Before SAI starts the with-SAI world *is* the no-SAI world, so the delta is now zero
over the whole projection (icefrac's with-SAI field is its no-SAI field), the histogram is
fed the full projection map (it slices the decade itself; identical for every post-start
decade), and an empty test window yields an explicit NaN decade. Verified: for tas and
icefrac, the 2041–2050 / start-2060 p-values, histogram and delta map equal the no-SAI ones
exactly, and post-start requests are unchanged (max |Δ| 6e-14 vs the harness dumps).

### Production cache — regenerated and verified

Eight scenarios × 470 files (29 no-SAI + 21 targets × 10 variables × 2 + 21 SO₂) plus 20
historical files = **3,781 files, 59 GB, stamped v1.5.0**, zero errors in any log; ~3 minutes
per scenario, **nine minutes wall-clock with four in parallel** (FaIR is served from its
pickles). Production-path check: the default deployment for tas is served in 1.0 s and its
with-SAI p-values match the harness dump to 6e-14; a non-default start year is computed live
in 0.3 s.

## Histograms

### State (`get_outputs.py:_compute_distribution`, `launch_gradio.py` / `variable_launch_gradio.py`)

| # | Finding | Decision |
|---|---|---|
| H1 | Every grid cell counts once on a regular lat/lon grid, so high latitudes are over-represented: **land north of 60 °N is 45.9% of land cells but 19.7% of land area.** | **Keep** (bars = share of grid cells); y-axis relabelled "Fraction of grid cells". |
| H2 | Default-branch plotting passes `bins[:-1]` as both data and edges (`launch_gradio.py:367-369`, same in the other launcher): N counts into N−1 bars, the last two bins merge. | **Leave**, documented. |
| H3 | Exposure bins `[1, 5, 10, 20, 50, ∞]`, unequal width, open top, plotted as fractions per bin. | **Keep** as designed (ordered categories). |
| H4 | The distribution shown is of the 2-model **mean** field (narrower than either model's). | **Keep** the mean field. |
| H5 | icefrac series restricted to cells with historical ice > 0. | **Keep**. |
| H6 | `np.concatenate([..., pdf_historical])` for the default bins would fail if a variable had no historical series (unreachable today). | Documented. |

### What area weighting would have changed (measured, 2091–2100 no-SAI, combined)

| | tas: share of land > +4 K | tas 90th pct | tasmax > +4 K | exposure_above_35 90th pct |
|---|---|---|---|---|
| counting cells (current) | 20.8% | 7.68 K | 14.1% | 5.5 M person-days |
| weighting by cell area | 11.4% | 4.14 K | 13.9% | 15.5 M person-days |

A 70 °N cell has 0.34× the area of an equatorial one. For tas the equal-cell histogram roughly
doubles the apparent prevalence of extreme (Arctic) warming; for tasmax it hardly matters.
Between-model differences on the equal-cell histograms (Wasserstein-1 vs `cesm-only`, tas):
no-SAI `miroc-only` 0.854 K, `combined` 0.371 K.

## Cross-cutting

| # | Finding | Status |
|---|---|---|
| X1 | icefrac historical reference was CESM-only (comment predating the per-model sigmoids). | **Fixed** (`aeb8549`): mean of both models; 1985–2014 extent 5.55 → 5.81 M km² (NSIDC ≈ 6.5). |
| X2 | `scripts/fit_delta.py` docstring references a `model` dim the pooled fit no longer writes; `backend/delta.py:56-62` carries matching dead code. | **Fixed** (`c03b3f8`). |
| X3 | `output_data["geotiff_paths"]` / `["image_paths"]` are *assigned* per variable in the loop (`get_outputs.py`), so a multi-variable call keeps only the last variable's paths. The API calls one variable at a time, so it is latent. | **Fixed** (`de7060f`): merged. |
| X4 | The S3 verification harness has **no numeric tolerance gate** — it passes on any drift (only crashes and shape changes fail). | Documented; out of scope. |
| X5 | CESM's annual series starts 1851, MIROC's 1850: the "1850–1900" reference is 50 vs 51 years, and the combined rebase drops MIROC's 1850 (≤ 0.07 K at Arctic cells). | Documented; design follow-up. |
| X6 | Artifacts carry latitudes from different sources (`area.nc` vs `correct_lat.npy`) that differ in the last digits; xarray arithmetic between them inner-joins to 125 of 192 latitudes. The backend works in `.values`; the harness pins coordinates. | Documented; a normalization at write time would remove the hazard. |

### Follow-ups recorded, not done

- Absolute variables (exposure counts, icefrac) combine model *levels*; a per-model-anomaly
  combination would separate change disagreement from level bias (U9).
- MIROC p-e pairs P and E from different realizations (inflated σ).
- Reference-window definition across models (X5); coordinate normalization at write time (X6).
- Front-end adoption of the new `band_*` keys; API bump of the vendored backend.
- A numeric tolerance gate for the S3 verification harness (X4).

### X5 — the "1850–1900" reference window is 50 years for CESM and 51 for MIROC

Found while validating the harness. The processed CESM annual series runs **1851–2099**
(248 years); MIROC's runs **1850–2099** (250 years). CESM stamps monthly means at the end of
the month, so its 1850 has only 11 months and is dropped by the yearly aggregation. Effects:

- `cesm-only` rebases on 1851–1900 (50 yr); `miroc-only` on 1850–1900 (51 yr).
- `combined` goes through `_load_multi_model_baseline`'s `xr.align(join="inner")`, which
  drops MIROC's 1850 — so the combined rebase is **not** the mean of the two single-model
  rebases. Measured on `tas`: up to **0.069 K** at one Arctic ocean cell (75.9 °N, 41.3 °E),
  constant in time (it is a reference-level shift). The unrebased maps agree to 5.7e-13.
- `scripts/fit_map.py`'s `PREINDUSTRIAL_YEARS = (1850, 1900)` likewise covers different year
  counts per model.

Severity: low (single Arctic cells, ~0 in the global mean). Design question for later: pin
the reference window to years both models have, or fix the CESM 1850 aggregation upstream.

## Harness

`scripts/compare_runs.py` + `scripts/compare/` (commits `da28906`, plus render). Read-only
w.r.t. `backend/`.

**Configs.** `combined`, `cesm-only`, `miroc-only`. For each, the harness builds a reduced
`model_dir` (symlinks to that config's emulator joblibs, the pooled delta interpolators
unchanged, the icefrac interpolator re-cut on its `model` dim) and, for the duration of a
run, rebinds `REGIONAL_MODEL_NAMES` in `backend.constants`/`baseline`/`historical` and clears
the `lru_cache` loaders. If a rebinding is ever missed the reduced dir makes the loader fail
loudly rather than mix models.

**Two stages.** `dump` persists, once per config, everything `get_outputs` computes before
the audited quantities (projection map, SAI delta, historical series and rebase, per-model
sea-ice predictions) under `harness/<config>/intermediates/<params>/<var>/`; then recomputes
the decadal p-values (all decades, bypassing the production cache), the mean-over-time
series and band, the internal-variability scalar and the histogram under
`harness/<config>/<git-sha>/<params>/<var>/`. `render` / `render_matrix` draw N dumps
against each other — configs at one sha (isolation) or shas of one config (before/after) —
into `figs/<step>/<var>/` with a `summary.json` of every number drawn.

**What the harness cannot isolate.** The SAI delta slope is one pooled fit with no `model`
dim, so it is identical in every config; the harness prints that caveat on every run.
Measured on `tas`: `cesm-only` vs `miroc-only` delta max |Δ| = 1.7e-6 K (float noise);
`combined` differs by 1.3e-3 K, entirely through `temp_diff` (target − projected) moving
with the config's projection — the slope itself does not change.

### Anchor leverage — verified

`python scripts/compare_runs.py leverage`, from the regenerated `data/processed/tas/`:

| Anchor | Model | Cooling (K) | x² share |
|---|---|---|---|
| gauss-1.5 | CESM2-WACCM | −0.869 | 7.3% |
| gauss-1.0 | CESM2-WACCM | −1.440 | 20.1% |
| gauss-0.5 | CESM2-WACCM | −1.840 | 32.9% |
| g6-1.5-miroc | MIROC-ES2H | −0.822 | 6.6% |
| g6-0.5-miroc | MIROC-ES2H | −1.845 | 33.1% |
| **CESM2-WACCM total** | | | **60.4%** |
| **MIROC-ES2H total** | | | **39.6%** |

The `fit_delta.py` docstring's figure is confirmed. The two models' deep anchors cool almost
identically (−1.840 vs −1.845 K), as do their shallow ones.

## Baseline measurements — before any fix

Request: SSP2-4.5, target 1.0 °C, SAI start 2035, ramp-up 10, decade 2091–2100 (the
dashboard defaults). Code at `5c91df4`. Figures: `figs/phase2-baseline/<var>/` (regenerate
with `render_matrix --configs cesm-only,miroc-only,combined --shas 5c91df4 --step phase2-baseline`).

### tas

**Isolation sanity.** Global-mean 2091–2100 anomaly is **2.742 / 2.742 / 2.743 K** for
`cesm-only` / `miroc-only` / `combined` — identical by construction, since `fit_map`
regresses each model on its own global anomaly so every model's map tracks FaIR's global
mean. Configs therefore differ in *pattern*, not in the global series. Historical
1985–2014 global mean: 0.589 / 0.455 / 0.523 K (combined = mean, as expected).

**P-values (P1–P5).** 18,378 valid (land) cells.

| | 2050 | 2060 | 2070 | 2080 | 2090 | 2100 |
|---|---|---|---|---|---|---|
| no-SAI, every config | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| with-SAI, cesm-only | 98.9% | 98.5% | 97.8% | 96.4% | 95.2% | 94.3% |
| with-SAI, miroc-only | 97.8% | 96.8% | 96.0% | 95.1% | 94.2% | 93.6% |
| with-SAI, combined | 99.9% | 99.7% | 99.5% | 99.1% | 98.7% | 98.4% |

- The no-SAI stipple carries **no information** for tas: every land cell is significant in
  every decade. With SAI at a +1.0 °C target, 94–98% still are — expected physically (the
  target is above pre-industrial), but the map cannot distinguish "warmed 1 K" from "warmed 3 K".
- A Benjamini–Hochberg control at q = 0.05 changes the counts by ≤ 0.1% — the p-values are
  so far below threshold that P4 is moot for tas. It will matter for variables with weaker
  signals.
- **The combined mean is "more significant" than either model alone** (98.4% vs 94.3% /
  93.6% in 2100; +884 cells gained, −126 lost vs `cesm-only`). Averaging two patterns
  smooths the field, so fewer cells sit near zero anomaly — while the test's σ does not
  change at all: `model_internal_variability` = **0.119 K in all three configs**, because it
  is the CESM-only file (P3 / U4 made visible).
- No `std == 0` cells (P5) for tas.

**Series and band (U1–U6).** No band from inter-model spread exists for tas today; all three
configs draw the same ±2 × 0.119 K CESM-only scalar. `no_sai` differs across configs by
≤ 0.002 K and `with_sai` by ≤ 3e-4 K (both FaIR-tracked); the historical series differs by
up to 0.20 K (MIROC cooler around 1880–1920).

**Histogram (H1–H4).** Data-driven 0.1 K bins; `combined` spans −2.37 … +15.23 K, the
single-model tails reach ~+22 K. Wasserstein-1 distance from `cesm-only`, in K:

| series | miroc-only | combined |
|---|---|---|
| historical | 0.192 | 0.169 |
| no-SAI | 0.854 | 0.371 |
| with-SAI | 0.655 | 0.275 |

CESM shows a secondary no-SAI peak near +8 K; MIROC is broader with a long warm tail. Both
tails are Arctic cells counted with equal weight (H1); the MIROC tail is a genuine feature of
that model's Arctic warming (an earlier skin-temperature explanation was wrong; see Data
provenance). The `combined` distribution is the distribution of the mean field and sits
between the two (H4).

## Round 3 (2026-09-28 to 2026-10-01): sea-ice rebuild, cache fixes, merge with main, UKESM, Welch test, replaceable band

### MIROC sea ice: the tripolar seam (store rebuild 2026-09-22)

MIROC-ES2H's sea ice lives on its ocean model's tripolar grid but was labelled with 1-D
lat/lon arrays, so the regrid treated it as a regular grid. The result was exact-zero
"pizza-slice" strips along two antipodal meridians (≈59°E / 239°E) north of 84°N in the
`baseline`, `ssp245` and `G6-1.5K-SAI` stores (62 always-zero cells north of 84°N, zero even in
March), and the seam carried into the processed files, the MIROC sigmoid (`b`, `x0`) and the
SAI response. All five MIROC `ICEFRAC` arrays were rebuilt with conservative binning on the
true 2-D vertices. After reprocessing and refitting: processed NH September ice area
`baseline` 2020s 3.88 → 4.20 M km², `G6-1.5` 2050-69 3.21 → 3.48, `ssp245` and `G6-0.5`
(already conservatively built) within 0.04; CESM bit-identical. Figures (local):
`figs/miroc-seaice/` (polar maps of the stores, the seam cell by cell) and
`figs/seaice-rebuild/` (processed maps, SAI response, MIROC sigmoid parameters and area series,
before / after).

### Three cache bugs, fixed by building the cache through the live path (`9c0ff86`, `842ee14`)

| Bug | Found by | Effect at the default deployment |
|---|---|---|
| With-SAI p-values for exposure counts added the 1850-1900 level to already-absolute maps | John (review) | `exposure_above_10` 65% of land significant vs 4-8% live |
| Sea-ice cache used the shared-temperature sigmoid, not the end-to-end one | John (review) | cached shading disagreed with the map drawn |
| `cache.py` overwrote the unramped delta file with the ramped, start-sliced delta (regression from `9004185`) | this review | first 10 SAI years ramped twice: 0.01× instead of 0.1× in 2035, 0.36× instead of 0.6× in 2040 (tas with SAI 2040: 1.44 → 1.28 K) |

`_with_sai_comparison_field` is now the single definition of the field the test compares, and
`cache.py` calls `_load_variable_fields` / `_compute_regional_delta` rather than re-implementing
them. `tests/integration/test_cache_matches_live.py` checks cached = live p-values for an
exposure variable, icefrac and tas, and that the delta on disk is unramped.

### Merge with main: #53, #54, #55 (`ddf4217`, `f0f360b`)

`main` squash-merged `feature/add-miroc-inputs` (#51), so GitHub saw 26 conflicting files; the
squash commit is tree-identical to this branch's fork point, and recording it as merged
(`-s ours`) left 7 real conflicts. Resolutions: the detectability test kept, with #53's
tagged cache paths; #53's `significance_reference` (1995-2014) and `display_offset` added to the
variable fields, and every p-value (live and cached) now tests against 1995-2014; #55's
per-variable roster (`regional_models_for_var`) used for the historical references; one
v1.5.0 note. #54 (FaIR from the `fair==2.1.3` package) merged without conflict.

### UKESM1-1-LL as the third model

UKESM takes part in **every** variable; its sea ice and p-e come from one member (r3, the only
one with historical sea ice and evaporation), the rest from three. Processed from R2,
emulators fitted, the pooled SAI delta refit over six anchors (leverage CESM 51.1% / MIROC 33.6%
/ UKESM 15.4%, UKESM's G6-1.5 cooling 1.37 K), the sea-ice sigmoids refit for all three (with
#55's change that keeps ice-free years, which also moves CESM and MIROC). The sea-ice hindcast
behind its σ now pairs each ice member with its own temperature (`05425ad`; UKESM's single ice
member against three tas members had failed to broadcast). Per-model σ, extent σ in M km²:

| | tas (K) | tasmax (K) | p-e (mm/day) | icefrac extent |
|---|---|---|---|---|
| CESM2-WACCM | 0.119 | 0.161 | 0.0015 | 0.583 |
| MIROC-ES2H | 0.181 | 0.262 | 0.0200 | 0.308 |
| UKESM1-1-LL | 0.146 | 0.184 | 0.0018 | 0.303 (one member) |

1985-2014 September extent: CESM 5.55, MIROC 6.11, UKESM 4.89, three-model mean 5.52 M km²
(NSIDC 6.08).

### Significance: Welch two-sample t-test against 1995-2014 (decision 2026-09-30, `fc1f61a`, `bb03111`)

The old dashboard test was a two-sample t with a supplied CESM σ (2025-2039, detrended members,
45 samples) and n = 45 for both means. This branch first replaced it with a known-σ z-test of the
decade mean; it now tests the **difference of the decade mean and the 1995-2014 mean** with each
side's own noise:

    t = Δ / sqrt(v_d + v_r),  df = (v_d + v_r)² / (v_d²/dof_d + v_r²/dof_r)

* decade side — the emulated decade is smooth (no samples), so its noise is one realization's,
  `c_ρ σ²/k`, σ pooled over models from 2025-2039 member residuals; dof = the samples behind σ;
* reference side — measured from the members over 1995-2014 (each detrended within the window),
  `(1/M²) Σ_m c_ρ,m s²_m / (n N_m)` (new artifact `grid_level_reference_variability_<model>.nc`).

The 1995-2014 member scatter matches the 2025-2039 σ (tas median 0.542 vs 0.544 K with two
models; 0.539 vs 0.547 with three, Susanne's check), the reference adds about 3% to the noise
(median 2.7%) and the Welch dof is ≈ 147 with three models (≈ 91 with two). Only members with
data in a window are counted (`579fe82`, review on #56): CESM's tasmax / tasmin files carry
three historical-only and three projection-only members, which had halved CESM's reference
scatter by √2; corrected, it is 1.01 × its σ. Share of cells significant, three models,
SSP2-4.5 / +1.0 °C at the default deployment (start 2035, ramp-up 10) (A = z-test vs
1850-1900, this branch before the merge; B = z-test vs 1995-2014, #53 only; C = Welch vs
1995-2014, now):

| Variable | World | Decade | A | B | C |
|---|---|---|---|---|---|
| tas | no SAI | 2091-2100 | 100.0% | 100.0% | 100.0% |
| tas | with SAI | 2041-2050 | 99.4% | 42.9% | 39.2% |
| tas | with SAI | 2091-2100 | 97.7% | 46.1% | 43.0% |
| tasmax | no SAI | 2041-2050 | 95.3% | 66.7% | 63.8% |
| tasmax | with SAI | 2091-2100 | 35.0% | 3.8% | 3.4% |
| p-e | no SAI | 2091-2100 | 42.0% | 36.8% | 35.3% |
| p-e | with SAI | 2041-2050 | 23.6% | 4.7% | 3.7% |
| p-e | with SAI | 2091-2100 | 26.8% | 19.0% | 17.2% |
| exposure_above_35 | with SAI | 2041-2050 | 20.3% | 12.0% | 11.5% |
| exposure_above_35 | with SAI | 2091-2100 | 28.7% | 21.3% | 21.0% |

(The 2041-2050 with-SAI rows include the ramp-up years; an earlier version of this table had
them at ramp-up 0.)

The reference move (A → B) dominates: with SAI holding warming near 1 K, much of the land is
no longer distinguishable from today's climate. Welch (B → C) removes a few marginal cells.
Maps: `figs/significance-change/*_3model.png` (local).

### Replaceable band and decade noise for InVERT (decision 2026-09-30, `a7e8332`)

InVERT (#59) will provide both the line-plot band and the per-decade σ. Both are now single,
swappable inputs:

* **Band:** `mean_over_time["band"] = {low, high, label, source}`, per-year offsets added to each
  series. Default: `±2σ_internal(mean)` from the members (`source="members"`). A function
  registered in `get_outputs.BAND_SOURCES[var]` (called with `data_dir`, `regional_map`,
  `spatial_gdf`, `spatial_item`, `time`) replaces it; returning `None` falls back. The API
  forwards `band_low` / `band_high` per series and the front end shades `[value + low,
  value + high]`, so an asymmetric per-year envelope needs no API or front-end change.
* **Decade noise:** `p_values.DECADE_NOISE_SOURCES[var]` replaces `member_decade_noise`; the
  reference noise stays from the members (InVERT does not cover 1995-2014).

InVERT covers tas, tasmin, tasmax and pr (p-e via pr); the exposure counts and sea ice keep the
member-based defaults.

`setup/steps.sh` Step E now runs `scripts/variability.py` for every variable (`05ca7ee`): it is
the only writer of the per-model σ, ρ and reference-scatter files, and a fresh setup that ran it
for icefrac alone left `get_outputs` failing for every variable (review on #56).

### Memory: emulators averaged by coefficients (`29f4d91`)

`get_smip` predicted all 100 bootstrap emulators of every model before averaging: 86 years ×
300 regressions × 55,296 cells ≈ 11 GB plus a same-sized temporary, 25.7 GB per cache scenario,
and four scenarios in parallel froze a 48 GB laptop. The regressors are linear, so each model's
emulators are now averaged by coefficients first: peak 25.7 → 1.8 GB for the emulator step,
6.7 GB for a whole scenario, about 90 s per scenario, maps unchanged to ≤ 3e-12.

### Three-model values on the dashboard (SSP2-4.5, +1.0 °C, start 2035 / ramp 10, 2100)

| Variable | With SAI | No SAI | Band (±2σ internal, mean of 3) |
|---|---|---|---|
| tas (K) | 1.00 | 2.77 | ±0.17 (two models: ±0.22) |
| icefrac (M km²) | 3.27 | 0.38 | ±0.48 |
| exposure_above_10 (person-days) | 186,200 | 191,100 | ±1,400 |

Sea ice with SAI falls from 4.2 (two models) to 3.3 M km²: UKESM's Arctic recovers less under the
pooled SAI delta, and #55's sigmoid change moves the other two.

## Figures (local only, by decision)

No figure is committed or published: `figs/` and `*.png` are gitignored. Every figure this
document cites is regenerated from the harness dumps in seconds — dumps first
(`compare_runs.py dump`, ~1 s per variable for the quantities stage), then:

| Set | What it shows | Command |
|---|---|---|
| `figs/final/<var>/` | **Baseline → final**, every variable: series with legacy vs nested bands, p-value fractions per decade, with-SAI masks + changed cells, histograms | `python scripts/compare_runs.py render_matrix --configs combined --shas baseline,final --step final` |
| `figs/phase2-baseline/<var>/` | Model isolation on the baseline code: `cesm-only` vs `miroc-only` vs `combined` | `... render_matrix --configs cesm-only,miroc-only,combined --shas baseline --step phase2-baseline` |
| `figs/fix-3b/<var>/` | Uncertainty track alone (baseline → `3b-bands`) | `... render_matrix --configs combined --shas baseline,3b-bands --step fix-3b` |
| `figs/fix-4-pvalues/<var>/` | P-value track alone (baseline → `4-pvalues`) | `... render_matrix --configs combined --shas baseline,4-pvalues --step fix-4-pvalues` |
| `figs/p3-sigma-options/` | The three candidate test σ's and their stipple, per variable | `python scripts/compare_runs.py sigma_options` |
| `figs/miroc-seaice/`, `figs/seaice-rebuild/` | MIROC sea-ice seam in the stores; processed fields, SAI response, sigmoid and area before / after the rebuild | ad-hoc scripts, Round 3 |
| `figs/significance-change/` | Cells significant under A / B / C (reference period and Welch), tas and p-e, 2091-2100 | ad-hoc script, Round 3 |

Baseline dumps are the `feature/add-miroc-inputs` code (`--sha baseline`, git HEAD `5c91df4`);
`final` is this branch's HEAD. Each figure directory carries a `summary.json` with every
number drawn.

## Regenerating everything

Order matters (P−E needs `pr`; variability needs the all-members files; the cache stamps the
version, so bump first):

Launch scripts run from `nohup` must call `.venv/bin/python` explicitly — a detached shell
does not inherit the activated environment (this bit once).

```bash
source .venv/bin/activate
python scripts/data_sanity_checks.py                                   # stores reachable
bash scripts/process_all_monthly.sh data/raw data/processed            # both models, 11 vars, ~1 h
bash scripts/fit_all.sh data/processed data/models                     # emulators, deltas, sigmoids
for v in tas pr tasmax tasmin tas_above_35 tas_above_40 tas_below_0 pr_above_10 pr_above_20 p-e icefrac; do
  python scripts/variability.py --var_name $v --data_dir data/processed --model_dir data/models
done                                                                    # per-model sigma, rho, geojson columns
python scripts/cache.py --data_dir data/processed --model_dir data/models --output_dir data/cache --historical_only
for s in SSP1-1.9 SSP1-2.6 SSP2-4.5 SSP3-Baseline SSP4-3.4 SSP4-6.0 SSP5-3.4 SSP5-Baseline; do
  python scripts/cache.py --data_dir data/processed --model_dir data/models --output_dir data/cache --scenario "$s"
done                                                                    # ~3 min each; 4 in parallel -> ~9 min
```

Scenario keys are `REVERSE_FANCY_SSP_TITLES`'s (`SSP3-Baseline`, `SSP5-Baseline`, `SSP5-3.4`
— not "SSP3-7.0" / "SSP5-8.5"). The harness (`scripts/compare_runs.py`) does not need the
production cache; `tests/integration/` pins the numbers above and skips without local data.

Deployment notes: the API vendors this repository and must be bumped to pick up any of this;
the front-end needs no change for the p-values (band-2 mask) and a change to show the new
`band_*` keys (until then it keeps drawing the legacy `model_internal_variability` band).

## Change log (branch `feature/pvalues-uncertainty-histograms`, off `feature/add-miroc-inputs`)

| Commit | Track | What |
|---|---|---|
| `def548a` | setup | `.gitignore` for pipeline outputs and audit dirs; broken archive symlinks removed |
| `1740a16` | setup | README: ESMPy/ESMF version pitfall, lightweight test env |
| `5c91df4` `41d7302` `a5b65a1` `fdc423f` `5e2d3b0` `236ee95` | doc | This review, section by section as numbers landed |
| `da28906` `ff4aa7c` `7554e65` `d3e67c9` `ee577e5` | harness | `scripts/compare_runs.py`: configs, dump, render, sync fix, band rendering, σ what-if |
| `be03341` | 3a | All-members files for every variable's no-SAI runs; per-member P−E |
| `a1625f4` | 3a | `scripts/variability.py` standalone, per model, from regridded members; icefrac unit attr |
| `331c6c3` | 3a | Per-model `variability_<var>_<model>` geojson columns (data) |
| `8dd66ee` | 3b | `backend/uncertainty.py`: prediction + confidence bands (tested) |
| `aeb8549` | 3b | Per-model maps and per-model historical through the backend; X1 |
| `22e27f8` | 3b | Bands wired into `get_outputs` (end-to-end per-model series, smoothed first) |
| `856663f` | 3b | Launchers draw nested bands with labels |
| `f64977d` | 3b | Confidence interval removed; one 2σ prediction band everywhere (see 3b decision) |
| `dec6c38` | 4 | Detectability z-test on pooled per-model σ with autocorrelation; `SIGNIFICANCE_THRESHOLD`; land-mask cache; tests |
| `9004185` | 4 | With-SAI p-values cached for the default deployment only, live otherwise |
| `860ec4d` | 4 | Legacy zero-noise rule kept |
| `50682eb` | 5 | Histogram y-axis labelled "Fraction of grid cells" |
| `c03b3f8` `5fd90b0` | X2 | Pooled-slope contract made explicit in `fit_delta.py` / `delta.py` |
| `de7060f` | X3 | `geotiff_paths` / `image_paths` merged across variables |
| `252b9f5` | release | `SIMULATOR_VERSION` v1.4.0 → v1.5.0 |
| `ae95141` | tests | Real-data integration tests pinning the measurements |
| `d9f341d` | doc | Final status, regeneration recipe, change log |
| `644ceae` | P9 | Pre-start decades are the unmodified projection (p-values, histogram, delta map) + integration test |
| `f64977d` `72935f7` | 3b | Band: confidence interval dropped, then internal variability of the mean only (spread kept as a diagnostic; frozen branch `aa-feature/multi-model-mean-uncertainty-both-internal-and-modelspread`) |
| `9c0ff86` `842ee14` | cache | One with-SAI comparison field; `cache.py` built through the live helpers (exposure rebase, sea-ice end-to-end, double ramp-up) |
| `8cb43c1` | data | Pins for the rebuilt MIROC sea ice |
| `ddf4217` `f0f360b` | merge | `main` (#53 1995-2014 reference, #54 FaIR package, #55 UKESM) |
| `665caf7` | tests | Real-data tests require every roster model's data |
| `fc1f61a` `bb03111` | P | Reference-period scatter artifact; Welch t-test with replaceable decade / reference noise |
| `a7e8332` | U | `mean_over_time["band"]` per-year low/high, `BAND_SOURCES` |
| `05425ad` `153ddac` | UKESM | Sea-ice hindcast member pairing; UKESM region variability columns |
| `29f4d91` | memory | Emulators averaged by coefficients (25.7 → 1.8 GB) |
| `0ea2f60` | tests | Real-data pins for three models |
| `05ca7ee` `579fe82` `8b2de06` | review | Setup Step E over every variable; only members with data in a window counted; docstring |

