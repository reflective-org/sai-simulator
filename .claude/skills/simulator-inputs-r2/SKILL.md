---
name: simulator-inputs-r2
description: How to open, inspect and correctly interpret the SAI simulator's climate-model training data — the `simulator-inputs` Zarr stores on Reflective's public Cloudflare R2 bucket (CESM2-WACCM-MA, MIROC-ES2H, UKESM1-1-LL; historical, baseline/ssp245 and SAI/G6 experiments). Use this whenever someone wants to load, plot, compare, subset or sanity-check CESM, MIROC or UKESM data, mentions the R2 bucket, r2.dev, simulator-inputs, ZARR_STORES_URI, TREFHT/PRECT/QFLX/ICEFRAC, SAI-minus-baseline deltas, or asks "where is the model data" — even if they don't name the bucket. The stores have several traps (CESM's shifted time stamps, MIROC's skin-temperature TREFHT, partial UKESM historical fields, PRECT in m/s) that silently produce wrong numbers if you go in cold.
---

# simulator-inputs Zarr stores on R2

The SAI simulator trains on monthly fields from three Earth-system models. Each
(model, experiment) is one Zarr v3 store on a **public, anonymous** R2 bucket —
no credentials, no `storage_options`:

```
https://pub-d127229e707d485c80965ab41706e316.r2.dev/simulator-inputs/<MODEL>_<EXPERIMENT>.zarr
```

(In the `sai-simulator-internal` repo this is `ZARR_STORES_URI` in `version.py`.)
Writing to the bucket needs R2 credentials and is a producer task — this skill is
for reading.

## Quick start

`scripts/simstores.py` (next to this file) opens a store and normalizes time so
the models line up. Prefer it over raw `xr.open_zarr` unless you specifically need
the raw stamps — the CESM time trap below is easy to miss.

```python
import sys; sys.path.insert(0, "<this skill dir>/scripts")
from simstores import open_store, STORES

ds = open_store("UKESM1-1-LL", "G6-1.5K-SAI")                 # Mon/atmos_2d
mx = open_store("CESM2-WACCM-MA", "baseline", "tasminmax")    # TREFHTMN/MX
ct = open_store("MIROC-ES2H", "ssp245", "derived")            # day counts
```

Requirements: `xarray`, `zarr>=3`, `fsspec`, `aiohttp`, `cftime`, `pandas`.
If the user wants code they can paste without the helper, inline its ~15 lines
of time-fixing rather than dropping them.

Everything opens lazily. A full global field for one store is several GB, so
slice (`.sel(time=...)`, `.isel(member=0)`, a lat/lon box) **before**
`.compute()`/`.load()`.

## Inventory (verified against the live bucket, 2026-10-02)

| Store | Members | Years | Grid (lat×lon) |
|---|---|---|---|
| `CESM2-WACCM-MA_historical` | 001 002 003 | 1850–2014 | 192×288 |
| `CESM2-WACCM-MA_baseline` | 001 002 003 | 2015–2099 | 192×288 |
| `CESM2-WACCM-MA_1.5K-SAI`, `_1.0K-SAI` | 001 002 003 | 2035–2069 | 192×288 |
| `CESM2-WACCM-MA_0.5K-SAI` | 001 002 003 | 2035–2070 | 192×288 |
| `MIROC-ES2H_historical` | r01 r02 r03 | 1850–2014 | 128×256 |
| `MIROC-ES2H_baseline` | r01 r02 r03 | **2020**–2084 | 128×256 |
| `MIROC-ES2H_ssp245` | r01 r02 r03 | 2015–2100 | 128×256 |
| `MIROC-ES2H_G6-1.5K-SAI`, `_G6-0.5K-SAI` | r01 r02 r03 | 2035–2084 | 128×256 |
| `UKESM1-1-LL_historical` | r2 r3 r12 | 1850–2014 | 144×192 |
| `UKESM1-1-LL_baseline` | r2 r3 r12 | 2015–2100 | 144×192 |
| `UKESM1-1-LL_G6-1.5K-SAI` | r2 r3 r12 | **2034**–2084 | 144×192 |

Years are after CESM's time fix. Note the SAI naming: CESM has no `G6-` prefix.
The same prefix also holds `InVERT_50yrs_SSP370_100members.zarr`, a different
product (a 100-member internal-variability emulator), not a model run. If the user asks about a store not in this table, probe it
(`curl -s <url>/zarr.json | head`) rather than assuming it doesn't exist; the
table can go stale.

**Experiments:** `historical` = CMIP6 historical. `baseline` = the no-SAI run
the SAI runs branch from (SSP2-4.5 forcing; for CESM it is also the SSP2-4.5
projection). `ssp245` (MIROC only) = the ScenarioMIP SSP2-4.5 run — a
*different realization* from MIROC `baseline`. SAI runs (`*K-SAI`) hold global
temperature at roughly that many K above pre-industrial (CESM GAUSS / GeoMIP G6).

## Groups and variables

Each store has three groups, all dims `(member, time, lat, lon)` on the model's
**native** grid:

| Group | Variables | Units |
|---|---|---|
| `Mon/atmos_2d` | `TREFHT` (near-surface temperature) | K |
| | `PRECT` (precipitation) | **m/s** — ×8.64e7 for mm/day |
| | `QFLX` (evaporation) | kg m⁻² s⁻¹ — ×86400 for mm/day |
| | `ICEFRAC` (sea-ice fraction) | 0–1, land = 0 |
| `Mon/atmos_2d_tasminmax` | `TREFHTMN`, `TREFHTMX` | K |
| `Mon/atmos_2d_derived` | `TREFHT_above_35`, `TREFHT_above_40`, `TREFHT_below_0`, `PRECT_above_10`, `PRECT_above_20` | days in month |

- `units` attrs are **missing** on most CESM and MIROC arrays — don't infer
  units from attrs; use the table. (PRECT really is m/s: values ~3e-8.)
- `TREFHTMN`/`TREFHTMX` are the **absolute monthly extremes** of daily min/max
  (coldest night / hottest day in the month), not monthly means of daily
  extremes. Don't compare them to CMIP `tasmin`/`tasmax` Amon.
- Day counts use **daily-mean** TREFHT (>35 °C, >40 °C, <0 °C) and daily PRECT
  (>10, >20 mm/day). Dtype is int64, except CESM SAI stores (float64).
- Read each array's `note` attr — it is the producer's provenance record
  (source, conversion, caveats) and is the authority when it disagrees with
  this file. `"PLACEHOLDER"` in a note = deliberately all-NaN; `"PARTIAL"` =
  real for some members, NaN for others.

## Traps — check these before reporting a number

1. **CESM time stamps are shifted one month.** CESM labels each month with the
   first day of the *next* month (Jan 2015 → `2015-02-01`, Dec 2099 →
   `2100-01-01`). Verified: raw CESM Arctic sea ice bottoms out on the
   "October" label. With raw stamps, `groupby("time.year")` gives Feb–Jan years
   and "September" is really August. **This is per group, not per model.**
   Shifted: `Mon/atmos_2d` (every experiment) and `Mon/atmos_2d_tasminmax`
   (baseline and SAI), which are raw CAM history and still carry CAM's
   `datesec` variable. Correct as stored: `Mon/atmos_2d_derived` (every
   experiment) and historical `tasminmax` (mid-month stamps). Verified
   2026-10-02 by lagged anomaly correlation against TREFHT. `simstores.open_store`
   applies the same rule as `scripts/process_monthly_gauss.py` (CAM `datesec`
   present, every stamp the 1st at 00:00, and the series running Feb→Jan — a
   store corrected at the source would run Jan→Dec and is not shifted again).
   If you use raw `xr.open_zarr`, shift only those groups back one month yourself.
   CESM baseline `derived` also ends in a `2100-01` entry that holds a single
   day (its count maxes at 1); drop it rather than treating it as a month.

2. **Calendars differ across stores.** CESM/most MIROC: proleptic_gregorian.
   MIROC `G6-1.5K-SAI`: noleap. UKESM: 360_day (cftime objects). Raw time coords
   from different models won't align or subtract. Align on year/month (the
   helper rebuilds time as first-of-month datetime64 for exactly this).

3. **MIROC TREFHT is not one quantity.** In MIROC `historical` and `ssp245`,
   TREFHT is true 2 m air temperature. In MIROC `baseline`, `G6-1.5K-SAI` and
   `G6-0.5K-SAI` it is **surface skin temperature** (JAMSTEC's GeoMIP "tas" is
   SurfT). The same goes for TREFHTMN/MX and the TREFHT day counts. So:
   - SAI − baseline within MIROC GeoMIP stores is self-consistent (both skin).
   - Splicing `historical` onto `baseline` creates an artificial ~0.4 K step.
     Splice onto `ssp245` instead.
   - MIROC G6 − MIROC ssp245 mixes the two quantities — don't.

4. **UKESM historical QFLX and ICEFRAC are PARTIAL:** real for member `r3` only,
   all-NaN for `r2` and `r12`. A member mean with `skipna=True` silently becomes
   r3-only; a plain mean becomes NaN. Use `.sel(member="r3")` deliberately and
   say so.

5. **Different models use different member labels** (`001`, `r01`, `r2`) and
   different grids. To combine models, average over members within each model
   first, then regrid (e.g. xESMF) or compare area-weighted regional/global
   means. Never `xr.concat` models along `member`.

6. **Experiments cover different years.** SAI−baseline deltas only work on years
   both stores have. Check overlap before choosing a window: CESM SAI ends
   2069/2070, MIROC `baseline` starts 2020, UKESM SAI starts 2034. MIROC
   `ssp245` QFLX is NaN for 2015–2019 and 2085–2100. It was filled from the
   `baseline` run, a different realization, so P−E there pairs precipitation and
   evaporation from different runs.

7. **Area-weight everything spatial.** These are lat-lon grids:
   `da.weighted(np.cos(np.deg2rad(da.lat))).mean(("lat", "lon"))`. For land-only
   P−E or precipitation statistics, say it's a land mean. The global-mean P−E is
   ≈0 by water balance, so a large "global P−E" means something is mislabeled.

8. **ICEFRAC is a fraction per cell, not an area.** For Arctic sea-ice extent or
   area, multiply by cell area and sum over lat > 0. On these regular grids,
   cell area ≈ R²·Δλ·Δφ·cos(φ) with R = 6.371e6 m and Δλ, Δφ in radians. The simulator's sea-ice output is NH
   **September** (the annual minimum), not an annual mean. Compare like with
   like.

## Worked example: SAI cooling by model

```python
import numpy as np
from simstores import open_store

PAIRS = {  # (no-SAI, SAI) pairs that are internally consistent
    "CESM2-WACCM-MA": ("baseline", "1.5K-SAI"),
    "MIROC-ES2H":     ("baseline", "G6-1.5K-SAI"),   # both skin temp — consistent
    "UKESM1-1-LL":    ("baseline", "G6-1.5K-SAI"),
}
WINDOW = slice("2050", "2069")  # inside every pair's overlap

def gmean(da):
    w = np.cos(np.deg2rad(da.lat))
    return da.weighted(w).mean(("lat", "lon"))

for model, (ref, sai) in PAIRS.items():
    a = gmean(open_store(model, ref).TREFHT.sel(time=WINDOW)).mean()
    b = gmean(open_store(model, sai).TREFHT.sel(time=WINDOW)).mean()
    print(model, f"{float((b - a).compute()):+.2f} K")
```

Expected output (verified 2026-10-02, all members): CESM −0.87 K, MIROC
−0.82 K, UKESM −1.37 K. This makes a good check that you're reading the stores
correctly. It pulls about 1–2 GB per store. For a quick look, add `.isel(member=0)` or
coarsen the window.

## Answering well

- When you give a number, say which store, members, years and variable
  definition it came from (for example "UKESM r3 only" or "MIROC skin
  temperature"). Most mistakes with these stores are mislabeling, not bad
  arithmetic.
- Start with metadata: open the store and check `ds.member`, the time range and
  the `note` attrs before writing analysis code.
- If a result looks physically odd (cooling across a historical→baseline seam,
  sea ice minimum in August or October, PRECT of 1e-8 "mm/day"), suspect one of
  the traps above before suspecting the model.
