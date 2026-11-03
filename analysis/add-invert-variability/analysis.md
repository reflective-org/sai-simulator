# Add InVERT internal variability to the emulated tas, tasmin, tasmax and pr responses

## Scope (decided 2026-09-29)

For now InVERT variability feeds **only the dashboard's global annual-mean line plot**, as
an envelope across the 20 realized members around the historical, no-SAI and
with-SAI series. The gridded ensemble (`get_regional_map_with_variability`) stays in the
backend for the future climate-velocity work, but the spatial questions raised below
(negative precipitation, historical smoothing, the New Guinea variance hotspot) are
**deferred** and not resolved on this branch. Regional (per-country) envelopes are also
deferred: in regional mode the dashboard keeps the legacy scalar band.

**Update 2026-09-30:** regional envelopes are now in scope. When a region is selected, the
envelope is computed from the 20 gridded members averaged over that region (see
"Dashboard integration" below), not from the global mean. The spatial questions above stay
deferred. Also decided on 2026-09-30: the envelope is the full range of the 20 members
(lowest to highest) for every variable instead of 5-95 %, and each member's yearly tasmin /
tasmax is its coldest / warmest month instead of the 12-month mean (see
"tasmin/tasmax: coldest / warmest month" below).

## Branch base (rebased 2026-09-29)

The InVERT commits now sit on top of `feature/add-ukesm-pt1` (three regional models,
AR6-constrained FaIR, stamped cache dirs). The branch was originally built on
`feature/current-baseline`; the three commits from that branch (cooling slope, fixed
cooling ramp, 1995-2014 display baseline) are **not** carried here and live on their own
branch. Consequences for this document:

- All series and figures are anomalies vs 1850-1900 (the UKESM base has no 1995-2014
  display frame).
- `get_regional_map` refuses an unstamped cache dir, so the scripts must be pointed at
  a stamped one, e.g. `--model-dir data/models_pr55_3model --cache-dir data/cache_pr55_3model`.
- The smooth mean under the envelope is the three-model mean, not CESM2-WACCM alone.
- On the UKESM base the dashboard also carries an icefrac-only multi-model spread band;
  the InVERT envelope is drawn for its four variables as the next branch of that
  `if/elif` in both Gradio launchers.

## Motivation

The simulator's emulated `tas` and `pr` fields are **smooth deterministic means**: the FaIR->SMIP
regression pipeline in `backend/baseline.py` produces a single `(time, lat, lon)`
trajectory per SSP (yearly, 2015-2100), and internal variability is only reported as a
scalar +/-sigma error-band (`_compute_internal_variability` in
`backend/get_outputs.py`). Realized noise is never added.

For climate-velocity analysis (dT/dt divided by |grad T|) this smoothness is a
problem: the numerator dT/dt on a noiseless signal is essentially the forced trend
alone, and the velocity map ends up unphysically clean. Real climate velocity work needs
many realistic realizations so that the year-to-year noise and the resulting velocity
uncertainty can be assessed. **This branch prepares that ingredient.** Climate velocity
itself is out of scope -- it will be a follow-up branch that consumes the ensemble
built here.

## Data source

Four InVERT netCDFs (4.4 GB each, float32), one per variable. The originals sit zipped in
the team Google Drive. They are published, unchanged, as the four variables of one public
Zarr store on R2, `simulator-inputs/InVERT_50yrs_SSP370_100members.zarr` under
`version.ZARR_STORES_URI` (`INVERT_STORE`), written and verified bit for bit by
`scripts/zarrify_invert.py`. Each store variable records the netCDF it came from
(`source_file`, `source_size_bytes`, `source_sha256`). The backend reads the store
anonymously; a netCDF under `<data_dir>/invert/`, if present, takes precedence.

| simulator var | file | InVERT var | raw units | scale applied |
|---|---|---|---|---|
| `tas` | `InVERT_TREFHT_50yrs_SSP370_100members.nc` | `TREFHT` | K | 1 |
| `tasmin` | `InVERT_TREFHTMN_50yrs_SSP370_100members_float32.nc` | `TREFHTMN` | K | 1 |
| `tasmax` | `InVERT_TREFHTMX_50yrs_SSP370_100members_float32.nc` | `TREFHTMX` | K | 1 |
| `pr` | `InVERT_PRECT_50yrs_SSP370_100members_float32.nc` | `PRECT` | m/s (raw CESM) | 86400 x 1000 -> mm/day |

No file carries a `units` attribute. The monthly member std is 1.4 K (TREFHT), 1.6 K
(TREFHTMN, TREFHTMX). The PRECT scale was inferred from the data
(member std ~1.6e-8 m/s, i.e. ~1.4 mm/day monthly), which is the CESM `PRECT`
convention. Both files share the same dimensions:

- `ensemble = 100` initial-condition members
- `time = 600` monthly steps (50 years)
- `lat = 96`, `lon = 192` (coarse CESM F09-like Gaussian grid, ~1.9 deg lat x 1.875 deg lon)
- values are anomalies (already detrended internal variability; the SSP3-7.0 label on
  the filename is informational -- the forced signal has been removed upstream, so the
  residual is **scenario-agnostic**. The same 20-member ensemble is added on top of the
  smooth mean regardless of which SSP the simulator is projecting).

The per-variable mapping lives in `INVERT_VARIABLES` in `backend/invert_variability.py`.

The emulator grid is finer (192 x 288, ~0.94 deg lat x 1.25 deg lon), so InVERT is
regridded up to the emulator resolution before being combined.

## Stitching scheme

The simulator's user-facing timeframe is **1851-2100 (250 yrs)**, built by
stitching the historical CESM ensemble-mean field (1851-2014, from
`output_gauss-baseline.nc`) with the FaIR->SMIP emulator output (2015-2100). InVERT
provides only 50 yrs per member, so no single member can cover the full timeline.

We tile 5 InVERT members end-to-end per output ensemble member:

  output_member[i] = InVERT[5i] || InVERT[5i+1] || InVERT[5i+2] || InVERT[5i+3] || InVERT[5i+4]

for `i = 0..19`. This gives **20 output ensemble members**, each 250 yrs long
(1851-2100), and every one of the 100 InVERT members is used exactly once.

Each output member has 4 seam years (at 1901, 1951, 2001, 2051) where two adjacent
50-year blocks come from different InVERT members. Temporal autocorrelation is broken
at these seams -- adjacent noise years become uncorrelated where they wouldn't be
within a single member. The forced signal is completely unaffected (the mean field is
untouched), and the ensemble spread at any given year matches InVERT's stationary
variability. For most downstream diagnostics -- and specifically for climate velocity
computed over 10-30 year windows -- the seam effect is negligible as long as no window
straddles a seam year at exactly the wrong offset.

## No double-counting concern

InVERT is added on top of the smooth mean, which includes the historical CESM segment.
That segment is *not* a single realization -- it is the CESM ensemble-mean (verified:
year-to-year global-mean |dT| is ~0.07 K, compared to ~0.15 K for a single member).
The residual variability baked into the historical field is small compared to the
InVERT amplitude we add on top, so double-counting is negligible in practice.

If a future validation reveals visible residual noise on the historical segment, a
20-year rolling mean or LOWESS smoother can be applied to the historical field before
combining -- this is a one-line change in `get_regional_map_with_variability`.

## Regridding

InVERT is bilinearly interpolated from its native 96 x 192 grid to the emulator's
192 x 288 grid via `xarray.DataArray.interp(method="linear")`. Before interpolation
the source array is padded:

- Longitude gains a synthetic column at `lon=360` copied from `lon=0` (periodic wrap),
  so target longitudes just below 360 do not extrapolate.
- Latitude gains synthetic rows at `lat=+/-90` copied from the pole-adjacent source
  rows, so the emulator's target latitudes (which reach +/-89.76) do not extrapolate
  beyond the source range (+/-88.57).

xesmf is not a repo dependency, and `xarray.interp` is sufficient here because both
grids are regular rectilinear -- no unstructured or curvilinear source geometry to
worry about.

## Public API

The one entry point downstream code should use is:

```python
from backend.baseline import get_regional_map_with_variability

mean, ensemble = get_regional_map_with_variability(
    "tas",  # or "pr"
    data_dir=..., model_dir=..., cache_dir=..., ssp_scenario="SSP2-4.5",
)
# mean:     dims (time=250, lat=192, lon=288), K (tas) or mm/day (pr) vs 1850-1900
# ensemble: dims (ensemble=20, time=250, lat=192, lon=288), same units
```

`get_regional_tas_with_variability(...)` remains as a tas-only alias. For `pr` the
emulator mean is land-only (NaN over ocean, via `clip_to_land`), so the ensemble is
land-only too.

## Dashboard integration (global and regional line plots)

**Merged with `main` on 2026-10-02 (after #56 and #57).** `main` now draws a single
`mean_over_time["band"]` around the three series, filled by `_line_band` in
`backend/get_outputs.py`: a per-variable entry in `BAND_SOURCES` if one exists, else the
members' ±2σ internal variability of the multi-model mean. This branch registers
`_invert_band` for tas, tasmin, tasmax, pr and p-e, so the band for those variables is the
InVERT members' range (also on the custom-injection / CIDER path, which goes through the
same call), and every other variable keeps the members' band. Both are labelled
"Natural variability" in the legend, with the method in parentheses. The InVERT source
returns `None` when no valid cache exists, which hands the band back to the members, so a
deployment without the InVERT caches is unchanged from `main`. Where this document still
mentions `mean_over_time["invert_variability"]`, read `mean_over_time["band"]` with
`source == "invert"`: the key was replaced by the band in the merge.

`get_outputs` now adds, in global and in regional mode,, for every variable in `INVERT_VARIABLES`:

```python
output_data[var]["mean_over_time"]["invert_variability"] = {
    "low":  <DataArray, time 1851-2100>,   # lowest of the 20 members
    "high": <DataArray, time 1851-2100>,   # highest of the 20 members
}
```

Both are **offsets** (anomalies around zero, in the variable's units) to add to each of
`historical_model`, `no_sai` and `with_sai`. They come from
`backend.invert_variability.get_global_mean_variability`, which takes the no-SAI regional
map as a template so the InVERT global mean uses the same grid and the same NaN mask
(land-only for pr) as the dashboard's own aggregation. The `(20, 250)` global-mean
ensemble is cached to `<cache_dir>/invert/invert_global_mean_<var>_1851_2100.nc`, so
after the one-time offline build the extra cost per `get_outputs` call is negligible.

In regional mode the offsets come from
`backend.invert_variability.get_regional_mean_variability` instead. It takes the same
template plus the region mask from `create_mask`, drops cells outside the region or NaN
in the template (ocean for pr), and takes the `cos(lat)`-weighted mean of each member, so
the envelope is aggregated exactly like the regional line it surrounds. It reads only the
latitude rows the region touches from the gridded-ensemble cache
(`invert_annual_ensemble_<var>_1851_2100.nc`), which takes about a second, so nothing is
cached per region.

The key is only added when a **valid pre-built cache** is present under
`<cache_dir>/invert/`: the global-mean cache in global mode, the gridded-ensemble cache in
regional mode. On a machine without one, `get_outputs` output and the dashboard are
exactly as on the UKESM branch. The request path never builds a cache (see "Cache
layout" below). The legacy scalar `model_internal_variability` is still
returned unchanged. Both Gradio launchers draw the InVERT envelope when the key is present
and fall back to the scalar +/-2 sigma band otherwise. Note that `launch_gradio.py` does not draw a
global line plot for `pr` at all in global mode; the pr envelope is still produced for
the API.

Full range rather than percentiles (changed 2026-09-30): the envelope was first the 5-95 %
range, chosen because with 20 members the min/max is a single draw and jumps around year to
year, while the 5-95 % range interpolates between the two most extreme members. It is now
the full range of the 20 members, so every realized member lies inside it. With 20 members
the expected coverage of the full range is about 90 % (19/21), so it is only slightly wider
than 5-95 %, and noisier from year to year.

Before/after figures: `plots/dashboard/<var>_global_mean_line.png`, generated by
`plot_dashboard_lines.py`, which runs the real `get_outputs` pipeline (SSP2-4.5, 1.5 C
target, SAI from 2035):

```
python analysis/add-invert-variability/plot_dashboard_lines.py \
    --model-dir data/models_pr55_3model --cache-dir data/cache_pr55_3model
```

Envelope half-width (mean over years of (high - low) / 2) against the legacy band:

| var | InVERT full-range half-width | InVERT centre | legacy sigma | legacy band half-width (2 sigma) |
|---|---|---|---|---|
| tas | 0.235 K | +0.01 K | 0.119 K | 0.238 K |
| tasmin | 0.337 K | -0.03 K | 0.184 K | 0.368 K |
| tasmax | 0.305 K | +0.04 K | 0.161 K | 0.321 K |
| pr | 0.118 mm/day | | 0.013 mm/day | 0.027 mm/day |

Numbers as of 2026-09-30 (full range, centred coldest / warmest month for tasmin / tasmax). The
centre is the mean over years of (high + low) / 2, i.e. how far the envelope sits from the
line.

The legacy scalar (`scripts/process_monthly_gauss.py`) is the std of the linearly
detrended **ocean + land** global mean of the GAUSS baseline members over 2025-2099.
For the temperature variables that is the same quantity InVERT emulates, and the widths
agree to within ~10 %. For pr the legacy scalar averages over the ocean while the
dashboard's pr series is land-only, so it under-sizes the band by a factor of ~7; the
InVERT envelope is computed with the same land mask as the series and is the consistent
one. Consequence visible in `plots/dashboard/pr_global_mean_line.png`: the realized
year-to-year variability of land-mean precipitation is larger than the SAI-vs-no-SAI
separation for most of the century.

The InVERT band is jagged because it is realized noise (a different draw each year),
whereas the legacy band was a constant width around a smoothed curve.

The ensemble is the smooth mean plus 20 realized InVERT variability trajectories.
`tas_ensemble.isel(ensemble=0)` has dims `(time, lat, lon)` with integer yearly time
coord -- i.e. it matches the `compute_climate_speed(tas)` contract in
`backend/climate_velocity.py` exactly.

`get_outputs.py` is **not** modified: exposing the ensemble there would trigger a
~1-minute first-run cost on every simulator run (one-time cache build), even for users
who never need variability. The future climate-velocity module can import
`get_regional_map_with_variability` directly.

## Cache layout (changed 2026-10-01 after PR review)

Each 20-member x 250-year ensemble is cached to
`<cache_dir>/invert/invert_annual_ensemble_<var>[_min|_max]_1851_2100.nc` (~1.1 GB
float32) and its global mean to `<cache_dir>/invert/invert_global_mean_<var>..._1851_2100.nc`
(~50 KB). Three rules, each answering a point from the review of PR #59:

1. **Atomic writes.** Files are written through `backend.utils.save_processed_data`
   (temp file in the same directory, then rename), so a reader never sees a half-written
   cache after an interrupted build or a concurrent one. (Reading a region's latitude
   rows from the zlib-compressed gridded cache still takes about a second.)
2. **Self-describing caches, checked on every read.** Every cache carries
   `invert_cache_version`, the source file name and size, the scale and annual rule, the
   target grid shape and a hash of its coordinates, and (global-mean cache) a hash of the
   land mask it was averaged over. A cache whose attributes do not match what the code
   expects now is treated as missing. Bumping `INVERT_CACHE_VERSION` therefore retires
   every old cache without touching the simulator's own `cache_meta.json` fingerprint,
   and a cache built for a different grid or mask can never be served by mistake.
3. **Never built inside a request.** `get_outputs` goes through
   `get_area_mean_variability`, which only reads valid caches and returns `None`
   otherwise (legacy band). Caches are built once, offline:

   ```
   python -m backend.invert_variability --data-dir data/processed --cache-dir <cache_dir>
   ```

   `setup.sh` runs this as step F2 right after the regular cache step. It reads the R2
   store (or local netCDFs), and skips, with the reason, any variable whose source cannot
   be read. The build uses the emulator grid from
   `correct_lat.npy` and, for land-only variables, the same `clip_to_land` mask the
   dashboard applies, so the mask hash matches at request time. The analysis scripts and
   `get_regional_map_with_variability` still build on demand (`build=True` paths), since
   they are not run inside a web worker.

The InVERT source is the R2 store (local netCDFs optional); only derived caches live in
`<cache_dir>/invert/`. Cache provenance records the source netCDF's size, taken from the
local file or from the store's `source_size_bytes`, so caches built either way are
interchangeable. The request path never reads the source.

## pr-specific caveat: additive anomalies can go below zero

InVERT anomalies are added, not multiplied. Where the pre-industrial climatology is
near zero (deserts) a negative InVERT anomaly can push absolute precipitation below
0 mm/day. The ensemble is left unclipped on this branch; `07_map_negative_pr_fraction.png`
shows where and how often this happens so a clipping decision can be made deliberately.

Numbers from the SSP2-4.5 run (`generate_plots.py --var pr`):

| quantity | value |
|---|---|
| InVERT pr annual anomaly std, all land cells | 0.35 mm/day (median cell 0.09, 95th pct 0.73, max 4.2) |
| InVERT pr global-land-mean std | 0.063 mm/day |
| (member, year, land cell) values below 0 mm/day | 0.21 % |
| land cells that ever go negative | 2141 of 18378 |
| worst cell (Atacama) | 20.9 % of member-years negative |
| smooth mean alone below 0 | never |

Negatives cluster where the climatology is under 0.5 mm/day (Sahara, Arabia, Pakistan,
Atacama). The most negative absolute value (-3.4 mm/day) is not a desert cell, though:
it is New Guinea (4 S, 135 E), where the InVERT annual std reaches 4 mm/day and one
member-year carries a -15.8 mm/day anomaly against a 12.8 mm/day climatology. That
cell is the hotspot in `04_map_ensemble_spread_2081_2100.png`; whether InVERT's
variance there matches CESM2-LENS has not been checked against the training data.

## pr-specific caveat: the historical segment is noisier than for tas

The historical field is the CESM ensemble mean, which for tas is smooth enough that
adding InVERT on top does not double-count (see above). For pr this holds less well:
after removing a 21-year running mean, the global-land-mean historical pr still has a
year-to-year std of 0.031 mm/day, about half the InVERT global-land-mean std
(0.063 mm/day). The same measure for tas gives 0.083 K vs 0.128 K, but that number is
inflated by volcanic years, which are forced rather than noise. The historical pr field
is visibly jagged in `01_global_annual_mean_line.png`. No smoothing is applied on this
branch; that is a decision to make before the ensemble is used downstream.

The small step at the 2014/2015 join (historical to emulator) is pre-existing and
appears for tas as well.

## Water availability (p-e) uses the pr envelope (added 2026-09-30)

InVERT has no evaporation field, so p-e borrows the pr envelope (`INVERT_PROXIES` in
`backend/invert_variability.py`). This treats evaporation variability, and its covariance
with precipitation, as negligible. Both are in mm/day and clipped to land by
`clip_to_land`; their land masks are identical in all three models (18,378 cells), so the
cached pr global mean is reused for p-e as is, and regional mode uses the p-e map's own
mask. Mean half-widths (SSP2-4.5, 1.5 C target), against the legacy p-e 2 sigma:

| region | InVERT half-width | legacy 2 sigma |
|---|---|---|
| Global (land) | 0.118 mm/day | 0.026 mm/day |
| S.Asia | 0.807 mm/day | 0.320 mm/day |
| N.Europe | 0.217 mm/day | 0.240 mm/day |

As for pr, the legacy global value averages over the ocean while the line is land-only.

## tasmin/tasmax: coldest / warmest month (changed 2026-09-30)

The simulator's yearly `tasmin` / `tasmax` are the coldest / warmest month of each year
(`AGGREGATION_RULES` in `scripts/process_monthly_gauss.py`). `_annualise_monthly` first
averaged all 12 InVERT months for every variable, which described the much smaller
variability of the annual-mean daily minimum / maximum: across the 58 IPCC-WGI regions the
5-95 % half-width was a median 0.21 (tasmin) and 0.28 (tasmax) of the legacy 2 sigma,
against 0.55 for tas.

Each member's yearly InVERT value for tasmin / tasmax is now the lowest / highest of its 12
monthly anomalies, per grid cell, before regridding and area averaging (`annual` in
`INVERT_VARIABLES`), then centred (below). The gridded and global-mean caches for these two
variables carry a `_min` / `_max` tag in the file name, so the older annual-mean caches are
never reused.

**Centring (decided 2026-09-30):** InVERT anomalies have the seasonal cycle removed (every
calendar month averages ~0, checked at three grid cells), so the lowest / highest of 12
zero-mean months is biased low / high. Without correction the global tasmin envelope sat
1.61 K below its line and the tasmax envelope 1.50 K above it, so neither contained its line;
regionally the offset reached +3.7 K (tasmax, West & Central Europe). The line already is
the coldest / warmest month, so the noise added to it must average zero. Each grid cell's
mean over all members and years of the lowest / highest month is therefore subtracted
before regridding. This keeps the larger spread of the extreme month and removes the bias:
the global envelope centre is now -0.03 K (tasmin) and +0.04 K (tasmax). Regionally the
envelope midpoint sits +0.1 to +0.3 K from the tasmax line because the warmest month is
skewed towards warm years; the members still average zero.

It is an approximation: the highest of 12 anomalies is not necessarily the anomaly of the
climatologically warmest month. Using that month's own anomaly would need a monthly
climatology to identify it, which InVERT does not provide. Across the 58 IPCC-WGI regions
the full-range half-width is a median 0.43 (tasmin) and 0.52 (tasmax) of the legacy
2 sigma, against 0.72 for tas. The climatological detrending check
(`CLIMATOLOGY_TOLERANCE`) is applied to the annual means before this reduction.

## Known pre-existing issue: pr for ocean-only regions

A pr request for a region with no land cells (six ocean regions of IPCC-WGI and five
small-islands regions) already failed before the regional envelope with
`ValueError: arange: cannot compute length`; it now fails earlier with
`The region mask selects no grid cells with pr data.`

## Regional figures

See `plots/regional/`, generated by `plot_regional_envelopes.py` (same directory arguments
as below, with a data dir that has both `invert/` and the three-model inputs):

- `<var>_regional_lines.png` -- dashboard line for six IPCC-WGI regions, InVERT envelope
  over the region shaded, legacy +/-2 sigma dotted.
- `all_regions_invert_vs_legacy.png` -- time-mean envelope half-width against the legacy
  2 sigma for every IPCC-WGI region, per variable.

## Before / after figures

See `plots/<var>/`. Generated by `generate_plots.py`, run from the repo root:

```
DIRS="--model-dir data/models_pr55_3model --cache-dir data/cache_pr55_3model"
python analysis/add-invert-variability/generate_plots.py --var tas $DIRS
python analysis/add-invert-variability/generate_plots.py --var pr  $DIRS
python analysis/add-invert-variability/generate_plots.py --var tasmin --line-only $DIRS
python analysis/add-invert-variability/generate_plots.py --var tasmax --line-only $DIRS
```

`tasmin` and `tasmax` only have the line plot (`--line-only`), in keeping with the
global-mean-only scope; their maps can be generated the same way if the spatial work
is picked up.

Default scenario is `SSP2-4.5` (the simulator's own baseline). This choice only
affects the smooth mean shown behind the ensemble -- the 20-member spread is
identical for any SSP because InVERT is detrended.

- `01_global_annual_mean_line.png` -- smooth global-mean (land-mean for pr) with the
  InVERT envelope, full range of the 20 ensemble members, 1851-2100. The black line is
  the raw CESM ensemble-mean historical field plus the emulator, without the dashboard's
  boundary smoothing, so its own year-to-year wiggle is visible.
- `02_map_mean_2081_2100.png` -- map of the smooth mean over the final 20 yrs
- `03_map_one_member_2081_2100.png` -- same period, one ensemble member (mean +
  InVERT)
- `04_map_ensemble_spread_2081_2100.png` -- std-dev across the 20 members, averaged
  over 2081-2100
- `05_map_diff_member_minus_mean_2081_2100.png` -- signed difference, single member
  minus the smooth mean
- `06_global_annual_ensemble_std.png` -- per-year global-mean std across the ensemble;
  should be roughly stationary with no visible jumps at seam years (1901, 1951, 2001,
  2051)
- `07_map_negative_pr_fraction.png` (pr only) -- % of (member, year) values per land
  cell where mean + InVERT anomaly is below 0 mm/day absolute precipitation
