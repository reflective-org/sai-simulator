# Model comparison

A static page comparing the simulator's three climate models (CESM2-WACCM, MIROC-ES2H,
UKESM1-1-LL) with each other and with their multi-model mean. Everything shown is the
**raw model output**: each model's own no-SAI runs (historical joined to SSP2-4.5), averaged
over its ensemble members. Nothing goes through the emulator.

The runs are the ones the simulator trains on, as processed by
`scripts/process_monthly_gauss.py` onto the shared 192 × 288 grid: CESM's `baseline`,
MIROC's `ssp245` and UKESM's `baseline` store.

Integration into the online simulator (sai-simulator-fe) is separate work and is not part of
this directory.

## What the page shows

- **Global Mean:** for each model, the annual ensemble mean, each member as a faint line
  (checkbox) and a ±2σ band, plus the multi-model mean and observations where available.
  Each series is aggregated as the simulator aggregates its lines: cos-lat global mean
  (water availability over land only), population-weighted land sum for exposure, and
  September Arctic sea-ice area. Temperature, min/max temperature and water availability
  are anomalies against the model's own 1850–1900 mean.
- **Spatial Comparison:** a decade-mean difference map between any two responses (a model's
  ensemble mean or the multi-model mean). The anomaly variables are each taken relative to
  their own 1995–2014 mean before subtracting. Colours follow the simulator's scales: blue
  is colder and red is warmer; brown is drier and blue is wetter.

## Bands and gaps

- **Each model:** ±2σ, where σ is the standard deviation across its members that year
  (ddof = 1), drawn only where at least two members have data.
- **Multi-model mean:** the mean of the three ensemble means, drawn only in years every
  model has. Its band is ±2·sqrt(Σσ²)/3, drawn only where every model has a σ.
- **Members are not complete everywhere.** A year a member lacks is left out of that
  year's ensemble mean:
  - CESM exposure: one member after 2014 (the `baseline` store's day counts have one
    member), so its band ends in 2014.
  - CESM min/max temperature: 2015–2069 only (the `baseline` store's tasminmax group), so
    its line and the multi-model mean stop in 2069.
  - CESM tas and water availability: member 003 has no 2015–2019 and 002 no 2099; sea ice:
    003 ends in 2069 and 002 in 2098.
  - MIROC water availability: 2015–2019 and 2085–2099 are missing (its `ssp245`
    evaporation is published for 2020–2084 only, from the GeoMIP baseline run).
  - UKESM water availability and sea ice: one member (r3), so no band.
- **Map decades:** 2090–2099, except water availability (2075–2084) and min/max
  temperature (2060–2069), the last decade every model has. The page notes this on those
  maps.
- **MIROC evaporation.** The data uses the store's 2026-10-21 seam correction, which removes
  the jump between MIROC's historical and GeoMIP evaporation (see the QFLX `note` attribute).

## View it locally

The page loads its data with `fetch`, so serve it rather than opening the file directly:

```bash
cd model-comparison && python -m http.server 8765
# open http://localhost:8765/
```

## Regenerate the data

`data/` is generated and committed so the page works without the model data. To rebuild it
you need the repo's processed data (`data/processed`; see the main README). Run from the
repo root with the `sai-simulator` conda env:

```bash
python model-comparison/pipeline/preview_data.py --out model-comparison/data   # ~1 min
python model-comparison/pipeline/coastlines.py model-comparison/data/coastlines.json
```

`--data-root <checkout>` reads another checkout's `data/processed` instead.
