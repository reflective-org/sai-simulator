# sai-simulator

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15531372.svg)](https://doi.org/10.5281/zenodo.15531372)

Official repository for [the SAI Simulator](https://simulator.reflective.org/), a new tool to explore the effects of stratospheric aerosol injection on the climate.

![SAI Simulator](img/preview.png)

## Table of Contents

- [Introduction](#introduction)
- [Climate Model Data](#climate-model-data)
- [Setup Options](#setup-options)
- [Option 1: Automated Setup (Recommended)](#-option-1-automated-setup-recommended)
- [Option 2: Manual Setup](#-option-2-manual-setup)
  - [Environment Setup](#a-environment-setup)
  - [Data Download](#b-data-download)
  - [Data Processing](#c-data-processing)
  - [Fit the Regression Models](#d-fit-the-regression-models)
  - [Variability Calculation](#e-variability-calculation)
  - [Cache the Data](#f-cache-the-data)
  - [Launch Gradio](#g-launch-gradio)
- [Adding a New Model](#-adding-a-new-model)
- [License](#-license)
- [Citation](#%EF%B8%8F-citation)

## Introduction

The SAI Simulator is a web-based tool that runs in your browser at [simulator.reflective.org](https://simulator.reflective.org/). If you would like to reproduce the steps to prepare the data presented in the simulator, you can follow the instructions below.

## Climate Model Data

The simulator is trained on monthly output from three Earth system models, combined as
an equal-weight multi-model mean of **patterns per kelvin**: each model's regional
fields are regressed on **its own global-mean tas anomaly** (vs its own 1850–1900 mean),
and at request time FaIR's global anomaly (AR6 calibrated-constrained ensemble mean by
default; see `FAIR_CALIBRATION` in `backend/baseline.py`) is fed through every model's
fitted line and the lines are averaged equally — that average is the final mapping from
global temperature to grid-cell values, so every pattern counts equally even though the
models' transient sensitivities differ. (A model can sit out a variable it has no data
for — see `REGIONAL_MODEL_VAR_EXCLUSIONS` in `backend/constants.py`; today UKESM sits
out p−e on the projection side.) One Zarr store per (model, experiment) is served from
the public bucket configured in `version.py` (`ZARR_STORES_URI`); the table lists what
each store contains.

| Model | Experiment | Members | Years | Data coverage | Native grid (lat × lon) |
|-------|------------|---------|-------|---------------|--------------------------|
| CESM2-WACCM-MA | historical | 3 (`001`–`003`) | 165 | 1850–2014 | 192 × 288 (~0.9° × 1.25°, fv09) |
| CESM2-WACCM-MA | baseline (SSP2-4.5) | 3 | 85 | 2015–2099 | 192 × 288 |
| CESM2-WACCM-MA | 1.5K-SAI (GAUSS) | 3 | 35 | 2035–2069 | 192 × 288 |
| CESM2-WACCM-MA | 1.0K-SAI (GAUSS) | 3 | 35 | 2035–2069 | 192 × 288 |
| CESM2-WACCM-MA | 0.5K-SAI (GAUSS) | 3 | 36 | 2035–2070 | 192 × 288 |
| MIROC-ES2H | historical | 3 (`r01`–`r03`) | 165 | 1850–2014 | 128 × 256 (T85, ~1.4°) |
| MIROC-ES2H | baseline (GeoMIP no-SAI reference, SSP2-4.5 forcing) | 3 | 65 | 2020–2084 | 128 × 256 |
| MIROC-ES2H | ssp245 (ScenarioMIP SSP2-4.5) | 3 | 86 | 2015–2100 | 128 × 256 |
| MIROC-ES2H | G6-1.5K-SAI (GeoMIP) | 3 | 50 | 2035–2084 | 128 × 256 |
| MIROC-ES2H | G6-0.5K-SAI (GeoMIP) | 3 | 50 | 2035–2084 | 128 × 256 |
| UKESM1-1-LL | historical | 3 (`r2`/`r3`/`r12`) | 165 | 1850–2014 | 144 × 192 (N96, 1.25° × 1.875°) |
| UKESM1-1-LL | baseline (het-chem SSP2-4.5, GeoMIP branch parent) | 3 | 86 | 2015–2100 | 144 × 192 |
| UKESM1-1-LL | G6-1.5K-SAI (GeoMIP) | 3 | 51 | 2034–2084 | 144 × 192 |

Notes:

- **Common grid.** All model output is regridded bilinearly onto the CESM2-WACCM
  192 × 288 grid during processing (`scripts/process_monthly_gauss.py`), so every
  downstream product shares one grid.
- **Coverage vs. timestamps.** CESM stores stamp monthly means at the *end* of the
  month (e.g. the January 1850 mean is stamped `1850-02-01`); the table states data
  coverage, not raw stamp ranges. MIROC stamps are start-of-month.
- **Historical merge.** Each model's historical run is stitched, per member, into every
  no-SAI run it publishes, so each of those files carries the full 1850-onwards series
  its consumer expects.
- **Two no-SAI runs, two roles (MIROC).** `baseline` and `ssp245` are distinct
  realizations under the same SSP2-4.5 forcing (global means agree to ~0.1–0.2 K, but
  the weather differs), and the pipeline uses each where it belongs:

  | Role | Run used | Why |
  |------|----------|-----|
  | **SAI delta** (`scripts/fit_delta.py`) | `baseline` | The G6 experiments branched from it, so rebasing the anchors against it isolates the SAI signal from inter-run drift. |
  | **Climate-change-only fit** (`scripts/fit_map.py`, dashboard projection + historical series) | `ssp245` | The standard ScenarioMIP run: 2015–2100, continuous with historical, and the same scenario definition CESM's baseline provides. |

  CESM publishes one run that serves both roles, so only MIROC has the split. The
  selection lives in `backend.constants.REGIONAL_MODEL2PROJECTION_BASELINE_FILES`
  (projection side) and `scripts/fit_delta.ANCHORS` (delta side).
- **`ssp245` evaporation and sea ice.** Native ScenarioMIP `hfls` for this run lives
  only on the offline ESGF node `esgf-data02.diasjp.net`, so the store's `QFLX` is
  **filled from the GeoMIP baseline run's hfls** over 2020–2084 (NaN for 2015–2019 and
  2085–2100; see the array's `note` attr) — same SSP2-4.5 forcing but a different
  realization, so monthly P−E covariance is approximate while climatological means are
  consistent; MIROC's `ssp245` p−e output covers 65 years accordingly. `ICEFRAC` is
  regridded from the run's **own CMOR `siconc`** (ESGF ORNL mirror; full 2015–2100,
  bilinear from the curvilinear ocean grid; validated same-run against the shipped
  historical ICEFRAC to 0.1–0.8% on September Arctic ice area — ice-edge cells differ
  slightly from the producer's GeoMIP seaice conversion, see the array's `note` attr).
  The sea-ice **sigmoid fit** still trains on the GeoMIP family (branch-consistency
  with the G6 anchors), not on `ssp245`.
- **The sea-ice fit keeps ice-free years.** `filter_icefrac_data` excludes marginal
  grid cells — those with too few valid years, or whose September ice never reaches
  `intensity_threshold` — but it does **not** discard individual ice-free
  observations. It once set every value ≤ 0.01 to NaN as noise, which censored the
  training set precisely where a decreasing logistic needs data: the zeros are what
  pin its warm asymptote. The cost scaled with how ice-free a model gets (CESM lost
  1.8% of its observations, MIROC 8.0%), leaving the warm end unconstrained and the
  fitted curve resting on a floor rather than decaying. Removing that step improved
  the Arctic-wide September ice bias for every model against its own output.
- **The two G6 experiments share one branch-point baseline.** JAMSTEC publishes
  `*_baseline_*` files under both `G6-1.5K-SAI/` and `G6-0.5K-SAI/`; they are
  byte-identical (verified bitwise for hfls, size-identical for every other variable),
  so `MIROC-ES2H_baseline.zarr` is the correct rebase reference for **both** G6 anchors
  in the delta fit — no per-experiment baseline stores are needed.
- **SAI experiment naming.** The CESM GAUSS runs and MIROC G6 runs are labeled by
  their temperature *target* above pre-industrial (e.g. `0.5K` targets +0.5 °C, which
  requires the largest cooling), and each is rebased against its own model's baseline
  in the delta fit (`scripts/fit_delta.py`).
- **MIROC evaporation.** MIROC's `QFLX` (used for water availability, P−E) is derived
  from JAMSTEC's monthly latent heat flux: `QFLX = hfls / 2.5e6`.
- **UKESM1.1 provenance.** The UKESM runs are the **het-chem** UKESM1.1 variant
  (heterogeneous chemistry on injected stratospheric aerosols; UM 13.1 for the
  scenarios, 11.7 for historical), distributed by the Met Office/Exeter via the
  public JASMIN MACLOUD workspace rather than ESGF, on a 360-day calendar. Its
  single het-chem SSP2-4.5 run is both the G6 branch parent and the projection
  baseline (CESM-style). Monthly TREFHT/PRECT are derived as 30-day means of the
  daily fields in every experiment (the shipped monthly files mix 6-hourly- and
  1-hourly-sampled duplicates inconsistently across experiments); TREFHTMX/MN and
  the exposure counts likewise come from the daily fields; QFLX = evap / 2.5e6
  (latent heat flux, as for MIROC); ICEFRAC is the shipped atmosphere-grid
  fraction (no regridding). The historical suites carry no realization metadata,
  so the suite-to-member pairing (`r2=u-bz502`, `r3=u-by791`, `r12=u-bz897` —
  not alphabetical) was established by cross-boundary weather continuity at the
  2015 stitch, then confirmed independently by Matthew Henry on 2026-09-24. It
  is documented in the stores' note attrs and is what `--historical-map`
  expects.
- **UKESM historical sea ice comes from ESGF, for one member.** MACLOUD
  publishes none, but CMIP6 ESGF carries `siconca` for UKESM1-1-LL historical
  `r1i1p1f2` (CEDA node, `v20220512`, atmosphere grid, no regridding needed).
  ESGF publishes historical for that variant alone — `r2i1p1f2` and `r3i1p1f2`
  have `ssp126` only, and there is no `r12` at all — so it covers one of our
  three members. It is the same run as this store's **r3**: monthly `tas` agrees
  with r3 to 0.0001 K RMS and with r2/r12 to only 1.97–1.99 K, which is the
  scale our own members differ from each other by. The CMIP variant label and
  MACLOUD's member numbering therefore do **not** correspond; matching on the
  label alone pairs the wrong run. The historical store's ICEFRAC is `PARTIAL`
  as a result: real for r3, all-NaN for r2 and r12. Two conversions matter —
  ESGF ships percent where the store holds a 0–1 fraction, and marks land with
  a real `_FillValue` where the store holds 0.
- **UKESM missing sources.** The MACLOUD historical publishes no evaporative
  flux or sea ice, so the historical store ships all-NaN QFLX/ICEFRAC
  placeholders: UKESM's p−e and icefrac series start in 2015. Its p−e therefore
  feeds only the SAI delta (branch-consistent baseline/G6 pair) and sits out the
  projection side (`REGIONAL_MODEL_VAR_EXCLUSIONS`). Its **icefrac now trains
  the sea-ice sigmoid** alongside the other two: once ESGF supplied historical
  sea ice (see above) the training window reaches back to 1900 and the logistic's
  full-ice plateau is fitted rather than extrapolated. It held out while its ice
  began in 2015, which left no cold regime in the training data. UKESM's
  September Arctic sea-ice **area** is low-biased against observations —
  0.2–2.4 Mkm² over 2015–34 against 2.8–3.5 Mkm² observed over 2015–25 (the
  `area` column of `data/processed/september-sea-ice-extent-observation.csv`) —
  which is a property of the model, and shows up as the lowest of the three
  curves rather than as a fitting artefact. `python scripts/plot_icefrac_fit.py`
  renders a diagnostic figure of the fitted sigmoids, each model's training
  coverage, and what including UKESM does to the mean (written to
  `img/icefrac_fit.png`, a generated artefact that is not committed).
- **UKESM icefrac is one realization.** ESGF publishes its historical sea ice for
  a single member, so `variable_members` in `process_monthly_gauss.MODELS`
  restricts icefrac to `r3` in **every** experiment. A three-member mean after
  2015 against one member before it would step the variance at the stitch, and
  the sigmoid trains on exactly that series. Its `tas` stays a three-member mean,
  so the fit pairs a one-member ice record against an ensemble-mean temperature;
  the forced response they share is what the logistic fits.
- **UKESM 360-day calendar and exposure counts.** UKESM years have 360 days, so
  its exposure counts saturate at 360 days/year where CESM's reach 365 and
  MIROC's 366. Nothing rescales by days-per-year; in fully saturated cells
  UKESM is therefore ~1.4% low in the multi-model mean (the CESM/MIROC leap-year
  gap is the same class of effect at 0.3%), and the SAI delta largely cancels
  it. Documented rather than corrected — counts are stored as integers.
- **SO2 injection.** The SO2-by-latitude controller data exists only for the CESM
  GAUSS runs, so the SO2 fit is CESM-only.

## Setup Options

There are two ways to set up the SAI Simulator locally:

| Option | Description | Best For |
|--------|-------------|----------|
| **[Automated Setup](#-option-1-automated-setup-recommended)** | Interactive script that handles everything | Most users |
| **[Manual Setup](#-option-2-manual-setup)** | Step-by-step commands you run yourself | Advanced users who want full control |

Both options require ~140GB of disk space for data storage.

---

## 🚀 Option 1: Automated Setup (Recommended)

The automated setup script guides you through the entire installation process interactively. It asks all configuration questions up front, then executes the setup steps.

```bash
./setup.sh
```

The script will:

1. Ask all configuration questions up front (environment type, data directories, which steps to run)
2. Set up your Python environment (Conda or venv)
3. Optionally compile and install ESMF (default version: release/8.9.0)
4. Download ~140GB of data from S3
5. Process the data and fit regression models
6. Cache the data for the simulator

After setup completes, launch the simulator with:

```bash
python launch_gradio.py --use_local_cache
```

**Note:** Use Firefox browser (not Safari) for the web UI.

---

## 📦 Option 2: Manual Setup

Use this option if you want full control over each step or need to troubleshoot specific parts of the setup.

### A. Environment Setup

We recommend using **Python 3.11** or higher (all code is tested using Python 3.12.12).

#### Option 1: Using Conda

Set up a new Conda environment and install required packages:

```bash
conda create -n sai-simulator python=3.12.12 -c conda-forge
conda activate sai-simulator
pip install -r requirements.txt
```

#### Option 2: Using venv (Standard Python Virtual Environment)

If you prefer to use `venv` instead of Conda, follow these steps:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Install ESMF Library (Required for both Conda and venv)

The SAI Simulator requires the ESMF (Earth System Modeling Framework) library to be compiled from source.

**Prerequisites:**

- gfortran: `brew install gcc`
- Xcode Command Line Tools: `xcode-select --install`

**Installation Steps:**

1. Clone the ESMF repository:

   ```bash
   git clone https://github.com/esmf-org/esmf.git
   cd esmf
   ```

2. Checkout a release branch (e.g., release/8.9.0):

   ```bash
   git checkout release/8.9.0
   ```

3. Clean previous builds and set up the build environment:

   ```bash
   make distclean
   export ESMF_DIR=$(pwd)
   export ESMF_BOPT=O
   export ESMF_CXX=/usr/bin/clang++
   export ESMF_C=/usr/bin/clang
   export ESMF_F90=gfortran
   export ESMF_COMPILER=gfortranclang
   export ESMF_SHARED_LIB_BUILD=ON
   export ESMF_CXXLINKLIBS="-lc++"
   export ESMF_F90LINKLIBS="-lc++"
   export ESMF_COMM=mpiuni
   ```

4. Build ESMF (this takes 10-20 minutes):

   ```bash
   make
   ```

5. Verify the libraries exist in `lib/libO/<os-system>/`:

   - `libesmf_fullylinked.dylib` (or `.so` on Linux)
   - `libesmf.dylib` (or `.so` on Linux)
   - `esmf.mk`

6. Set the ESMFMKFILE environment variable and add it to your shell configuration:

   ```bash
   export ESMFMKFILE=$(find $ESMF_DIR/lib/libO -name "esmf.mk" | head -1)
   echo "export ESMFMKFILE=\"$ESMFMKFILE\"" >> ~/.zshrc  # or ~/.bashrc
   source ~/.zshrc  # or ~/.bashrc
   ```

7. Install ESMPy:
   ```bash
   cd src/addon/esmpy
   python -m pip install .
   ```

> **Version pitfall.** ESMPy takes its version from the git checkout it is installed from
> and refuses to load against an ESMF library of a different version
> (`VersionMismatch: ESMF installation version 8.7.0 differs from ESMPy version 8.9.0`).
> If the source tree has moved past the tag you compiled, install ESMPy from a worktree at
> the matching tag instead of the tree head:
>
> ```bash
> grep ESMF_VERSION_STRING "$ESMFMKFILE"                 # e.g. 8.7.0
> git -C /path/to/esmf worktree add ../esmf-8.7.0 v8.7.0
> python -m pip install ../esmf-8.7.0/src/addon/esmpy
> ```

#### Lightweight environment for tests and the backend

`requirements.txt` pins the full stack (dashboard, S3 verification, pipeline). To run the
unit tests or `backend.get_outputs` you only need `requirements-test.txt` — the same set CI
installs — plus, for the processing scripts, `zarr fsspec s3fs dask fire rioxarray xesmf`
(and ESMPy as above). `cf-units` and `scitools-iris` have no macOS arm64 wheels, and nothing
in `backend/`, `fair_wrap/` or `scripts/` imports them.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-test.txt
pytest -q tests/test_*.py          # exactly what CI runs
```

### B. Verify Data Access

The training data lives in per-experiment Zarr stores hosted on a public
Cloudflare R2 bucket (`reflective-data-store/simulator-inputs/`), exposed
anonymously over HTTPS via the bucket's `r2.dev` subdomain — see
`version.ZARR_STORES_URI`. The processing pipeline reads them lazily, so
there is nothing to pre-download and no `data/raw/` directory to populate.

Verify the stores are reachable and have the expected schema before
processing:

```bash
python scripts/data_sanity_checks.py
```

A few-second metadata-only smoke test; exits non-zero if any store is
unreachable or missing an expected variable.

### C. Data Processing

Process the monthly Zarr stores into per-experiment annual NetCDF files ready
for model fitting. The Zarr stores already carry the precomputed
exposure-count fields (`TREFHT_above_*`, `PRECT_above_*`) under
`Mon/atmos_2d_derived`, so there is no separate daily-processing step on the
consumer side. Run all variables with:

   ```bash
   sh scripts/process_all_monthly.sh data/raw data/processed
   ```

   Or you can process the monthly data for a specific variable using:

   ```bash
   python scripts/process_monthly_gauss.py  --var tas --data_dir data/raw --output_dir data/processed
   ```

   Then calculate model internal variability for sea ice:

   ```bash
   python scripts/variability.py icefrac ./data/processed ./data/models
   ```

### D. Fit the Regression Models

We fit linear regression models to (1) estimate a gridded map of each variable given a global temperature output by [FaIR](https://github.com/OMS-NetZero/FAIR) and (2) estimate a gridded delta of each variable given a global temperature delta. The models are trained using the GAUSS simulation data.

You can fit the regression models for all variables using:

```bash
sh scripts/fit_all.sh data/processed data/models
```

Or you can fit the regression models for a specific variable using:

```bash
python scripts/fit_map.py --var tas --data_dir data/processed --output_dir data/models
python scripts/fit_delta.py --var tas --data_dir data/processed --output_dir data/models
```

### E. Variability calculation

To calculate grid cell level and global model internal variability, we need to run the variability script.

```bash
python scripts/variability.py --var_name icefrac --data_dir data/processed data/models
```

### F. Cache the Data

The simulator caches all the data to be loaded more efficiently by the frontend. You can cache the data by running the following command:

```bash
python scripts/cache.py --data_dir data/processed --model_dir data/models --output_dir data/cache
```

### G. Launch gradio

Gradio handles the connection between the frontend and backend and builds the interactive web interface for the simulator. Run the prompt below and use the web UI that it returns in the Firefox browser (not Safari).

```bash
python launch_gradio.py --use_local_cache
```

For custom injection

```bash
python variable_launch_gradio.py --use_local_cache
```

## 🧩 Adding a New Model

The pipeline is multi-model: each climate model is processed independently and
contributes its SAI experiment(s) as additional training **anchors** to the
delta fit. CESM2-WACCM (GAUSS), MIROC-ES2H (G6-1.5K/0.5K) and UKESM1-1-LL
(G6-1.5K) are wired in today; adding another model is mostly configuration —
copy the entries the existing models have in the files below.

**Producer side (off-repo).** Build per-experiment Zarr stores for the new
model in R2 under the `/simulator-inputs/` prefix, matching the existing
convention: CESM-style variable names (`TREFHT`, `PRECT`, `ICEFRAC`,
`TREFHTMN`/`TREFHTMX`, the `*_above_*`/`*_below_*` exposure counts), the same
group layout (`Mon/atmos_2d`, `Mon/atmos_2d_derived`, `Mon/atmos_2d_tasminmax`),
all available members, and an SSP2-4.5 baseline plus the model's G6 SAI run(s).
Regrid sea ice to `ICEFRAC` (land filled with 0) and precompute the exposure
counts so the consumer stays model-agnostic. The consumer bilinearly regrids
every model onto the CESM2-WACCM target grid, so the native grid needs no
special handling. UKESM's producer chain lives in this repo as a worked
example: `scripts/mirror_macloud_to_r2.py` (JASMIN MACLOUD → R2 raw staging;
`scripts/copy_ukesm_to_r2.py` does the same from the 2i2c hub bucket, and
`scripts/mirror_esgf_to_r2.py` from ESGF) and `scripts/zarrify_ukesm11.py`
(raw NetCDF → the three `UKESM1-1-LL_*.zarr` stores, with schema helpers in
`scripts/zarrify_common.py`).

**Repo side (five edits, all by analogy to the MIROC entries):**

1. `scripts/data_sanity_checks.py` — add the new store filenames to
   `STORE_NAMES`. If the model publishes no evaporative-flux source at all (no
   `QFLX`, which means no `p-e` downstream), add it to `ALLOWED_MISSING` so the
   absence is treated as allowed rather than a failure.
2. `scripts/process_monthly_gauss.py` — add a `MODELS` entry: the
   `(store, label)` experiment pairs, `model_coord`, a unique `output_suffix`
   (e.g. `"-ukesm"`), `has_historical`, `no_sai_experiments` (every no-SAI run
   the model publishes — the historical run is stitched into each), and
   `experiment_missing_vars` for any variable a given store ships as a
   placeholder, plus `compute_variability: False`.
3. `scripts/fit_delta.py` — add one `Anchor` to `ANCHORS` per SAI cooling label,
   pointing at the consumer's `output_gauss-<label><output_suffix>.nc` and, as
   the baseline, **the run those SAI experiments branched from**.
4. `backend/constants.py` — add the model to `REGIONAL_MODEL_NAMES` and give it
   an entry in `REGIONAL_MODEL2PROJECTION_BASELINE_FILES` listing its
   projection-baseline file(s) in preference order (the ScenarioMIP run first,
   then any fallback for variables that run lacks).
5. Run the pipeline (`process_monthly_gauss.py --model all`, then
   `scripts/fit_all.sh`) and validate.

**Weighting note.** The SAI slope is one **pooled** through-origin least
squares over every model's anchors, which weights each anchor by the *square*
of its cooling magnitude (currently CESM 51.1% / MIROC 33.6% / UKESM 15.4% overall) — unlike
the no-SAI emulators and sea-ice sigmoids, which average per-model results
equally at prediction time. The pooled form is deliberate: fitting per model and
averaging would make model totals equal but concentrate 41.7% of all influence
in MIROC's single G6-0.5K run (see the ``scripts/fit_delta.py`` docstring).
A new model contributing only a lone G6 run adds leverage proportional to the
square of that run's actual cooling — for a nominal "1.5K target" that depends
on the model's sensitivity: UKESM's single G6-1.5K anchor cools 1.37 K
(high-ECS model, large overshoot to pull back) and lands at 15.4%, whereas the
same target in MIROC cools only 0.82 K (5.6%). Recompute the split when adding
an anchor if you care about balanced coverage across cooling magnitudes.

## 🔒 License

This project is released under the Apache 2.0 License - see the [LICENSE](https://github.com/reflective-org/sai-simulator/blob/main/LICENSE) file for details.

## ✏️ Citation

If you use the SAI Simulator in your work, please cite the following:

```
Ali Akherati, Susanne Baur, Charlotte DeWald, Jake Dexheimer, Alistair Duffey, Jared Farley, Dakota Gruener, Jeremy Irvin, Emily Kuhn, Douglas MacMartin, John Orcutt, Kate Pellegrino, Daniele Visioni, Duncan Watson-Parris, Kion Yaghoobzadeh. (2026). SAI Simulator (Version 1.3.0) [Software]. Available at GitHub: https://github.com/Reflective-org/sai-simulator. Hosted at: https://simulator.reflective.org and https://planetparasol.ai.  DOI: https://doi.org/10.5281/zenodo.15531372
```
