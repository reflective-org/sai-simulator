# Model validation

An interactive website that checks the simulator's emulator against the GCM runs it
was trained on, for CESM2-WACCM, MIROC-ES2H and UKESM1-1-LL. It shows RMSE, MAE and
R² maps per variable over 2050–2069; click a cell to see the data behind its numbers.

- **Baseline (SSP2-4.5):** the dashboard's no-SAI map (the equal-weight multi-model
  mean, `backend.baseline.get_regional_map`) against each model's own projection
  baseline, anomalies against 1850–1900.
- **SAI delta:** the dashboard's 1.5 K SAI delta (`backend.delta.get_regional_delta`)
  against each model's own 1.5 K SAI run minus its reference run. RMSE and MAE are per
  model. R² is one map pooled over all six anchor runs in `scripts/fit_delta.py`
  (CESM 0.5/1.0/1.5 K, MIROC 0.5/1.5 K, UKESM 1.5 K) and is the same under every model.
- **Stippling** on R² maps marks cells where the GCM's signal-to-noise ratio is below 1
  (3 members × 20 years, detrended noise).

## Contents

- `build_website.py` builds the site.
- `website/` is the built site: `index.html` (maps and page code) and one folder per
  model holding the click-through data as gzipped JSON (`data_<var>.json.gz`,
  `data_<var>_snr_{baseline,sai_delta}.json.gz`). The page unzips each file as it
  loads it, which needs a browser from roughly 2023 or later.

## Viewing

The page fetches its data files, so serve the folder rather than opening the file:

```bash
cd model-validation/website
python -m http.server 8000
```

then open http://localhost:8000.

## Rebuilding

The builder reads `data/processed/` and the fitted models, so run the usual pipeline
first (see the repo README). Stippling also needs the per-member
`*-all-members.nc` files, which `scripts/process_monthly_gauss.py` writes for
tas, tasmin, tasmax, pr, p-e and icefrac.

1. `scripts/process_monthly_gauss.py` for the variables above (and the rest of the
   site's variables), for all three models.
2. `scripts/fit_all.sh`, then `scripts/variability.py`.
3. From the repo root:

   ```bash
   python model-validation/build_website.py
   ```

   This rewrites `model-validation/website/`.
