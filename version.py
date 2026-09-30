"""Project-wide version constants."""

# v1.4.0: FaIR default climate calibration switched from CESM-tuned to the AR6
# calibrated-constrained ensemble (mean across members) — every FaIR-dependent
# cache artifact and pregenerated golden changes; regenerate under this version.
# v1.5.0: UKESM1-1-LL joins the regional-model roster (3-model no-SAI mean for
# most variables, 2-model for p-e; a sixth anchor in the pooled SAI delta). The
# cache fingerprint now also records the roster and per-variable exclusions, so
# every pregenerated cache and verification golden regenerates under this version.
# Also under v1.5.0: the sea-ice sigmoid fit stops discarding ice-free
# observations (see scripts/fit_icefrac_arctic.filter_icefrac_data), which shifts
# the Arctic September sea-ice output for the EXISTING models too — the two-model
# mean falls 0.011 million km^2 at pre-industrial and 0.129 by +10 K. v1.5.0 has
# not shipped, so this folds into it rather than taking a version of its own; the
# regeneration it requires was already on the list. Merging main also brings
# the 1995-2014 map display baseline (#53) under this version.
SIMULATOR_VERSION = "v1.5.0"

# Public Cloudflare R2 location of the per-experiment Zarr training-data
# stores consumed by scripts/process_monthly_gauss.py and verified by
# scripts/data_sanity_checks.py. Served via the bucket's r2.dev subdomain,
# which is fully anonymous — no credentials or storage_options needed
# anywhere in the consumer path.
ZARR_STORES_URI = "https://pub-d127229e707d485c80965ab41706e316.r2.dev/simulator-inputs"
