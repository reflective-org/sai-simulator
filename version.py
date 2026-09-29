"""Project-wide version constants."""

# v1.4.0: FaIR default climate calibration switched from CESM-tuned to the AR6
# calibrated-constrained ensemble (mean across members) — every FaIR-dependent
# cache artifact and pregenerated golden changes; regenerate under this version.
SIMULATOR_VERSION = "v1.4.0"

# Public Cloudflare R2 location of the per-experiment Zarr training-data
# stores consumed by scripts/process_monthly_gauss.py and verified by
# scripts/data_sanity_checks.py. Served via the bucket's r2.dev subdomain,
# which is fully anonymous — no credentials or storage_options needed
# anywhere in the consumer path.
ZARR_STORES_URI = "https://pub-d127229e707d485c80965ab41706e316.r2.dev/simulator-inputs"
