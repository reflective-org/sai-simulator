"""Project-wide version constants."""

# v1.4.0: FaIR default climate calibration switched from CESM-tuned to the AR6
# calibrated-constrained ensemble (mean across members) — every FaIR-dependent
# cache artifact and pregenerated golden changes; regenerate under this version.
# v1.5.0: multi-model uncertainty and significance rework (PR #56) -- per-model internal
# variability for every model (P-E's from P-E, not evaporation), end-to-end per-model
# maps/series, the dashboard band as the internal variability of the multi-model mean (the
# model spread kept as a diagnostic), icefrac historical and centre as the mean of
# end-to-end outcomes, a detectability z-test on pooled per-model sigma with
# autocorrelation for the stipple, and with-SAI p-values cached for the default deployment
# only, built through the live request path. Every FaIR-independent cache artifact changes
# meaning; regenerate under this version (docs/pvalues-uncertainty-histograms-review.md).
# v1.5.0 also: UKESM1-1-LL joins the regional-model roster (3-model no-SAI mean for
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
# Also under v1.5.0: CESM2-WACCM's monthly stamps are corrected on load. Its CAM
# history groups (Mon/atmos_2d, and Mon/atmos_2d_tasminmax outside historical)
# stamp each month at the END of its interval, so processing read August as
# September and built every yearly aggregate over Feb-Jan, which also dropped
# 1850, 2015 and each SAI run's 2035 as 11-month years. Corrected, CESM's
# processed Arctic September sea ice falls 1.06 -> 0.45 million km^2 over
# 2050-2069 (baseline) and the multi-model sigmoid area at +4 K falls
# 3.27 -> 3.02. Every CESM processed file, fit, cache and golden regenerates;
# that folds into the v1.5.0 regeneration already pending.
SIMULATOR_VERSION = "v1.5.0"

# Public Cloudflare R2 location of the per-experiment Zarr training-data
# stores consumed by scripts/process_monthly_gauss.py and verified by
# scripts/data_sanity_checks.py. Served via the bucket's r2.dev subdomain,
# which is fully anonymous — no credentials or storage_options needed
# anywhere in the consumer path.
ZARR_STORES_URI = "https://pub-d127229e707d485c80965ab41706e316.r2.dev/simulator-inputs"
