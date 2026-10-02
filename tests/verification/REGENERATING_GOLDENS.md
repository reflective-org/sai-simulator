# Regenerating the S3 validation goldens

The goldens this harness compares against are **not produced by this repo**. They are the
production pregenerated API cache, written by `sai-simulator-api` →
`api/scripts/pregenerate_cache`. `tests/verification` only ever downloads them. Nothing
here regenerates them, which is why it is easy to forget where they come from.

## The contract between producer and consumer

| | code | key it builds |
|---|---|---|
| Producer | `api/s3_manager.py` → `get_s3_key` | `v{S3_CACHE_VERSION}/{geotiffs\|plot_data}/{filename}` |
| Consumer | [validate_outputs.py](validate_outputs.py) → `default_s3_key` | `{VALIDATION_S3_PREFIX}/{filetype}/{var}_{no_sai\|with_sai}_{hash}.tif` |

`S3_CACHE_VERSION` has **no** leading `v` (the producer prepends it); `VALIDATION_S3_PREFIX`
does. The prefix resolves to the most recently created git tag, falling back to
`VALIDATION_S3_PREFIX_FALLBACK` in [.github/validation/self-hosted.env](../../.github/validation/self-hosted.env).
The repo normally has no tags, so **the fallback is what actually runs** — and if a tag is
ever pushed, the newest by `creatordate` silently wins, whatever it is called.

The filename hash covers `ssp_scenario`, `temp_target`, both decade years, `start_year`,
`ramp_up`, and the display baseline period (`base_{DISPLAY_BASELINE_START}_{DISPLAY_BASELINE_END}`).
Any change to that tuple renames every object, so goldens from before such a change are not
"stale numbers" — they are unreachable keys.

## Where it runs

`github-sai-validation-runner-ec2` (`i-0c6e8d3830b17ea1b`, c5a.12xlarge, 48 vCPU / 93 GB,
300 GB gp3, us-west-2). Its instance role carries `AmazonS3FullAccess` and
`AmazonSSMManagedInstanceCore`, so it can write the goldens and be driven over SSM without
SSH. Target bucket: `simulator-pregenerated-datacache`.

## Procedure

1. **Start from a merged commit.** Goldens built from a branch under review cannot
   independently validate that branch, and these objects are also what production serves.
2. **Bump the submodule** in `sai-simulator-api` to that commit, and run from the pinned
   checkout rather than a working tree, so the goldens are reproducible from a recorded SHA.
   The box has no GitHub credentials and both repos are private — `git archive` of the
   pinned SHAs, shipped through S3, avoids putting a token on a shared machine.
3. **Ship `data/processed` and `data/models`** (~4.6 GB) to the box. Check what is already
   there first; it has been years out of date before.
4. **Rebuild `data/cache`, do not copy it.** The fingerprint records
   `{simulator_version, fair_calibration, regional_models, var_exclusions}`, and
   `validate_cache_dir` refuses a dir stamped under a different roster. Historical pass
   first — it is shared and must not be written concurrently — then scenarios:
   ```bash
   python scripts/cache.py data/processed data/models data/cache --historical_only
   python scripts/cache.py data/processed data/models data/cache --scenario "SSP2-4.5"
   ```
5. **Set `api/.env`**: `S3_BUCKET_NAME`, `S3_CACHE_VERSION`, and the `SIMULATOR_*_DIR` paths.
6. **Delete `api/scripts/pregenerate_cache/progress.json`**, or the run skips everything as
   already done. (Conversely: leave it in place to resume an interrupted run.)
7. **Run it**: Redis up, N Celery workers with `--pool=solo` (the geospatial stack is not
   fork-safe, so parallelism comes from separate processes), then
   `python -m api.scripts.pregenerate_cache`.
8. **Verify** before believing it — see below.
9. **Point the harness at the new prefix** via `VALIDATION_S3_PREFIX_FALLBACK`.

## Scale and resources

`generate_parameter_combinations()` returns **321,312** combinations: 320,880 standard
(8 SSPs × 21 temp targets × 56 start years × 10 vars, decades filtered to those starting at
or after the SAI start year) plus 432 variable-injection. Each yields 2 GeoTIFFs and 1
plot_data file, so **963,936 objects, ~85 GB**.

Disk on the box is dominated by `data/cache` (~59 GB), **not** by the pregeneration output:
`bulk_upload_to_s3_task` unlinks each file after a successful upload, so `api/output_cache`
drains continuously and ends empty. An empty `output_cache` after a run is correct, not a
symptom.

Throughput plateaus around 20 workers — beyond that the driver's polling loop is the
bottleneck, not worker capacity.

## Verifying a run

"The driver exited" is not "the goldens are correct". Check all of:

- `Errors encountered: 0` in the driver log, and `len(progress.json) == 321312`.
- No `Failed to upload` across the worker logs, and the Celery queue drained.
- Object counts per subprefix: `geotiffs` = 2 × combinations, `plot_data` = 1 ×.
- **Key agreement**: sample combinations across all SSPs, variables and decades, build each
  key with the consumer's own `default_s3_key`, and head it. This is the check that catches
  a producer/consumer hash divergence, and it is the one that matters.

## Traps

- **SSM runs as root.** Files written that way are unreadable to the validation workflow,
  whose steps run as `ec2-user`. Run builds under `su ec2-user` and `chown -R` anything
  root touched.
- **Set `POOCH_HOME`** to `/home/ec2-user/.cache/pooch`, which already holds the RCMIP
  inputs. Without it FaIR re-downloads from Zenodo, and parallel processes doing that at
  once get a 403. Warm pooch serially before fanning out.
- **Cache scenario builds peak around 25 GB RSS.** Four at once OOM-killed one on the 93 GB
  box; use `-P 2`.
- **`scripts/cache.py` has no atomic write.** It writes with a bare `to_netcdf()` and
  resumes on `path.exists()`, so a process killed mid-write leaves a truncated file that the
  next run accepts as complete. After any kill, open every `.nc` before resuming.
- **Opening a PR against `main` stops the instance.** `s3-validation-self-hosted.yml` runs
  on `pull_request` and its `stop-runner` job is `if: always()`, so any PR — including other
  people's — will stop the box out from under a long job. The cleanup watchdog also stops it
  once `gh_validation_lease_expires_at` passes; extend that tag before a long run. Note the
  watchdog resolves its workflow from the **default branch**, so changes to its behaviour do
  nothing until merged.

## What this does and does not buy you

Regenerating the goldens restores a **key-existence and shape check**. It does not give you
a numerical regression test, because the harness does not have one:

- Nothing compares `max_rel_err` (or any metric) against a threshold.
  `CaseVerificationResult.ok` is `not result.errors`, and errors are appended only when
  something *raises*.
- The metrics are NaN regardless. Every GeoTIFF is 2-band — band 1 data, band 2 a
  significance mask that is `1.0` where `p >= 0.05` and **NaN where the cell is
  significant** (`backend/get_outputs.py:_significance_mask`). `compare_geotiff_files` reads
  all bands and reduces with plain `np.mean` / `np.median` / `np.max` / `np.percentile`,
  none of which skip NaN, so one significant cell NaNs every metric.

## Run record

**v1.5.0 — 2026-10-01.** 321,312 combinations, 0 errors, 0 upload failures, 963,936 objects
/ 85.0 GB, ~8 hours at ~20 batches/hour with 28 workers. Built from `sai-simulator-api`
pinned to this repo's `4e9383e` ("Add UKESM1-1-LL as a third climate model"), against a
cache stamped `{v1.5.0, constrained, [CESM2-WACCM, MIROC-ES2H, UKESM1-1-LL], {}}`. Counts
matched expectation exactly and 400 sampled keys across all SSPs, variables and decades
resolved.

No `v1.4.0/` prefix exists — that regeneration never happened, so before this run the
goldens predated both the AR6-constrained FaIR default and the 1995-2014 display baseline.
