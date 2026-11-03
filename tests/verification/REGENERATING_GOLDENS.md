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
does. The prefix comes from `VALIDATION_S3_PREFIX` in
[.github/validation/self-hosted.env](../../.github/validation/self-hosted.env). Only if that is
empty does the workflow fall back to the most recently created git tag. The config file wins
because release tags (`v1.3.0`, ...) can predate the goldens the code is meant to match.

The filename hash covers `ssp_scenario`, `temp_target`, both decade years, `start_year`,
`ramp_up`, and the display baseline period (`base_{DISPLAY_BASELINE_START}_{DISPLAY_BASELINE_END}`).
Any change to that tuple renames every object, so goldens from before such a change are not
"stale numbers" — they are unreachable keys.

## Where it runs

`github-sai-validation-runner-ec2` (the `VALIDATION_RUNNER_INSTANCE_ID` repository variable; c5a.12xlarge, 48 vCPU / 93 GB,
300 GB gp3, us-west-2). Its instance role carries `AmazonS3FullAccess` and
`AmazonSSMManagedInstanceCore`, so it can write the goldens and be driven over SSM without
SSH. Target bucket: the `VALIDATION_S3_BUCKET` repository secret.

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
5. **Build the InVERT caches into the same `data/cache`.** The line-plot envelope for `tas`,
   `tasmin`, `tasmax` and `pr` (and `p-e` through `pr`) is served from `data/cache/invert/`,
   which `scripts/cache.py` does not build. Without it every request **silently falls back**
   to the members' band (`band_source: "members"`) — nothing fails. From the pinned
   checkout (`setup.sh` step F2 runs the same command):
   ```bash
   python -m backend.invert_variability --data-dir data/processed --cache-dir data/cache
   ```
   It reads the public InVERT store on R2 (~15 GB, once) and took ~4 minutes on the runner;
   every variable should report `built`, not `skipped`. Spot-check before the run: a `tas`
   request's `plot_data` should carry `band_source: "invert"` (`icefrac` and the exposure
   counts stay `members` — InVERT does not cover them). The band only reaches `plot_data`
   from sai-simulator-api#23 on; before it the API dropped it whatever its source.
6. **Set `api/.env`**: `S3_BUCKET_NAME`, `S3_CACHE_VERSION`, and the `SIMULATOR_*_DIR` paths.
7. **Delete `api/scripts/pregenerate_cache/progress.json`**, or the run skips everything as
   already done. (Conversely: leave it in place to resume an interrupted run.)
8. **Raise Redis's pub/sub output-buffer limit** before starting more than a few dozen
   workers. The driver subscribes to every task of a 2,000-task batch on the result backend,
   and results (whole `plot_data` payloads) arrive faster than it reads them. Redis's default
   `pubsub 32mb 8mb 60` then disconnects it — logged in `/var/log/redis6/redis6.log` as
   "closed for overcoming of output buffer limits" — and the driver dies on the reconnect
   (see Traps). As root:
   ```bash
   redis6-cli CONFIG SET client-output-buffer-limit "pubsub 1073741824 536870912 60"
   ```
   This is a runtime setting: it is lost when Redis restarts (including an instance
   stop/start), so set it again after any restart.
9. **Run it**: Redis up, N Celery workers with `--pool=solo` (the geospatial stack is not
   fork-safe, so parallelism comes from separate processes), then
   `python -m api.scripts.pregenerate_cache`.
10. **Verify** before believing it — see below.
11. **Point the harness at the new prefix** via `VALIDATION_S3_PREFIX` in `self-hosted.env`.

## Scale and resources

`generate_parameter_combinations()` returns **321,600** combinations: 320,880 standard
(8 SSPs × 21 temp targets × 56 start years × 10 vars, decades filtered to those starting at
or after the SAI start year) plus 720 variable-injection (144 each for `tas`, `tasmin`,
`tasmax`, `p-e` and `icefrac`; `tasmin`/`tasmax` joined in sai-simulator-api#23 — 321,312
and 432 before it). Each yields 2 GeoTIFFs and 1 plot_data file, so **964,800 objects,
~85 GB**.

Disk on the box is dominated by `data/cache` (~59 GB), **not** by the pregeneration output:
`bulk_upload_to_s3_task` unlinks each file after a successful upload, so `api/output_cache`
drains continuously and ends empty. An empty `output_cache` after a run is correct, not a
symptom.

Throughput is bound by the machine, not the driver. On c5a.12xlarge and r6a.12xlarge (48
vCPU = 24 physical cores) it plateaus at **~2.6 tasks/s whether 11 or 40 workers run** —
per-task time grows in step with the worker count while the driver sits at ~47% CPU — so a
full run takes ~30 h there. Each worker also grows from ~4 GB to 6–8 GB, so the 93 GB c5a
holds only ~11–12 before the kernel OOM-kills them (a killed worker's task is redelivered
after the 1 h visibility timeout; nothing is lost, the batch just stalls). What worked:
r6a.48xlarge (192 vCPU / 96 cores, 1.5 TB) with 96 workers and the Redis-shared global-tas
memo from sai-simulator-api#23 ran at **~11 tasks/s (~8 h)**. Resize with stop → modify
instance type → start (same instance, disk and runner registration), and put it back to
c5a.12xlarge afterwards — the PR validation workflow starts whatever type it finds.

## Verifying a run

"The driver exited" is not "the goldens are correct". Check all of:

- `Errors encountered: 0` in the driver log, and `len(progress.json)` equals the
  combination count (321,600 with sai-simulator-api#23).
- No `Failed to upload` across the worker logs, `skipped_existing` summing to 0 over the
  `bulk_upload_to_s3_task` results, and the Celery queue drained.
- **Freshness**: every object under the prefix has a `LastModified` from this run. A rerun
  under an existing `S3_CACHE_VERSION` that leaves old objects behind is the failure the
  counts alone do not show.
- Object counts per subprefix: `geotiffs` = 2 × combinations, `plot_data` = 1 ×.
- **Key agreement**: sample combinations across all SSPs, variables and decades, build each
  key with the consumer's own `default_s3_key`, and head it. This is the check that catches
  a producer/consumer hash divergence, and it is the one that matters.
- **Band source**: download a few `plot_data` objects; `tas` / `tasmin` / `tasmax` / `p-e`
  should report `band_source: "invert"` (step 5), with `band_low` / `band_high` on each series.

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
- **Re-uploads used to be silent no-ops.** Before sai-simulator-api#23, `bulk_upload` skipped
  every key that already existed, so a rerun under the same `S3_CACHE_VERSION` uploaded almost
  nothing, kept the previous run's objects, and reported no error (the 2026-10-05 run
  uploaded 864 of 963,936 files). The bucket has no versioning, so an overwrite cannot be
  undone either. Pregenerate only from an API that includes #23, and check freshness.
- **The driver dies when Redis drops its pub/sub connection.** Celery's Redis result backend
  reconnects through `connection.register_connect_callback`, which the installed redis-py no
  longer has (`AttributeError ... Did you mean: '_register_connect_callback'?`). Any pub/sub
  disconnect — in practice the output-buffer limit of step 8 — kills the driver. Progress is
  saved per batch, so restarting just `python -m api.scripts.pregenerate_cache` resumes from
  `progress.json` and redoes only the interrupted batch; the workers can keep running.
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

**v1.5.0 — 2026-10-08.** 321,600 combinations, 0 errors, 964,800 files uploaded (0 failed,
0 skipped), every `plot_data` object and the 643,200 expected GeoTIFFs fresh. Built from
`sai-simulator-api` `4afcf85` (#23, stacked on #21) pinned to this repo's `36729dc` (#65,
CESM time-stamp fix), on r6a.48xlarge with 96 workers, InVERT caches built and the pub/sub
limit raised (after one driver crash at 156,000, resumed). 240 standard and 50 injection
keys sampled across all SSPs, variables and decades resolved; sampled `plot_data` carried
the band with the expected sources. 864 GeoTIFFs from 2026-10-01 remain under the prefix:
the old injection keys for `tas` / `p-e` / `icefrac`, unreachable since the CESM params file
(part of the injection key) changed.

A dry run on 2026-10-05–07 (`b43d7c9`, c5a.12xlarge, 11–20 workers, ~29 h) generated
everything but published 864 files, because of the skip-existing upload (see Traps).

**v1.5.0 — 2026-10-01.** 321,312 combinations, 0 errors, 0 upload failures, 963,936 objects
/ 85.0 GB, ~8 hours at ~20 batches/hour with 28 workers. Built from `sai-simulator-api`
pinned to this repo's `4e9383e` ("Add UKESM1-1-LL as a third climate model"), against a
cache stamped `{v1.5.0, constrained, [CESM2-WACCM, MIROC-ES2H, UKESM1-1-LL], {}}`. Counts
matched expectation exactly and 400 sampled keys across all SSPs, variables and decades
resolved.

No `v1.4.0/` prefix exists — that regeneration never happened, so before this run the
goldens predated both the AR6-constrained FaIR default and the 1995-2014 display baseline.
