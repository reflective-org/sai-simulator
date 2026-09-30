#!/usr/bin/env python3
"""Stage raw UKESM NetCDF files from the hub bucket into Cloudflare R2.

The Met Office side uploads UKESM1-1-LL output to the 2i2c hub's AWS bucket
(``s3://reflective-persistent-prod-large/UKESM1-1/{experiment}/{member}/*.nc``),
which is only reachable with hub credentials. The simulator's zarrification
step runs elsewhere with R2 credentials, so this script mirrors that prefix
into the R2 bucket that already holds the simulator's data
(``reflective-data-store``), preserving the key layout 1:1. The producer-side
zarrification (see the ``netCDF_to_zarr_scripts`` repo) then reads the staged
NetCDFs from R2 and writes the ``simulator-inputs/UKESM1-1-LL_*.zarr`` stores.

Modeled on ``jamstec-s3-mirror/mirror_jamstec.py``: the destination bucket is
the state, a diff decides what moves, and short/failed transfers never leave a
completed object behind — so the script is idempotent and safe to re-run as
the remaining UKESM files (tasmax/tasmin/siconc as of Sep 2026) land on the
hub. Unlike the JAMSTEC mirror there is no politeness machinery: both ends are
ours. Objects stream S3 → R2 through memory; nothing touches local disk.

Where to run it
    On the hub (JupyterLab terminal), where the ambient web-identity role can
    read the source bucket and boto3 auto-refreshes it. Off-hub, short-lived
    ``issue-creds`` credentials via ``--profile`` work for runs that finish
    within their lifetime.

Credentials
    Source: boto3's normal chain (ambient hub role, or ``--profile``).
    Destination: the team's usual R2 env vars —
        R2_ACCOUNT_ID   Cloudflare account id (endpoint subdomain)
        R2_ACCESS_KEY   R2 access key id
        R2_SECRET_KEY   R2 secret access key

Cost note
    AWS charges egress per byte copied out of the source bucket (R2 ingress is
    free), so always start with ``--dry-run`` — it prints exactly what would
    move and how big it is, and ``--exclude`` can drop the heavy 3D fields the
    simulator never reads (85-level ozone/temperature; "ta_" does not match
    the 2m "tas_" files).

Examples
    python copy_ukesm_to_r2.py --dry-run
    python copy_ukesm_to_r2.py --exclude o3_ --exclude ta_ --log-file copy.log
    python copy_ukesm_to_r2.py --only SSP245/ --limit 10
"""

import argparse
import fcntl
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

# boto3/botocore are imported lazily, inside the S3 helpers that need them. The
# pure planning core below is unit-tested in CI, whose requirements-test.txt
# deliberately stays minimal — the same reason ``fire`` and ``xesmf`` are lazy
# elsewhere in scripts/.

SRC_BUCKET = "reflective-persistent-prod-large"
DEST_BUCKET = "reflective-data-store"
# The hub-side prefix the Met Office uploads under; mirrored verbatim into R2.
DEFAULT_PREFIX = "UKESM1-1/"

RETRY_SETTINGS = {"max_attempts": 8, "mode": "adaptive"}


def retry_config():
    """Shared botocore retry policy (lazy import — see the note above).

    Copies RETRY_SETTINGS because botocore rewrites the dict it is handed
    (``max_attempts`` becomes ``total_max_attempts``), which would leave the
    module constant reading differently after the first client is built.
    """
    from botocore.config import Config

    return Config(retries=dict(RETRY_SETTINGS))


log = logging.getLogger("copy-ukesm")


# --- pure planning core (unit-tested in tests/test_copy_ukesm_to_r2.py) -----

def ensure_dir_prefix(prefix: str) -> str:
    """Normalize to exactly one trailing slash; '' stays '' (whole bucket)."""
    return prefix.rstrip("/") + "/" if prefix.rstrip("/") else ""


def matches_filters(rel_key: str, only, exclude) -> bool:
    """Substring filters on the prefix-relative key; exclude beats only."""
    if any(s in rel_key for s in exclude):
        return False
    if only and not any(s in rel_key for s in only):
        return False
    return True


@dataclass
class Plan:
    """What one run would move: keys are relative to the source prefix."""

    to_copy: list = field(default_factory=list)      # missing + mismatched, sorted
    mismatched: list = field(default_factory=list)   # exist at dest with wrong size
    up_to_date: int = 0                              # present with matching size
    filtered_out: int = 0                            # dropped by --only/--exclude
    bytes_to_copy: int = 0


def plan_transfers(src_sizes: dict, dest_sizes: dict, only=(), exclude=()) -> Plan:
    """Diff source against destination; copy what's missing or size-mismatched.

    Size is the only integrity signal used (multipart uploads make ETags
    incomparable) — the same contract the team's JAMSTEC mirror relies on. A
    destination object whose size differs is recopied, so a corrected upstream
    upload propagates on the next run.
    """
    plan = Plan()
    for rel, size in sorted(src_sizes.items()):
        if rel.endswith("/"):  # zero-byte "directory" placeholder objects
            continue
        if not matches_filters(rel, only, exclude):
            plan.filtered_out += 1
            continue
        if rel not in dest_sizes:
            plan.to_copy.append(rel)
            plan.bytes_to_copy += size
        elif dest_sizes[rel] != size:
            plan.to_copy.append(rel)
            plan.mismatched.append(rel)
            plan.bytes_to_copy += size
        else:
            plan.up_to_date += 1
    return plan


# --- boto3 plumbing ----------------------------------------------------------

def list_sizes(client, bucket: str, prefix: str) -> dict:
    """Map of prefix-relative key -> byte size, via paginated ListObjectsV2."""
    sizes = {}
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=prefix
    ):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix):]
            if rel:
                sizes[rel] = obj["Size"]
    return sizes


def make_src_client(profile: str | None):
    """Source S3 client from the default chain (hub role) or a named profile."""
    import boto3

    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    return session.client("s3", config=retry_config())


def make_r2_client():
    """Destination client for R2 from the team's usual env vars."""
    import boto3

    env = {name: os.environ.get(name) for name in
           ("R2_ACCOUNT_ID", "R2_ACCESS_KEY", "R2_SECRET_KEY")}
    missing = [name for name, value in env.items() if not value]
    if missing:
        raise SystemExit(
            "missing environment variable(s): " + ", ".join(missing)
            + " — export the same R2 credentials the zarrification notebooks use"
        )
    return boto3.client(
        "s3",
        endpoint_url="https://{}.r2.cloudflarestorage.com".format(env["R2_ACCOUNT_ID"]),
        aws_access_key_id=env["R2_ACCESS_KEY"],
        aws_secret_access_key=env["R2_SECRET_KEY"],
        region_name="auto",
        config=retry_config(),
    )


def copy_one(src, dest, src_key: str, dest_key: str, expected: int) -> int:
    """Stream one object S3 -> R2; verify size; never leave a short object."""
    from boto3.s3.transfer import TransferConfig

    body = src.get_object(Bucket=SRC_BUCKET, Key=src_key)["Body"]
    try:
        # use_threads=False: the source body is a non-seekable stream, so the
        # multipart uploader must consume it strictly sequentially.
        dest.upload_fileobj(
            body, DEST_BUCKET, dest_key, Config=TransferConfig(use_threads=False)
        )
    finally:
        body.close()
    landed = dest.head_object(Bucket=DEST_BUCKET, Key=dest_key)["ContentLength"]
    if landed != expected:
        dest.delete_object(Bucket=DEST_BUCKET, Key=dest_key)
        raise OSError(f"size mismatch after upload: got {landed} of {expected} bytes")
    return landed


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--src-prefix", default=DEFAULT_PREFIX,
                   help=f"source prefix in {SRC_BUCKET} (default {DEFAULT_PREFIX})")
    p.add_argument("--dest-prefix", default=None,
                   help="destination prefix in R2 (default: same as --src-prefix)")
    p.add_argument("--only", action="append", default=[],
                   help="only copy keys containing this substring (repeatable)")
    p.add_argument("--exclude", action="append", default=[],
                   help="skip keys containing this substring (repeatable; beats --only)")
    p.add_argument("--limit", type=int, default=0,
                   help="max files this run (0 = everything missing)")
    p.add_argument("--workers", type=int, default=8,
                   help="concurrent transfers (default 8; both endpoints are ours)")
    p.add_argument("--profile", default=None,
                   help="AWS profile for the SOURCE bucket (default: ambient hub role)")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be copied; move nothing")
    p.add_argument("--log-file", default=None)
    p.add_argument("--lock-file", default="/tmp/copy-ukesm-to-r2.lock")
    args = p.parse_args()

    handlers = [logging.StreamHandler(sys.stderr)]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )

    # One run at a time: overlapping invocations exit instead of double-copying.
    lock = open(args.lock_file, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log.info("previous run still active (lock held); exiting")
        return 0

    src_prefix = ensure_dir_prefix(args.src_prefix)
    dest_prefix = ensure_dir_prefix(
        args.src_prefix if args.dest_prefix is None else args.dest_prefix
    )

    src = make_src_client(args.profile)
    dest = make_r2_client()

    src_sizes = list_sizes(src, SRC_BUCKET, src_prefix)
    dest_sizes = list_sizes(dest, DEST_BUCKET, dest_prefix)
    plan = plan_transfers(src_sizes, dest_sizes, args.only, args.exclude)
    log.info(
        "%d in s3://%s/%s, %d in r2://%s/%s: %d to copy "
        "(%d size-mismatched recopies), %d up to date, %d filtered out",
        len(src_sizes), SRC_BUCKET, src_prefix,
        len(dest_sizes), DEST_BUCKET, dest_prefix,
        len(plan.to_copy), len(plan.mismatched), plan.up_to_date, plan.filtered_out,
    )
    for rel in plan.mismatched:
        log.warning("size mismatch at destination, will recopy: %s", rel)

    todo = plan.to_copy[: args.limit] if args.limit else plan.to_copy
    todo_bytes = sum(src_sizes[rel] for rel in todo)
    if not todo:
        log.info("nothing to do — R2 is up to date")
        return 0
    if args.dry_run:
        for rel in todo:
            log.info("COPY %s%s (%.1f MB)", dest_prefix, rel, src_sizes[rel] / 1e6)
        log.info("dry run: %d file(s), %.2f GB would be copied (AWS egress applies)",
                 len(todo), todo_bytes / 1e9)
        return 0

    ok, failed, total_bytes = 0, 0, 0
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(copy_one, src, dest,
                        src_prefix + rel, dest_prefix + rel, src_sizes[rel]): rel
            for rel in todo
        }
        for fut in as_completed(futures):
            rel = futures[fut]
            try:
                size = fut.result()
                ok += 1
                total_bytes += size
                log.info("done %s (%.1f MB) [%d/%d]",
                         rel, size / 1e6, ok + failed, len(todo))
            except Exception as exc:
                failed += 1
                log.warning("FAILED %s: %s", rel, exc)

    elapsed = time.monotonic() - started
    log.info("finished: %d ok, %d failed, %.2f GB in %.0f min",
             ok, failed, total_bytes / 1e9, elapsed / 60)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
