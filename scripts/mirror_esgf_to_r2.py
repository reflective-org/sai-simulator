#!/usr/bin/env python3
"""Mirror the UKESM1-0-LL simulator subset from ESGF (CEDA) into Cloudflare R2.

Option B of the UKESM plan: the new-generation ``g6-1p5-*`` runs and every
UKESM1-1-LL ssp245 run are unpublished, so the simulator's UKESM entry starts
as a projection-side-only UKESM1-0-LL contribution built from public CMIP6
data. This script stages exactly the files the zarrification needs into
``r2:reflective-data-store/UKESM1-0/{experiment}/{member}/{table}/{variable}/``.

What it selects (see ``SELECTIONS``), for historical + ssp245, members
r2/r3/r12 i1p1f2:

- ``Amon``  tas, pr, hfls          (monthly fields; hfls -> QFLX)
- ``SImon`` siconc, siconca        (-> ICEFRAC; r12 ssp245 has only
                                    ocean-grid siconc)
- ``day``   tas, pr                (exposure counts: days above 35/40 °C,
                                    below 0 °C on daily-mean tas; above
                                    10/20 mm/day on pr — the recipe from the
                                    retired scripts/process_daily_gauss.py)
- ``day``   tasmax, tasmin         (the stores' TREFHTMX/MN are ABSOLUTE
                                    monthly extremes of the daily fields —
                                    per the shipped MIROC store attrs — so
                                    CMOR's monthly-mean Amon tasmax/tasmin
                                    are the wrong quantity and are not
                                    mirrored)

Known upstream gaps (verified on the CEDA + DKRZ indexes, Sep 2026): ssp245
r12i1p1f2 publishes no hfls and no siconca in any table. The zarrifier handles
those (QFLX placeholder for that member; ICEFRAC from ocean-grid siconc).

Like the team's other mirrors, the destination is the state: each run diffs
the ESGF file manifest against R2 and transfers only what is missing or
size-changed, verifying both byte count and ESGF's SHA256 checksum while
streaming (the digest is also stored as R2 object metadata ``sha256``).
Short or corrupt uploads are deleted (multipart uploads aborted), never left
behind. CEDA is a public third-party server, so the JAMSTEC-mirror politeness
defaults apply: few workers, paced starts, retries that honor Retry-After, an
identifying User-Agent.

Files larger than ``PART_SIZE`` move as sequential ranged GETs feeding an R2
multipart upload, because CEDA drops or stalls multi-GB single connections
(observed 2026-09-08: truncation around ~0.5-0.9 GB and 300 s read timeouts
on the ``day`` files). A dropped connection then costs one 64 MiB part —
retried on its own — instead of the whole file.

Credentials: only the team's R2 env vars (``R2_ACCOUNT_ID``, ``R2_ACCESS_KEY``,
``R2_SECRET_KEY``) — ESGF is anonymous. Runs from anywhere; no hub needed.

Examples
    python scripts/mirror_esgf_to_r2.py --dry-run
    python scripts/mirror_esgf_to_r2.py --log-file mirror.log
    python scripts/mirror_esgf_to_r2.py --only historical/ --limit 5
"""

import argparse
import fcntl
import hashlib
import http.client
import json
import logging
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.copy_ukesm_to_r2 import (  # noqa: E402
    DEST_BUCKET,
    ensure_dir_prefix,
    list_sizes,
    make_r2_client,
    plan_transfers,
)

INDEX_NODE = "esgf.ceda.ac.uk"
DEFAULT_DEST_PREFIX = "UKESM1-0/"

MEMBERS = "r2i1p1f2,r3i1p1f2,r12i1p1f2"
BASE_QUERY = {
    "type": "File",
    "project": "CMIP6",
    "source_id": "UKESM1-0-LL",
    "latest": "true",
    "retracted": "false",
    "replica": "false",
    "fields": "title,url,size,checksum,checksum_type,variable_id,table_id,"
              "experiment_id,variant_label",
    "format": "application/solr+json",
}

# One entry per File-API query; merged over BASE_QUERY. Daily hfls (~19 GB) is
# deliberately NOT selected (QFLX is monthly, from Amon hfls), and neither are
# Amon tasmax/tasmin (monthly MEANS of the daily extremes — the stores carry
# absolute monthly extremes computed from the daily fields instead).
SELECTIONS = [
    {"experiment_id": "historical,ssp245", "variant_label": MEMBERS,
     "table_id": "Amon", "variable_id": "tas,pr,hfls"},
    {"experiment_id": "historical,ssp245", "variant_label": MEMBERS,
     "table_id": "SImon", "variable_id": "siconc,siconca"},
    {"experiment_id": "historical,ssp245", "variant_label": MEMBERS,
     "table_id": "day", "variable_id": "tas,pr,tasmax,tasmin"},
]

USER_AGENT = "reflective-esgf-mirror/0.1 (https://reflective.org; john@reflective.org)"

# Files above this size transfer as ranged parts (uniform size, R2 requires it
# for multipart; 64 MiB clears R2's 5 MiB minimum comfortably).
PART_SIZE = 64 * 1024 * 1024

log = logging.getLogger("esgf-mirror")

_pace_lock = Lock()


# --- pure helpers (unit-tested in tests/test_mirror_esgf_to_r2.py) -----------

def _one(value):
    """ESGF Solr fields arrive list-wrapped as often as not; unwrap."""
    return value[0] if isinstance(value, list) else value


def pick_http_url(urls) -> str:
    """The HTTPServer entry of an ESGF ``url`` field ('url|mime|service')."""
    for entry in urls:
        parts = entry.split("|")
        if parts[-1] == "HTTPServer":
            return parts[0]
    return ""


@dataclass(frozen=True)
class FileRecord:
    """One source file: where it lives, where it goes, and how to verify it.

    ``sha256`` may be empty for sources that publish no checksums (e.g. the
    JASMIN macloud listings) — verification is then byte-count only, matching
    the team's other HTML mirrors.
    """

    rel_key: str   # {experiment}/{member}/.../{filename}
    url: str
    size: int
    sha256: str = ""


def doc_to_record(doc: dict) -> FileRecord:
    """Map one File-API Solr doc to a :class:`FileRecord` (raises on gaps)."""
    url = pick_http_url(doc.get("url", []))
    if not url:
        raise ValueError(f"no HTTPServer url for {doc.get('title')}")
    checksum_type = _one(doc.get("checksum_type"))
    if checksum_type != "SHA256":
        raise ValueError(f"unexpected checksum type {checksum_type!r} for {doc.get('title')}")
    rel_key = "/".join((
        _one(doc["experiment_id"]),
        _one(doc["variant_label"]),
        _one(doc["table_id"]),
        _one(doc["variable_id"]),
        _one(doc["title"]),
    ))
    return FileRecord(rel_key, url, int(_one(doc["size"])), _one(doc["checksum"]))


class HashingReader:
    """Wrap a stream: count bytes and accumulate SHA256 as it is read."""

    def __init__(self, raw):
        self._raw = raw
        self._digest = hashlib.sha256()
        self.count = 0

    def read(self, amt=None):
        chunk = self._raw.read(amt)
        if chunk:
            self.count += len(chunk)
            self._digest.update(chunk)
        return chunk

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def part_ranges(total_size: int, part_size: int):
    """Inclusive (start, end) byte ranges: uniform ``part_size``, smaller tail."""
    if total_size <= 0 or part_size <= 0:
        raise ValueError("sizes must be positive")
    return [
        (start, min(start + part_size, total_size) - 1)
        for start in range(0, total_size, part_size)
    ]


# --- ESGF + transfer plumbing -------------------------------------------------

def fetch_manifest(index_node: str) -> dict:
    """rel_key -> :class:`FileRecord` for every file the selections match."""
    records = {}
    for selection in SELECTIONS:
        params = {**BASE_QUERY, **selection, "limit": 1000}
        url = f"https://{index_node}/esg-search/search?" + urllib.parse.urlencode(
            params, safe=","
        )
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=90) as resp:
            payload = json.load(resp)
        response = payload["response"]
        if response["numFound"] > len(response["docs"]):
            raise RuntimeError(
                f"selection {selection} matched {response['numFound']} files; "
                "paging not implemented because selections were sized under the "
                "1000-file limit — split the selection"
            )
        for doc in response["docs"]:
            record = doc_to_record(doc)
            existing = records.get(record.rel_key)
            if existing is not None and existing.sha256 != record.sha256:
                raise RuntimeError(f"conflicting manifest entries for {record.rel_key}")
            records[record.rel_key] = record
    return records


class RangeUnsupported(Exception):
    """The server ignored the Range header — never retry, never stream the body."""


def _ranged_get(url: str, start: int, end: int, attempts: int = 5) -> bytes:
    """One inclusive byte range, retried on its own with backoff."""
    expected = end - start + 1
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"}
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                # A 200 here means the server is sending the WHOLE file for
                # every part; reading it would multiply the transfer by the
                # part count. Fail loudly instead.
                if resp.status != 206:
                    raise RangeUnsupported(
                        f"{url.rsplit('/', 1)[-1]}: expected 206 Partial Content, got {resp.status}"
                    )
                data = resp.read()
            if len(data) != expected:
                raise OSError(f"short range read: {len(data)}/{expected} bytes")
            return data
        except RangeUnsupported:
            raise
        except Exception as exc:
            retriable = not (
                isinstance(exc, urllib.error.HTTPError)
                and exc.code not in (429, 500, 502, 503, 504)
            )
            if not retriable or attempt == attempts - 1:
                raise
            wait = 2.0 ** (attempt + 1)
            if isinstance(exc, urllib.error.HTTPError) and exc.headers.get("Retry-After"):
                wait = float(exc.headers["Retry-After"])
            log.info("range %d-%d of %s: %s — retrying in %.0fs",
                     start, end, url.rsplit("/", 1)[-1], type(exc).__name__, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


def _verification_error(record: FileRecord, got_bytes: int, got_sha: str) -> str:
    sha_state = "not checked" if not record.sha256 else (
        "ok" if got_sha == record.sha256 else "MISMATCH"
    )
    return (f"verification failed: {got_bytes}/{record.size} bytes, "
            f"sha256 {sha_state}")


def _verified(record: FileRecord, got_bytes: int, got_sha: str) -> bool:
    if got_bytes != record.size:
        return False
    return not record.sha256 or got_sha == record.sha256


def _transfer_single(dest, record: FileRecord, dest_key: str) -> int:
    """Small files: one streaming GET into a plain PUT-style upload."""
    # Lazy: keeps boto3 out of the import path CI's unit tests exercise.
    from boto3.s3.transfer import TransferConfig

    extra = {"Metadata": {"sha256": record.sha256}} if record.sha256 else {}
    req = urllib.request.Request(record.url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        stream = HashingReader(resp)
        dest.upload_fileobj(
            stream, DEST_BUCKET, dest_key,
            ExtraArgs=extra,
            Config=TransferConfig(use_threads=False),
        )
    if not _verified(record, stream.count, stream.hexdigest()):
        dest.delete_object(Bucket=DEST_BUCKET, Key=dest_key)
        raise OSError(_verification_error(record, stream.count, stream.hexdigest()))
    return stream.count


def _transfer_multipart(dest, record: FileRecord, dest_key: str) -> int:
    """Big files: sequential ranged GETs into an R2 multipart upload."""
    kwargs = {"Metadata": {"sha256": record.sha256}} if record.sha256 else {}
    mpu = dest.create_multipart_upload(Bucket=DEST_BUCKET, Key=dest_key, **kwargs)
    upload_id = mpu["UploadId"]
    digest = hashlib.sha256()
    sent = 0
    parts = []
    try:
        for number, (start, end) in enumerate(part_ranges(record.size, PART_SIZE), start=1):
            data = _ranged_get(record.url, start, end)
            digest.update(data)
            sent += len(data)
            resp = dest.upload_part(
                Bucket=DEST_BUCKET, Key=dest_key, UploadId=upload_id,
                PartNumber=number, Body=data,
            )
            parts.append({"ETag": resp["ETag"], "PartNumber": number})
        if not _verified(record, sent, digest.hexdigest()):
            raise OSError(_verification_error(record, sent, digest.hexdigest()))
        dest.complete_multipart_upload(
            Bucket=DEST_BUCKET, Key=dest_key, UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )
    except BaseException:
        # Never let a failing abort mask the real error (e.g. the upload had
        # in fact completed and the abort returns NoSuchUpload).
        try:
            dest.abort_multipart_upload(Bucket=DEST_BUCKET, Key=dest_key, UploadId=upload_id)
        except Exception as abort_exc:  # noqa: BLE001
            log.warning("abort of multipart upload %s failed: %s", dest_key, abort_exc)
        raise
    return sent


def transfer(dest, record: FileRecord, dest_key: str, delay: float) -> int:
    """Move one ESGF file into R2; verify size + SHA256; never leave junk."""
    with _pace_lock:
        time.sleep(delay)
    if record.size <= PART_SIZE:
        return _transfer_single(dest, record, dest_key)
    return _transfer_multipart(dest, record, dest_key)


def transfer_with_retry(dest, record, dest_key, delay, attempts=4) -> int:
    """Politely retry transient failures with exponential backoff."""
    for attempt in range(attempts):
        try:
            return transfer(dest, record, dest_key, delay)
        except urllib.error.HTTPError as exc:
            retriable = exc.code in (429, 500, 502, 503, 504)
            if not retriable or attempt == attempts - 1:
                raise
            wait = float(exc.headers.get("Retry-After") or 2 ** (attempt + 1))
            log.info("retrying %s in %.0fs (HTTP %d)", dest_key, wait, exc.code)
            time.sleep(wait)
        except RangeUnsupported:
            raise
        except (OSError, http.client.HTTPException):
            # IncompleteRead and friends are HTTPException, not OSError.
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    raise AssertionError("unreachable")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--index-node", default=INDEX_NODE)
    p.add_argument("--dest-prefix", default=DEFAULT_DEST_PREFIX,
                   help=f"destination prefix in R2 (default {DEFAULT_DEST_PREFIX})")
    p.add_argument("--only", action="append", default=[],
                   help="only copy keys containing this substring (repeatable)")
    p.add_argument("--exclude", action="append", default=[],
                   help="skip keys containing this substring (repeatable; beats --only)")
    p.add_argument("--limit", type=int, default=0,
                   help="max files this run (0 = everything missing)")
    p.add_argument("--workers", type=int, default=3,
                   help="concurrent transfers (keep low; CEDA is a shared public server)")
    p.add_argument("--delay", type=float, default=1.0,
                   help="seconds between download starts (default 1.0)")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be copied; move nothing")
    p.add_argument("--log-file", default=None)
    p.add_argument("--lock-file", default="/tmp/esgf-ukesm10-mirror.lock")
    args = p.parse_args()

    handlers = [logging.StreamHandler(sys.stderr)]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )

    lock = open(args.lock_file, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log.info("previous run still active (lock held); exiting")
        return 0

    dest_prefix = ensure_dir_prefix(args.dest_prefix)
    dest = make_r2_client()

    manifest = fetch_manifest(args.index_node)
    src_sizes = {rel: rec.size for rel, rec in manifest.items()}
    dest_sizes = list_sizes(dest, DEST_BUCKET, dest_prefix)
    plan = plan_transfers(src_sizes, dest_sizes, args.only, args.exclude)
    log.info(
        "%d in ESGF manifest, %d in r2://%s/%s: %d to copy "
        "(%d size-mismatched recopies), %d up to date, %d filtered out",
        len(manifest), len(dest_sizes), DEST_BUCKET, dest_prefix,
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
        log.info("dry run: %d file(s), %.2f GB would be copied from %s",
                 len(todo), todo_bytes / 1e9, args.index_node)
        return 0

    ok, failed, total_bytes = 0, 0, 0
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(transfer_with_retry, dest, manifest[rel],
                        dest_prefix + rel, args.delay): rel
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
