#!/usr/bin/env python3
"""Mirror the UKESM1.1 GeoMIP simulator subset from JASMIN MACLOUD into R2.

Source: Matthew Henry's public group workspace
``https://gws-access.jasmin.ac.uk/public/macloud/`` (see its README.txt),
which carries the full UKESM1-1-LL G6 family the hub bucket only partially
mirrored: G6-1p5k-SAI, its het-chem SSP2-4.5 branch parents, and per-member
historical.

Destination keys extend the layout the hub mirror established under
``r2:reflective-data-store/UKESM1-1/`` — ``{EXPERIMENT}/{member}/{filename}``
with the hub's experiment/member directory names — so files already staged
from the hub (e.g. the big ``seaice_day`` files) are skipped by the diff
rather than re-downloaded. Historical suites are keyed by SUITE ID
(``historical/u-by791/…``): their r-label mapping is not published anywhere in
the files, and pinning it into the keys would bake it in — the zarrifier takes
the mapping explicitly instead. It is r2=u-bz502, r3=u-by791, r12=u-bz897,
derived from weather continuity and confirmed by Matthew Henry on 2026-09-24.

Only the files the simulator stores need are selected (see ``WANTED`` /
``SUITES``): monthly tas/pr/evap/siconca, the daily tas mean-min-max triple,
daily pr, and daily seaice. MACLOUD publishes no checksums, so verification
is byte-count only (the ESGF mirror's machinery with an empty ``sha256``);
filename year ranges there are known to overstate actual coverage — the
zarrifier validates real time axes.

JASMIN is a shared public service: same politeness defaults as the other
mirrors (3 workers, paced starts, ranged 64 MiB parts retried individually).

Examples
    python scripts/mirror_macloud_to_r2.py --dry-run
    python scripts/mirror_macloud_to_r2.py --log-file macloud.log
    python scripts/mirror_macloud_to_r2.py --only historical/ --limit 2
"""

import argparse
import fcntl
import logging
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.copy_ukesm_to_r2 import (  # noqa: E402
    DEST_BUCKET,
    ensure_dir_prefix,
    list_sizes,
    make_r2_client,
    plan_transfers,
)
from scripts.mirror_esgf_to_r2 import (  # noqa: E402
    USER_AGENT,
    FileRecord,
    transfer_with_retry,
)

BASE_URL = "https://gws-access.jasmin.ac.uk/public/macloud"
DEFAULT_DEST_PREFIX = "UKESM1-1/"

# (macloud experiment dir, suite dir) -> destination "{EXPERIMENT}/{member}".
# SAI/SSP245 members follow the hub bucket's existing directory names; the
# suite->r-label pairs come from MACLOUD's README.txt (dp583/690/691 confirmed
# against the hub's own SAI member dirs). The MCB family (u-dh561/u-di183/
# u-di184 parents) is deliberately absent — MCB is not an SAI input.
SUITES = {
    ("G6-1p5k-SAI", "u-dp583"): "G6-1p5K-SAI/r2i1p1f1",
    ("G6-1p5k-SAI", "u-dp690"): "G6-1p5K-SAI/r3i1p1f1",
    ("G6-1p5k-SAI", "u-dp691"): "G6-1p5K-SAI/r12i1p1f1",
    ("SSP245_ukesm1p1", "u-dp902"): "SSP245/r2i1p1f1",
    ("SSP245_ukesm1p1", "u-dp903"): "SSP245/r3i1p1f1",
    ("SSP245_ukesm1p1", "u-dp904"): "SSP245/r12i1p1f1",
    ("historical_ukesm1p1", "u-by791"): "historical/u-by791",
    ("historical_ukesm1p1", "u-bz502"): "historical/u-bz502",
    ("historical_ukesm1p1", "u-bz897"): "historical/u-bz897",
}

# Filename prefixes the simulator pipeline consumes. Prefix matching keeps the
# heavy 3D/chemistry files (o3_*, ta_all_*, so4*_, aero_*, …) out.
WANTED = (
    "tas_mon_", "pr_mon_", "evap_mon_", "siconca_mon_",   # monthly fields
    "tas_day_", "tas_mean_min_max_day_",                  # daily tas triple
    "tas-min-mn-max_daily_",                              # …historical spelling
    "pr_day_", "pr_daily_",                               # daily pr
    "seaice_day_",                                        # daily sea ice
)

log = logging.getLogger("macloud-mirror")


# --- pure helpers (unit-tested in tests/test_mirror_macloud_to_r2.py) --------

class _LinkExtractor(HTMLParser):
    """Collect href values from anchor tags (same as the JAMSTEC mirror)."""

    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value:
                    self.hrefs.append(value)


def extract_nc_names(html: str) -> list:
    """Filenames of every .nc link in a directory-listing page."""
    parser = _LinkExtractor()
    parser.feed(html)
    names = []
    for href in parser.hrefs:
        if href.startswith("?") or href.startswith("/") or href.endswith("/"):
            continue
        name = href.rsplit("/", 1)[-1]
        if name.endswith(".nc"):
            names.append(name)
    return names


def wanted(filename: str) -> bool:
    """Keep only the files the simulator stores are built from."""
    return filename.startswith(WANTED)


# --- manifest ------------------------------------------------------------------

def _get(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req, timeout=timeout)


def _head_size(url: str, attempts: int = 4) -> int:
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(
                url, method="HEAD", headers={"User-Agent": USER_AGENT}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                return int(resp.headers["Content-Length"])
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    raise AssertionError("unreachable")


def fetch_manifest() -> dict:
    """rel_key -> FileRecord for every wanted file across all suites."""
    records = {}
    for (exp_dir, suite), dest in SUITES.items():
        listing_url = f"{BASE_URL}/{exp_dir}/{suite}/"
        with _get(listing_url) as resp:
            names = extract_nc_names(resp.read().decode("utf-8", "replace"))
        kept = [n for n in names if wanted(n)]
        log.info("%s/%s: %d files listed, %d selected", exp_dir, suite, len(names), len(kept))
        if not kept:
            # An empty listing must never turn into a quiet "nothing to do":
            # a changed page layout or a moved directory would look like success.
            raise RuntimeError(
                f"{listing_url}: no wanted .nc files found ({len(names)} .nc links parsed) — "
                "listing format changed or directory moved?"
            )
        for name in kept:
            url = f"{BASE_URL}/{exp_dir}/{suite}/{name}"
            rel_key = f"{dest}/{name}"
            records[rel_key] = FileRecord(rel_key, url, _head_size(url))
            time.sleep(0.2)  # gentle on the listing server
    return records


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dest-prefix", default=DEFAULT_DEST_PREFIX)
    p.add_argument("--only", action="append", default=[],
                   help="only copy keys containing this substring (repeatable)")
    p.add_argument("--exclude", action="append", default=[],
                   help="skip keys containing this substring (repeatable; beats --only)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=3,
                   help="concurrent transfers (keep low; JASMIN is a shared public server)")
    p.add_argument("--delay", type=float, default=1.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--log-file", default=None)
    p.add_argument("--lock-file", default="/tmp/macloud-mirror.lock")
    args = p.parse_args()

    handlers = [logging.StreamHandler(sys.stderr)]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", handlers=handlers)

    lock = open(args.lock_file, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log.info("previous run still active (lock held); exiting")
        return 0

    dest_prefix = ensure_dir_prefix(args.dest_prefix)
    dest = make_r2_client()

    manifest = fetch_manifest()
    src_sizes = {rel: rec.size for rel, rec in manifest.items()}
    dest_sizes = list_sizes(dest, DEST_BUCKET, dest_prefix)
    plan = plan_transfers(src_sizes, dest_sizes, args.only, args.exclude)
    log.info(
        "%d in MACLOUD manifest, %d under r2://%s/%s: %d to copy "
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
        log.info("dry run: %d file(s), %.2f GB would be copied from MACLOUD",
                 len(todo), todo_bytes / 1e9)
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
