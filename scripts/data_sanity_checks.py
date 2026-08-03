"""Sanity-check the Zarr training-data stores at :data:`version.ZARR_STORES_URI`
before any fitting or dashboard work runs.

For every ``(store, group, variable)`` triple the consumer pipeline
(:mod:`scripts.process_monthly_gauss`) reads, verify:

- the store opens via :func:`xarray.open_zarr` (consolidated metadata),
- the group exists,
- the expected variable is present,
- the variable carries the canonical ``(member, time, lat, lon)`` dims and
  neither ``member`` nor ``time`` is empty.

This is a few-second metadata-only smoke test — no chunk reads — designed
to surface bucket / credential / schema issues up front rather than several
minutes into a processing run.

Exits with status ``2`` on any failure so callers (e.g. ``setup/steps.sh``)
can distinguish "checks failed" from generic Python errors.
"""
import argparse
import sys
from pathlib import Path

import xarray as xr

# Allow running as a top-level script (``python scripts/data_sanity_checks.py``)
# without installing the project.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from version import ZARR_STORES_URI

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"


STORE_NAMES = [
    "CESM2-WACCM-MA_baseline.zarr",
    "CESM2-WACCM-MA_historical.zarr",
    "CESM2-WACCM-MA_1.5K-SAI.zarr",
    "CESM2-WACCM-MA_1.0K-SAI.zarr",
    "CESM2-WACCM-MA_0.5K-SAI.zarr",
]

# Per-group required variables. Mirrors the routing logic in
# :func:`scripts.process_monthly_gauss._group_for_var` so the consumer-side
# expectations stay in sync.
REQUIRED_GROUPS: dict[str, list[str]] = {
    "Mon/atmos_2d": ["TREFHT", "PRECT", "QFLX", "ICEFRAC"],
    "Mon/atmos_2d_tasminmax": ["TREFHTMN", "TREFHTMX"],
    "Mon/atmos_2d_derived": [
        "TREFHT_above_35",
        "TREFHT_above_40",
        "TREFHT_below_0",
        "PRECT_above_10",
        "PRECT_above_20",
    ],
}

# Canonical dims the consumer pipeline aligns on.
EXPECTED_DIMS = {"member", "time", "lat", "lon"}


def check_store(store_uri: str) -> list[str]:
    """Return failure messages for one store. Empty list ⇒ all checks passed."""
    failures: list[str] = []
    for group, expected_vars in REQUIRED_GROUPS.items():
        try:
            ds = xr.open_zarr(store_uri, group=group, consolidated=True)
        except Exception as exc:
            failures.append(
                f"  [{group}] open failed — {type(exc).__name__}: {exc}"
            )
            continue
        for var in expected_vars:
            if var not in ds.data_vars:
                failures.append(f"  [{group}] missing variable: {var}")
                continue
            da = ds[var]
            missing_dims = EXPECTED_DIMS - set(da.dims)
            if missing_dims:
                failures.append(
                    f"  [{group}/{var}] dims {da.dims} missing "
                    f"{sorted(missing_dims)}"
                )
                continue
            if da.sizes["member"] == 0:
                failures.append(f"  [{group}/{var}] empty member dim")
            if da.sizes["time"] == 0:
                failures.append(f"  [{group}/{var}] empty time dim")
    return failures


def run_checks(zarr_uri: str = ZARR_STORES_URI) -> int:
    """Iterate the 5 stores and check each. Returns the count of failing stores."""
    print(f"Zarr root: {zarr_uri}\n")
    failed_stores = 0
    for store_name in STORE_NAMES:
        store_uri = f"{zarr_uri}/{store_name}"
        print(f"{store_name}:")
        failures = check_store(store_uri)
        if failures:
            failed_stores += 1
            for msg in failures:
                print(f"{RED}{msg}{RESET}")
        else:
            print(f"  {GREEN}OK{RESET}")
        print()
    return failed_stores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanity-check the Zarr training-data stores."
    )
    parser.add_argument(
        "--zarr_uri",
        default=ZARR_STORES_URI,
        help=f"Override the Zarr store root (default: {ZARR_STORES_URI}).",
    )
    # Legacy CLI flags from the pre-Zarr file-tree world. Accepted but ignored
    # so an older ``setup/steps.sh`` invocation doesn't crash with
    # ``unrecognized arguments``.
    parser.add_argument("--data_dir", default=None, help="Ignored (legacy).")
    parser.add_argument(
        "--expected_bucket_uri", default=None, help="Ignored (legacy)."
    )
    parser.add_argument(
        "--include_lens_files", action="store_true", help="Ignored (legacy)."
    )
    parser.add_argument(
        "--required_variables", default=None, help="Ignored (legacy)."
    )
    parser.add_argument(
        "--min_ensemble_members", type=int, default=3, help="Ignored (legacy)."
    )
    args = parser.parse_args()

    failed = run_checks(args.zarr_uri)
    if failed:
        print(
            f"{RED}Sanity checks failed for {failed} store(s).{RESET}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    print(f"{GREEN}All sanity checks passed.{RESET}")


if __name__ == "__main__":
    main()
