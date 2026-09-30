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
    "MIROC-ES2H_baseline.zarr",
    "MIROC-ES2H_ssp245.zarr",
    "MIROC-ES2H_historical.zarr",
    "MIROC-ES2H_G6-1.5K-SAI.zarr",
    "MIROC-ES2H_G6-0.5K-SAI.zarr",
    # UKESM1-1-LL (MACLOUD het-chem runs; built by scripts/zarrify_ukesm11.py).
    # The historical store ships all-NaN QFLX/ICEFRAC placeholders (MACLOUD
    # publishes no evap or sea ice for historical) — present with PLACEHOLDER
    # note attrs, so they warn rather than fail below.
    "UKESM1-1-LL_baseline.zarr",
    "UKESM1-1-LL_historical.zarr",
    "UKESM1-1-LL_G6-1.5K-SAI.zarr",
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

# Variables permitted to be absent for stores whose name contains the key.
# (MIROC used to be listed here for QFLX, but its stores now ship QFLX derived
# from JAMSTEC's monthly hfls — QFLX = hfls / 2.5e6 — so it is required again.)
ALLOWED_MISSING: dict[str, set[str]] = {
    # UKESM ships QFLX (= evap / 2.5e6) in its baseline and G6 stores; its
    # historical store carries QFLX/ICEFRAC as all-NaN PLACEHOLDER arrays,
    # which the note-attr check below reports as warnings, so nothing is
    # allowed-missing today. Add a model here only when a variable is absent
    # from the store entirely.
}

# Canonical dims the consumer pipeline aligns on.
EXPECTED_DIMS = {"member", "time", "lat", "lon"}


def check_store(store_uri: str) -> list[str]:
    """Return failure messages for one store. Empty list ⇒ all checks passed."""
    failures: list[str] = []
    # Vars this store is allowed to be missing (see ALLOWED_MISSING).
    allowed_missing: set[str] = set()
    for substring, vars_ in ALLOWED_MISSING.items():
        if substring in store_uri:
            allowed_missing |= vars_

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
                if var in allowed_missing:
                    print(f"  [{group}] {var} absent (allowed for this store)")
                else:
                    failures.append(f"  [{group}] missing variable: {var}")
                continue
            da = ds[var]
            # Producer-declared all-NaN placeholders (e.g. MIROC's ScenarioMIP store
            # carries QFLX/ICEFRAC stubs because hfls/siconc for that run are only on
            # an offline ESGF node). The array is well-formed, so this is a warning:
            # consumers fall back to the model's other no-SAI run for those variables.
            note = str(da.attrs.get("note", ""))
            if "PLACEHOLDER" in note.upper():
                print(f"{YELLOW}  [{group}/{var}] all-NaN placeholder — {note}{RESET}")
            elif "PARTIAL" in note.upper():
                # Real for some ensemble members and NaN for the rest, because the
                # source only covers those members (UKESM historical ICEFRAC: ESGF
                # publishes sea ice for one of the three). Well-formed, so a warning.
                print(f"{YELLOW}  [{group}/{var}] partial — {note}{RESET}")
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
    """Iterate the stores in :data:`STORE_NAMES` and check each. Returns the count of failing stores."""
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
