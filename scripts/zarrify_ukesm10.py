#!/usr/bin/env python3
"""Build the UKESM1-0-LL simulator-input Zarr stores from the mirrored raw files.

Consumes the CMIP6 NetCDFs staged by ``scripts/mirror_esgf_to_r2.py`` under
``r2:reflective-data-store/UKESM1-0/`` and writes two stores next to the
existing CESM/MIROC ones:

    simulator-inputs/UKESM1-0-LL_historical.zarr   (1850-2014)
    simulator-inputs/UKESM1-0-LL_ssp245.zarr       (2015-2100)

Schema matches the shipped MIROC-ES2H stores exactly (verified against
``MIROC-ES2H_ssp245.zarr``): zarr v3 with consolidated metadata, groups

- ``Mon/atmos_2d``            TREFHT [K] = Amon tas
                              PRECT  [m/s] = Amon pr / 1000
                              QFLX   [kg/m2/s] = Amon hfls / 2.5e6
                              ICEFRAC [0-1] = SImon siconc / 100, xESMF-
                              bilinear-regridded from the eORCA1 ocean grid
                              to the n96 atmos grid, land NaN -> 0
- ``Mon/atmos_2d_tasminmax``  TREFHTMX/MN = ABSOLUTE monthly max/min of
                              daily tasmax/tasmin (CESM convention, per the
                              MIROC store attrs — not CMOR's Amon monthly
                              means)
- ``Mon/atmos_2d_derived``    TREFHT_above_35/40, TREFHT_below_0,
                              PRECT_above_10/20: monthly day-counts from
                              daily tas / pr (thresholds 35/40/0 °C and
                              10/20 mm/day — the recipe of the retired
                              scripts/process_daily_gauss.py)

with dims ``(member, time, lat, lon)``, member coord = the true variant
labels (identical across both stores — the consumer stitches historical into
ssp245 by member key), monthly chunks ``(1, 120, lat, lon)``, and time
normalized to start-of-month stamps on the model's native 360-day calendar.

ICEFRAC uses ONE recipe everywhere (regridded ocean-grid ``siconc``) rather
than mixing ``siconca`` for the members that have it with a regrid for the
one that doesn't (ssp245 r12i1p1f2 publishes no siconca) — no per-member or
cross-store method seams. The shipped ``siconca`` is used as a same-run
validation target instead; ``--validate-icefrac`` prints the comparison.

Known upstream gap baked in with a ``note``: ssp245 r12i1p1f2 publishes no
hfls anywhere, so that member's QFLX slice is NaN in the ssp245 store.

Credentials: the team's R2 env vars (R2_ACCOUNT_ID / R2_ACCESS_KEY /
R2_SECRET_KEY). Examples:

    python scripts/zarrify_ukesm10.py --experiment historical --dry-run
    python scripts/zarrify_ukesm10.py --experiment historical
    python scripts/zarrify_ukesm10.py --experiment ssp245 --validate-icefrac
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.zarrify_common import (  # noqa: E402
    DEST_BUCKET,
    PR_COUNTS,
    STORE_GROUPS,
    TAS_COUNTS,
    assert_360_day,
    finalize_store,
    monthly_count,
    monthly_extreme,
    month_starts,       # noqa: F401  (re-exported for tests)
    normalize_monthly_time,
    r2_fs,
    stack_members,
    to_icefrac,
    verify_store as _verify_store_generic,
    write_group as _write_group_generic,
)

RAW_PREFIX = "UKESM1-0"
MODEL = "UKESM1-0-LL"
MEMBERS = ["r2i1p1f2", "r3i1p1f2", "r12i1p1f2"]

MONTHLY_CHUNKS = {"member": 1, "time": 120, "lat": 144, "lon": 192}

TODAY = date.today().isoformat()
NOTES = {
    "TREFHT": "2 m air temperature (CMIP6 CMOR Amon tas, K).",
    "PRECT": "CMOR Amon pr converted kg m-2 s-1 -> m/s (pr/1000), matching CESM PRECT.",
    "QFLX": (
        "QFLX = Amon hfls / 2.5e6 (kg/m2/s), same conversion as the MIROC stores. "
        "ssp245 r12i1p1f2 publishes no hfls in any CMIP6 table (checked CEDA+DKRZ "
        f"indexes), so that member is NaN in the ssp245 store. Built {TODAY} per J. Orcutt."
    ),
    "ICEFRAC": (
        "ICEFRAC = CMOR SImon siconc/100 (CMIP6.*.MOHC.UKESM1-0-LL.*.gn via "
        "esgf.ceda.ac.uk), regridded xESMF bilinear from the eORCA1 ocean grid to "
        "the 144x192 atmos grid, land NaN->0, clipped [0,1]. One recipe for every "
        "member and both stores; ssp245 r12i1p1f2 publishes no siconca, so the "
        "producer-regridded siconca is used only as a same-run validation target "
        f"(--validate-icefrac). Built {TODAY} per J. Orcutt."
    ),
    "TREFHTMX": "monthly MAXIMUM of daily tasmax (absolute monthly extreme, K); CMOR day table",
    "TREFHTMN": "monthly MINIMUM of daily tasmin (absolute monthly extreme, K); CMOR day table",
}

EXPERIMENT_SPANS = {"historical": (1850, 2014), "ssp245": (2015, 2100)}

log = logging.getLogger("zarrify-ukesm")


# --- pure helpers (shared ones live in scripts/zarrify_common.py) -------------

def expected_months(experiment: str) -> int:
    y0, y1 = EXPERIMENT_SPANS[experiment]
    return (y1 - y0 + 1) * 12


def open_raw(fs, experiment: str, member: str, table: str, var: str,
             time_chunk: int) -> xr.DataArray:
    """Open one (experiment, member, table, variable) as a lazy time-concat."""
    pattern = f"{DEST_BUCKET}/{RAW_PREFIX}/{experiment}/{member}/{table}/{var}/*.nc"
    paths = sorted(fs.glob(pattern))
    if not paths:
        raise FileNotFoundError(pattern)
    pieces = []
    for p in paths:
        ds = xr.open_dataset(
            fs.open(p, "rb"), engine="h5netcdf", chunks={"time": time_chunk},
            use_cftime=True,
        )
        pieces.append(ds[var])
    da = xr.concat(pieces, dim="time") if len(pieces) > 1 else pieces[0]
    assert_360_day(da.time)
    return da


def build_icefrac_regridder(fs, experiment: str, member: str, target: xr.DataArray):
    """xESMF bilinear regridder from the eORCA1 siconc grid to the atmos grid."""
    import xesmf as xe

    sample = open_raw(fs, experiment, member, "SImon", "siconc", 120)
    src = sample.isel(time=0).to_dataset(name="siconc")
    dest = xr.Dataset({"lat": target.lat, "lon": target.lon})
    return xe.Regridder(src, dest, method="bilinear",
                        ignore_degenerate=True, unmapped_to_nan=True), sample


def compute_group_atmos_2d(fs, experiment: str) -> xr.Dataset:
    trefht, prect, qflx, icefrac = {}, {}, {}, {}
    regridder = None
    for member in MEMBERS:
        tas = open_raw(fs, experiment, member, "Amon", "tas", 120)
        trefht[member] = tas.astype("float32")
        prect[member] = (open_raw(fs, experiment, member, "Amon", "pr", 120) / 1000.0
                         ).astype("float32")
        try:
            qflx[member] = (open_raw(fs, experiment, member, "Amon", "hfls", 120) / 2.5e6
                            ).astype("float32")
        except FileNotFoundError:
            log.warning("%s/%s: no hfls — QFLX NaN for this member", experiment, member)
            qflx[member] = xr.full_like(trefht[member], np.nan, dtype="float32")
        if regridder is None:
            regridder, _ = build_icefrac_regridder(fs, experiment, member, tas)
        siconc = open_raw(fs, experiment, member, "SImon", "siconc", 120)
        regridded = regridder(siconc, keep_attrs=False)  # CMOR siconc is percent
        icefrac[member] = to_icefrac(regridded, "%").assign_coords(lat=tas.lat, lon=tas.lon)

    ds = xr.Dataset({
        "TREFHT": stack_members(trefht),
        "PRECT": stack_members(prect),
        "QFLX": stack_members(qflx),
        "ICEFRAC": stack_members(icefrac),
    })
    ds = normalize_monthly_time(ds)
    ds["QFLX"].attrs["units"] = "kg/m2/s"
    for var in ("TREFHT", "PRECT", "QFLX", "ICEFRAC"):
        ds[var].attrs["note"] = NOTES[var]
    return ds


def compute_group_tasminmax(fs, experiment: str, day_chunk: int) -> xr.Dataset:
    mx, mn = {}, {}
    for member in MEMBERS:
        tasmax = open_raw(fs, experiment, member, "day", "tasmax", day_chunk)
        tasmin = open_raw(fs, experiment, member, "day", "tasmin", day_chunk)
        mx[member] = monthly_extreme(tasmax, "max").astype("float32")
        mn[member] = monthly_extreme(tasmin, "min").astype("float32")
    ds = xr.Dataset({"TREFHTMX": stack_members(mx), "TREFHTMN": stack_members(mn)})
    ds["TREFHTMX"].attrs["note"] = NOTES["TREFHTMX"]
    ds["TREFHTMN"].attrs["note"] = NOTES["TREFHTMN"]
    return ds


def compute_group_derived(fs, experiment: str, day_chunk: int) -> xr.Dataset:
    out = {name: {} for name in list(TAS_COUNTS) + list(PR_COUNTS)}
    for member in MEMBERS:
        tas = open_raw(fs, experiment, member, "day", "tas", day_chunk)
        pr = open_raw(fs, experiment, member, "day", "pr", day_chunk)
        for name, (threshold, direction) in TAS_COUNTS.items():
            out[name][member] = monthly_count(tas, threshold, direction)
        for name, (threshold, direction) in PR_COUNTS.items():
            out[name][member] = monthly_count(pr, threshold, direction)
    return xr.Dataset({name: stack_members(per) for name, per in out.items()})


def write_group(store: str, group: str, ds: xr.Dataset) -> None:
    _write_group_generic(store, group, ds, MONTHLY_CHUNKS)


def verify_store(store: str, experiment: str) -> list:
    return _verify_store_generic(store, STORE_GROUPS, len(MEMBERS),
                                 EXPERIMENT_SPANS[experiment], 144, 192)


def validate_icefrac(fs, experiment: str, store_icefrac: xr.DataArray) -> None:
    """Same-run check: regridded-siconc ICEFRAC vs the producer's siconca."""
    for member in MEMBERS:
        try:
            siconca = open_raw(fs, experiment, member, "SImon", "siconca", 120)
        except FileNotFoundError:
            log.info("%s/%s: no siconca published — skipping validation", experiment, member)
            continue
        ours = store_icefrac.sel(member=member)
        theirs = to_icefrac(siconca, "%").assign_coords(time=ours.time, lat=ours.lat, lon=ours.lon)
        north = ours.lat > 0
        sept = ours.time.dt.month == 9
        weights = np.cos(np.deg2rad(ours.lat))
        area_ours = (ours.where(north).isel(time=sept) * weights).sum(("lat", "lon"))
        area_theirs = (theirs.where(north).isel(time=sept) * weights).sum(("lat", "lon"))
        rel = float((np.abs(area_ours - area_theirs) / area_theirs).mean().compute())
        log.info("%s/%s ICEFRAC validation: mean |Δ| Sept NH ice area vs siconca = %.2f%%",
                 experiment, member, 100 * rel)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--experiment", choices=list(EXPERIMENT_SPANS), required=True)
    p.add_argument("--groups", default="atmos_2d,tasminmax,derived",
                   help="comma list of groups to (re)build")
    p.add_argument("--day-chunk", type=int, default=1800,
                   help="dask chunk length along time for daily reads (default 1800 = 5y)")
    p.add_argument("--dry-run", action="store_true",
                   help="build the lazy datasets and report shapes; write nothing")
    p.add_argument("--validate-icefrac", action="store_true")
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument("--log-file", default=None)
    args = p.parse_args()

    handlers = [logging.StreamHandler(sys.stderr)]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", handlers=handlers)

    fs = r2_fs()
    store = f"{MODEL}_{args.experiment}.zarr"
    wanted = set(args.groups.split(","))
    n_expected = expected_months(args.experiment)

    builders = {
        "atmos_2d": ("Mon/atmos_2d", lambda: compute_group_atmos_2d(fs, args.experiment)),
        "tasminmax": ("Mon/atmos_2d_tasminmax",
                      lambda: compute_group_tasminmax(fs, args.experiment, args.day_chunk)),
        "derived": ("Mon/atmos_2d_derived",
                    lambda: compute_group_derived(fs, args.experiment, args.day_chunk)),
    }

    for key, (group, build) in builders.items():
        if key not in wanted:
            continue
        ds = build()
        if ds.sizes["time"] != n_expected:
            raise SystemExit(
                f"{group}: time={ds.sizes['time']}, expected {n_expected} — raw data incomplete?"
            )
        if args.dry_run:
            log.info("dry run %s :: %s -> %s vars %s", store, group,
                     list(ds.data_vars), dict(ds.sizes))
            continue
        write_group(store, group, ds)
        if key == "atmos_2d" and args.validate_icefrac:
            validate_icefrac(fs, args.experiment, ds["ICEFRAC"])

    if args.dry_run:
        return 0

    finalize_store(fs, store)
    if not args.skip_verify:
        failures = verify_store(store, args.experiment)
        for f in failures:
            log.error("VERIFY %s", f)
        if failures:
            return 2
        log.info("verify OK: %s reads back with the expected schema", store)
    return 0


if __name__ == "__main__":
    sys.exit(main())
