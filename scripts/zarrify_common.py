#!/usr/bin/env python3
"""Shared machinery for building simulator-input Zarr stores.

Used by ``scripts/zarrify_ukesm10.py`` (CMOR-file inputs) and
``scripts/zarrify_ukesm11.py`` (MACLOUD u-suite inputs). Everything here is
model-agnostic: 360-day-calendar monthly aggregation, unit-ish converters,
member stacking, and the zarr-v3 store writing/verification that matches the
shipped MIROC/CESM stores (see the store-schema notes in those scripts).

Alignment policy: every combine step uses ``join="exact"``. xarray's default
outer join would silently NaN-pad a member or variable whose time/lat/lon axis
differs from its siblings, and a length check on the joined result cannot see
that — so mismatches must raise at the combine, not be inspected afterwards.
"""

import logging
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parents[1]))

DEST_BUCKET = "reflective-data-store"
STORE_PREFIX = "simulator-inputs"

DAYS_PER_MONTH = 30  # UKESM runs on a 360-day calendar; enforced on every read.

# Anything this large is a fill value, not data: netCDF's default float fill is
# 9.97e36 and the UM's missing-data indicator is -1.07e9. Some u-suite files
# ship land as the netCDF default fill with NO _FillValue attribute, so xarray
# cannot mask it for us.
FILL_THRESHOLD = 1e30

# Attrs written on every store variable. Source-file attrs are deliberately NOT
# carried over: after the unit conversions they would be wrong (m/s data
# labelled kg m-2 s-1, day counts labelled K), and the shipped MIROC contract
# stores carry only ``units`` plus a provenance ``note``.
VAR_ATTRS = {
    "TREFHT": {"units": "K", "long_name": "2 m air temperature, monthly mean"},
    "PRECT": {"units": "m/s", "long_name": "total precipitation rate"},
    "QFLX": {"units": "kg/m2/s", "long_name": "surface evaporation (water flux)"},
    "ICEFRAC": {"units": "1", "long_name": "sea-ice area fraction, land = 0"},
    "TREFHTMX": {"units": "K", "long_name": "monthly maximum of daily maximum 2 m air temperature"},
    "TREFHTMN": {"units": "K", "long_name": "monthly minimum of daily minimum 2 m air temperature"},
    # Counts deliberately carry NO ``units``: a CF time unit such as "days"
    # makes xarray decode the int64 counts as timedelta64 on read.
    "TREFHT_above_40": {"long_name": "days per month with daily-mean 2 m air temperature > 40 degC"},
    "TREFHT_above_35": {"long_name": "days per month with daily-mean 2 m air temperature > 35 degC"},
    "TREFHT_below_0": {"long_name": "days per month with daily-mean 2 m air temperature < 0 degC"},
    "PRECT_above_10": {"long_name": "days per month with precipitation > 10 mm/day"},
    "PRECT_above_20": {"long_name": "days per month with precipitation > 20 mm/day"},
}
COUNT_VARS = {"TREFHT_above_40", "TREFHT_above_35", "TREFHT_below_0", "PRECT_above_10", "PRECT_above_20"}
STORE_COORDS = {"member", "time", "lat", "lon"}
UM_RMDI = -1073741824.0  # the Unified Model's real missing-data indicator

log = logging.getLogger("zarrify")


# --- pure helpers --------------------------------------------------------------

def assert_360_day(time) -> None:
    """Every downstream aggregation assumes exact 30-day months."""
    cal = getattr(time.values[0], "calendar", None) or time.dt.calendar
    if cal != "360_day":
        raise ValueError(f"expected 360_day calendar, got {cal!r}")


def assert_daily_axis(daily_time) -> None:
    """A daily axis must start on day 1 of a month and step by exactly one day.

    ``coarsen(time=30)`` labels 30-day BLOCKS; only with these two properties
    do the blocks coincide with calendar months.
    """
    first = daily_time.values[0]
    if first.day != 1:  # hour is irrelevant: UM daily means are stamped at 12:00
        raise ValueError(f"daily axis must start on day 1 of a month, starts {first}")
    idx = xr.CFTimeIndex(daily_time.values)
    steps = idx[1:] - idx[:-1]
    if len(steps) and (steps.min() != timedelta(days=1) or steps.max() != timedelta(days=1)):
        raise ValueError(
            f"daily axis is not contiguous: step range {steps.min()} .. {steps.max()}"
        )
    if daily_time.size % DAYS_PER_MONTH:
        raise ValueError(f"daily axis length {daily_time.size} is not a whole number of months")


def month_starts(daily_time):
    """Start-of-month cftime stamps for a daily 360-day axis (validated)."""
    assert_daily_axis(daily_time)
    stamps = daily_time.values[::DAYS_PER_MONTH]
    return [t.replace(day=1, hour=0, minute=0, second=0, microsecond=0) for t in stamps]


def normalize_monthly_time(obj):
    """Floor monthly stamps (UKESM stamps mid-month) to start-of-month."""
    floored = [t.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
               for t in obj.time.values]
    return obj.assign_coords(time=floored)


def monthly_count(daily: xr.DataArray, threshold: float, direction: str) -> xr.DataArray:
    """Days per month beyond ``threshold`` (30-day months, int64)."""
    exceed = daily > threshold if direction == "above" else daily < threshold
    counts = exceed.coarsen(time=DAYS_PER_MONTH).sum().astype("int64")
    return counts.assign_coords(time=month_starts(daily.time))


def monthly_extreme(daily: xr.DataArray, how: str) -> xr.DataArray:
    """Absolute monthly max/min of a daily field (30-day months)."""
    coarse = daily.coarsen(time=DAYS_PER_MONTH)
    agg = coarse.max() if how == "max" else coarse.min()
    return agg.assign_coords(time=month_starts(daily.time))


def monthly_mean(daily: xr.DataArray) -> xr.DataArray:
    """Monthly mean of a daily field (30-day months)."""
    mean = daily.coarsen(time=DAYS_PER_MONTH).mean()
    return mean.assign_coords(time=month_starts(daily.time))


def mask_fill_values(da: xr.DataArray) -> xr.DataArray:
    """Turn unmasked fill values into NaN.

    Covers the netCDF default fill (|x| >= 1e30) and the UM's missing-data
    indicator (-1.07e9); no simulator variable can legitimately take either.
    """
    is_fill = (np.abs(da) >= FILL_THRESHOLD) | (np.abs(da - UM_RMDI) < 1.0)
    return da.where(~is_fill)


def to_icefrac(sea_ice: xr.DataArray, units: str) -> xr.DataArray:
    """Sea-ice concentration -> [0,1] fraction, land/missing/fill -> 0.

    ``units`` must be given explicitly — ``'%'`` (CMOR siconc/siconca) or
    ``'1'`` (MACLOUD u-suite siconca). Range-based auto-detection is not safe:
    unmasked land fill values (9.97e36) once made a fraction field look like
    percent, which turned the store's ICEFRAC into a land mask.
    """
    if units == "%":
        scale = 100.0
    elif units in ("1", "fraction"):
        scale = 1.0
    else:
        raise ValueError(f"sea-ice units must be '%' or '1', got {units!r}")
    clean = mask_fill_values(sea_ice)
    top = clean.max()
    top = float(top.compute() if hasattr(top.data, "compute") else top)
    if np.isnan(top):
        raise ValueError("sea-ice field is entirely NaN/fill")
    if top > 1.5 * scale:
        raise ValueError(f"sea-ice max {top:g} inconsistent with units {units!r}")
    return (clean.fillna(0.0).clip(0.0, scale) / scale).astype("float32")


def stack_members(per_member: dict) -> xr.DataArray:
    """dict of member -> DataArray  =>  (member, time, lat, lon) DataArray.

    ``join="exact"``: a member with a different time/lat/lon axis is an error,
    never a NaN-padded row.
    """
    labeled = [da.expand_dims(member=[m]) for m, da in per_member.items()]
    return xr.concat(labeled, dim="member", join="exact", coords="minimal", compat="equals")


def exact_dataset(variables: dict) -> xr.Dataset:
    """Combine same-grid DataArrays into a Dataset; any axis mismatch raises."""
    return xr.merge(
        [da.rename(name) for name, da in variables.items()],
        join="exact", compat="equals", combine_attrs="drop_conflicts",
    )


def assert_group_axes(ds: xr.Dataset, span: tuple, nlat: int, nlon: int) -> None:
    """The finished group must cover exactly ``span`` on the expected grid."""
    y0, y1 = span
    n_months = (y1 - y0 + 1) * 12
    if ds.sizes["time"] != n_months:
        raise ValueError(f"time={ds.sizes['time']}, expected {n_months} for {y0}-{y1}")
    first, last = ds.time.values[0], ds.time.values[-1]
    if (first.year, first.month, first.day) != (y0, 1, 1):
        raise ValueError(f"time axis starts {first}, expected {y0}-01-01")
    if (last.year, last.month) != (y1, 12):
        raise ValueError(f"time axis ends {last}, expected {y1}-12")
    if ds.sizes["lat"] != nlat or ds.sizes["lon"] != nlon:
        raise ValueError(f"grid {ds.sizes['lat']}x{ds.sizes['lon']}, expected {nlat}x{nlon}")


# --- R2 I/O -------------------------------------------------------------------

def r2_fs():
    import os

    import s3fs

    env = {k: os.environ.get(k) for k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY", "R2_SECRET_KEY")}
    missing = [k for k, v in env.items() if not v]
    if missing:
        raise SystemExit("missing environment variable(s): " + ", ".join(missing))
    return s3fs.S3FileSystem(
        key=env["R2_ACCESS_KEY"], secret=env["R2_SECRET_KEY"],
        endpoint_url=f"https://{env['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        client_kwargs={"region_name": "auto"},
    )


def store_target(store: str):
    """A zarr-v3 FsspecStore on its OWN async s3fs instance.

    zarr 3 drives store I/O from a dedicated event loop; sharing the
    synchronous s3fs instance the NetCDF reads use raises "Future attached to
    a different loop". ``from_url`` builds a fresh asynchronous filesystem
    that lives on zarr's loop.
    """
    import os

    from zarr.storage import FsspecStore

    return FsspecStore.from_url(
        f"s3://{DEST_BUCKET}/{STORE_PREFIX}/{store}",
        storage_options={
            "key": os.environ["R2_ACCESS_KEY"],
            "secret": os.environ["R2_SECRET_KEY"],
            "endpoint_url": f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
            "client_kwargs": {"region_name": "auto"},
            "asynchronous": True,
        },
    )


def clean_for_store(ds: xr.Dataset) -> xr.Dataset:
    """Drop stray coords and replace inherited attrs with the store's own.

    Keeps each variable's provenance ``note``; everything else (source units,
    standard_name, cell_methods, UM stash ids, grid_mapping) is wrong after
    conversion and is dropped.
    """
    stray = [c for c in ds.coords if c not in STORE_COORDS]
    ds = ds.reset_coords(stray, drop=True)
    for var in ds.data_vars:
        if var not in VAR_ATTRS:
            raise ValueError(f"{var} has no entry in VAR_ATTRS")
        note = ds[var].attrs.get("note")
        ds[var].attrs = dict(VAR_ATTRS[var])
        if note:
            ds[var].attrs["note"] = note
        ds[var].encoding = {}
    return ds


def write_group(store: str, group: str, ds: xr.Dataset, chunks: dict, target=None) -> None:
    """Chunk to the store layout and write one group (zarr v3)."""
    import zarr

    ds = clean_for_store(ds).chunk(chunks)
    if target is None:
        target = store_target(store)
    log.info("writing %s :: %s %s", store, group, dict(ds.sizes))
    ds.to_zarr(target, group=group, mode="w", zarr_format=3, consolidated=False)
    zarr.consolidate_metadata(target, path=group)


def finalize_store(fs, store: str, target=None) -> None:
    """Root/Mon group markers + root-level consolidated metadata (R2 needs both).

    Markers are plain object PUTs through the synchronous filesystem; the
    consolidation goes through zarr's own async store.
    """
    import json

    import zarr

    if fs is not None:
        for sub in ("", "Mon"):
            key = f"{DEST_BUCKET}/{STORE_PREFIX}/{store}/{sub}/zarr.json".replace("//", "/")
            if not fs.exists(key):
                with fs.open(key, "w") as f:
                    json.dump({"zarr_format": 3, "node_type": "group", "attributes": {}}, f)
    zarr.consolidate_metadata(target if target is not None else store_target(store))


def verify_store(store: str, groups: dict, n_members: int, span: tuple,
                 nlat: int, nlon: int, target=None) -> list:
    """Read-back verification through the consolidated metadata.

    Checks, per group and variable: presence, dims, member/time/lat/lon sizes,
    time endpoints, coordinate set (no stray UM coords), int64 counts, and —
    for every variable not declared PLACEHOLDER or PARTIAL in its note — that no
    NaN survived (an outer join or unmasked fill would show up here). A
    PLACEHOLDER must be entirely NaN and a PARTIAL must be neither entirely NaN
    nor entirely present, so a note that has gone stale is caught too.
    """
    y0, y1 = span
    n_months = (y1 - y0 + 1) * 12
    failures = []
    for group, expected in groups.items():
        try:
            ds = xr.open_zarr(target if target is not None else store_target(store),
                              group=group, consolidated=True, decode_timedelta=False)
        except Exception as exc:
            failures.append(f"[{group}] open failed: {type(exc).__name__}: {exc}")
            continue
        stray = set(ds.coords) - STORE_COORDS
        if stray:
            failures.append(f"[{group}] stray coordinates {sorted(stray)}")
        for var in expected:
            if var not in ds.data_vars:
                failures.append(f"[{group}] missing {var}")
                continue
            da = ds[var]
            if tuple(da.dims) != ("member", "time", "lat", "lon"):
                failures.append(f"[{group}/{var}] dims {da.dims}")
                continue
            sizes = (da.sizes["member"], da.sizes["time"], da.sizes["lat"], da.sizes["lon"])
            if sizes != (n_members, n_months, nlat, nlon):
                failures.append(f"[{group}/{var}] sizes {sizes}, expected "
                                f"{(n_members, n_months, nlat, nlon)}")
                continue
            first, last = da.time.values[0], da.time.values[-1]
            if (first.year, first.month) != (y0, 1) or (last.year, last.month) != (y1, 12):
                failures.append(f"[{group}/{var}] time {first}..{last}, expected {y0}-01..{y1}-12")
            if var in COUNT_VARS and da.dtype != np.int64:
                failures.append(f"[{group}/{var}] dtype {da.dtype}, expected int64")
            if da.attrs.get("units") != VAR_ATTRS.get(var, {}).get("units"):
                failures.append(f"[{group}/{var}] units {da.attrs.get('units')!r}, "
                                f"expected {VAR_ATTRS.get(var, {}).get('units')!r}")
            note = str(da.attrs.get("note", "")).upper()
            nan_frac = float(da.isnull().mean().compute())
            if "PLACEHOLDER" in note:
                # Declared all-NaN. If real data turned up, the note is stale.
                if nan_frac < 1.0:
                    failures.append(f"[{group}/{var}] declared PLACEHOLDER but only "
                                    f"{100 * nan_frac:.2f}% NaN")
            elif "PARTIAL" in note:
                # Real for some members, NaN for the rest (e.g. UKESM historical
                # ICEFRAC: ESGF publishes one member). Both extremes mean the note
                # no longer describes the array.
                if nan_frac == 0.0:
                    failures.append(f"[{group}/{var}] declared PARTIAL but nothing is NaN")
                elif nan_frac == 1.0:
                    failures.append(f"[{group}/{var}] declared PARTIAL but everything is NaN")
            elif nan_frac > 0:
                failures.append(f"[{group}/{var}] {100 * nan_frac:.2f}% NaN in a non-placeholder")
    return failures


# The exposure-count recipe from the retired scripts/process_daily_gauss.py:
# (threshold in the daily variable's native units, direction).
TAS_COUNTS = {
    "TREFHT_above_40": (40.0 + 273.15, "above"),
    "TREFHT_above_35": (35.0 + 273.15, "above"),
    "TREFHT_below_0": (0.0 + 273.15, "below"),
}
PR_COUNTS = {  # daily pr is kg m-2 s-1; thresholds are 10 / 20 mm per day
    "PRECT_above_10": (10.0 / 86400.0, "above"),
    "PRECT_above_20": (20.0 / 86400.0, "above"),
}

STORE_GROUPS = {
    "Mon/atmos_2d": ["TREFHT", "PRECT", "QFLX", "ICEFRAC"],
    "Mon/atmos_2d_tasminmax": ["TREFHTMN", "TREFHTMX"],
    "Mon/atmos_2d_derived": list(TAS_COUNTS) + list(PR_COUNTS),
}
