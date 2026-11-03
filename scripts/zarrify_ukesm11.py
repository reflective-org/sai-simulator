#!/usr/bin/env python3
"""Build the UKESM1-1-LL simulator-input Zarr stores from the MACLOUD staging.

Consumes the u-suite NetCDFs mirrored by ``scripts/mirror_macloud_to_r2.py``
under ``r2:reflective-data-store/UKESM1-1/`` and writes the stores the repo's
UKESM stubs expect, next to the CESM/MIROC ones:

    simulator-inputs/UKESM1-1-LL_baseline.zarr     (het-chem SSP2-4.5, 2015-2100
                                                    — the G6 branch parent)
    simulator-inputs/UKESM1-1-LL_G6-1.5K-SAI.zarr  (2034-2084)
    simulator-inputs/UKESM1-1-LL_historical.zarr   (1850-2014)

Schema identical to the shipped MIROC stores (see scripts/zarrify_common.py
and the UKESM1-0 sibling script): zarr v3, groups Mon/atmos_2d (TREFHT [K],
PRECT [m/s], QFLX [kg/m2/s], ICEFRAC [0-1]), Mon/atmos_2d_tasminmax (absolute
monthly extremes of the daily fields), Mon/atmos_2d_derived (exposure counts),
dims (member, time, lat, lon) on the native n96 144x192 grid, member coord
["r2", "r3", "r12"] in every store (the consumer stitches historical into the
baseline by member label), start-of-month 360-day timestamps.

MACLOUD u-suite conventions this script handles (vs. the CMOR files the
UKESM1-0 script reads):

- variables are CF-named inside the files (e.g. ``air_temperature``); the
  daily tas file carries the mean/min/max triple disambiguated by
  ``cell_methods`` (never by position — the order varies between files);
- the monthly tas/pr files carry DUPLICATE fields sampled at different
  intervals (6-hourly and 1-hourly means side by side; the SAI tas_mon has
  only the 6-hourly one), so monthly TREFHT and PRECT are instead derived as
  30-day means of the daily fields in EVERY experiment — one 1-hourly-sampled
  recipe across historical/baseline/SAI, with no sampling-interval seams at
  the 2015 stitch or in the SAI-vs-baseline delta. The monthly files are used
  only for QFLX (evap_mon) and ICEFRAC (siconca_mon), which have single
  fields;
- dims are ``latitude``/``longitude`` (renamed to lat/lon here);
- QFLX comes from ``evap_mon`` (surface_upward_latent_heat_flux, W m-2):
  QFLX = evap / 2.5e6, the same conversion MIROC uses for hfls;
- ICEFRAC comes from ``siconca_mon``, already a [0,1] fraction on the atmos
  grid (no regridding; land NaN -> 0);
- filename year ranges overstate coverage (SAI files claim 2015-2100 but hold
  2034-2084) — real time axes are asserted against each experiment's span;
- u-dp903 ships a second, unlabeled evap file — files are picked by their
  experiment token, and ambiguity is an error;
- historical publishes only daily pr + the daily tas triple, so its monthly
  TREFHT/PRECT are 30-day means and both QFLX and ICEFRAC come from a second
  source — CMIP6 ESGF ``hfls`` and ``siconca`` for UKESM1-1-LL historical
  r1i1p1f2, staged alongside the MACLOUD files. Each covers ONE of the three
  members, so those arrays are PARTIAL rather than placeholders: real for the
  member they cover, NaN for the rest. ESGF publishes historical for r1i1p1f2
  only; the 2015-boundary continuity test identified that run as this store's
  r3. data_sanity_checks warns on both kinds instead of failing.
  Historical QFLX is ``hfls``, NOT ESGF's ``evspsbl``: the scenario stores take
  QFLX from ``evap_mon``, which IS the latent heat flux, so hfls/2.5e6 keeps one
  recipe on both sides of the 2015 stitch. evspsbl is the evaporation flux
  directly and differs from it by the latent-heat constant wherever sublimation
  matters — 0.04% in the tropics but 1.23% over 60-90N and 1.83% over 60-90S —
  which mixing the two would deposit into the seam as a step;

The historical suites (u-by791/u-bz502/u-bz897) carry no realization
metadata, so building the historical store REQUIRES ``--historical-map``.
The shipped store was built with ``r2=u-bz502,r3=u-by791,r12=u-bz897``,
established by cross-boundary weather continuity at the 2015 stitch and
recorded in the store's note attrs — NOT alphabetical. Matthew Henry confirmed
the same pairing independently on 2026-09-24. Every example below uses it; do not
copy a different one without re-deriving it, since the consumer stitches
historical into each no-SAI run per member.

Credentials: R2_ACCOUNT_ID / R2_ACCESS_KEY / R2_SECRET_KEY. Examples:

    python scripts/zarrify_ukesm11.py --experiment baseline --dry-run
    python scripts/zarrify_ukesm11.py --experiment baseline --validate-icefrac
    python scripts/zarrify_ukesm11.py --experiment g6-1.5k-sai
    python scripts/zarrify_ukesm11.py --experiment historical \
        --historical-map "r2=u-bz502,r3=u-by791,r12=u-bz897"
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
    assert_group_axes,
    exact_dataset,
    finalize_store,
    mask_fill_values,
    monthly_count,
    monthly_extreme,
    monthly_mean,
    normalize_monthly_time,
    r2_fs,
    stack_members,
    to_icefrac,
    verify_store,
    write_group,
)

RAW_PREFIX = "UKESM1-1"
MODEL = "UKESM1-1-LL"
MEMBERS = ["r2", "r3", "r12"]
NLAT, NLON = 144, 192  # native N96 grid

MONTHLY_CHUNKS = {"member": 1, "time": 120, "lat": 144, "lon": 192}

EXPERIMENTS = {
    "baseline": {
        "span": (2015, 2100),
        "store": f"{MODEL}_baseline.zarr",
        "dirs": {"r2": "SSP245/r2i1p1f1", "r3": "SSP245/r3i1p1f1",
                 "r12": "SSP245/r12i1p1f1"},
        "token": "_SSP245_",
        "daily_tas_prefix": "tas_mean_min_max_day_",
        "daily_pr_prefix": "pr_day_",
    },
    "g6-1.5k-sai": {
        "span": (2034, 2084),
        "store": f"{MODEL}_G6-1.5K-SAI.zarr",
        "dirs": {"r2": "G6-1p5K-SAI/r2i1p1f1", "r3": "G6-1p5K-SAI/r3i1p1f1",
                 "r12": "G6-1p5K-SAI/r12i1p1f1"},
        "token": "_G6-1p5K-SAI_",
        "daily_tas_prefix": "tas_day_",
        "daily_pr_prefix": "pr_day_",
    },
    "historical": {
        "span": (1850, 2014),
        "store": f"{MODEL}_historical.zarr",
        "dirs": None,  # from --historical-map; suites carry no realization attrs
        "token": "_historical_",
        "daily_tas_prefix": "tas-min-mn-max_daily_",
        "daily_pr_prefix": "pr_daily_",
    },
}

TODAY = date.today().isoformat()
SOURCE = ("JASMIN MACLOUD (gws-access.jasmin.ac.uk/public/macloud, "
          "Matthew Henry, README dated 2026-08-26), mirrored to R2 by "
          "scripts/mirror_macloud_to_r2.py")
NOTES = {
    "TREFHT": ("2 m air temperature: 30-day-month means of the daily tas triple's "
               "'time: mean' field (1-hourly sampled) — used for EVERY experiment "
               "because the u-suite tas_mon files mix 6-hourly- and 1-hourly-sampled "
               f"duplicates inconsistently across experiments. Source: {SOURCE}."),
    "PRECT": ("30-day-month means of daily pr, converted kg m-2 s-1 -> m/s (pr/1000), "
              f"matching CESM PRECT; same one-recipe rationale as TREFHT. Source: {SOURCE}."),
    "QFLX": ("QFLX = evap_mon (surface_upward_latent_heat_flux, W m-2) / 2.5e6 "
             f"(kg/m2/s), same conversion as the MIROC stores. Source: {SOURCE}. "
             f"Built {TODAY} per J. Orcutt."),
    "QFLX_hist": ("PLACEHOLDER — all NaN: UKESM1.1 historical publishes no "
                  "evaporative flux on MACLOUD (checked 2026-09-09) and no ESGF "
                  "hfls was staged for any member, so the historical store "
                  "carries no QFLX and p-e coverage starts in 2015 (baseline "
                  f"store). Built {TODAY} per J. Orcutt."),
    "ICEFRAC": ("ICEFRAC = u-suite siconca_mon (sea_ice_area_fraction, already a "
                "[0,1] fraction on the n96 atmos grid), land/missing -> 0, clipped. "
                f"No regridding. Source: {SOURCE}. Built {TODAY} per J. Orcutt."),
    "ICEFRAC_hist": ("PLACEHOLDER — all NaN: UKESM1.1 historical publishes no sea "
                     "ice on MACLOUD (checked 2026-09-09). Consumers fall back per "
                     f"data_sanity_checks. Built {TODAY} per J. Orcutt."),
    "TREFHTMX": ("monthly MAXIMUM of daily tasmax (absolute monthly extreme, K); "
                 "daily tas triple field with cell_methods 'time: maximum'"),
    "TREFHTMN": ("monthly MINIMUM of daily tasmin (absolute monthly extreme, K); "
                 "daily tas triple field with cell_methods 'time: minimum'"),
}

log = logging.getLogger("zarrify-ukesm11")


# --- pure helpers (unit-tested in tests/test_zarrify_ukesm11.py) ---------------

def expected_months(experiment: str) -> int:
    y0, y1 = EXPERIMENTS[experiment]["span"]
    return (y1 - y0 + 1) * 12


def pick_file(paths, token: str) -> str:
    """Choose among same-prefix files: the experiment token disambiguates.

    u-dp903 ships both ``evap_mon_..._SSP245_...`` and an unlabeled
    ``evap_mon_..._u-dp903_200001-210012.nc`` — silent concatenation of both
    would corrupt the time axis, so ambiguity is an error, never a guess.
    """
    if len(paths) == 1:
        return paths[0]
    tokened = [p for p in paths if token in p.rsplit("/", 1)[-1]]
    if len(tokened) == 1:
        return tokened[0]
    raise ValueError(f"cannot pick uniquely among {paths} with token {token!r}")


def split_tas_triple(ds: xr.Dataset) -> dict:
    """The daily tas file's mean/min/max fields, keyed by cell_methods.

    Field order varies between MACLOUD files (historical has max first,
    ssp245 differs), so position must never be used.
    """
    out = {}
    for var in ds.data_vars:
        methods = ds[var].attrs.get("cell_methods", "")
        for key, marker in (("mean", "time: mean"), ("min", "time: minimum"),
                            ("max", "time: maximum")):
            if marker in methods:
                if key in out:
                    raise ValueError(f"two fields claim cell_methods {marker!r}")
                out[key] = ds[var]
    missing = {"mean", "min", "max"} - set(out)
    if missing:
        raise ValueError(f"daily tas triple is missing {sorted(missing)} "
                         f"(vars: {list(ds.data_vars)})")
    # The store notes promise 1-hourly-sampled means in every experiment
    # (the monthly u-suite files mix 6-hourly and 1-hourly duplicates).
    methods = out["mean"].attrs.get("cell_methods", "")
    if "(interval: 1 hour)" not in methods:
        raise ValueError(f"daily-mean tas is not 1-hourly sampled: cell_methods={methods!r}")
    return out


def parse_historical_map(text: str) -> dict:
    """'r2=u-bz502,r3=u-by791,r12=u-bz897' -> {'r2': 'historical/u-bz502', ...}

    Every member must map to a distinct suite: a duplicated suite would
    silently halve the ensemble, a duplicated member would silently drop one.
    """
    pairs = {}
    for item in text.split(","):
        member, _, suite = item.strip().partition("=")
        member, suite = member.strip(), suite.strip()
        if not member or not suite:
            raise ValueError(f"bad historical-map entry {item!r}; expected member=suite")
        if member in pairs:
            raise ValueError(f"member {member} listed twice in the historical map")
        pairs[member] = suite
    if set(pairs) != set(MEMBERS):
        raise ValueError(f"historical map must cover exactly {MEMBERS}, got {sorted(pairs)}")
    if len(set(pairs.values())) != len(pairs):
        raise ValueError(f"historical map assigns one suite to several members: {pairs}")
    return {member: f"historical/{suite}" for member, suite in pairs.items()}


def assert_daily_span(time, span: tuple, what: str) -> None:
    """A daily axis must run from Jan 1 of span[0] to Dec 30 of span[1]."""
    first, last = time.values[0], time.values[-1]
    y0, y1 = span
    if (first.year, first.month, first.day) != (y0, 1, 1):
        raise ValueError(f"{what}: daily axis starts {first}, expected {y0}-01-01")
    if (last.year, last.month, last.day) != (y1, 12, 30):
        raise ValueError(f"{what}: daily axis ends {last}, expected {y1}-12-30")


# --- R2 input layer -------------------------------------------------------------

def open_usuite(fs, dir_key: str, prefix: str, token: str, time_chunk: int) -> xr.Dataset:
    """Open one u-suite file by filename prefix; normalize dims to lat/lon."""
    pattern = f"{DEST_BUCKET}/{RAW_PREFIX}/{dir_key}/{prefix}*.nc"
    paths = sorted(fs.glob(pattern))
    if not paths:
        raise FileNotFoundError(pattern)
    path = pick_file(paths, token)
    ds = xr.open_dataset(fs.open(path, "rb"), engine="h5netcdf",
                         chunks={"time": time_chunk}, use_cftime=True,
                         decode_timedelta=False)
    renames = {d: t for d, t in (("latitude", "lat"), ("longitude", "lon")) if d in ds.dims}
    ds = ds.rename(renames)
    drop = [v for v in ds.data_vars
            if v.endswith("_bnds") or ds[v].ndim == 0 or "time" not in ds[v].dims]
    ds = ds.drop_vars(drop)
    # UM auxiliary coords (forecast_period, forecast_reference_time, height)
    # are not part of the store schema and must not ride along.
    ds = ds.reset_coords([c for c in ds.coords if c not in ("time", "lat", "lon")], drop=True)
    # Some u-suite files ship land as the netCDF default fill with no
    # _FillValue attr, so xarray leaves 9.97e36 in the data. Mask it here.
    for var in ds.data_vars:
        attrs = dict(ds[var].attrs)  # units/cell_methods drive the asserts below
        ds[var] = mask_fill_values(ds[var])
        ds[var].attrs = attrs
    assert_360_day(ds.time)
    return ds


ESGF_SEAICE_PREFIX = "siconca_SImon_"
ESGF_HFLS_PREFIX = "hfls_Amon_"


def open_esgf(fs, dir_key: str, prefix: str, time_chunk: int) -> xr.DataArray | None:
    """One staged CMIP6 ESGF monthly field for one member, or None if absent.

    MACLOUD publishes no historical sea ice or evaporative flux, but ESGF does —
    for exactly one UKESM1-1-LL historical member, ``r1i1p1f2``, which the
    2015-boundary continuity test identified as the run this repo calls r3
    (u-by791): monthly tas agrees with our r3 to 0.0001 K RMS, against
    1.97-1.99 K for r2 and r12, the same scale our own members differ from each
    other by.

    Unlike the u-suite files these sources are split across two files, so they
    are concatenated rather than picked. Returns None when a member has no
    staged file, which is the normal case for r2 and r12 — nothing is published
    for them.

    Callers assert units and convert, because the CMOR conventions differ from
    MACLOUD's: ``siconca`` is percent rather than a [0,1] fraction and marks land
    with a real ``_FillValue`` (1e20) that xarray decodes to NaN on read, which
    ``to_icefrac`` sends back to 0; ``hfls`` is W m-2 like ``evap_mon`` and needs
    only the 2.5e6 division.
    """
    pattern = f"{DEST_BUCKET}/{RAW_PREFIX}/{dir_key}/{prefix}*.nc"
    paths = sorted(fs.glob(pattern))
    if not paths:
        return None
    parts = []
    for path in paths:
        ds = xr.open_dataset(fs.open(path, "rb"), engine="h5netcdf",
                             chunks={"time": time_chunk}, use_cftime=True,
                             decode_timedelta=False)
        ds = ds.rename({d: t for d, t in (("latitude", "lat"), ("longitude", "lon"))
                        if d in ds.dims})
        ds = ds.drop_vars([v for v in ds.data_vars
                           if v.endswith("_bnds") or ds[v].ndim == 0
                           or "time" not in ds[v].dims])
        ds = ds.reset_coords([c for c in ds.coords if c not in ("time", "lat", "lon")],
                             drop=True)
        assert_360_day(ds.time)
        parts.append(single_var(ds))
    combined = xr.concat(parts, dim="time", join="exact", combine_attrs="drop_conflicts")
    combined = combined.sortby("time")
    if combined.attrs.get("units") is None:
        combined.attrs["units"] = parts[0].attrs.get("units")
    log.info("%s: ESGF %s from %d file(s), %d months",
             dir_key, prefix, len(paths), combined.sizes["time"])
    return combined


def partial_note(real_members: list[str], body: str) -> str:
    """The PARTIAL note for an array real on some members and NaN on the rest.

    PARTIAL is its own convention, distinct from PLACEHOLDER: ``verify_store``
    allows the NaN but requires that real data is actually present, and
    ``data_sanity_checks`` reports it separately.
    """
    missing = [m for m in MEMBERS if m not in real_members]
    return (
        "PARTIAL — real for member(s) " + ", ".join(real_members)
        + "; all-NaN for " + (", ".join(missing) if missing else "none")
        + ". " + body
        + " ESGF publishes historical for that one variant only; r1i1p1f2 was shown "
        "to be the same run as this store's r3 by 2015-boundary continuity (monthly "
        "tas RMS 0.0001 K vs r3, 1.97-1.99 K vs r2/r12 — our own members differ by "
        f"1.97-2.06 K). Built {TODAY} per J. Orcutt."
    )


def single_var(ds: xr.Dataset) -> xr.DataArray:
    if len(ds.data_vars) != 1:
        raise ValueError(f"expected one data variable, got {list(ds.data_vars)}")
    return next(iter(ds.data_vars.values()))


def assert_units(da: xr.DataArray, expected: str, what: str) -> None:
    units = da.attrs.get("units")
    if units != expected:
        raise ValueError(f"{what}: expected units {expected!r}, file says {units!r}")


# --- group builders --------------------------------------------------------------

def daily_sources(fs, cfg, member: str, day_chunk: int):
    """(tas mean/min/max triple, daily pr) for one member."""
    dir_key = cfg["dirs"][member]
    triple = split_tas_triple(
        open_usuite(fs, dir_key, cfg["daily_tas_prefix"], cfg["token"], day_chunk)
    )
    pr = single_var(
        open_usuite(fs, dir_key, cfg["daily_pr_prefix"], cfg["token"], day_chunk)
    )
    assert_units(triple["mean"], "K", "daily tas")
    assert_units(pr, "kg m-2 s-1", "daily pr")
    # Filename year ranges are unreliable on MACLOUD; the axes themselves must
    # cover exactly the experiment span (and start on Jan 1, for coarsen).
    assert_daily_span(triple["mean"].time, cfg["span"], f"{member} daily tas")
    assert_daily_span(pr.time, cfg["span"], f"{member} daily pr")
    return triple, pr


def compute_all_groups(fs, experiment: str, cfg, day_chunk: int) -> dict:
    """All three store groups, reading each member's daily files exactly once.

    The daily fields feed TREFHT/PRECT (30-day means), TREFHTMX/MN (30-day
    extremes) AND the exposure counts, so building groups independently would
    stream the same ~10-20 GB per member three times over.
    """
    hist = experiment == "historical"
    real_icefrac_members: list[str] = []
    real_qflx_members: list[str] = []
    a2d = {v: {} for v in ("TREFHT", "PRECT", "QFLX", "ICEFRAC")}
    tmm = {"TREFHTMX": {}, "TREFHTMN": {}}
    derived = {name: {} for name in list(TAS_COUNTS) + list(PR_COUNTS)}

    for member in MEMBERS:
        dir_key = cfg["dirs"][member]
        triple, pr_daily = daily_sources(fs, cfg, member, day_chunk)

        a2d["TREFHT"][member] = monthly_mean(triple["mean"]).astype("float32")
        a2d["PRECT"][member] = (monthly_mean(pr_daily) / 1000.0).astype("float32")
        if hist:
            esgf_hfls = open_esgf(fs, dir_key, ESGF_HFLS_PREFIX, 120)
            if esgf_hfls is None:
                a2d["QFLX"][member] = xr.full_like(
                    a2d["TREFHT"][member], np.nan, dtype="float32")
            else:
                # Same 2.5e6 division as the scenario stores' evap_mon, which is
                # the same physical field under a different name.
                assert_units(esgf_hfls, "W m-2", ESGF_HFLS_PREFIX)
                a2d["QFLX"][member] = normalize_monthly_time(
                    (esgf_hfls / 2.5e6).astype("float32"))
                real_qflx_members.append(member)
            esgf_sic = open_esgf(fs, dir_key, ESGF_SEAICE_PREFIX, 120)
            if esgf_sic is None:
                a2d["ICEFRAC"][member] = xr.full_like(
                    a2d["TREFHT"][member], np.nan, dtype="float32")
            else:
                assert_units(esgf_sic, "%", ESGF_SEAICE_PREFIX)
                a2d["ICEFRAC"][member] = normalize_monthly_time(
                    to_icefrac(esgf_sic, esgf_sic.attrs["units"]))
                real_icefrac_members.append(member)
        else:
            evap = single_var(open_usuite(fs, dir_key, "evap_mon_", cfg["token"], 120))
            assert_units(evap, "W m-2", "evap_mon")
            sic = single_var(open_usuite(fs, dir_key, "siconca_mon_", cfg["token"], 120))
            assert_units(sic, "1", "siconca_mon")
            # Normalize each monthly axis on its own BEFORE any combine, so a
            # stamp mismatch between files is caught by the exact join later.
            a2d["QFLX"][member] = normalize_monthly_time((evap / 2.5e6).astype("float32"))
            a2d["ICEFRAC"][member] = normalize_monthly_time(to_icefrac(sic, sic.attrs["units"]))

        tmm["TREFHTMX"][member] = monthly_extreme(triple["max"], "max").astype("float32")
        tmm["TREFHTMN"][member] = monthly_extreme(triple["min"], "min").astype("float32")
        for name, (threshold, direction) in TAS_COUNTS.items():
            derived[name][member] = monthly_count(triple["mean"], threshold, direction)
        for name, (threshold, direction) in PR_COUNTS.items():
            derived[name][member] = monthly_count(pr_daily, threshold, direction)

    ds_a2d = exact_dataset({v: stack_members(per) for v, per in a2d.items()})
    suffix = "_hist" if hist else ""
    for var in a2d:
        ds_a2d[var].attrs["note"] = NOTES.get(var + suffix, NOTES[var])
    if hist and real_icefrac_members:
        ds_a2d["ICEFRAC"].attrs["note"] = partial_note(
            real_icefrac_members,
            "MACLOUD publishes no historical sea ice, so this comes from CMIP6 "
            "ESGF siconca (UKESM1-1-LL historical r1i1p1f2, v20220512, CEDA node), "
            "percent -> [0,1] fraction with land -> 0.")
    if hist and real_qflx_members:
        ds_a2d["QFLX"].attrs["note"] = partial_note(
            real_qflx_members,
            "MACLOUD publishes no historical evaporative flux, so this comes from "
            "CMIP6 ESGF hfls (UKESM1-1-LL historical r1i1p1f2, CEDA node), "
            "W m-2 / 2.5e6 -> kg/m2/s. hfls, NOT evspsbl: evap_mon on the scenario "
            "side IS the latent heat flux, so this keeps one recipe across the 2015 "
            "stitch; evspsbl differs by the latent-heat constant where sublimation "
            "matters (0.04% in the tropics, 1.23% over 60-90N, 1.83% over 60-90S).")

    ds_tmm = exact_dataset({v: stack_members(per) for v, per in tmm.items()})
    ds_tmm["TREFHTMX"].attrs["note"] = NOTES["TREFHTMX"]
    ds_tmm["TREFHTMN"].attrs["note"] = NOTES["TREFHTMN"]

    ds_derived = exact_dataset({name: stack_members(per) for name, per in derived.items()})

    return {
        "atmos_2d": ("Mon/atmos_2d", ds_a2d),
        "tasminmax": ("Mon/atmos_2d_tasminmax", ds_tmm),
        "derived": ("Mon/atmos_2d_derived", ds_derived),
    }


def validate_icefrac(fs, cfg, store_icefrac: xr.DataArray) -> None:
    """Same-run check: siconca_mon-based ICEFRAC vs monthly means of seaice_day."""
    for member in MEMBERS:
        daily = single_var(
            open_usuite(fs, cfg["dirs"][member], "seaice_day_", cfg["token"], 1800)
        )
        assert_units(daily, "1", "seaice_day")
        monthly = monthly_mean(to_icefrac(daily, daily.attrs["units"]))
        ours = store_icefrac.sel(member=member)
        # seaice_day files can span more than the experiment (SAI parent years);
        # align on the store's months.
        monthly = monthly.sel(time=ours.time)
        north = ours.lat > 0
        sept = ours.time.dt.month == 9
        weights = np.cos(np.deg2rad(ours.lat))
        ours_area = (ours.where(north).isel(time=sept) * weights).sum(("lat", "lon"))
        theirs_area = (monthly.where(north).isel(time=sept) * weights).sum(("lat", "lon"))
        rel = float((np.abs(ours_area - theirs_area) /
                     xr.where(theirs_area == 0, np.nan, theirs_area)).mean().compute())
        log.info("%s ICEFRAC validation: mean |Δ| Sept NH ice area, siconca_mon vs "
                 "seaice_day monthly means = %.2f%%", member, 100 * rel)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--experiment", choices=list(EXPERIMENTS), required=True)
    p.add_argument("--groups", default="atmos_2d,tasminmax,derived")
    p.add_argument("--day-chunk", type=int, default=1800)
    p.add_argument("--historical-map", default=None,
                   help='member=suite pairs; the shipped store used '
                        '"r2=u-bz502,r3=u-by791,r12=u-bz897"')
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--validate-icefrac", action="store_true")
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument("--log-file", default=None)
    args = p.parse_args()

    handlers = [logging.StreamHandler(sys.stderr)]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", handlers=handlers)

    cfg = dict(EXPERIMENTS[args.experiment])
    map_note = ""
    if args.experiment == "historical":
        if not args.historical_map:
            raise SystemExit(
                "--historical-map is required for the historical store: the MACLOUD "
                "suites carry no realization metadata (all attrs stripped to "
                "um_version; verified 2026-09-11), and the consumer stitches "
                "historical into the baseline PER MEMBER — pass the pairing "
                "explicitly rather than guessing."
            )
        cfg["dirs"] = parse_historical_map(args.historical_map)
        map_note = (
            f" Historical suite->member mapping: {args.historical_map} — the files "
            "carry no realization metadata, so the pairing was established by "
            "cross-boundary weather continuity (1-day-lag daily-tas anomaly pattern "
            "correlation across 2014-12-30 -> 2015-01-01: true pairs 0.70-0.83, at "
            "the within-run lag-1 level of 0.85; all false pairs negative; "
            f"2026-09-11, J. Orcutt). Independently confirmed by M. Henry "
            "(Exeter), 2026-09-24."
        )

    fs = r2_fs()
    store = cfg["store"]
    wanted = set(args.groups.split(","))

    built = compute_all_groups(fs, args.experiment, cfg, args.day_chunk)

    for key, (group, ds) in built.items():
        if key not in wanted:
            continue
        if map_note:
            for var in ds.data_vars:
                ds[var].attrs["note"] = ds[var].attrs.get("note", "") + map_note
        assert_group_axes(ds, cfg["span"], NLAT, NLON)
        if args.dry_run:
            log.info("dry run %s :: %s -> %s vars %s", store, group,
                     list(ds.data_vars), dict(ds.sizes))
            continue
        write_group(store, group, ds, MONTHLY_CHUNKS)
        if key == "atmos_2d" and args.validate_icefrac and args.experiment != "historical":
            validate_icefrac(fs, cfg, ds["ICEFRAC"])

    if args.dry_run:
        return 0

    finalize_store(fs, store)
    if not args.skip_verify:
        failures = verify_store(store, STORE_GROUPS, len(MEMBERS), cfg["span"], NLAT, NLON)
        for f in failures:
            log.error("VERIFY %s", f)
        if failures:
            return 2
        log.info("verify OK: %s reads back with the expected schema", store)
    return 0


if __name__ == "__main__":
    sys.exit(main())
