"""InVERT internal-variability integration for the SAI-simulator's emulated tas, tasmin, tasmax and pr responses.

The base emulator returns a smooth deterministic mean (see
``get_regional_map_from_global_temp`` in :mod:`backend.baseline`). This module adds a
realized noise trajectory on top by injecting the 100-member InVERT ensembles of detrended
monthly anomalies. They are read from the public R2 Zarr store :data:`INVERT_STORE` (one
variable per InVERT file, written by ``scripts/zarrify_invert.py``), or from a local copy
of the original netCDF under ``<data_dir>/invert/`` when one is present (see
:data:`INVERT_VARIABLES`).

InVERT lives on a coarse CESM F09-like grid (96 lat x 192 lon) and covers 50 years per
member. We annualise, regrid to the emulator's F05 grid (192 lat x 288 lon), and stitch
5 consecutive members end-to-end (5 x 50 yrs = 250 yrs) to produce 20 output ensemble
members covering the full 1851-2100 simulator timeframe. This uses all 100 InVERT
members exactly once.

The result is cached per variable under ``<cache_dir>/invert/`` (see :func:`_cache_path`)
because each raw file is ~4 GB; a cached ensemble is ~1.1 GB (float32, 20 x 250 x 192 x
288). Caches are written atomically and carry provenance attributes (cache version, source
file, grid, land mask) that are checked on every read, so a stale or foreign cache is
treated as missing rather than served. The request path (:func:`get_area_mean_variability`)
never builds a cache; build them once with

    python -m backend.invert_variability --data-dir <data_dir> --cache-dir <cache_dir>
"""
from __future__ import annotations

import hashlib
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import xarray as xr


# Per-variable source file, source variable name, and the factor that converts the raw
# InVERT units into the simulator's units. InVERT PRECT is raw CESM m/s; the simulator's
# pr is mm/day, so multiply by 86400 s/day * 1000 mm/m. ``annual`` is how the 12 monthly
# anomalies of a year are reduced to one value: the lowest (tasmin) or highest (tasmax)
# month, like the simulator's own yearly tasmin/tasmax, and the mean otherwise. Lowest /
# highest values are then centred per grid cell (see :func:`build_annual_ensemble`).
INVERT_VARIABLES = {
    "tas": {
        "filename": "InVERT_TREFHT_50yrs_SSP370_100members.nc",
        "source_var": "TREFHT",
        "scale": 1.0,
        "units": "K",
        "annual": "mean",
    },
    "tasmin": {
        "filename": "InVERT_TREFHTMN_50yrs_SSP370_100members_float32.nc",
        "source_var": "TREFHTMN",
        "scale": 1.0,
        "units": "K",
        "annual": "min",
    },
    "tasmax": {
        "filename": "InVERT_TREFHTMX_50yrs_SSP370_100members_float32.nc",
        "source_var": "TREFHTMX",
        "scale": 1.0,
        "units": "K",
        "annual": "max",
    },
    "pr": {
        "filename": "InVERT_PRECT_50yrs_SSP370_100members_float32.nc",
        "source_var": "PRECT",
        "scale": 86400.0 * 1000.0,
        "units": "mm/day",
        "annual": "mean",
    },
}
# Dashboard variables without an InVERT file of their own that borrow another variable's
# envelope. Water availability (p-e, mm/day, land-only like pr) uses the pr envelope:
# InVERT has no evaporation, so this treats evaporation variability, and its covariance
# with precipitation, as negligible.
INVERT_PROXIES = {"p-e": "pr"}
# Variables whose simulator field is land-only (``clip_to_land`` in ``baseline``), so their
# global-mean cache must be built with the same land mask.
LAND_ONLY_VARIABLES = {"pr"}
# Bump when the way the ensemble is built or stored changes; every cache written under an
# older version is then rebuilt (offline) or ignored (request path).
INVERT_CACHE_VERSION = 1
# Max |mean over ensemble and time| tolerated per grid cell before we conclude the input
# is not detrended. Same numeric guard for both variables (K for tas, mm/day for pr).
CLIMATOLOGY_TOLERANCE = 0.3

OUTPUT_START_YEAR = 1851
OUTPUT_END_YEAR = 2100
OUTPUT_N_YEARS = OUTPUT_END_YEAR - OUTPUT_START_YEAR + 1  # 250
INVERT_YEARS_PER_MEMBER = 50
MEMBERS_PER_OUTPUT = OUTPUT_N_YEARS // INVERT_YEARS_PER_MEMBER  # 5
N_OUTPUT_MEMBERS = 20  # 5 * 20 = 100 InVERT members used exactly


# The R2 store holding all four InVERT files as variables (named by ``source_var``), under
# :data:`version.ZARR_STORES_URI` like the training-data stores. Each variable carries the
# ``source_size_bytes`` of the netCDF it was copied from, so caches built from the store
# and from a local netCDF carry the same provenance.
INVERT_STORE = "InVERT_50yrs_SSP370_100members.zarr"


def _invert_path(data_dir: Path, var: str) -> Path:
    return Path(data_dir) / "invert" / INVERT_VARIABLES[var]["filename"]


def invert_store_uri() -> str:
    """URL of :data:`INVERT_STORE` (anonymous; no credentials needed)."""
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from version import ZARR_STORES_URI

    return f"{ZARR_STORES_URI}/{INVERT_STORE}"


@lru_cache(maxsize=None)
def _open_store(uri: str) -> xr.Dataset:
    return xr.open_zarr(uri, consolidated=True)


def _source_size(data_dir: Path, var: str) -> int:
    """Size in bytes of the InVERT netCDF ``var`` comes from: the local file, else the store's record.

    Raises when there is no local file and the store cannot be opened or lacks ``var``.
    """
    raw = _invert_path(data_dir, var)
    if raw.exists():
        return raw.stat().st_size
    return int(_open_store(invert_store_uri())[INVERT_VARIABLES[var]["source_var"]].attrs["source_size_bytes"])


def invert_source_status(data_dir: Path, var: str) -> str | None:
    """``None`` when ``var`` can be read (local netCDF or R2 store), else why it cannot."""
    if _invert_path(data_dir, var).exists():
        return None
    try:
        _source_size(data_dir, var)
    except Exception as exc:  # noqa: BLE001 -- any failure means the source is unusable
        return (
            f"no {_invert_path(data_dir, var)} and the store {invert_store_uri()} is unavailable "
            f"({type(exc).__name__}: {exc})"
        )
    return None


def _annual_suffix(var: str) -> str:
    """File-name tag for a non-mean annual reduction, so min/max caches never reuse mean ones."""
    annual = INVERT_VARIABLES[var]["annual"]
    return "" if annual == "mean" else f"_{annual}"


def _cache_path(cache_dir: Path, var: str) -> Path:
    return Path(cache_dir) / "invert" / f"invert_annual_ensemble_{var}{_annual_suffix(var)}_1851_2100.nc"


def _global_mean_cache_path(cache_dir: Path, var: str) -> Path:
    return Path(cache_dir) / "invert" / f"invert_global_mean_{var}{_annual_suffix(var)}_1851_2100.nc"


def _array_hash(values: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(values).tobytes()).hexdigest()[:16]


def _provenance(
    var: str,
    data_dir: Path,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    mask: np.ndarray | None = None,
    with_source: bool = False,
) -> dict:
    """Attributes a cache must carry to be served: everything its numbers depend on.

    ``source_size_bytes`` is only included with ``with_source`` (when building or refreshing
    a cache, from the local netCDF or the store's record of it), so the request path can
    validate the rest without reaching the source at all.
    """
    spec = INVERT_VARIABLES[var]
    attrs = {
        "invert_cache_version": INVERT_CACHE_VERSION,
        "source_file": spec["filename"],
        "source_var": spec["source_var"],
        "scale": spec["scale"],
        "annual_aggregation": spec["annual"],
        "grid_shape": f"{len(target_lat)}x{len(target_lon)}",
        "grid_hash": _array_hash(np.concatenate([np.asarray(target_lat, float), np.asarray(target_lon, float)])),
    }
    if with_source:
        attrs["source_size_bytes"] = _source_size(data_dir, var)
    if mask is not None:
        attrs["mask_hash"] = _array_hash(np.asarray(mask, bool))
    return attrs


def _cache_matches(path: Path, expected: dict) -> bool:
    """True when ``path`` exists, opens, and carries every attribute in ``expected``."""
    if not path.exists():
        return False
    try:
        with xr.open_dataarray(path) as da:
            found = dict(da.attrs)
    except Exception:
        return False
    return all(str(found.get(key)) == str(value) for key, value in expected.items())


def _write_cache(da: xr.DataArray, path: Path, attrs: dict) -> None:
    """Stamp ``attrs`` on ``da`` and write it atomically (temp file + rename, see utils)."""
    from .utils import save_processed_data  # local: utils pulls in geopandas/rioxarray

    da = da.copy(deep=False)
    da.attrs.update(attrs)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_processed_data(da, path, overwrite=True)


def load_invert_raw(data_dir: Path, var: str) -> xr.Dataset:
    """Open the raw InVERT data for ``var`` lazily (Dask-chunked one member at a time).

    A local netCDF under ``<data_dir>/invert/`` wins; otherwise ``var`` is read from the R2
    store, whose chunks are already one member each.
    """
    raw = _invert_path(data_dir, var)
    if raw.exists():
        return xr.open_dataset(raw, chunks={"ensemble": 1})
    return _open_store(invert_store_uri())[[INVERT_VARIABLES[var]["source_var"]]]


def _annualise_monthly(da: xr.DataArray, how: str = "mean") -> xr.DataArray:
    """Collapse the 600-step monthly ``time`` axis to a 50-step annual axis.

    The InVERT ``time`` coord is a bare integer 0..599 with no CF units, and a repeating
    ``month`` coord 1..12 attached. We group by ``time // 12`` and reduce each year's 12
    months with ``how`` (``"mean"``, ``"min"`` or ``"max"``).
    """
    n_time = da.sizes["time"]
    assert n_time % 12 == 0, f"expected multiple of 12 monthly steps, got {n_time}"
    year_idx = xr.DataArray(np.arange(n_time) // 12, dims=("time",), name="year")
    annual = getattr(da.groupby(year_idx), how)("time")
    annual = annual.rename({"year": "invert_year"})
    return annual


def _pad_for_regrid(da: xr.DataArray) -> xr.DataArray:
    """Pad the source array so linear interpolation can reach the emulator's edges.

    - Longitude is periodic: append a copy of ``lon=0`` at ``lon=360`` so the target
      longitudes just below 360 (e.g. 358.75) interpolate cleanly.
    - Latitude: prepend a row at ``lat=-90`` and append a row at ``lat=+90``, filled
      with the values at the pole-adjacent source latitudes. The emulator's target
      latitudes reach ~+/-89.76, which the raw source (~+/-88.57) cannot cover without
      this pad.
    """
    lon_wrap = da.isel(lon=0).assign_coords(lon=360.0)
    da_lon = xr.concat([da, lon_wrap], dim="lon")

    south_pad = da_lon.isel(lat=0).assign_coords(lat=-90.0)
    north_pad = da_lon.isel(lat=-1).assign_coords(lat=90.0)
    return xr.concat([south_pad, da_lon, north_pad], dim="lat")


def _regrid_to_emulator_grid(
    da: xr.DataArray, target_lat: np.ndarray, target_lon: np.ndarray
) -> xr.DataArray:
    """Bilinearly interpolate onto the emulator's (lat, lon) grid via xarray.interp."""
    padded = _pad_for_regrid(da)
    return padded.interp(lat=target_lat, lon=target_lon, method="linear")


def _splice_members(annual_regridded: xr.DataArray) -> xr.DataArray:
    """Build the 20 output ensemble members from the 100 annualised InVERT members.

    Output member ``i`` (i = 0..19) concatenates InVERT members
    ``[5i, 5i+1, 5i+2, 5i+3, 5i+4]`` end-to-end, giving 5 x 50 = 250 years covering
    1851-2100. Every InVERT member is used exactly once.

    Note: temporal autocorrelation breaks at the 4 seam years within each output
    member (1900/01, 1950/51, 2000/01, 2050/51). The forced signal is unaffected --
    only the residual noise trajectory has these seams.
    """
    n_ens = annual_regridded.sizes["ensemble"]
    assert n_ens == N_OUTPUT_MEMBERS * MEMBERS_PER_OUTPUT, (
        f"expected {N_OUTPUT_MEMBERS * MEMBERS_PER_OUTPUT} InVERT members, got {n_ens}"
    )

    output_members = []
    for i in range(N_OUTPUT_MEMBERS):
        chunk_ids = list(range(i * MEMBERS_PER_OUTPUT, (i + 1) * MEMBERS_PER_OUTPUT))
        chunks = [
            annual_regridded.isel(ensemble=cid).drop_vars(["ensemble", "month"], errors="ignore")
            for cid in chunk_ids
        ]
        stitched = xr.concat(chunks, dim="invert_year")
        stitched = stitched.rename({"invert_year": "time"})
        stitched = stitched.assign_coords(
            time=np.arange(OUTPUT_START_YEAR, OUTPUT_END_YEAR + 1)
        )
        output_members.append(stitched)

    ensemble = xr.concat(output_members, dim="ensemble")
    ensemble = ensemble.assign_coords(ensemble=np.arange(N_OUTPUT_MEMBERS))
    ensemble = ensemble.transpose("ensemble", "time", "lat", "lon")
    return ensemble


def build_annual_ensemble(
    data_dir: Path,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    var: str = "tas",
    cache: bool = True,
    cache_dir: Path | None = None,
) -> xr.DataArray:
    """Return the 1851-2100 InVERT variability ensemble for ``var`` on the emulator's grid.

    Dims: ``(ensemble=20, time=250, lat=len(target_lat), lon=len(target_lon))``.
    Values: detrended internal-variability anomalies in the simulator's units for ``var``
    (K for tas, mm/day for pr; see :data:`INVERT_VARIABLES`), reduced to one value per
    year with the variable's ``annual`` rule. The lowest / highest of 12 zero-mean monthly
    anomalies is biased low / high, so for tasmin / tasmax each grid cell's mean over all
    members and years is subtracted, centring the values on zero like the annual means.

    With ``cache=True`` (``cache_dir`` required) the result is written to
    ``<cache_dir>/invert/`` and a later call that finds a cache with matching provenance
    reads it instead of rebuilding.
    """
    spec = INVERT_VARIABLES[var]
    if cache:
        if cache_dir is None:
            raise ValueError("cache_dir is required when cache=True")
        cache_path = _cache_path(cache_dir, var)
        expected = _provenance(var, data_dir, target_lat, target_lon, with_source=True)
        if _cache_matches(cache_path, expected):
            with xr.open_dataarray(cache_path) as da:
                return da.load()

    ds = load_invert_raw(data_dir, var)
    raw = ds[spec["source_var"]] * spec["scale"]
    annual_mean = _annualise_monthly(raw).compute()

    # Detrending check on the annual means, before any min/max reduction shifts the values.
    climatology_abs_max = float(np.abs(annual_mean.mean(("ensemble", "invert_year"))).max().values)
    if climatology_abs_max > CLIMATOLOGY_TOLERANCE:
        raise ValueError(
            f"InVERT {var} climatology (mean over ensemble and time) deviates from zero by "
            f"{climatology_abs_max:.3f} {spec['units']} at some grid cell; expected pure "
            f"internal variability with |climatology| < {CLIMATOLOGY_TOLERANCE} everywhere. "
            "The upstream file may not be detrended after all."
        )

    if spec["annual"] == "mean":
        annual = annual_mean
    else:
        annual = _annualise_monthly(raw, spec["annual"]).compute()
        annual = annual - annual.mean(("ensemble", "invert_year"))
    regridded = _regrid_to_emulator_grid(annual, target_lat, target_lon)
    ensemble = _splice_members(regridded)
    ensemble = ensemble.astype(np.float32)
    ensemble.name = f"{var}_variability"
    ensemble.attrs["units"] = spec["units"]
    ensemble.attrs["annual_aggregation"] = spec["annual"]
    ensemble.attrs["description"] = (
        f"InVERT detrended internal-variability anomalies of {spec['source_var']} (annual "
        f"{spec['annual']} of the monthly values, centred per grid cell when not the mean), "
        "spliced into 20 members x 250 yrs "
        "(1851-2100) from 100 InVERT members x 50 yrs each."
    )

    if cache:
        _write_cache(ensemble, cache_path, expected)

    return ensemble


def add_variability_to_mean(
    mean_da: xr.DataArray, variability_ensemble: xr.DataArray
) -> xr.DataArray:
    """Add the InVERT ensemble on top of a smooth mean field.

    ``mean_da`` has dims ``(time, lat, lon)``; ``variability_ensemble`` has dims
    ``(ensemble, time, lat, lon)``. Returns ``(ensemble, time, lat, lon)``.
    The time axes must match exactly. lat/lon coords are taken from ``mean_da``.
    """
    if not np.array_equal(mean_da["time"].values, variability_ensemble["time"].values):
        raise ValueError(
            "time coord mismatch: mean_da spans "
            f"{mean_da['time'].values.min()}-{mean_da['time'].values.max()}, "
            f"ensemble spans {variability_ensemble['time'].values.min()}-"
            f"{variability_ensemble['time'].values.max()}."
        )
    aligned = variability_ensemble.assign_coords(lat=mean_da["lat"], lon=mean_da["lon"])
    combined = mean_da + aligned
    return combined.transpose("ensemble", "time", "lat", "lon")


def _template_mask(template: xr.DataArray) -> xr.DataArray:
    """Boolean ``(lat, lon)`` mask of the cells the simulator field has data for (land for pr)."""
    return template.isel(time=0, drop=True).notnull()


def get_global_mean_variability(
    var: str,
    data_dir: Path,
    cache_dir: Path,
    template: xr.DataArray,
    cache: bool = True,
) -> xr.DataArray:
    """Return the area-weighted global-mean InVERT anomaly per member and year.

    Dims: ``(ensemble=20, time=250)``, 1851-2100, in the simulator's units for ``var``.
    ``template`` is a ``(time, lat, lon)`` simulator field (e.g. the no-SAI regional map):
    its lat/lon define the target grid and its NaN pattern (land-only for pr) is applied
    before averaging, so the result matches the dashboard's own global mean.

    The full gridded ensemble is built (and cached) if needed; the small ``(20, 250)``
    result is cached separately, keyed on the grid and the mask, so later calls are cheap.
    """
    mask = _template_mask(template)
    cache_path = _global_mean_cache_path(cache_dir, var)
    expected = _provenance(
        var, data_dir, template["lat"].values, template["lon"].values, mask.values, with_source=cache
    )
    if cache and _cache_matches(cache_path, expected):
        with xr.open_dataarray(cache_path) as da:
            return da.load()

    ensemble = build_annual_ensemble(
        data_dir=data_dir,
        target_lat=template["lat"].values,
        target_lon=template["lon"].values,
        var=var,
        cache=cache,
        cache_dir=cache_dir,
    )
    ensemble = ensemble.assign_coords(lat=template["lat"], lon=template["lon"])
    weights = np.cos(np.deg2rad(template["lat"]))
    global_mean = ensemble.where(mask).weighted(weights).mean(("lat", "lon"))
    global_mean.name = f"{var}_global_mean_variability"
    global_mean.attrs["units"] = INVERT_VARIABLES[var]["units"]
    global_mean.attrs["annual_aggregation"] = INVERT_VARIABLES[var]["annual"]

    if cache:
        _write_cache(global_mean, cache_path, expected)
    return global_mean


def get_regional_mean_variability(
    var: str,
    data_dir: Path,
    cache_dir: Path,
    template: xr.DataArray,
    region_mask: xr.DataArray,
    build: bool = True,
) -> xr.DataArray | None:
    """Return the area-weighted InVERT anomaly over one region, per member and year.

    Dims: ``(ensemble=20, time=250)``, 1851-2100, in the simulator's units for ``var``.
    ``template`` is a ``(time, lat, lon)`` simulator field and ``region_mask`` a boolean
    ``(lat, lon)`` mask from :func:`backend.utils.create_mask`. Cells outside the region or
    NaN in ``template`` (ocean for pr) are dropped, and the rest are ``cos(lat)``-weighted,
    exactly as the dashboard's regional line is aggregated.

    Only the latitude rows the region touches are read from the gridded-ensemble cache, so
    a call takes about a second and nothing is cached per region. A missing or stale
    gridded cache is built when ``build`` is True and returns ``None`` otherwise.
    """
    cache_path = _cache_path(cache_dir, var)
    expected = _provenance(var, data_dir, template["lat"].values, template["lon"].values, with_source=build)
    if not _cache_matches(cache_path, expected):
        if not build:
            return None
        build_annual_ensemble(
            data_dir=data_dir,
            target_lat=template["lat"].values,
            target_lon=template["lon"].values,
            var=var,
            cache=True,
            cache_dir=cache_dir,
        )

    mask = region_mask & _template_mask(template)
    rows = np.flatnonzero(mask.any("lon").values)
    if rows.size == 0:
        raise ValueError(f"The region mask selects no grid cells with {var} data.")
    lat_slice = slice(int(rows[0]), int(rows[-1]) + 1)
    mask = mask.isel(lat=lat_slice)

    with xr.open_dataarray(cache_path) as ensemble:
        ensemble = ensemble.isel(lat=lat_slice).load()
    ensemble = ensemble.assign_coords(lat=mask["lat"], lon=mask["lon"])
    weights = np.cos(np.deg2rad(mask["lat"]))
    regional_mean = ensemble.where(mask).weighted(weights).mean(("lat", "lon"))
    regional_mean.name = f"{var}_regional_mean_variability"
    regional_mean.attrs["units"] = INVERT_VARIABLES[var]["units"]
    return regional_mean


def get_area_mean_variability(
    var: str,
    data_dir: Path,
    cache_dir: Path,
    template: xr.DataArray,
    region_mask: xr.DataArray | None = None,
) -> xr.DataArray | None:
    """Request-path accessor: the global (or regional) InVERT area mean from valid caches only.

    Returns ``None`` when the needed cache is missing or its provenance does not match the
    current code, grid or mask, so ``get_outputs`` falls back to the legacy band instead of
    building a multi-GB ensemble inside a web request. Build the caches offline with
    ``python -m backend.invert_variability``.
    """
    if region_mask is not None:
        return get_regional_mean_variability(var, data_dir, cache_dir, template, region_mask, build=False)
    cache_path = _global_mean_cache_path(cache_dir, var)
    expected = _provenance(
        var, data_dir, template["lat"].values, template["lon"].values, _template_mask(template).values
    )
    if not _cache_matches(cache_path, expected):
        return None
    with xr.open_dataarray(cache_path) as da:
        return da.load()


def build_caches(data_dir: Path, cache_dir: Path, variables: list[str] | None = None) -> dict[str, str]:
    """Build the gridded and global-mean caches for every InVERT variable whose raw file exists.

    The target grid is the emulator's (``correct_lat.npy`` x ``NUM_LON`` longitudes) and the
    global-mean mask is the simulator's land mask for land-only variables, i.e. the same grid
    and mask ``get_outputs`` will present at request time. Returns ``{var: status}``.
    """
    from .constants import NUM_LON
    from .utils import clip_to_land  # local: pulls in geopandas/rioxarray

    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    target_lat = np.load(data_dir / "correct_lat.npy")
    target_lon = np.linspace(0, 360, NUM_LON, endpoint=False)
    template = xr.DataArray(
        np.zeros((1, target_lat.size, target_lon.size), dtype=np.float32),
        dims=("time", "lat", "lon"),
        coords={"time": [OUTPUT_START_YEAR], "lat": target_lat, "lon": target_lon},
    )
    land_template = None

    status = {}
    for var in variables or sorted(INVERT_VARIABLES):
        unavailable = invert_source_status(data_dir, var)
        if unavailable is not None:
            status[var] = f"skipped: {unavailable}"
            continue
        if var in LAND_ONLY_VARIABLES:
            if land_template is None:
                land_template = clip_to_land(data_dir, template)
            var_template = land_template
        else:
            var_template = template
        gridded_fresh = _cache_matches(
            _cache_path(cache_dir, var), _provenance(var, data_dir, target_lat, target_lon, with_source=True)
        )
        get_global_mean_variability(var, data_dir, cache_dir, var_template, cache=True)
        status[var] = "already up to date" if gridded_fresh else "built"
    return status


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m backend.invert_variability",
        description="Build the InVERT variability caches used by the dashboard's line-plot envelope.",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="processed data dir (emulator grid, land mask); netCDFs in <data-dir>/invert/ override the R2 store",
    )
    parser.add_argument("--cache-dir", required=True, help="the simulator cache dir; caches go to <cache-dir>/invert/")
    parser.add_argument("--var", nargs="*", choices=sorted(INVERT_VARIABLES), help="subset of variables (default: all)")
    args = parser.parse_args(argv)
    for var, outcome in build_caches(args.data_dir, args.cache_dir, args.var or None).items():
        print(f"{var}: {outcome}")


if __name__ == "__main__":
    main()
