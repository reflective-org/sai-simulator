"""Unit tests for :mod:`backend.invert_variability`.

The heavy I/O path (opening the 4 GB raw InVERT file, annualising, regridding, splicing)
is exercised end-to-end via a small synthetic dataset written to ``tmp_path``. This keeps
the tests self-contained and independent of the real InVERT file being present.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from backend import invert_variability as iv


N_INVERT_MEMBERS = 100
N_INVERT_MONTHS = 600
# The member count and month count must match the real files (the splice asserts 100
# members and the annualiser needs whole years). The grids are shrunk from the real
# 96x192 / 192x288 so that one synthetic file is ~70 MB instead of 4.4 GB.
SRC_LAT = 12
SRC_LON = 24
TARGET_LAT_SIZE = 24
TARGET_LON_SIZE = 36


def _write_synthetic_invert(invert_dir, var, data):
    """Write ``data`` as an InVERT-shaped file for ``var`` (source variable name from the spec)."""
    spec = iv.INVERT_VARIABLES[var]
    da = xr.DataArray(
        data,
        dims=("ensemble", "time", "lat", "lon"),
        coords={
            "ensemble": np.arange(N_INVERT_MEMBERS),
            "time": np.arange(N_INVERT_MONTHS, dtype=float),
            "month": ("time", (np.arange(N_INVERT_MONTHS) % 12 + 1).astype(np.int64)),
            "lat": np.linspace(-88.57, 88.57, SRC_LAT),
            "lon": np.linspace(0, 360, SRC_LON, endpoint=False),
        },
        name=spec["source_var"],
    )
    da.to_dataset().to_netcdf(invert_dir / spec["filename"])


@pytest.fixture(autouse=True)
def no_r2_store(tmp_path, monkeypatch):
    """Point the R2 store at a path that does not exist, so no test ever reaches the network.

    Tests that exercise the store path write a local Zarr and re-point the URI themselves.
    """
    monkeypatch.setattr(iv, "invert_store_uri", lambda: str(tmp_path / "no_such_store.zarr"))
    iv._open_store.cache_clear()
    yield
    iv._open_store.cache_clear()


def _write_synthetic_store(netcdf_dir, store_path):
    """Copy the synthetic netCDFs in ``netcdf_dir`` into one store the way ``zarrify_invert`` does."""
    ds = xr.Dataset()
    for var, spec in iv.INVERT_VARIABLES.items():
        src = netcdf_dir / spec["filename"]
        with xr.open_dataset(src) as one:
            da = one[spec["source_var"]].load()
        da.attrs = {"source_file": src.name, "source_size_bytes": src.stat().st_size}
        ds[spec["source_var"]] = da
    for name in ds.variables:
        ds[name].encoding = {}
    ds.chunk({"ensemble": 1}).to_zarr(store_path, mode="w", zarr_format=3, consolidated=True)


@pytest.fixture
def synthetic_invert_dir(tmp_path):
    """Write tiny InVERT-shaped tas and pr files into ``tmp_path/invert/`` and return ``tmp_path``.

    The synthetic TREFHT field is Gaussian white noise per (ensemble, time, lat, lon),
    scaled so that annualised values are O(1 K). This matches the real InVERT amplitude
    closely enough for downstream sanity checks. The PRECT field is the same noise in raw
    CESM m/s (1e-8 m/s ~ 0.86 mm/day), so the unit conversion is exercised too.
    """
    invert_dir = tmp_path / "invert"
    invert_dir.mkdir()

    rng = np.random.default_rng(42)
    data = rng.standard_normal(
        (N_INVERT_MEMBERS, N_INVERT_MONTHS, SRC_LAT, SRC_LON)
    ).astype(np.float32)
    _write_synthetic_invert(invert_dir, "tas", data)
    _write_synthetic_invert(invert_dir, "tasmin", data)
    _write_synthetic_invert(invert_dir, "tasmax", data)
    _write_synthetic_invert(invert_dir, "pr", data * np.float32(1e-8))
    return tmp_path


def _emulator_target_grid():
    target_lat = np.linspace(-89.76, 89.76, TARGET_LAT_SIZE)
    target_lon = np.linspace(0, 360, TARGET_LON_SIZE, endpoint=False)
    return target_lat, target_lon


@pytest.mark.parametrize("var", sorted(iv.INVERT_VARIABLES))
def test_load_raw_dims(synthetic_invert_dir, var):
    ds = iv.load_invert_raw(synthetic_invert_dir, var)
    assert iv.INVERT_VARIABLES[var]["source_var"] in ds
    assert dict(ds.sizes) == {
        "ensemble": N_INVERT_MEMBERS,
        "time": N_INVERT_MONTHS,
        "lat": SRC_LAT,
        "lon": SRC_LON,
    }


def test_annualise_collapses_monthly_to_annual():
    rng = np.random.default_rng(0)
    da = xr.DataArray(
        rng.standard_normal((3, 24, 4, 5)).astype(np.float32),
        dims=("ensemble", "time", "lat", "lon"),
    )
    annual = iv._annualise_monthly(da)
    assert dict(annual.sizes) == {"ensemble": 3, "invert_year": 2, "lat": 4, "lon": 5}
    # Each annual value is the mean of a contiguous 12-month block.
    expected_year0 = da.isel(time=slice(0, 12)).mean("time").values
    np.testing.assert_allclose(annual.isel(invert_year=0).values, expected_year0, atol=1e-6)


@pytest.mark.parametrize("how", ["min", "max"])
def test_annualise_takes_min_or_max_month(how):
    rng = np.random.default_rng(1)
    da = xr.DataArray(
        rng.standard_normal((2, 24, 3, 4)).astype(np.float32),
        dims=("ensemble", "time", "lat", "lon"),
    )
    annual = iv._annualise_monthly(da, how)
    expected_year1 = getattr(da.isel(time=slice(12, 24)), how)("time").values
    np.testing.assert_array_equal(annual.isel(invert_year=1).values, expected_year1)


def test_min_max_caches_have_their_own_file_names(tmp_path):
    # tasmin/tasmax caches built from monthly min/max must never be confused with the
    # annual-mean caches of earlier versions, which keep the plain names.
    assert iv._cache_path(tmp_path, "tas").name == "invert_annual_ensemble_tas_1851_2100.nc"
    assert iv._cache_path(tmp_path, "tasmax").name == "invert_annual_ensemble_tasmax_max_1851_2100.nc"
    assert iv._global_mean_cache_path(tmp_path, "tasmin").name == "invert_global_mean_tasmin_min_1851_2100.nc"


def test_annualise_rejects_partial_year():
    da = xr.DataArray(
        np.zeros((1, 13, 2, 2), dtype=np.float32),
        dims=("ensemble", "time", "lat", "lon"),
    )
    with pytest.raises(AssertionError):
        iv._annualise_monthly(da)


def test_pad_for_regrid_adds_lon_wrap_and_lat_poles():
    da = xr.DataArray(
        np.arange(3 * 4).reshape(3, 4).astype(np.float32),
        dims=("lat", "lon"),
        coords={
            "lat": np.array([-60.0, 0.0, 60.0]),
            "lon": np.array([0.0, 90.0, 180.0, 270.0]),
        },
    )
    padded = iv._pad_for_regrid(da)
    assert padded.sizes == {"lat": 5, "lon": 5}
    np.testing.assert_array_equal(padded["lat"].values, [-90.0, -60.0, 0.0, 60.0, 90.0])
    np.testing.assert_array_equal(padded["lon"].values, [0.0, 90.0, 180.0, 270.0, 360.0])
    # lon=360 must be a copy of lon=0 in the padded array (periodic wrap).
    np.testing.assert_array_equal(
        padded.sel(lon=360.0).values, padded.sel(lon=0.0).values
    )
    # lat=-90 / lat=+90 must copy the pole-adjacent source latitude values.
    np.testing.assert_array_equal(
        padded.sel(lat=-90.0).values, padded.sel(lat=-60.0).values
    )
    np.testing.assert_array_equal(
        padded.sel(lat=90.0).values, padded.sel(lat=60.0).values
    )


def test_regrid_produces_target_shape():
    da = xr.DataArray(
        np.zeros((SRC_LAT, SRC_LON), dtype=np.float32) + 1.5,
        dims=("lat", "lon"),
        coords={
            "lat": np.linspace(-88.57, 88.57, SRC_LAT),
            "lon": np.linspace(0, 360, SRC_LON, endpoint=False),
        },
    )
    target_lat, target_lon = _emulator_target_grid()
    regridded = iv._regrid_to_emulator_grid(da, target_lat, target_lon)
    assert regridded.sizes == {"lat": TARGET_LAT_SIZE, "lon": TARGET_LON_SIZE}
    # Constant field must be preserved exactly (up to FP tolerance).
    np.testing.assert_allclose(regridded.values, 1.5, atol=1e-6)


def test_splice_covers_1851_2100_with_20_members():
    # Ensemble dim holds 100 unique integers so we can verify which member ended up where.
    ens = xr.DataArray(
        np.tile(
            np.arange(N_INVERT_MEMBERS, dtype=np.float32)[:, None, None, None],
            (1, iv.INVERT_YEARS_PER_MEMBER, 4, 5),
        ),
        dims=("ensemble", "invert_year", "lat", "lon"),
        coords={
            "ensemble": np.arange(N_INVERT_MEMBERS),
            "invert_year": np.arange(iv.INVERT_YEARS_PER_MEMBER),
            "lat": np.linspace(-89, 89, 4),
            "lon": np.linspace(0, 360, 5, endpoint=False),
        },
    )
    spliced = iv._splice_members(ens)
    assert spliced.dims == ("ensemble", "time", "lat", "lon")
    assert spliced.sizes["ensemble"] == iv.N_OUTPUT_MEMBERS
    assert spliced.sizes["time"] == iv.OUTPUT_N_YEARS
    np.testing.assert_array_equal(
        spliced["time"].values,
        np.arange(iv.OUTPUT_START_YEAR, iv.OUTPUT_END_YEAR + 1),
    )
    # Output member 0 must draw its first 50 years from InVERT member 0, the next 50
    # from member 1, ..., the last 50 from member 4.
    m0 = spliced.isel(ensemble=0)
    for chunk_i in range(iv.MEMBERS_PER_OUTPUT):
        start = chunk_i * iv.INVERT_YEARS_PER_MEMBER
        end = start + iv.INVERT_YEARS_PER_MEMBER
        expected_member = float(chunk_i)  # InVERT members 0..4 for output member 0
        assert np.all(m0.isel(time=slice(start, end)).values == expected_member)
    # Output member 19 must use InVERT members 95..99.
    m19 = spliced.isel(ensemble=19)
    assert m19.isel(time=0).values.min() == 95.0
    assert m19.isel(time=-1).values.max() == 99.0


def test_add_variability_shape_and_ensemble_mean_round_trip():
    time = np.arange(1851, 2101)
    lat = np.linspace(-89.76, 89.76, 8)
    lon = np.linspace(0, 360, 10, endpoint=False)
    mean_da = xr.DataArray(
        np.zeros((time.size, lat.size, lon.size), dtype=np.float32) + 5.0,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
    )
    rng = np.random.default_rng(1)
    ensemble_values = rng.standard_normal((5, time.size, lat.size, lon.size)).astype(np.float32)
    ensemble = xr.DataArray(
        ensemble_values,
        dims=("ensemble", "time", "lat", "lon"),
        coords={
            "ensemble": np.arange(5),
            "time": time,
            "lat": lat,
            "lon": lon,
        },
    )
    combined = iv.add_variability_to_mean(mean_da, ensemble)
    assert combined.dims == ("ensemble", "time", "lat", "lon")
    assert combined.sizes == {"ensemble": 5, "time": time.size, "lat": lat.size, "lon": lon.size}
    # Averaging the combined ensemble over the ensemble dim must recover the mean plus
    # the ensemble-mean noise (which is not zero for a finite draw).
    np.testing.assert_allclose(
        combined.mean("ensemble").values,
        mean_da.values + ensemble.mean("ensemble").values,
        atol=1e-5,
    )


def test_add_variability_rejects_time_mismatch():
    mean_da = xr.DataArray(
        np.zeros((3, 2, 2), dtype=np.float32),
        dims=("time", "lat", "lon"),
        coords={"time": np.array([2000, 2001, 2002])},
    )
    ensemble = xr.DataArray(
        np.zeros((1, 3, 2, 2), dtype=np.float32),
        dims=("ensemble", "time", "lat", "lon"),
        coords={"time": np.array([1900, 1901, 1902])},
    )
    with pytest.raises(ValueError, match="time coord mismatch"):
        iv.add_variability_to_mean(mean_da, ensemble)


@pytest.mark.parametrize("var", sorted(iv.INVERT_VARIABLES))
def test_build_annual_ensemble_end_to_end(synthetic_invert_dir, var):
    target_lat, target_lon = _emulator_target_grid()
    ensemble = iv.build_annual_ensemble(
        data_dir=synthetic_invert_dir,
        target_lat=target_lat,
        target_lon=target_lon,
        var=var,
        cache=False,
    )
    assert ensemble.name == f"{var}_variability"
    assert ensemble.attrs["units"] == iv.INVERT_VARIABLES[var]["units"]
    assert ensemble.dims == ("ensemble", "time", "lat", "lon")
    assert ensemble.sizes == {
        "ensemble": iv.N_OUTPUT_MEMBERS,
        "time": iv.OUTPUT_N_YEARS,
        "lat": TARGET_LAT_SIZE,
        "lon": TARGET_LON_SIZE,
    }
    np.testing.assert_array_equal(
        ensemble["time"].values,
        np.arange(iv.OUTPUT_START_YEAR, iv.OUTPUT_END_YEAR + 1),
    )
    assert ensemble.attrs["annual_aggregation"] == iv.INVERT_VARIABLES[var]["annual"]
    # White-noise input averaged over both the ensemble and time dims yields the InVERT
    # climatology per grid cell -- must be near zero everywhere (well under the 0.3 K
    # guard that would otherwise flag a forced trend). For tasmin / tasmax the coldest /
    # warmest month is biased, and the centring removes that bias.
    climatology = ensemble.mean(("ensemble", "time")).values
    assert float(np.abs(climatology).max()) < iv.CLIMATOLOGY_TOLERANCE


def test_pr_ensemble_is_converted_from_m_per_s_to_mm_per_day(synthetic_invert_dir):
    """The pr ensemble must equal the tas ensemble times 1e-8 m/s * 86400000 (mm/day per m/s)."""
    target_lat, target_lon = _emulator_target_grid()
    kwargs = dict(data_dir=synthetic_invert_dir, target_lat=target_lat, target_lon=target_lon, cache=False)
    tas = iv.build_annual_ensemble(var="tas", **kwargs)
    pr = iv.build_annual_ensemble(var="pr", **kwargs)
    np.testing.assert_allclose(pr.values, tas.values * 1e-8 * 86400.0 * 1000.0, rtol=1e-4, atol=1e-5)


def test_cache_path_is_per_variable(tmp_path):
    assert iv._cache_path(tmp_path, "tas") != iv._cache_path(tmp_path, "pr")
    assert iv._cache_path(tmp_path, "tas").name == "invert_annual_ensemble_tas_1851_2100.nc"


def test_get_global_mean_variability_shape_and_mask(synthetic_invert_dir, tmp_path):
    target_lat, target_lon = _emulator_target_grid()
    # Template with the western half of the grid masked out (like a land-only pr field).
    template = xr.DataArray(
        np.zeros((3, TARGET_LAT_SIZE, TARGET_LON_SIZE), dtype=np.float32),
        dims=("time", "lat", "lon"),
        coords={"time": np.array([2015, 2016, 2017]), "lat": target_lat, "lon": target_lon},
    )
    template = template.where(template["lon"] >= 180)

    gm = iv.get_global_mean_variability("tas", synthetic_invert_dir, tmp_path / "cache", template, cache=False)
    assert gm.dims == ("ensemble", "time")
    assert gm.sizes == {"ensemble": iv.N_OUTPUT_MEMBERS, "time": iv.OUTPUT_N_YEARS}
    assert gm.attrs["units"] == "K"

    # Must equal the cos(lat)-weighted mean over the unmasked cells only.
    ensemble = iv.build_annual_ensemble(
        data_dir=synthetic_invert_dir, target_lat=target_lat, target_lon=target_lon, var="tas", cache=False
    )
    weights = np.cos(np.deg2rad(ensemble["lat"]))
    expected = ensemble.sel(lon=slice(180, None)).weighted(weights).mean(("lat", "lon"))
    np.testing.assert_allclose(gm.values, expected.values, rtol=1e-4, atol=1e-5)


def test_get_global_mean_variability_uses_cache(synthetic_invert_dir, tmp_path):
    cache_dir = tmp_path / "cache"
    target_lat, target_lon = _emulator_target_grid()
    template = xr.DataArray(
        np.zeros((1, TARGET_LAT_SIZE, TARGET_LON_SIZE), dtype=np.float32),
        dims=("time", "lat", "lon"),
        coords={"time": np.array([2015]), "lat": target_lat, "lon": target_lon},
    )
    first = iv.get_global_mean_variability("tas", synthetic_invert_dir, cache_dir, template, cache=True)
    assert iv._global_mean_cache_path(cache_dir, "tas").exists()
    assert iv._cache_path(cache_dir, "tas").exists()
    second = iv.get_global_mean_variability("tas", synthetic_invert_dir, cache_dir, template, cache=True)
    np.testing.assert_array_equal(first.values, second.values)
    # Caches live under <cache_dir>/invert/, never next to the raw data.
    assert not (synthetic_invert_dir / "invert" / iv._cache_path(cache_dir, "tas").name).exists()
    # Atomic writes leave no temp files behind.
    assert not list((cache_dir / "invert").glob("*.tmp*"))


def _template(n_time=1, land_only=False):
    target_lat, target_lon = _emulator_target_grid()
    template = xr.DataArray(
        np.zeros((n_time, TARGET_LAT_SIZE, TARGET_LON_SIZE), dtype=np.float32),
        dims=("time", "lat", "lon"),
        coords={"time": 2015 + np.arange(n_time), "lat": target_lat, "lon": target_lon},
    )
    return template.where(template["lon"] >= 180) if land_only else template


def test_request_path_never_builds_and_serves_only_valid_caches(synthetic_invert_dir, tmp_path):
    cache_dir = tmp_path / "cache"
    template = _template()
    # No cache yet: the request path returns None and builds nothing, even with the raw file present.
    assert iv.get_area_mean_variability("tas", synthetic_invert_dir, cache_dir, template) is None
    assert not (cache_dir / "invert").exists()
    # Offline build, then the request path serves it.
    built = iv.get_global_mean_variability("tas", synthetic_invert_dir, cache_dir, template, cache=True)
    served = iv.get_area_mean_variability("tas", synthetic_invert_dir, cache_dir, template)
    np.testing.assert_array_equal(served.values, built.values)
    # The raw file may be removed afterwards (cache-only deployment): still served.
    (synthetic_invert_dir / "invert" / iv.INVERT_VARIABLES["tas"]["filename"]).unlink()
    assert iv.get_area_mean_variability("tas", synthetic_invert_dir, cache_dir, template) is not None


def test_store_is_read_when_there_is_no_local_netcdf(synthetic_invert_dir, tmp_path, monkeypatch):
    """Without local netCDFs the R2 store is the source, and it yields the same ensemble."""
    target_lat, target_lon = _emulator_target_grid()
    kwargs = dict(target_lat=target_lat, target_lon=target_lon, var="tasmax", cache=False)
    from_netcdf = iv.build_annual_ensemble(data_dir=synthetic_invert_dir, **kwargs)

    store = tmp_path / "store.zarr"
    _write_synthetic_store(synthetic_invert_dir / "invert", store)
    monkeypatch.setattr(iv, "invert_store_uri", lambda: str(store))
    empty_data_dir = tmp_path / "empty"
    empty_data_dir.mkdir()
    assert iv.invert_source_status(empty_data_dir, "tasmax") is None
    from_store = iv.build_annual_ensemble(data_dir=empty_data_dir, **kwargs)
    np.testing.assert_array_equal(from_store.values, from_netcdf.values)


def test_caches_from_netcdf_and_store_share_provenance(synthetic_invert_dir, tmp_path, monkeypatch):
    """A cache built from the local netCDF is up to date for a machine that reads the store."""
    cache_dir = tmp_path / "cache"
    template = _template()
    iv.get_global_mean_variability("tas", synthetic_invert_dir, cache_dir, template, cache=True)

    store = tmp_path / "store.zarr"
    _write_synthetic_store(synthetic_invert_dir / "invert", store)
    monkeypatch.setattr(iv, "invert_store_uri", lambda: str(store))
    empty_data_dir = tmp_path / "empty"
    empty_data_dir.mkdir()
    target_lat, target_lon = _emulator_target_grid()
    for data_dir in (synthetic_invert_dir, empty_data_dir):
        expected = iv._provenance("tas", data_dir, target_lat, target_lon, with_source=True)
        assert iv._cache_matches(iv._cache_path(cache_dir, "tas"), expected)


def test_source_status_reports_a_missing_store(tmp_path):
    reason = iv.invert_source_status(tmp_path, "tas")
    assert reason is not None and "no_such_store.zarr" in reason


def test_stale_cache_is_ignored_on_version_bump(synthetic_invert_dir, tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    template = _template()
    iv.get_global_mean_variability("tas", synthetic_invert_dir, cache_dir, template, cache=True)
    assert iv.get_area_mean_variability("tas", synthetic_invert_dir, cache_dir, template) is not None
    monkeypatch.setattr(iv, "INVERT_CACHE_VERSION", iv.INVERT_CACHE_VERSION + 1)
    # Request path: treated as missing. Offline path: rebuilt and stamped with the new version.
    assert iv.get_area_mean_variability("tas", synthetic_invert_dir, cache_dir, template) is None
    iv.get_global_mean_variability("tas", synthetic_invert_dir, cache_dir, template, cache=True)
    with xr.open_dataarray(iv._global_mean_cache_path(cache_dir, "tas")) as da:
        assert int(da.attrs["invert_cache_version"]) == iv.INVERT_CACHE_VERSION
    assert iv.get_area_mean_variability("tas", synthetic_invert_dir, cache_dir, template) is not None


def test_global_mean_cache_is_keyed_on_the_land_mask(synthetic_invert_dir, tmp_path):
    cache_dir = tmp_path / "cache"
    iv.get_global_mean_variability("tas", synthetic_invert_dir, cache_dir, _template(), cache=True)
    # Same variable and grid but a different NaN pattern: the cache does not match.
    assert iv.get_area_mean_variability("tas", synthetic_invert_dir, cache_dir, _template(land_only=True)) is None
    assert iv.get_area_mean_variability("tas", synthetic_invert_dir, cache_dir, _template()) is not None


def test_corrupt_cache_file_is_treated_as_missing(synthetic_invert_dir, tmp_path):
    cache_dir = tmp_path / "cache"
    (cache_dir / "invert").mkdir(parents=True)
    iv._global_mean_cache_path(cache_dir, "tas").write_bytes(b"not a netcdf file")
    assert iv.get_area_mean_variability("tas", synthetic_invert_dir, cache_dir, _template()) is None


def test_build_caches_cli_builds_from_the_emulator_grid(synthetic_invert_dir, tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    target_lat, target_lon = _emulator_target_grid()
    np.save(synthetic_invert_dir / "correct_lat.npy", target_lat)
    import backend.constants as constants
    monkeypatch.setattr(constants, "NUM_LON", TARGET_LON_SIZE)
    status = iv.build_caches(synthetic_invert_dir, cache_dir, ["tas"])
    assert status == {"tas": "built"}
    assert iv.get_area_mean_variability("tas", synthetic_invert_dir, cache_dir, _template()) is not None
    # Second run finds everything up to date; a variable without a raw file is skipped.
    (synthetic_invert_dir / "invert" / iv.INVERT_VARIABLES["tasmin"]["filename"]).unlink()
    status = iv.build_caches(synthetic_invert_dir, cache_dir, ["tas", "tasmin"])
    assert status["tas"] == "already up to date"
    assert status["tasmin"].startswith("skipped")


def test_get_regional_mean_variability_matches_masked_weighted_mean(synthetic_invert_dir, tmp_path):
    cache_dir = tmp_path / "cache"
    target_lat, target_lon = _emulator_target_grid()
    # Template with the western half masked out (like a land-only pr field) and a region
    # that straddles that edge, so both the region and the template mask must apply.
    template = xr.DataArray(
        np.zeros((2, TARGET_LAT_SIZE, TARGET_LON_SIZE), dtype=np.float32),
        dims=("time", "lat", "lon"),
        coords={"time": np.array([2015, 2016]), "lat": target_lat, "lon": target_lon},
    )
    template = template.where(template["lon"] >= 180)
    region_mask = (
        (template["lat"] >= 20) & (template["lat"] <= 60)
        & (template["lon"] >= 120) & (template["lon"] <= 240)
    ).transpose("lat", "lon")

    # Request path: nothing cached yet, so None and no build.
    assert iv.get_area_mean_variability("tas", synthetic_invert_dir, cache_dir, template, region_mask) is None
    rm = iv.get_regional_mean_variability("tas", synthetic_invert_dir, cache_dir, template, region_mask)
    assert rm.dims == ("ensemble", "time")
    assert rm.sizes == {"ensemble": iv.N_OUTPUT_MEMBERS, "time": iv.OUTPUT_N_YEARS}
    assert rm.attrs["units"] == "K"
    # The gridded ensemble is cached on first use so later regions are cheap, and the
    # request path now serves the same numbers.
    assert iv._cache_path(cache_dir, "tas").exists()
    served = iv.get_area_mean_variability("tas", synthetic_invert_dir, cache_dir, template, region_mask)
    np.testing.assert_array_equal(served.values, rm.values)

    ensemble = iv.build_annual_ensemble(
        data_dir=synthetic_invert_dir, target_lat=target_lat, target_lon=target_lon, var="tas", cache=False
    )
    weights = np.cos(np.deg2rad(ensemble["lat"]))
    expected = (
        ensemble.sel(lat=slice(20, 60), lon=slice(180, 240)).weighted(weights).mean(("lat", "lon"))
    )
    np.testing.assert_allclose(rm.values, expected.values, rtol=1e-4, atol=1e-5)


def test_get_regional_mean_variability_rejects_empty_region(synthetic_invert_dir, tmp_path):
    target_lat, target_lon = _emulator_target_grid()
    template = xr.DataArray(
        np.zeros((1, TARGET_LAT_SIZE, TARGET_LON_SIZE), dtype=np.float32),
        dims=("time", "lat", "lon"),
        coords={"time": np.array([2015]), "lat": target_lat, "lon": target_lon},
    )
    region_mask = xr.zeros_like(template.isel(time=0, drop=True), dtype=bool)
    with pytest.raises(ValueError, match="no grid cells"):
        iv.get_regional_mean_variability("tas", synthetic_invert_dir, tmp_path / "cache", template, region_mask)


def test_regional_mode_needs_the_gridded_cache(synthetic_invert_dir, tmp_path):
    cache_dir = tmp_path / "cache"
    template = _template()
    region_mask = (template["lat"] > 0).broadcast_like(template.isel(time=0, drop=True)).transpose("lat", "lon")
    # A global-mean cache alone serves global mode but not regional mode.
    iv.get_global_mean_variability("tas", synthetic_invert_dir, cache_dir, template, cache=True)
    iv._cache_path(cache_dir, "tas").unlink()
    assert iv.get_area_mean_variability("tas", synthetic_invert_dir, cache_dir, template) is not None
    assert iv.get_area_mean_variability("tas", synthetic_invert_dir, cache_dir, template, region_mask) is None


def test_min_max_members_are_centred_and_spread_like_the_extreme_month(synthetic_invert_dir):
    target_lat, target_lon = _emulator_target_grid()
    kwargs = dict(data_dir=synthetic_invert_dir, target_lat=target_lat, target_lon=target_lon, cache=False)
    tas = iv.build_annual_ensemble(var="tas", **kwargs)
    tasmax = iv.build_annual_ensemble(var="tasmax", **kwargs)
    # Same monthly noise in both synthetic files: the 12-month mean has std ~1/sqrt(12),
    # the warmest of 12 months a much larger one, and after centring both average ~0.
    assert float(np.abs(tasmax.mean(("ensemble", "time"))).max()) < 1e-4
    assert float(tasmax.std()) > 1.5 * float(tas.std())


def test_p_e_borrows_the_pr_envelope_in_the_same_units():
    source = iv.INVERT_PROXIES["p-e"]
    assert source == "pr"
    assert iv.INVERT_VARIABLES[source]["units"] == "mm/day"
    assert iv.INVERT_VARIABLES[source]["annual"] == "mean"
