"""Unit tests for the pure / near-pure helpers in ``backend.get_outputs``.

Scope rationale
---------------
The helpers in ``backend/get_outputs.py`` fall into three groups:

1. **Pure / near-pure helpers** — operate on numpy arrays, xarray DataArrays, or
   strings, and have no I/O. These are unit-tested here against synthetic
   inputs constructed inline. They are the bulk of this file.

2. **Smoke-testable I/O helpers** — write a single artifact (GeoTIFF or NetCDF)
   from synthetic DataArrays. These are tested against ``tmp_path`` here, which
   gives us regression coverage on the file format and option threading without
   needing real cached netCDF inputs.

3. **Heavy-I/O / orchestrator helpers** — ``get_outputs``,
   ``_load_variable_fields``, ``_compute_regional_delta``, ``_compute_sai_p_values``,
   ``_compute_internal_variability``, ``get_temp_diff``, ``create_vectorized_files``,
   and ``_borders_50m``. **Intentionally not tested here.** They depend on real
   ``data_dir`` / ``model_dir`` / ``cache_dir`` contents (multi-megabyte cached
   netCDF inputs, fitted SO2 interpolators, IPCC continental geojsons), on
   cartopy network downloads, or on the GeoPandas region table. End-to-end
   coverage for these paths is provided by ``tests/verification/`` against the
   v1.3.0 reference bucket on S3 — running 40 randomised cases there exercises
   every branch in the orchestrator. Reproducing that fixture set inline would
   be both fragile and redundant.

   The three small dataclasses (``_VariableFields``, ``_DeltaResult``,
   ``_MeanOverTimeResult``) are plain data containers with no behaviour worth
   testing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr

from backend.get_outputs import (
    _adjust_polar_lonlat,
    _compute_distribution,
    _compute_mean_over_time,
    _compute_params_hash,
    _geotiff_coord_step_degrees,
    _hash_variable_injection,
    _reproject_to_arctic_stereo,
    _significance_mask,
    create_geotiff_files,
    generate_sai_netcdf,
    get_so2_by_latitude,
)


# ----------------------------------------------------------------------------
# Synthetic-fixture builders
# ----------------------------------------------------------------------------


def _lat_lon_da(values: np.ndarray) -> xr.DataArray:
    """Wrap a 2D array as a ``(lat, lon)`` DataArray with sensible default coords."""
    arr = np.asarray(values, dtype=np.float64)
    lat = np.linspace(-30.0, 30.0, arr.shape[0])
    lon = np.linspace(0.0, 350.0, arr.shape[1])
    return xr.DataArray(arr, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})


def _time_lat_lon_da(values: np.ndarray, time: np.ndarray) -> xr.DataArray:
    """Wrap a 3D array as a ``(time, lat, lon)`` DataArray."""
    arr = np.asarray(values, dtype=np.float64)
    lat = np.linspace(-30.0, 30.0, arr.shape[1])
    lon = np.linspace(0.0, 350.0, arr.shape[2])
    return xr.DataArray(
        arr,
        dims=("time", "lat", "lon"),
        coords={"time": np.asarray(time), "lat": lat, "lon": lon},
    )


# ============================================================================
# _compute_params_hash
# ============================================================================


def test_compute_params_hash_is_deterministic() -> None:
    h1 = _compute_params_hash("SSP1-2.6", 1.5, 2091, 2100, 2050, 10, None)
    h2 = _compute_params_hash("SSP1-2.6", 1.5, 2091, 2100, 2050, 10, None)
    assert h1 == h2
    assert len(h1) == 10
    # Matches an MD5 prefix of the joined params (encodes the contract at
    # tests/verification/validate_outputs.py:67-85 which depends on it).
    expected = hashlib.md5("SSP1-2.6_1.5_2091_2100_2050_10".encode()).hexdigest()[:10]
    assert h1 == expected


@pytest.mark.parametrize(
    "ssp,temp,d0,d1,sy,r",
    [
        ("SSP5-Baseline", 1.5, 2091, 2100, 2050, 10),  # SSP differs
        ("SSP1-2.6", 2.0, 2091, 2100, 2050, 10),       # temp_target differs
        ("SSP1-2.6", 1.5, 2081, 2100, 2050, 10),       # decade_start differs
        ("SSP1-2.6", 1.5, 2091, 2090, 2050, 10),       # decade_end differs
        ("SSP1-2.6", 1.5, 2091, 2100, 2060, 10),       # start_year differs
        ("SSP1-2.6", 1.5, 2091, 2100, 2050, 20),       # ramp_up differs
    ],
)
def test_compute_params_hash_param_sensitivity(
    ssp: str, temp: float, d0: int, d1: int, sy: int, r: int
) -> None:
    base = _compute_params_hash("SSP1-2.6", 1.5, 2091, 2100, 2050, 10, None)
    other = _compute_params_hash(ssp, temp, d0, d1, sy, r, None)
    assert base != other


def test_compute_params_hash_injection_path_differs_from_analytic() -> None:
    inj = np.zeros((7, 66), dtype=np.float64)
    h_inj = _compute_params_hash("SSP1-2.6", 1.5, 2091, 2100, 2050, 10, inj)
    h_ana = _compute_params_hash("SSP1-2.6", 1.5, 2091, 2100, 2050, 10, None)
    assert h_inj != h_ana
    assert len(h_inj) == 10


def test_compute_params_hash_injection_array_sensitivity() -> None:
    a = np.zeros((7, 66), dtype=np.float64)
    b = a.copy()
    b[3, 10] = 1.0
    # When injection is provided, temp_target/start_year/ramp_up are ignored.
    h_a = _compute_params_hash("SSP1-2.6", None, 2091, 2100, None, None, a)
    h_b = _compute_params_hash("SSP1-2.6", None, 2091, 2100, None, None, b)
    assert h_a != h_b


def test_compute_params_hash_injection_drops_analytic_params() -> None:
    # With the same injection array, varying temp_target / start_year / ramp_up
    # must NOT change the hash (they're excluded from the injection-path key).
    inj = np.zeros((7, 66), dtype=np.float64)
    h_a = _compute_params_hash("SSP1-2.6", 1.5, 2091, 2100, 2050, 10, inj)
    h_b = _compute_params_hash("SSP1-2.6", 9.9, 2091, 2100, 2099, 99, inj)
    assert h_a == h_b


# ============================================================================
# _hash_variable_injection
# ============================================================================


def test_hash_variable_injection_same_array_same_hash() -> None:
    a = np.arange(14, dtype=np.float64).reshape(7, 2)
    b = np.arange(14, dtype=np.float64).reshape(7, 2)
    assert _hash_variable_injection(a) == _hash_variable_injection(b)
    assert len(_hash_variable_injection(a)) == 8


def test_hash_variable_injection_different_arrays_different_hashes() -> None:
    a = np.zeros((7, 2), dtype=np.float64)
    b = np.zeros((7, 2), dtype=np.float64)
    b[0, 0] = 1.0
    assert _hash_variable_injection(a) != _hash_variable_injection(b)


def test_hash_variable_injection_shape_folded_into_hash() -> None:
    # Same float bytes, different shape — the docstring guarantees these hash
    # differently because the shape is struct-packed into the payload.
    arr = np.arange(12, dtype=np.float64)
    assert _hash_variable_injection(arr.reshape(3, 4)) != _hash_variable_injection(
        arr.reshape(4, 3)
    )


# ============================================================================
# _significance_mask
# ============================================================================


def _pvalues_da(values: list[list[float]]) -> xr.DataArray:
    """Build a small lat×lon p-values DataArray suitable for ``_significance_mask``."""
    arr = np.asarray(values, dtype=np.float64)
    return xr.DataArray(
        arr,
        dims=("lat", "lon"),
        coords={"lat": np.linspace(-30.0, 30.0, arr.shape[0]),
                "lon": np.linspace(0.0, 350.0, arr.shape[1])},
    )


def test_significance_mask_none_input_returns_none() -> None:
    assert _significance_mask(None) is None


def test_significance_mask_above_threshold_marks_one() -> None:
    pv = _pvalues_da([[0.10, 0.50, 0.99, 0.06]])
    mask = _significance_mask(pv)
    # All p >= 0.05 → all 1.0, no NaN.
    assert mask is not None
    assert not np.any(np.isnan(mask))
    assert np.all(mask == 1.0)


def test_significance_mask_below_threshold_marks_nan() -> None:
    pv = _pvalues_da([[0.01, 0.04, 0.001, 0.0001]])
    mask = _significance_mask(pv)
    assert mask is not None
    assert np.all(np.isnan(mask))


def test_significance_mask_nan_pvalues_pass_through_as_nan() -> None:
    pv = _pvalues_da([[np.nan, np.nan, np.nan, np.nan]])
    mask = _significance_mask(pv)
    assert mask is not None
    assert np.all(np.isnan(mask))


def test_significance_mask_dtype_is_float32() -> None:
    pv = _pvalues_da([[0.10, 0.04, 0.99, 0.001]])
    mask = _significance_mask(pv)
    assert mask is not None
    assert mask.dtype == np.float32


def test_significance_mask_applies_polar_lonlat_adjustment() -> None:
    # lon=200 wraps to -160, so columns reorder. Build a 1-row pattern where
    # the lon-200 cell has p<0.05 and the lon=10 cell has p>=0.05; after
    # reorder the mask's first column should be NaN, second 1.0.
    pv = xr.DataArray(
        np.array([[1.0, 0.0]], dtype=np.float64),
        dims=("lat", "lon"),
        coords={"lat": [0.0], "lon": [10.0, 200.0]},
    )
    mask = _significance_mask(pv)
    assert mask is not None
    assert mask.shape == (1, 2)
    assert np.isnan(mask[0, 0])  # came from lon=200 (p=0.0 < 0.05)
    assert mask[0, 1] == 1.0     # came from lon=10  (p=1.0 >= 0.05)


# ============================================================================
# _adjust_polar_lonlat
# ============================================================================


def test_adjust_polar_lonlat_wraps_lon_to_pm180_and_sorts() -> None:
    da = xr.DataArray(
        [[1.0, 2.0]],
        dims=("lat", "lon"),
        coords={"lat": [0.0], "lon": [10.0, 200.0]},
    )
    out = _adjust_polar_lonlat(da)
    assert list(out.lon.values) == [-160.0, 10.0]


def test_adjust_polar_lonlat_flips_lat_when_increasing() -> None:
    da = xr.DataArray(
        [[1.0], [2.0]],
        dims=("lat", "lon"),
        coords={"lat": [-30.0, 30.0], "lon": [0.0]},
    )
    out = _adjust_polar_lonlat(da)
    assert list(out.lat.values) == [30.0, -30.0]


def test_adjust_polar_lonlat_keeps_lat_when_already_decreasing() -> None:
    da = xr.DataArray(
        [[1.0], [2.0]],
        dims=("lat", "lon"),
        coords={"lat": [30.0, -30.0], "lon": [0.0]},
    )
    out = _adjust_polar_lonlat(da)
    assert list(out.lat.values) == [30.0, -30.0]


# ============================================================================
# _geotiff_coord_step_degrees
# ============================================================================


def test_coord_step_uniform_spacing() -> None:
    coord = xr.DataArray([0.0, 1.0, 2.0, 3.0], dims="x")
    assert _geotiff_coord_step_degrees(coord, axis_name="x") == 1.0


def test_coord_step_single_value_returns_one() -> None:
    coord = xr.DataArray([5.0], dims="x")
    assert _geotiff_coord_step_degrees(coord, axis_name="x") == 1.0


def test_coord_step_skips_nan_entries() -> None:
    # Span = 3 - 0 = 3, finite n = 3 → step = 3 / (3-1) = 1.5.
    coord = xr.DataArray([0.0, 1.0, np.nan, 3.0], dims="x")
    assert _geotiff_coord_step_degrees(coord, axis_name="x") == 1.5


def test_coord_step_all_nan_raises() -> None:
    coord = xr.DataArray([np.nan, np.nan], dims="x")
    with pytest.raises(ValueError, match="lat"):
        _geotiff_coord_step_degrees(coord, axis_name="lat")


# ============================================================================
# get_so2_by_latitude
# ============================================================================


def _so2_interpolator(slope_per_lat: np.ndarray) -> xr.DataArray:
    """Build a minimal interpolator with a 'features' coord including 'slope'."""
    arr = np.zeros((1, len(slope_per_lat)), dtype=np.float64)
    arr[0] = slope_per_lat
    return xr.DataArray(
        arr,
        dims=("features", "lat"),
        coords={"features": ["slope"], "lat": np.linspace(-60.0, 60.0, len(slope_per_lat))},
    )


def test_get_so2_by_latitude_zeroes_when_temp_diff_positive() -> None:
    # temp_diff > 0 ⇒ projected temp is BELOW target, no SAI needed.
    time = np.arange(2050, 2060)
    interp = _so2_interpolator(np.array([0.5, 0.5, 0.5]))
    temp_diff = xr.DataArray(np.full_like(time, 1.0, dtype=np.float64),
                             dims=("time",), coords={"time": time})
    out = get_so2_by_latitude(interp, temp_diff, ramp_up=5)
    # All output should be zero.
    assert np.all(out.values == 0.0)


def test_get_so2_by_latitude_applies_slope_when_sai_needed() -> None:
    # temp_diff < 0 ⇒ SAI needed. so2 = slope * temp_diff (negative result).
    time = np.arange(2050, 2070)  # 20 years, longer than ramp_up so non-ramp years are stable
    interp = _so2_interpolator(np.array([0.5, 1.0, 0.5]))
    temp_diff = xr.DataArray(np.full_like(time, -2.0, dtype=np.float64),
                             dims=("time",), coords={"time": time})
    out = get_so2_by_latitude(interp, temp_diff, ramp_up=0)  # no ramp-up: clean slope*diff

    expected = (interp.sel(features="slope") * temp_diff).values
    np.testing.assert_allclose(out.values, expected)


def test_get_so2_by_latitude_ramp_up_scales_first_years() -> None:
    # With ramp_up=4 and 20 years of SAI, years 0-3 of SAI are linearly scaled
    # (1/4, 2/4, 3/4, 4/4); subsequent years are unscaled.
    time = np.arange(2050, 2070)
    interp = _so2_interpolator(np.array([1.0, 1.0, 1.0]))
    temp_diff = xr.DataArray(np.full_like(time, -1.0, dtype=np.float64),
                             dims=("time",), coords={"time": time})
    out = get_so2_by_latitude(interp, temp_diff, ramp_up=4)

    # Take any latitude band. After ramp, value is slope * temp_diff = 1 * -1 = -1.
    # During ramp, each year is multiplied by linspace(1/4, 1, 4).
    series = out.isel(lat=1).values
    np.testing.assert_allclose(series[:4], np.linspace(0.25, 1.0, 4) * -1.0)
    np.testing.assert_allclose(series[4:], -1.0)


# ============================================================================
# _compute_mean_over_time
#
# Three representative cases per the agreed scope: regular variable with and
# without historical_model in global mode, and the icefrac branch (which uses
# sum-op aggregation and the area weighting). The is_above_below path is
# excluded because it calls clip_to_land which needs a real geojson directory,
# and the regional path is excluded because it uses create_mask on a real
# GeoDataFrame; both are exercised by the verification harness.
# ============================================================================


def _build_mot_inputs(
    *,
    n_lat: int = 3,
    n_lon: int = 4,
) -> dict:
    """Build a baseline set of inputs for ``_compute_mean_over_time`` cases.

    Returns time-aligned DataArrays for historical (1850–2014), recent
    (2015–2049), and projection (2050–2100) periods so the helper's
    concatenate-and-smooth pass has plenty of samples.
    """
    rng = np.random.default_rng(0)
    hist_time = np.arange(1850, 2015)
    recent_time = np.arange(2015, 2050)
    proj_time = np.arange(2050, 2101)

    historical = _time_lat_lon_da(
        rng.normal(0.0, 0.5, size=(len(hist_time), n_lat, n_lon)), hist_time
    )
    recent = _time_lat_lon_da(
        rng.normal(0.5, 0.3, size=(len(recent_time), n_lat, n_lon)), recent_time
    )
    regional_map = _time_lat_lon_da(
        rng.normal(2.0, 0.3, size=(len(proj_time), n_lat, n_lon)), proj_time
    )
    regional_delta = _time_lat_lon_da(
        rng.normal(-1.5, 0.1, size=(len(proj_time), n_lat, n_lon)), proj_time
    )
    return {
        "historical_model": historical,
        "recent_regional_map": recent,
        "regional_map": regional_map,
        "regional_delta": regional_delta,
    }


def test_compute_mean_over_time_global_with_history_returns_smoothed_series() -> None:
    inputs = _build_mot_inputs()
    res = _compute_mean_over_time(
        "tas",
        regional_map=inputs["regional_map"],
        regional_delta=inputs["regional_delta"],
        recent_regional_map=inputs["recent_regional_map"],
        historical_model=inputs["historical_model"],
        area=None,
        spatial_gdf=None,
        spatial_item=None,
        start_year=2050,
        variable_injection=None,
        data_dir=Path("/unused-in-global-non-above-below"),
    )

    # The historical+recent series + projection no-SAI series are reassembled
    # and smoothed; the with-SAI series shares the pre-SAI window with no-SAI.
    assert "time" in res.mean_no_sai.dims
    assert "time" in res.mean_with_sai.dims
    assert "time" in res.historical_model_global_mean.dims
    # mean_no_sai is single-valued per year (already aggregated over lat/lon).
    assert res.mean_no_sai.ndim == 1
    # Pre-SAI years of with_sai should equal no_sai (anchored splice).
    pre_sai = res.mean_with_sai.sel(time=slice(None, 2049))
    np.testing.assert_allclose(
        pre_sai.values, res.mean_no_sai.sel(time=pre_sai.time).values
    )
    # Global mode → masked outputs are None.
    assert res.masked_regional_map is None
    assert res.masked_regional_delta is None
    assert res.masked_historical_model is None


def test_compute_mean_over_time_global_without_history_uses_recent_as_prefix() -> None:
    inputs = _build_mot_inputs()
    res = _compute_mean_over_time(
        "tas",
        regional_map=inputs["regional_map"],
        regional_delta=inputs["regional_delta"],
        recent_regional_map=inputs["recent_regional_map"],
        historical_model=None,
        area=None,
        spatial_gdf=None,
        spatial_item=None,
        start_year=2050,
        variable_injection=None,
        data_dir=Path("/unused"),
    )

    # Without historical_model, historical_model_global_mean is the recent series
    # (post-smooth boundary) — its first year should be 2015, not 1850.
    assert int(res.historical_model_global_mean.time.values[0]) >= 2015
    # mean_with_sai is aggregated from regional_map (start_year:SIM_END_YEAR), so
    # its time axis starts at start_year and matches mean_no_sai post-split.
    assert int(res.mean_with_sai.time.values[0]) == 2050
    assert int(res.mean_no_sai.time.values[0]) == 2050
    assert res.mean_with_sai.shape == res.mean_no_sai.shape


def test_compute_mean_over_time_icefrac_uses_sum_op_with_area_weighting() -> None:
    inputs = _build_mot_inputs()
    # Make icefrac "fractions" (0–1 ish) to keep numbers stable.
    for k in ("historical_model", "recent_regional_map", "regional_map", "regional_delta"):
        inputs[k] = inputs[k].clip(0.0, 1.0)
    # Synthetic per-cell area: all-ones makes "sum" a count of ice-covered cells.
    area = _lat_lon_da(np.ones((3, 4)))

    res = _compute_mean_over_time(
        "icefrac",
        regional_map=inputs["regional_map"],
        regional_delta=inputs["regional_delta"],
        recent_regional_map=inputs["recent_regional_map"],
        historical_model=inputs["historical_model"],
        area=area,
        spatial_gdf=None,
        spatial_item=None,
        start_year=2050,
        variable_injection=None,
        data_dir=Path("/unused"),
    )

    # All three series are present; in icefrac mode aggregation op is "sum" (not
    # "mean"), so values should be on the order of n_cells * fraction (i.e.,
    # bounded by 12 for our 3x4 grid of fractions). This is a loose check —
    # the smoothing pass can shift values modestly.
    assert res.mean_no_sai.max().item() <= 12.5
    assert res.mean_with_sai.max().item() <= 12.5


# ============================================================================
# _compute_distribution
# ============================================================================


def _build_distribution_inputs(values_no_sai: float, values_delta: float) -> dict:
    """Build minimal time-varying inputs covering the 1950-1959 historical slice
    and the decade-of-interest 2091-2100."""
    hist_time = np.arange(1850, 2015)
    proj_time = np.arange(2050, 2101)
    rng = np.random.default_rng(1)
    hist = _time_lat_lon_da(
        rng.normal(0.0, 0.2, size=(len(hist_time), 3, 4)), hist_time
    )
    rmap = _time_lat_lon_da(
        np.full((len(proj_time), 3, 4), values_no_sai), proj_time
    )
    rdelta = _time_lat_lon_da(
        np.full((len(proj_time), 3, 4), values_delta), proj_time
    )
    return {
        "regional_map": rmap,
        "regional_delta": rdelta,
        "historical_model": hist,
    }


def test_compute_distribution_general_bins_with_history_normalizes_to_one() -> None:
    inputs = _build_distribution_inputs(values_no_sai=2.5, values_delta=-1.0)
    dist = _compute_distribution(
        "tas",
        regional_map=inputs["regional_map"],
        regional_delta=inputs["regional_delta"],
        historical_model=inputs["historical_model"],
        masked_regional_map=None,
        masked_regional_delta=None,
        masked_historical_model=None,
        decade_start_year=2091,
        decade_end_year=2100,
        spatial_gdf=None,
        spatial_item=None,
    )

    assert set(dist) == {"no_sai", "with_sai", "historical"}
    # Counts are normalized (sum to 1) inside the helper.
    assert sum(dist["no_sai"]["counts"]) == pytest.approx(1.0)
    assert sum(dist["with_sai"]["counts"]) == pytest.approx(1.0)
    assert sum(dist["historical"]["counts"]) == pytest.approx(1.0)
    # Bins are the data-driven 0.1-wide grid; should contain the underlying values.
    assert dist["no_sai"]["bin_edges"][0] <= 2.5 <= dist["no_sai"]["bin_edges"][-1]


def test_compute_distribution_above_below_uses_fixed_bins_and_drops_zeros() -> None:
    inputs = _build_distribution_inputs(values_no_sai=15.0, values_delta=-10.0)
    # Historical inputs for exposure variables are absolute counts; replace the
    # near-zero default fixture with values inside the [1, 50] bin range so the
    # historical-counts normalization doesn't divide by zero.
    hist_time = inputs["historical_model"].time.values
    inputs["historical_model"] = _time_lat_lon_da(
        np.full((len(hist_time), 3, 4), 12.0), hist_time
    )
    dist = _compute_distribution(
        "exposure_above_40",
        regional_map=inputs["regional_map"],
        regional_delta=inputs["regional_delta"],
        historical_model=inputs["historical_model"],
        masked_regional_map=None,
        masked_regional_delta=None,
        masked_historical_model=None,
        decade_start_year=2091,
        decade_end_year=2100,
        spatial_gdf=None,
        spatial_item=None,
    )

    # is_above_below variants use the fixed [1, 5, 10, 20, 50, ∞] bin scheme.
    expected_bins = [1, 5, 10, 20, 50, float("inf")]
    assert dist["no_sai"]["bin_edges"] == expected_bins
    # no_sai cells are uniformly 15 → all fall in the [10, 20) bucket (index 2).
    counts = dist["no_sai"]["counts"]
    assert counts[2] == pytest.approx(1.0)
    assert sum(counts[:2]) == 0.0 and sum(counts[3:]) == 0.0
    # with_sai cells are 15 + (-10) = 5 → they fall in the [5, 10) bucket (index 1).
    assert dist["with_sai"]["counts"][1] == pytest.approx(1.0)


def test_compute_distribution_icefrac_uses_unit_interval_bins_and_filters_to_history_positive() -> None:
    # Build an inputs set where historical_model is a mix of zero and positive
    # cells; the helper should drop the zero-historical cells from all three
    # series before histogramming.
    proj_time = np.arange(2050, 2101)
    hist_time = np.arange(1850, 2015)

    hist_arr = np.zeros((len(hist_time), 2, 2))
    hist_arr[:, 0, 0] = 0.4  # only this cell is historically icy
    rmap_arr = np.full((len(proj_time), 2, 2), 0.6)
    rdelta_arr = np.full((len(proj_time), 2, 2), -0.2)

    historical = _time_lat_lon_da(hist_arr, hist_time)
    rmap = _time_lat_lon_da(rmap_arr, proj_time)
    rdelta = _time_lat_lon_da(rdelta_arr, proj_time)

    dist = _compute_distribution(
        "icefrac",
        regional_map=rmap,
        regional_delta=rdelta,
        historical_model=historical,
        masked_regional_map=None,
        masked_regional_delta=None,
        masked_historical_model=None,
        decade_start_year=2091,
        decade_end_year=2100,
        spatial_gdf=None,
        spatial_item=None,
    )

    # Bins are [0, 0.1, ..., 1.0].
    np.testing.assert_allclose(dist["no_sai"]["bin_edges"],
                               np.arange(0.0, 1.01, 0.1))
    # Only the (0,0) cell survives the historical>0 filter, so all density is
    # in one bin per series.
    counts_no_sai = np.asarray(dist["no_sai"]["counts"])
    counts_with_sai = np.asarray(dist["with_sai"]["counts"])
    counts_hist = np.asarray(dist["historical"]["counts"])
    assert (counts_no_sai > 0).sum() == 1
    assert (counts_with_sai > 0).sum() == 1
    assert (counts_hist > 0).sum() == 1


def test_compute_distribution_regional_path_uses_masked_inputs() -> None:
    # In regional mode, the helper must consume ``masked_*`` inputs rather than
    # the unmasked ``regional_map`` / ``regional_delta``. We assert this by
    # passing different values in each pair and checking which one shows up.
    inputs = _build_distribution_inputs(values_no_sai=2.5, values_delta=-1.0)
    # Add small per-cell variation so the data-driven bins span a real range
    # (uniform values would put every sample on an open bin edge and wreck the
    # counts.sum() normalization).
    rng = np.random.default_rng(42)
    masked_map = inputs["regional_map"] * 0 + 7.0 + rng.normal(
        0, 0.2, size=inputs["regional_map"].shape
    )
    masked_delta = inputs["regional_delta"] * 0 - 3.0 + rng.normal(
        0, 0.2, size=inputs["regional_delta"].shape
    )
    masked_hist = inputs["historical_model"] * 0 + 5.0 + rng.normal(
        0, 0.2, size=inputs["historical_model"].shape
    )

    # Pass a stub gdf+item; helper only checks that *both* are non-None.
    class _StubGDF:
        pass

    dist = _compute_distribution(
        "tas",
        regional_map=inputs["regional_map"],
        regional_delta=inputs["regional_delta"],
        historical_model=inputs["historical_model"],
        masked_regional_map=masked_map,
        masked_regional_delta=masked_delta,
        masked_historical_model=masked_hist,
        decade_start_year=2091,
        decade_end_year=2100,
        spatial_gdf=_StubGDF(),
        spatial_item="anything",
    )

    # If the masked inputs were used, the data-driven bins span around the
    # masked values (no_sai = 7, with_sai = 4, historical = 5), not the
    # unmasked ones (which would have been around 2.5 / 1.5 / ~0). Loose
    # bounds because float-stepped np.arange edges drift slightly.
    edges = dist["no_sai"]["bin_edges"]
    assert min(edges) <= 4.0 + 1e-6
    assert max(edges) >= 7.0 - 0.1


# ============================================================================
# _reproject_to_arctic_stereo (smoke test on synthetic ndarray)
# ============================================================================


def test_reproject_to_arctic_stereo_returns_512_square_stereographic() -> None:
    # 90 lat × 180 lon equirectangular grid covering the full globe.
    h, w = 90, 180
    data = np.full((h, w), 0.5, dtype=np.float32)
    sig = np.full((h, w), np.nan, dtype=np.float32)
    src_transform = rasterio.transform.from_bounds(-180, -90, 180, 90, w, h)

    dst_data, dst_sig, dst_transform, dst_crs = _reproject_to_arctic_stereo(
        data, sig, src_transform
    )

    assert dst_data.shape == (512, 512)
    assert dst_sig.shape == (512, 512)
    assert dst_data.dtype == np.float32
    assert "stere" in dst_crs.to_proj4()
    # Some pixels should have been written (i.e., the projection produced output).
    assert np.any(~np.isnan(dst_data))


# ============================================================================
# create_geotiff_files (smoke test on tmp_path)
# ============================================================================


def _decadal_da() -> xr.DataArray:
    """Small (lat, lon) DataArray suitable as a decadal-mean input."""
    return _lat_lon_da(np.arange(16, dtype=np.float64).reshape(4, 4))


def test_create_geotiff_files_writes_two_band_tifs(tmp_path: Path) -> None:
    rmap = _decadal_da()
    rdelta = _decadal_da() * 0.1
    pvals = _lat_lon_da(np.full((4, 4), 0.50))  # all not-significant

    paths = create_geotiff_files(
        var="tas",
        regional_mean=rmap,
        regional_delta_mean=rdelta,
        regional_no_sai_p_values=pvals,
        regional_sai_p_values=pvals,
        ssp_scenario="SSP1-2.6",
        temp_target=1.5,
        decade_start_year=2091,
        decade_end_year=2100,
        start_year=2050,
        ramp_up=10,
        variable_injection=None,
        output_config={"output_dir": str(tmp_path)},
    )

    # Two outputs (no_sai, with_sai) keyed by var.
    assert set(paths) == {"tas_no_sai", "tas_with_sai"}
    for path in paths.values():
        p = Path(path)
        assert p.exists() and p.is_file()
        # Each GeoTIFF is 2-band (data + significance mask).
        with rasterio.open(p) as src:
            assert src.count == 2

    # Filename hash matches the public _compute_params_hash.
    expected_hash = _compute_params_hash(
        "SSP1-2.6", 1.5, 2091, 2100, 2050, 10, None
    )
    assert all(expected_hash in v for v in paths.values())


def test_create_geotiff_files_caches_existing_outputs(tmp_path: Path) -> None:
    rmap = _decadal_da()
    rdelta = _decadal_da() * 0.0  # zero delta
    common_kwargs = dict(
        var="tas",
        regional_mean=rmap,
        regional_delta_mean=rdelta,
        regional_no_sai_p_values=None,
        regional_sai_p_values=None,
        ssp_scenario="SSP1-2.6",
        temp_target=1.5,
        decade_start_year=2091,
        decade_end_year=2100,
        start_year=2050,
        ramp_up=10,
        variable_injection=None,
        output_config={"output_dir": str(tmp_path)},
    )

    paths_first = create_geotiff_files(**common_kwargs)
    mtimes_first = {k: Path(v).stat().st_mtime_ns for k, v in paths_first.items()}

    # Second call with identical inputs reuses cached files (same mtimes).
    paths_second = create_geotiff_files(**common_kwargs)
    assert paths_first == paths_second
    for k, v in paths_second.items():
        assert Path(v).stat().st_mtime_ns == mtimes_first[k]


# ============================================================================
# generate_sai_netcdf (smoke test on tmp_path)
# ============================================================================


def test_generate_sai_netcdf_writes_expected_variables(tmp_path: Path) -> None:
    proj_time = np.arange(2050, 2101)
    recent_time = np.arange(2015, 2050)
    hist_time = np.arange(1850, 2015)

    rng = np.random.default_rng(2)
    historical = _time_lat_lon_da(
        rng.normal(287.0, 0.5, size=(len(hist_time), 2, 2)), hist_time
    )
    historical_rebase = historical.sel(time=slice(1850, 1900)).mean("time")
    historical_anomaly = historical - historical_rebase
    recent = _time_lat_lon_da(
        rng.normal(0.5, 0.2, size=(len(recent_time), 2, 2)), recent_time
    )
    rmap = _time_lat_lon_da(
        rng.normal(2.5, 0.1, size=(len(proj_time), 2, 2)), proj_time
    )
    rdelta = _time_lat_lon_da(
        rng.normal(-1.0, 0.05, size=(len(proj_time), 2, 2)), proj_time
    )

    ds = generate_sai_netcdf(
        var="tas",
        recent_regional_map=recent,
        regional_map=rmap,
        regional_delta=rdelta,
        historical_model=historical_anomaly,
        historical_rebase=historical_rebase,
        model_internal_variability=0.5,
        ssp_scenario="SSP1-2.6",
        temp_target=1.5,
        decade_start_year=2091,
        decade_end_year=2100,
        start_year=2050,
        ramp_up=10,
        variable_injection=None,
        output_config={"output_dir": str(tmp_path)},
    )

    # Returned dataset has the three core data variables plus the std dev scalar.
    assert {"tas_historical", "tas_nosai", "tas_sai", "model_std_dev"}.issubset(
        set(ds.data_vars)
    )
    # File written under tmp_path/netcdfs/.
    nc_files = list((tmp_path / "netcdfs").glob("tas_*.nc"))
    assert len(nc_files) == 1
    # Filename hash matches the public _compute_params_hash.
    expected_hash = _compute_params_hash(
        "SSP1-2.6", 1.5, 2091, 2100, 2050, 10, None
    )
    assert expected_hash in nc_files[0].name
    # Global metadata propagated.
    assert ds.attrs["scenario"] == "SSP1-2.6"
    assert ds.attrs["sai_start_year"] == 2050
