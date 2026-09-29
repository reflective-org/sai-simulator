"""Unit tests for the multi-model icefrac sigmoid machinery.

Covers the per-model ``MODEL_SCENARIOS`` config in
:mod:`scripts.fit_icefrac_arctic`, the ``model``-dimension handling (including
legacy single-model files) in :func:`backend.icefrac.load_icefrac_params`, and
the mean-of-sigmoids behaviour of :func:`backend.icefrac.predict_icefrac` —
the across-model mean by default, per-model predictions when requested.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import scripts.fit_icefrac_arctic as fia
from backend.icefrac import (
    create_arctic_dataset,
    load_icefrac_params,
    predict_icefrac,
)


# One latitude below the Arctic cutoff plus one in each of three lat bands;
# one longitude in each of the four Arctic lon regions.
LAT = np.array([55.0, 62.0, 72.0, 82.0])
LON = np.array([10.0, 100.0, 200.0, 300.0])
TIME = pd.date_range("2050-01-01", periods=10, freq="YS")


def _make_temp_ds() -> xr.Dataset:
    """Constant global temperature field so sigmoid inputs are finite everywhere."""
    data = np.full((len(TIME), len(LAT), len(LON)), 275.0)
    return xr.Dataset(
        {"tas": (["time", "lat", "lon"], data)},
        coords={"time": TIME, "lat": LAT, "lon": LON},
    )


def _flat_params(temp_ds: xr.Dataset, b_value: float) -> xr.DataArray:
    """Band-sparse ``(features, lon_band, lat_band, lat, lon)`` params for one model.

    ``L = 0`` makes the logistic constant at ``b_value``, so predictions are
    known exactly. Params are NaN outside each cell's own band — the same
    sparse structure the real fit produces.
    """
    arctic_ref = create_arctic_dataset(temp_ds)
    member_mask = arctic_ref["tas"].isel(time=0, drop=True).notnull()

    per_feature = {"L": 0.0, "x0": 0.0, "k": 1.0, "b": b_value}
    features = [
        xr.where(member_mask, value, np.nan) for value in per_feature.values()
    ]
    params = xr.concat(features, dim="features").assign_coords(
        features=list(per_feature)
    )
    # Expand the Arctic-only lat axis back to the global grid (NaN elsewhere).
    return params.reindex(lat=temp_ds.lat, lon=temp_ds.lon)


def _write_interpolator(
    model_dir: Path, params: xr.DataArray
) -> None:
    out_dir = model_dir / "icefrac_arctic"
    out_dir.mkdir(parents=True)
    params.rename("logistic_params").to_netcdf(out_dir / "interpolator.nc")


def _arctic_cells(prediction: xr.DataArray) -> xr.DataArray:
    return prediction.sel(lat=slice(60, 90))


def test_model_scenarios_cover_cesm_and_miroc() -> None:
    assert set(fia.MODEL_SCENARIOS) == {"CESM2-WACCM", "MIROC-ES2H"}
    assert fia.PRIMARY_MODEL in fia.MODEL_SCENARIOS
    for scenarios in fia.MODEL_SCENARIOS.values():
        assert "baseline" in scenarios
        for config in scenarios.values():
            assert set(config) == {"file", "start", "end"}


def test_miroc_scenarios_point_at_miroc_files() -> None:
    miroc = fia.MODEL_SCENARIOS["MIROC-ES2H"]
    assert {"sai_g6_0_5", "sai_g6_1_5"} <= set(miroc)
    for config in miroc.values():
        assert config["file"].endswith("-miroc.nc")


def test_load_icefrac_params_promotes_legacy_single_model_file(tmp_path) -> None:
    temp_ds = _make_temp_ds()
    _write_interpolator(tmp_path, _flat_params(temp_ds, b_value=0.5))

    params = load_icefrac_params(tmp_path)
    assert "model" in params.dims
    assert list(params.model.values) == ["CESM2-WACCM"]


def test_load_icefrac_params_keeps_existing_model_dim(tmp_path) -> None:
    temp_ds = _make_temp_ds()
    stacked = xr.concat(
        [_flat_params(temp_ds, 0.2), _flat_params(temp_ds, 0.6)], dim="model"
    ).assign_coords(model=["CESM2-WACCM", "MIROC-ES2H"])
    _write_interpolator(tmp_path, stacked)

    params = load_icefrac_params(tmp_path)
    assert list(params.model.values) == ["CESM2-WACCM", "MIROC-ES2H"]


def test_predict_icefrac_averages_the_sigmoids(tmp_path) -> None:
    temp_ds = _make_temp_ds()
    stacked = xr.concat(
        [_flat_params(temp_ds, 0.2), _flat_params(temp_ds, 0.6)], dim="model"
    ).assign_coords(model=["CESM2-WACCM", "MIROC-ES2H"])
    _write_interpolator(tmp_path, stacked)

    prediction = predict_icefrac(temp_ds["tas"], tmp_path)

    assert "model" not in prediction.dims
    assert prediction.dims == ("time", "lat", "lon")
    # Mean of the two flat sigmoids (0.2 and 0.6) in the Arctic, 0 elsewhere.
    np.testing.assert_allclose(_arctic_cells(prediction).values, 0.4)
    np.testing.assert_allclose(prediction.sel(lat=55.0).values, 0.0)


def test_predict_icefrac_per_model_keeps_model_dim(tmp_path) -> None:
    temp_ds = _make_temp_ds()
    stacked = xr.concat(
        [_flat_params(temp_ds, 0.2), _flat_params(temp_ds, 0.6)], dim="model"
    ).assign_coords(model=["CESM2-WACCM", "MIROC-ES2H"])
    _write_interpolator(tmp_path, stacked)

    per_model = predict_icefrac(temp_ds["tas"], tmp_path, per_model=True)

    assert per_model.dims == ("model", "time", "lat", "lon")
    np.testing.assert_allclose(
        _arctic_cells(per_model.sel(model="CESM2-WACCM")).values, 0.2
    )
    np.testing.assert_allclose(
        _arctic_cells(per_model.sel(model="MIROC-ES2H")).values, 0.6
    )
    # The default prediction is exactly the across-model mean.
    mean = predict_icefrac(temp_ds["tas"], tmp_path)
    xr.testing.assert_allclose(mean, per_model.mean(dim="model"))


def test_predict_icefrac_legacy_file_matches_its_single_model(tmp_path) -> None:
    temp_ds = _make_temp_ds()
    _write_interpolator(tmp_path, _flat_params(temp_ds, b_value=0.5))

    prediction = predict_icefrac(temp_ds["tas"], tmp_path)
    np.testing.assert_allclose(_arctic_cells(prediction).values, 0.5)


# ============================================================================
# Across-model uncertainty band in _compute_mean_over_time
# ============================================================================


def _constant_field(value: float, time: np.ndarray) -> xr.DataArray:
    return xr.DataArray(
        np.full((len(time), 3, 4), value),
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": [0.0, 10.0, 20.0], "lon": [0.0, 90.0, 180.0, 270.0]},
    )


def _per_model_field(values: tuple[float, float], time: np.ndarray) -> xr.DataArray:
    return xr.concat(
        [_constant_field(v, time) for v in values], dim="model"
    ).assign_coords(model=["CESM2-WACCM", "MIROC-ES2H"])


def test_mean_over_time_computes_2sigma_band_across_models() -> None:
    from backend.get_outputs import _compute_mean_over_time

    hist_time = np.arange(1850, 2015)
    recent_time = np.arange(2015, 2050)
    proj_time = np.arange(2050, 2101)
    full_time = np.arange(2015, 2101)
    area = _constant_field(1.0, np.array([0])).isel(time=0, drop=True)

    # Per-model fractions 0.2 / 0.6 (no-SAI) and 0.1 / 0.3 (with-SAI) over a
    # 12-cell all-ones-area grid → extent sums 2.4 / 7.2 and 1.2 / 3.6.
    per_model = _per_model_field((0.2, 0.6), full_time)
    per_model_with_sai = _per_model_field((0.1, 0.3), proj_time)

    res = _compute_mean_over_time(
        "icefrac",
        regional_map=_constant_field(0.4, proj_time),
        regional_delta=_constant_field(-0.2, proj_time),
        recent_regional_map=_constant_field(0.4, recent_time),
        historical_model=_constant_field(0.5, hist_time),
        area=area,
        spatial_gdf=None,
        spatial_item=None,
        start_year=2050,
        variable_injection=None,
        data_dir=Path("/unused"),
        icefrac_per_model=per_model,
        icefrac_with_sai_per_model=per_model_with_sai,
    )

    band_no_sai = res.model_spread_2sigma_no_sai
    band_with_sai = res.model_spread_2sigma_with_sai
    assert band_no_sai is not None and band_with_sai is not None
    assert list(band_no_sai.time.values) == list(res.mean_no_sai.time.values)
    assert list(band_with_sai.time.values) == list(res.mean_with_sai.time.values)

    # 2σ (ddof=1) of two constant extent series {2.4, 7.2}: 2 * 4.8/√2.
    np.testing.assert_allclose(band_no_sai.values, 2 * 4.8 / np.sqrt(2))
    # With-SAI band from {1.2, 3.6} after the SAI start...
    np.testing.assert_allclose(
        band_with_sai.sel(time=slice(2050, None)).values, 2 * 2.4 / np.sqrt(2)
    )
    # ...and anchored to the no-SAI band before it (the seam year).
    seam_years = band_with_sai.time.values[band_with_sai.time.values < 2050]
    if len(seam_years):
        np.testing.assert_allclose(
            band_with_sai.sel(time=seam_years).values, 2 * 4.8 / np.sqrt(2)
        )


def test_mean_over_time_band_absent_for_single_model() -> None:
    from backend.get_outputs import _compute_mean_over_time

    proj_time = np.arange(2050, 2101)
    recent_time = np.arange(2015, 2050)
    area = _constant_field(1.0, np.array([0])).isel(time=0, drop=True)
    single_model = _per_model_field((0.4, 0.4), np.arange(2015, 2101)).isel(
        model=[0]
    )

    res = _compute_mean_over_time(
        "icefrac",
        regional_map=_constant_field(0.4, proj_time),
        regional_delta=_constant_field(-0.2, proj_time),
        recent_regional_map=_constant_field(0.4, recent_time),
        historical_model=None,
        area=area,
        spatial_gdf=None,
        spatial_item=None,
        start_year=2050,
        variable_injection=None,
        data_dir=Path("/unused"),
        icefrac_per_model=single_model,
        icefrac_with_sai_per_model=None,
    )

    # A one-model ensemble has no across-model spread to report.
    assert res.model_spread_2sigma_no_sai is None
    assert res.model_spread_2sigma_with_sai is None
