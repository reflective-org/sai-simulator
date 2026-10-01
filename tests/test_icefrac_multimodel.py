"""Unit tests for the multi-model icefrac sigmoid machinery.

Covers the per-model ``MODEL_SCENARIOS`` config in
:mod:`scripts.fit_icefrac_arctic`, the ``model``-dimension handling (including
legacy single-model files) in :func:`backend.icefrac.load_icefrac_params`, and
the mean-of-sigmoids behaviour of :func:`backend.icefrac.predict_icefrac` —
the across-model mean by default, per-model predictions when requested.
"""
from pathlib import Path

import numpy as np
import pytest
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


def test_model_scenarios_cover_all_three_models() -> None:
    assert set(fia.MODEL_SCENARIOS) == {"CESM2-WACCM", "MIROC-ES2H", "UKESM1-1-LL"}
    assert fia.PRIMARY_MODEL in fia.MODEL_SCENARIOS
    for scenarios in fia.MODEL_SCENARIOS.values():
        assert "baseline" in scenarios
        for config in scenarios.values():
            assert set(config) == {"file", "start", "end"}


def test_ukesm_trains_from_the_same_window_as_the_others() -> None:
    """UKESM only joined once ESGF supplied historical sea ice. A baseline window
    starting after 1900 would mean its ice record still lacks a cold regime, which
    is what left its full-ice plateau extrapolated rather than fitted."""
    ukesm = fia.MODEL_SCENARIOS["UKESM1-1-LL"]["baseline"]
    cesm = fia.MODEL_SCENARIOS["CESM2-WACCM"]["baseline"]
    assert (ukesm["start"], ukesm["end"]) == (cesm["start"], cesm["end"])


def test_no_model_is_currently_excluded() -> None:
    """The mechanism stays (it is what the plot script reads to show a rejected
    fit), but nothing should be sitting in it unnoticed."""
    assert fia.EXCLUDED_MODEL_SCENARIOS == {}
    assert not set(fia.EXCLUDED_MODEL_SCENARIOS) & set(fia.MODEL_SCENARIOS)


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


def test_mean_over_time_combined_bands_from_end_to_end_per_model_series() -> None:
    """Two models, no internal variability: the spread is half the model gap (ddof=0), reported
    as sigma_model, and the band (internal variability only) is zero."""
    from backend.get_outputs import _compute_mean_over_time

    hist_time = np.arange(1850, 2015)
    recent_time = np.arange(2015, 2050)
    proj_time = np.arange(2050, 2101)
    full_time = np.arange(2015, 2101)
    area = _constant_field(1.0, np.array([0])).isel(time=0, drop=True)

    # Per-model fractions 0.2 / 0.6 (no-SAI) and 0.1 / 0.3 (with-SAI) over a
    # 12-cell all-ones-area grid -> extent sums 2.4 / 7.2 and 1.2 / 3.6.
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
        sigma_internal_per_model={"CESM2-WACCM": 0.0, "MIROC-ES2H": 0.0},
    )

    bands = res.bands
    assert bands is not None
    assert bands["band_models"] == ["CESM2-WACCM", "MIROC-ES2H"]
    assert bands["band_n_models"] == 2 and bands["band_ddof"] == 0
    for segment in ("historical", "no_sai", "with_sai"):
        for key in ("band_internal", "sigma_model", "band_mean"):
            assert f"{key}_{segment}" in bands
    assert "M=2" in bands["band_labels"]["internal"]

    # No-SAI: per-model extents {2.4, 7.2} -> mean 4.8, sigma_model (ddof=0) 2.4.
    np.testing.assert_allclose(bands["band_mean_no_sai"].values, 4.8)
    np.testing.assert_allclose(bands["sigma_model_no_sai"].values, 2.4)
    np.testing.assert_allclose(bands["band_internal_no_sai"].values, 0.0)  # zero internal σ here
    assert list(bands["band_mean_no_sai"].time.values) == list(res.mean_no_sai.time.values)

    # With-SAI: each model's own sigmoid -> {1.2, 3.6} after the SAI start...
    with_sai = bands["sigma_model_with_sai"]
    np.testing.assert_allclose(with_sai.sel(time=slice(2050, None)).values, 1.2)
    # ...and anchored to the model's own no-SAI series before it (the seam year).
    seam_years = with_sai.time.values[with_sai.time.values < 2050]
    if len(seam_years):
        np.testing.assert_allclose(with_sai.sel(time=seam_years).values, 2.4)

    # The per-model series the bands were built from are exposed for figures.
    assert set(bands["per_model_series"]) == {"historical", "no_sai", "with_sai"}
    assert bands["per_model_series"]["no_sai"].sizes["model"] == 2


def test_mean_over_time_band_is_internal_variability_not_model_spread() -> None:
    from backend.get_outputs import _compute_mean_over_time
    from backend.uncertainty import Z

    proj_time = np.arange(2050, 2101)
    recent_time = np.arange(2015, 2050)
    area = _constant_field(1.0, np.array([0])).isel(time=0, drop=True)
    per_model = _per_model_field((0.2, 0.6), np.arange(2015, 2101))

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
        icefrac_per_model=per_model,
        icefrac_with_sai_per_model=None,
        sigma_internal_per_model={"CESM2-WACCM": 0.6, "MIROC-ES2H": 0.4},
    )
    bands = res.bands
    sigma_int_mean = np.sqrt(0.6**2 + 0.4**2) / 2
    assert bands["sigma_internal_mean"] == pytest.approx(sigma_int_mean)
    np.testing.assert_allclose(
        bands["band_internal_no_sai"].values, Z * sigma_int_mean
    )  # the 2.4 model spread (sigma_model) is not in the band


def test_mean_over_time_bands_absent_without_per_model_sigma_and_zero_spread_for_one_model() -> None:
    from backend.get_outputs import _compute_mean_over_time
    from backend.uncertainty import Z

    proj_time = np.arange(2050, 2101)
    recent_time = np.arange(2015, 2050)
    area = _constant_field(1.0, np.array([0])).isel(time=0, drop=True)
    single_model = _per_model_field((0.4, 0.4), np.arange(2015, 2101)).isel(model=[0])
    common = dict(
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

    # No per-model internal sigma supplied -> no bands at all (the caller decides).
    assert _compute_mean_over_time("icefrac", **common).bands is None

    # One model: no across-model spread, prediction interval is internal variability only.
    res = _compute_mean_over_time(
        "icefrac", **common, sigma_internal_per_model={"CESM2-WACCM": 0.5}
    )
    assert res.bands["band_n_models"] == 1
    np.testing.assert_allclose(res.bands["sigma_model_no_sai"].values, 0.0)
    np.testing.assert_allclose(res.bands["band_internal_no_sai"].values, Z * 0.5)


class TestIceFreeObservationsAreKept:
    """The fit must see ice-free years — they are what pin a decreasing logistic's
    warm asymptote. An earlier filter discarded every value <= 0.01 as noise, which
    censored the training set worst for the models that actually go ice-free."""

    @staticmethod
    def _cell(values):
        """One (lon_band, lat_band, time, cell) column from a list of values."""
        return np.asarray(values, dtype=float).reshape(1, 1, -1, 1)

    def test_zeros_survive_the_filter(self):
        # A real ice cell: icy early, ice-free later. Every year must be kept.
        series = [0.9] * 10 + [0.0] * 10
        out, _, _ = fia.filter_icefrac_data(self._cell(series))
        assert not np.isnan(out).any()
        assert out.ravel().tolist() == series

    def test_low_intensity_cells_are_still_dropped(self):
        # Never exceeds intensity_threshold (0.15) -> the marginal-cell guard fires.
        out, too_few, low = fia.filter_icefrac_data(self._cell([0.05] * 20))
        assert np.isnan(out).all()
        assert int(low.sum()) == 1 and int(too_few.sum()) == 0

    def test_cells_with_too_few_observations_are_still_dropped(self):
        series = [0.9] * 5 + [np.nan] * 15          # only 5 valid, below min_valid_obs
        out, too_few, low = fia.filter_icefrac_data(self._cell(series))
        assert np.isnan(out).all()
        assert int(too_few.sum()) == 1 and int(low.sum()) == 0

    def test_an_all_ice_free_cell_is_dropped_as_low_intensity(self):
        # Keeping zeros must not admit cells that never had ice: max 0 < 0.15.
        out, too_few, low = fia.filter_icefrac_data(self._cell([0.0] * 20))
        assert np.isnan(out).all()
        assert int(low.sum()) == 1


class TestYearPairing:
    """tas and icefrac files need not cover the same years, so the fit pairs them
    explicitly rather than by index position."""

    @staticmethod
    def _write(path, years):
        da = xr.DataArray(
            np.ones((len(years), 2, 2)),
            dims=("time", "lat", "lon"),
            coords={"time": np.asarray(years), "lat": [60.0, 70.0], "lon": [0.0, 180.0]},
            name="icefrac",
        )
        da.to_dataset().to_netcdf(path)
        return path

    def test_window_years_respects_the_window_and_file_order(self, tmp_path):
        path = self._write(tmp_path / "a.nc", range(2010, 2021))
        assert fia._window_years(path, 2012, 2015) == [2012, 2013, 2014, 2015]

    def test_shared_years_exclude_one_missing_from_either_file(self, tmp_path):
        # CESM's real asymmetry: tas has no 2015, icefrac does.
        tas = self._write(tmp_path / "tas.nc", [y for y in range(2013, 2018) if y != 2015])
        ice = self._write(tmp_path / "ice.nc", range(2013, 2018))
        ty = fia._window_years(tas, 2013, 2017)
        iy = fia._window_years(ice, 2013, 2017)
        assert 2015 not in ty and 2015 in iy
        assert sorted(set(ty) & set(iy)) == [2013, 2014, 2016, 2017]

    def test_a_model_carrying_2015_in_both_files_keeps_it(self, tmp_path):
        # UKESM's case. The old unconditional 2015 drop would have removed it from
        # the ice side alone, shifting every later year against tas.
        tas = self._write(tmp_path / "tas.nc", range(2013, 2018))
        ice = self._write(tmp_path / "ice.nc", range(2013, 2018))
        shared = sorted(set(fia._window_years(tas, 2013, 2017))
                        & set(fia._window_years(ice, 2013, 2017)))
        assert shared == [2013, 2014, 2015, 2016, 2017]

    def test_reshape_no_longer_drops_2015(self, tmp_path):
        ds = xr.open_dataset(self._write(tmp_path / "ice.nc", range(2013, 2018)))
        arctic = create_arctic_dataset(ds)
        out = fia.reshape_icefrac_data(arctic, "icefrac", "2013", "2017")
        assert out.shape[-2] == 5
