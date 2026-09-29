"""Unit tests for the own-coordinates warming-level fit and equal-weight averaging.

Covers :mod:`scripts.fit_map`'s regression of each model's fields on **its own**
global-mean tas anomaly (slope = pattern per K of own warming; slope and
intercept jointly consistent) and the equal-weight ``model``-dim mean in
:func:`backend.delta.get_regional_delta_without_rampup`.
"""
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from joblib import load

import scripts.fit_map as fm
from backend.delta import get_regional_delta_without_rampup

LAT = np.array([0.0, 10.0])
LON = np.array([0.0, 90.0])
TIME = np.arange(1850, 2100)  # merged historical + projection, like real files


def _ramp(time: np.ndarray) -> np.ndarray:
    """The model's own global warming trajectory (K above its 1850 state)."""
    return 0.01 * (time - 1850)


PRE_MEAN = float(_ramp(np.arange(1850, 1901)).mean())  # own 1850-1900 reference


def _write_baseline(
    var_dir: Path, filename: str, model: str, response: float, base: float, var: str = "tas"
) -> None:
    """Field = base + response x own-warming-ramp, spatially uniform, noiseless."""
    t = _ramp(TIME)
    field = base + response * t[:, None, None] * np.ones((len(TIME), len(LAT), len(LON)))
    xr.Dataset(
        {var: (["model", "ssp", "time", "lat", "lon"], field[None, None])},
        coords={"model": [model], "ssp": ["ssp245"], "time": TIME, "lat": LAT, "lon": LON},
    ).to_netcdf(var_dir / filename)


def test_own_global_tas_anomaly_rebases_to_preindustrial(tmp_path) -> None:
    tas_dir = tmp_path / "tas"
    tas_dir.mkdir()
    _write_baseline(tas_dir, "output_gauss-baseline.nc", "CESM2-WACCM", 1.0, 287.0)

    anom = fm._own_global_tas_anomaly(
        tas_dir / "output_gauss-baseline.nc", "CESM2-WACCM", "ssp245"
    )
    # Anomaly = own ramp minus its own 1850-1900 mean; zero at the reference.
    np.testing.assert_allclose(
        anom.sel(time=slice(1850, 1900)).mean().item(), 0.0, atol=1e-9
    )
    assert anom.sel(time=2099).item() == pytest.approx(_ramp(np.array([2099]))[0] - PRE_MEAN)


def test_own_global_tas_anomaly_requires_preindustrial_window(tmp_path) -> None:
    tas_dir = tmp_path / "tas"
    tas_dir.mkdir()
    t = np.arange(2015, 2100)
    field = 287.0 + 0.01 * (t - 1850)[:, None, None] * np.ones((len(t), len(LAT), len(LON)))
    xr.Dataset(
        {"tas": (["model", "ssp", "time", "lat", "lon"], field[None, None])},
        coords={"model": ["CESM2-WACCM"], "ssp": ["ssp245"], "time": t, "lat": LAT, "lon": LON},
    ).to_netcdf(tas_dir / "output_gauss-baseline.nc")

    with pytest.raises(ValueError, match="pre-industrial window"):
        fm._own_global_tas_anomaly(
            tas_dir / "output_gauss-baseline.nc", "CESM2-WACCM", "ssp245"
        )


def test_own_global_tas_anomaly_rejects_flat_series(tmp_path) -> None:
    tas_dir = tmp_path / "tas"
    tas_dir.mkdir()
    _write_baseline(tas_dir, "output_gauss-baseline.nc", "CESM2-WACCM", 0.0, 287.0)

    with pytest.raises(ValueError, match="Implausibly flat"):
        fm._own_global_tas_anomaly(
            tas_dir / "output_gauss-baseline.nc", "CESM2-WACCM", "ssp245"
        )


def test_fit_map_tas_slope_is_unity_in_own_coordinates(tmp_path) -> None:
    """Regressing tas on its own global mean gives slope exactly 1 and an
    intercept equal to the field's value at own-anomaly zero."""
    tas_dir = tmp_path / "tas"
    tas_dir.mkdir()
    _write_baseline(tas_dir, "output_gauss-baseline.nc", "CESM2-WACCM", 1.0, 280.0)

    out_dir = tmp_path / "models"
    fm.fit_map("tas", tmp_path, out_dir, num_bootstrap_replicates=2, ignore_existing=True)

    reg = load(out_dir / "tas" / "fair_to_smip_CESM2-WACCM_0.joblib")
    np.testing.assert_allclose(reg.coef_.ravel(), 1.0, atol=1e-9)
    # Line passes through the training centroid; at own-anomaly 0 the field is
    # base + 1.0 x (own 1850-1900 ramp mean).
    np.testing.assert_allclose(reg.intercept_, 280.0 + PRE_MEAN, atol=1e-6)


def test_fit_map_other_vars_regress_on_same_run_tas(tmp_path) -> None:
    """A non-tas variable's slope is its response per K of the model's own
    warming, with X taken from the same-run tas file."""
    tas_dir = tmp_path / "tas"
    tas_dir.mkdir()
    pr_dir = tmp_path / "pr"
    pr_dir.mkdir()
    _write_baseline(tas_dir, "output_gauss-baseline.nc", "CESM2-WACCM", 1.0, 280.0)
    _write_baseline(pr_dir, "output_gauss-baseline.nc", "CESM2-WACCM", 3.0, 5.0, var="pr")

    out_dir = tmp_path / "models"
    fm.fit_map("pr", tmp_path, out_dir, num_bootstrap_replicates=2, ignore_existing=True)

    reg = load(out_dir / "pr" / "fair_to_smip_CESM2-WACCM_0.joblib")
    np.testing.assert_allclose(reg.coef_.ravel(), 3.0, atol=1e-9)
    np.testing.assert_allclose(reg.intercept_, 5.0 + 3.0 * PRE_MEAN, atol=1e-6)


def _write_interpolator(model_dir: Path, var: str, slopes: dict[str, float] | float) -> None:
    var_dir = model_dir / var
    var_dir.mkdir(parents=True)
    if isinstance(slopes, dict):
        data = np.array([[np.full((len(LAT), len(LON)), s)] for s in slopes.values()])
        da = xr.DataArray(
            data,
            dims=["model", "features", "lat", "lon"],
            coords={"model": list(slopes), "features": ["slope"], "lat": LAT, "lon": LON},
        )
    else:
        da = xr.DataArray(
            np.full((1, len(LAT), len(LON)), slopes),
            dims=["features", "lat", "lon"],
            coords={"features": ["slope"], "lat": LAT, "lon": LON},
        )
    da.to_dataset(name=var).to_netcdf(var_dir / "interpolator.nc")


def _temp_diff() -> xr.DataArray:
    return xr.DataArray(
        np.linspace(-1.0, -2.0, 5), dims=["time"], coords={"time": np.arange(2050, 2055)}
    )


def test_delta_averages_models_equally(tmp_path) -> None:
    model_dir = tmp_path / "models"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _write_interpolator(model_dir, "tas", {"CESM2-WACCM": 1.0, "MIROC-ES2H": 3.0})

    delta = get_regional_delta_without_rampup(
        "tas", tmp_path, model_dir, cache_dir, "SSP2-4.5", 1.5, _temp_diff()
    )
    # Equal-weight mean slope = 2.0; no residual model dim.
    assert "model" not in delta.dims
    np.testing.assert_allclose(
        delta.isel(lat=0, lon=0).values, 2.0 * _temp_diff().values
    )


def test_delta_handles_legacy_single_slope_interpolator(tmp_path) -> None:
    model_dir = tmp_path / "models"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _write_interpolator(model_dir, "tas", 1.5)

    delta = get_regional_delta_without_rampup(
        "tas", tmp_path, model_dir, cache_dir, "SSP2-4.5", 1.5, _temp_diff()
    )
    np.testing.assert_allclose(
        delta.isel(lat=0, lon=0).values, 1.5 * _temp_diff().values
    )
