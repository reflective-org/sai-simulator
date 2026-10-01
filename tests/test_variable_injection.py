"""Unit tests for the custom-injection (CIDER) delta in ``backend.variable``.

``get_variable_regional_delta`` reads the CESM parameter file and ``correct_lat.npy`` from
``data/processed``. Both are tracked in git, so these tests run on the real parameters.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import scipy.io

from backend.variable import (
    CESM_PARAMS_FILE,
    CESM_PARAMS_SHA1,
    get_variable_regional_delta,
    pattern_from_all_injections_and_CO2,
)

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data" / "processed"
N_LATS, N_YEARS = 7, 66


def _injection(tg_per_year: float) -> np.ndarray:
    """``tg_per_year`` at every injection latitude from 2040 on, zero before."""
    injection = np.zeros((N_LATS, N_YEARS))
    injection[:, 5:] = tg_per_year
    return injection


def _global_mean(da):
    return float(da.weighted(np.cos(np.deg2rad(da.lat))).mean())


def test_params_checksum_matches_file():
    # CESM_PARAMS_SHA1 keys the saved custom-injection outputs; it must follow the file.
    digest = hashlib.sha1((DATA_DIR / CESM_PARAMS_FILE).read_bytes()).hexdigest()
    assert digest == CESM_PARAMS_SHA1


def test_zero_injection_gives_zero_delta():
    delta = get_variable_regional_delta("tas", DATA_DIR, DATA_DIR, _injection(0.0))
    assert float(abs(delta).max()) == 0.0


def test_output_grid_and_years():
    delta = get_variable_regional_delta("tas", DATA_DIR, DATA_DIR, _injection(1.0))
    assert delta.dims == ("time", "lat", "lon")
    assert delta.time.values.tolist() == list(range(2035, 2101))
    np.testing.assert_array_equal(delta.lat.values, np.load(DATA_DIR / "correct_lat.npy"))
    assert delta.sizes["lon"] == 288


@pytest.mark.parametrize("var", ["tas", "tasmin", "tasmax"])
def test_positive_injection_cools(var):
    delta = get_variable_regional_delta(var, DATA_DIR, DATA_DIR, _injection(3.0))
    assert _global_mean(delta.sel(time=slice(2080, 2100))) < 0


@pytest.mark.parametrize("var, key", [("tasmin", "TMN"), ("tasmax", "TMX")])
def test_tasmin_tasmax_use_tmn_tmx_entries(var, key):
    """The delta equals the convolution chain run directly on the TMN / TMX entries."""
    injection = _injection(2.0)
    delta = get_variable_regional_delta(var, DATA_DIR, DATA_DIR, injection)

    mat = scipy.io.loadmat(DATA_DIR / CESM_PARAMS_FILE)
    monthly = np.repeat(injection, 12, axis=1) / 12
    forcing = np.concatenate([monthly.T, np.zeros((monthly.shape[1], 1))], axis=1)
    expected = pattern_from_all_injections_and_CO2(
        forcing, mat["param_AOD_all"], mat[f"param_{key}_all"], mat[f"pattern_{key}_all"]
    )
    expected = expected.reshape(expected.shape[0], expected.shape[1], -1, 12).mean(-1)
    np.testing.assert_allclose(delta.values, expected.transpose(2, 1, 0))


def test_tasmin_and_tasmax_differ_from_tas():
    injection = _injection(2.0)
    tas = get_variable_regional_delta("tas", DATA_DIR, DATA_DIR, injection)
    for var in ["tasmin", "tasmax"]:
        other = get_variable_regional_delta(var, DATA_DIR, DATA_DIR, injection)
        assert not np.allclose(tas.values, other.values)


@pytest.mark.parametrize("tg_per_year", [
    3.0,
    pytest.param(10.0, marks=pytest.mark.xfail(strict=True, reason=(
        "Known limit: tasmin, tas and tasmax are emulated independently, and from about 5 Tg/yr "
        "per latitude their responses cross in ocean cells where TMN and TMX are close together."
    ))),
])
def test_tasmin_tas_tasmax_stay_ordered_with_sai(tg_per_year):
    """With SAI, tasmin <= tas <= tasmax in every cell and year.

    The deltas themselves may cross (tasmax can cool more than tas), but the temperatures they
    produce on the CESM baselines in the parameter file must not. 10 Tg/yr per latitude is the
    dashboard's slider maximum; if a new parameter file keeps the order there, the strict xfail
    fails and the case should become a normal one.
    """
    injection = _injection(tg_per_year)
    mat = scipy.io.loadmat(DATA_DIR / CESM_PARAMS_FILE)
    with_sai = {}
    for var, key in [("tasmin", "TMN"), ("tas", "T"), ("tasmax", "TMX")]:
        delta = get_variable_regional_delta(var, DATA_DIR, DATA_DIR, injection)
        base = np.asarray(mat[f"pattern_{key}_base"], dtype=float).T  # (lon, lat) -> (lat, lon)
        with_sai[var] = base + delta.values
    assert (with_sai["tasmin"] <= with_sai["tas"]).all()
    assert (with_sai["tas"] <= with_sai["tasmax"]).all()


def test_unsupported_variable_raises():
    with pytest.raises(ValueError, match="not supported"):
        get_variable_regional_delta("icefrac", DATA_DIR, DATA_DIR, _injection(1.0))
