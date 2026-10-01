"""Unit tests for backend.uncertainty — the combined-band maths, on synthetic series."""
import numpy as np
import pytest
import xarray as xr

from backend.uncertainty import (
    Z,
    band_label,
    combine_series,
    internal_sigma_of_mean,
    model_spread_ddof,
)


def _series(values_by_model: dict[str, list[float]]) -> xr.DataArray:
    models = list(values_by_model)
    data = np.array([values_by_model[m] for m in models], dtype=float)
    return xr.DataArray(
        data,
        dims=("model", "time"),
        coords={"model": models, "time": np.arange(2000, 2000 + data.shape[1])},
    )


def test_ddof_is_population_below_three_models_and_sample_from_three():
    assert model_spread_ddof(1) == 0
    assert model_spread_ddof(2) == 0
    assert model_spread_ddof(3) == 1
    assert model_spread_ddof(7) == 1


def test_internal_sigma_of_mean_shrinks_with_model_count():
    # Two equal sigmas: sqrt(2 s^2) / 2 = s / sqrt(2)
    assert internal_sigma_of_mean([0.5, 0.5]) == pytest.approx(0.5 / np.sqrt(2))
    assert internal_sigma_of_mean([0.3]) == pytest.approx(0.3)
    assert internal_sigma_of_mean([]) == 0.0


def test_model_spread_is_reported_but_not_in_the_band():
    per_model = _series({"A": [1.0, 1.0, 1.0], "B": [3.0, 3.0, 3.0]})
    res = combine_series(per_model, {"A": 0.0, "B": 0.0})

    assert res.n_models == 2 and res.ddof == 0
    np.testing.assert_allclose(res.mean.values, 2.0)
    # ddof=0 std of {1, 3} is 1: half the gap, returned as a diagnostic ...
    np.testing.assert_allclose(res.sigma_model.values, 1.0)
    # ... while the band is internal variability only, here zero
    np.testing.assert_allclose(res.internal.values, 0.0)


def test_band_is_the_internal_variability_of_the_mean():
    per_model = _series({"A": [1.0, 1.0], "B": [3.0, 3.0]})
    res = combine_series(per_model, {"A": 0.5, "B": 0.5})

    sigma_int_mean = np.sqrt(0.25 + 0.25) / 2
    assert res.sigma_internal_mean == pytest.approx(sigma_int_mean)
    np.testing.assert_allclose(res.internal.values, Z * sigma_int_mean)
    # the same band whatever the models' gap
    wide = combine_series(_series({"A": [0.0, 0.0], "B": [10.0, 10.0]}), {"A": 0.5, "B": 0.5})
    np.testing.assert_allclose(wide.internal.values, res.internal.values)


def test_three_models_switch_to_sample_std():
    per_model = _series({"A": [0.0], "B": [1.0], "C": [2.0]})
    res = combine_series(per_model, [0.0, 0.0, 0.0])
    assert res.ddof == 1
    np.testing.assert_allclose(res.sigma_model.values, np.std([0, 1, 2], ddof=1))
    np.testing.assert_allclose(res.internal.values, 0.0)


def test_single_model_band_is_its_own_internal_variability():
    per_model = _series({"A": [5.0, 6.0]})
    res = combine_series(per_model, {"A": 0.25})
    np.testing.assert_allclose(res.sigma_model.values, 0.0)
    np.testing.assert_allclose(res.internal.values, Z * 0.25)


def test_nan_in_one_model_is_skipped_at_that_time():
    per_model = _series({"A": [1.0, np.nan], "B": [3.0, 3.0]})
    res = combine_series(per_model, {"A": 0.0, "B": 0.0})
    np.testing.assert_allclose(res.mean.values, [2.0, 3.0])
    # only one model at the second time -> no spread there
    np.testing.assert_allclose(res.sigma_model.values, [1.0, 0.0])


def test_sequence_sigmas_follow_model_order_and_mapping_must_cover_all_models():
    per_model = _series({"A": [1.0], "B": [3.0]})
    res_seq = combine_series(per_model, [0.1, 0.2])
    res_map = combine_series(per_model, {"B": 0.2, "A": 0.1})
    assert res_seq.sigma_internal_mean == pytest.approx(res_map.sigma_internal_mean)
    with pytest.raises(KeyError):
        combine_series(per_model, {"A": 0.1})
    with pytest.raises(ValueError):
        combine_series(per_model, [0.1])


def test_as_dict_and_label_name_the_model_count():
    per_model = _series({"A": [1.0], "B": [3.0]})
    res = combine_series(per_model, {"A": 0.0, "B": 0.0})
    d = res.as_dict("no_sai")
    assert set(d) == {"band_internal_no_sai", "sigma_model_no_sai"}
    assert "M=2" in band_label(2) and "internal variability" in band_label(2)
