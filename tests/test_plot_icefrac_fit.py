"""Unit tests for the aggregation helpers in ``scripts/plot_icefrac_fit.py``.

The figure itself is not asserted on; what matters is that the numbers behind it
follow the same conventions as the predictor it illustrates — a cell whose fit
failed contributes zero ice rather than dropping out of the average, and every
aggregate is weighted by true cell area.
"""
import numpy as np
import pytest
import xarray as xr

import scripts.plot_icefrac_fit as pif


def _params(L, x0, k, b, shape=(1, 1, -1)):
    """Build a ``(features, lon_band, lat_band, n_cells)`` parameter block.

    Scalars broadcast, so a test can vary one parameter and hold the rest fixed.
    """
    blocks = []
    for v in (L, x0, k, b):
        arr = np.array(v, dtype=float)
        if arr.ndim == 0:
            arr = np.full(np.shape(np.array(L, dtype=float)) or (1,), float(arr))
        blocks.append(arr.reshape(shape))
    return np.array(blocks)


class TestFailedFitsCountAsIceFree:
    def test_unfitted_cell_predicts_zero_not_nan(self):
        # Cell 0 fitted (L=1, x0=0, k=1, b=0 → 0.5 at x=0); cell 1 never converged.
        params = _params([1.0, np.nan], [0.0, np.nan], [1.0, np.nan], [0.0, np.nan])
        cells = np.array([True, True])
        out = pif.evaluate_cells(params, [0.0], 0, 0, cells)
        assert out.shape == (1, 2)
        assert out[0, 0] == 0.5
        assert out[0, 1] == 0.0

    def test_fitted_mask_tracks_finite_amplitude(self):
        params = _params([1.0, np.nan], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0])
        assert pif.fitted_mask(params).tolist() == [[[True, False]]]

    def test_region_response_is_area_weighted(self):
        params = _params([1.0, np.nan], [0.0, np.nan], [1.0, np.nan], [0.0, np.nan])
        cells = np.array([True, True])
        area = np.array([1.0, 3.0])
        # (0.5 * 1 + 0.0 * 3) / 4
        assert pif.region_response(params, [0.0], 0, 0, cells, area)[0] == 0.125


class TestAreaAggregation:
    def test_ice_area_sums_fraction_times_area_and_skips_nan(self):
        member = np.array([[[True, True, False]]])
        area = np.array([2.0, 5.0, 100.0])
        values = np.array([[[[0.5, np.nan, 1.0], [1.0, 0.2, 1.0]]]])
        # t0: 0.5*2 + (NaN→0)*5 = 1.0 ; t1: 1.0*2 + 0.2*5 = 3.0
        # The out-of-region cell never contributes despite its huge area.
        assert pif.ice_area_from_data(values, member, area).tolist() == [1.0, 3.0]

    def test_arctic_mean_anomaly_weights_regions_by_area(self):
        # Region (0,0) holds 1 unit of area, region (0,1) holds 3.
        member = np.array([[[True, False, False, False], [False, True, True, True]]])
        area = np.ones(4)
        x = np.array([[[0.0], [4.0]]])
        assert pif.arctic_mean_anomaly(x, member, area)[0] == 3.0

    def test_uniform_response_scales_with_region_area(self):
        params = _params([0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0])  # constant 1.0
        member = np.array([[[True, True]]])
        area = np.array([1.5, 2.5])
        assert pif.uniform_response_area(params, [0.0, 5.0], member, area).tolist() == [4.0, 4.0]


class TestCommonFittedCells:
    def test_intersects_models_and_respects_membership(self):
        member = np.array([[[True, True, True]]])
        models = {
            "a": {"params": _params([1.0, 1.0, np.nan], [0, 0, 0], [1, 1, 1], [0, 0, 0])},
            "b": {"params": _params([1.0, np.nan, np.nan], [0, 0, 0], [1, 1, 1], [0, 0, 0])},
        }
        assert pif.common_fitted_cells(models, member).tolist() == [[[True, False, False]]]

    def test_falls_back_to_membership_when_intersection_is_empty(self):
        member = np.array([[[True, True]]])
        models = {
            "a": {"params": _params([1.0, np.nan], [0, 0], [1, 1], [0, 0])},
            "b": {"params": _params([np.nan, 1.0], [0, 0], [1, 1], [0, 0])},
        }
        assert pif.common_fitted_cells(models, member).tolist() == member.tolist()

    def test_no_params_at_all_falls_back_to_membership(self):
        member = np.array([[[True, False]]])
        assert pif.common_fitted_cells({"a": {"params": None}}, member).tolist() == member.tolist()


class TestParameterOrderIsNotAssumed:
    """A permuted ``features`` axis must not silently reinterpret the parameters.

    Every aggregator unpacks a parameter block positionally as (L, x0, k, b), so
    the loader is the single place that has to get the order right. These tests
    evaluate away from ``x == x0``: at ``x == 0`` with ``x0 == 0`` the logistic is
    invariant under swapping ``x0`` and ``k``, so a degenerate case cannot tell a
    correct unpack from a transposed one.
    """

    def test_evaluation_is_sensitive_to_x0_versus_k(self):
        # b + L*expit(-(x - x0)*k) at x=1, x0=2, k=3 -> expit(3).
        params = _params([1.0], [2.0], [3.0], [0.0])
        got = pif.evaluate_cells(params, [1.0], 0, 0, np.array([True]))[0, 0]
        assert got == pytest.approx(0.9525741268, abs=1e-9)
        # The same numbers read in the wrong order give expit(4) - a value this
        # assertion would reject.
        assert got != pytest.approx(0.9820137900, abs=1e-9)

    def test_loader_reorders_a_permuted_features_axis(self, tmp_path):
        lat = np.array([60.0, 70.0])
        lon = np.array([0.0, 180.0])
        stored_order = ["k", "b", "L", "x0"]          # deliberately not canonical
        values = {"L": 1.0, "x0": 2.0, "k": 3.0, "b": 0.25}
        block = np.stack([
            np.full((1, 1, len(lat), len(lon)), values[f]) for f in stored_order
        ])[None]                                       # (model, features, lon_band, lat_band, ...)
        da = xr.DataArray(
            block,
            dims=["model", "features", "lon_band", "lat_band", "lat", "lon"],
            coords={"model": ["CESM2-WACCM"], "features": stored_order,
                    "lon_band": ["Pacific"], "lat_band": ["60-70"], "lat": lat, "lon": lon},
            name="logistic_params",
        )
        out_dir = tmp_path / f"{pif.VAR_ICE}_arctic"
        out_dir.mkdir()
        da.to_netcdf(out_dir / "interpolator.nc")

        params = pif.load_fitted_params(tmp_path, {"lat": lat, "lon": lon})["CESM2-WACCM"]
        assert [float(p.ravel()[0]) for p in params] == [1.0, 2.0, 3.0, 0.25]
        # And the values survive as a usable sigmoid: b + L*expit(-(1-2)*3).
        got = pif.evaluate_cells(params, [1.0], 0, 0, np.ones(params.shape[-1], dtype=bool))
        assert got[0, 0] == pytest.approx(0.25 + 0.9525741268, abs=1e-9)

    def test_loader_rejects_a_foreign_file(self, tmp_path):
        out_dir = tmp_path / f"{pif.VAR_ICE}_arctic"
        out_dir.mkdir()
        xr.DataArray([1.0], dims="x", name="something_else").to_netcdf(
            out_dir / "interpolator.nc")
        with pytest.raises(ValueError, match="logistic_params"):
            pif.load_fitted_params(tmp_path, {"lat": np.array([60.0]), "lon": np.array([0.0])})


class TestCommonFittedCellsFallsBackPerRegion:
    def test_an_empty_region_keeps_its_own_membership(self):
        # Region (0,0) has no commonly fitted cell; region (0,1) has one.
        member = np.array([[[True, True], [True, True]]])
        models = {
            "a": {"params": _params([[1.0, np.nan], [1.0, 1.0]], 0.0, 1.0, 0.0,
                                    shape=(1, 2, 2))},
            "b": {"params": _params([[np.nan, 1.0], [1.0, np.nan]], 0.0, 1.0, 0.0,
                                    shape=(1, 2, 2))},
        }
        out = pif.common_fitted_cells(models, member)
        assert out[0, 0].tolist() == [True, True]     # fell back to membership
        assert out[0, 1].tolist() == [True, False]    # kept the real intersection
