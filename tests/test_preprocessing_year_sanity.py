"""Unit tests for :mod:`scripts.preprocessing_sanity`.

Exercises the per-experiment year-count tables, the NaN / non-negative / variable-specific
range checks, the monthly-cell-bound builder + validator (including the SAI-vs-historical
asymmetry where SAI runs may legitimately escape the historical-baseline envelope), and the
strict / max-gap year-continuity validators.
"""
import pytest
import xarray as xr
import numpy as np

from scripts.preprocessing_sanity import (
    SanityCheckError,
    build_monthly_cell_bounds,
    expected_year_count_for_experiment,
    validate_monthly_cell_bounds,
    validate_no_nans,
    validate_no_negative_values,
    validate_year_continuity_strict,
    validate_year_continuity_with_max_gap,
    validate_variable_specific_ranges,
    validate_year_count_for_experiment,
)


def test_expected_year_count_for_experiment_values() -> None:
    assert expected_year_count_for_experiment("historical") == (164, 165)
    # Upper bound 86: UKESM's het-chem SSP2-4.5 baseline spans 2015-2100 and its
    # 360-day calendar never drops an edge year.
    assert expected_year_count_for_experiment("baseline") == (50, 86)
    # Lower bound covers p-e, whose ssp245 QFLX fill spans only 2020-2084 (65 yr).
    assert expected_year_count_for_experiment("ssp245") == (63, 86)
    assert expected_year_count_for_experiment("1.5") == (33, 36)
    assert expected_year_count_for_experiment("1.0") == (33, 36)
    assert expected_year_count_for_experiment("0.5") == (33, 36)


def test_expected_year_count_for_unknown_experiment() -> None:
    assert expected_year_count_for_experiment("unknown-exp") is None


@pytest.mark.parametrize(
    ("exp", "count"),
    [
        ("historical", 164),
        ("historical", 165),
        ("baseline", 50),
        ("baseline", 85),
        ("baseline", 84),
        ("1.5", 33),
        ("1.5", 35),
        ("1.0", 34),
        ("0.5", 36),
    ],
)
def test_validate_year_count_for_experiment_passes(exp: str, count: int) -> None:
    validate_year_count_for_experiment(exp=exp, year_count=count, var_name="tas", member="001")


def test_validate_year_count_for_experiment_fails_for_incorrect_count() -> None:
    with pytest.raises(SanityCheckError, match="Unexpected yearly count"):
        validate_year_count_for_experiment(
            exp="baseline", year_count=49, var_name="tas", member="001"
        )


def test_validate_year_count_for_experiment_fails_sai_out_of_range() -> None:
    with pytest.raises(ValueError, match="expected between 33 and 36"):
        validate_year_count_for_experiment(
            exp="1.5", year_count=32, var_name="tas", member="001"
        )


def test_validate_no_nans_passes_for_clean_dataarray() -> None:
    data = xr.DataArray([1.0, 2.0, 3.0], dims=["time"], name="tas")
    validate_no_nans(data=data, var_name="tas", exp="baseline", member="001", stage="test")


def test_validate_no_nans_fails_for_dataarray_with_nan() -> None:
    data = xr.DataArray([1.0, float("nan"), 3.0], dims=["time"], name="tas")
    with pytest.raises(ValueError, match="NaN sanity check failed"):
        validate_no_nans(data=data, var_name="tas", exp="baseline", member="001", stage="test")


def test_validate_no_nans_fails_for_dataset_with_nan() -> None:
    ds = xr.Dataset(
        {
            "tas": xr.DataArray([1.0, 2.0], dims=["time"]),
            "pr": xr.DataArray([0.1, float("nan")], dims=["time"]),
        }
    )
    with pytest.raises(ValueError, match="NaN sanity check failed"):
        validate_no_nans(data=ds, var_name="tas", exp="historical", member="002", stage="test")


def test_validate_no_negative_values_passes_for_clean_required_var() -> None:
    data = xr.DataArray([0.0, 1.0, 2.0], dims=["time"], name="tas")
    validate_no_negative_values(
        data=data, var_name="tas", exp="baseline", member="001", stage="test"
    )


def test_validate_no_negative_values_fails_for_required_var_with_negative() -> None:
    data = xr.DataArray([0.0, -1.0, 2.0], dims=["time"], name="pr")
    with pytest.raises(ValueError, match="Negative-value sanity check failed"):
        validate_no_negative_values(
            data=data, var_name="pr", exp="baseline", member="001", stage="test"
        )


def test_validate_no_negative_values_ignores_non_required_var() -> None:
    data = xr.DataArray([-10.0, -5.0], dims=["time"], name="aod")
    validate_no_negative_values(
        data=data, var_name="aod", exp="baseline", member="001", stage="test"
    )


def _make_monthly_da(value: float, name: str = "tas") -> xr.DataArray:
    """Build a tiny ``(time=2, lat=1, lon=1)`` DataArray filled with ``value`` for bounds-check fixtures."""
    return xr.DataArray(
        np.full((2, 1, 1), value, dtype=float),
        dims=["time", "lat", "lon"],
        coords={"time": [0, 1], "lat": [10.0], "lon": [20.0]},
        name=name,
    )


def test_build_monthly_cell_bounds_for_temperature() -> None:
    exp2member2data = {
        "historical": {"001": [_make_monthly_da(270.0, name="tas")]},
        "baseline": {"001": [_make_monthly_da(300.0, name="tas")]},
    }
    bounds = build_monthly_cell_bounds("tas", exp2member2data)
    assert bounds is not None
    assert float(bounds["min"].item()) == 270.0 - 30.0
    assert float(bounds["max"].item()) == 310.0


def test_build_monthly_cell_bounds_spans_every_no_sai_run() -> None:
    """A model with two no-SAI runs bounds against the hotter of the two."""
    exp2member2data = {
        "historical": {"r01": [_make_monthly_da(270.0, name="tas")]},
        "baseline": {"r01": [_make_monthly_da(300.0, name="tas")]},
        "ssp245": {"r01": [_make_monthly_da(305.0, name="tas")]},
    }
    bounds = build_monthly_cell_bounds(
        "tas", exp2member2data, no_sai_experiments=("baseline", "ssp245")
    )
    assert bounds is not None
    assert float(bounds["max"].item()) == 315.0  # from ssp245, the hotter run


def test_validate_monthly_cell_bounds_checks_upper_for_every_no_sai_run() -> None:
    """The upper bound applies to each declared no-SAI run, not just 'baseline'."""
    bounds = {"min": _make_monthly_da(10.0), "max": _make_monthly_da(20.0)}
    data = _make_monthly_da(25.0)
    with pytest.raises(SanityCheckError, match="Monthly grid-cell bound check failed"):
        validate_monthly_cell_bounds(
            data=data, var_name="tas", exp="ssp245", member="r01", bounds=bounds,
            no_sai_experiments=("baseline", "ssp245"),
        )


def test_validate_monthly_cell_bounds_ignores_upper_for_undeclared_experiment() -> None:
    """An experiment outside the no-SAI set keeps the SAI-style exemption."""
    bounds = {"min": _make_monthly_da(10.0), "max": _make_monthly_da(20.0)}
    data = _make_monthly_da(25.0)
    validate_monthly_cell_bounds(
        data=data, var_name="tas", exp="ssp245", member="r01", bounds=bounds,
        no_sai_experiments=("baseline",),
    )


def test_baseline_polar_cold_extreme_within_margin_passes() -> None:
    """MIROC's SSP2-4.5 runs contain Antarctic winter events well below the per-cell
    historical minimum — 19.6 K in the GeoMIP baseline and 21.3 K in ssp245 (tasmin) —
    physical variability that must not trip the plausibility bound."""
    exp2member2data = {
        "historical": {"r01": [_make_monthly_da(250.0, name="tas")]},
        "baseline": {"r01": [_make_monthly_da(280.0, name="tas")]},
    }
    bounds = build_monthly_cell_bounds("tas", exp2member2data)
    data = _make_monthly_da(250.0 - 21.3)
    validate_monthly_cell_bounds(
        data=data, var_name="tas", exp="baseline", member="r01", bounds=bounds
    )


def test_baseline_unit_error_still_fails_low_bound() -> None:
    """A gross error (e.g. °C shipped as K → ~273 K offset) must still be caught."""
    exp2member2data = {
        "historical": {"r01": [_make_monthly_da(250.0, name="tas")]},
        "baseline": {"r01": [_make_monthly_da(280.0, name="tas")]},
    }
    bounds = build_monthly_cell_bounds("tas", exp2member2data)
    data = _make_monthly_da(250.0 - 273.0)
    with pytest.raises(SanityCheckError, match="Monthly grid-cell bound check failed"):
        validate_monthly_cell_bounds(
            data=data, var_name="tas", exp="baseline", member="r01", bounds=bounds
        )


def test_build_monthly_cell_bounds_for_pr() -> None:
    exp2member2data = {
        "historical": {"001": [_make_monthly_da(1.0, name="pr")]},
        "baseline": {"001": [_make_monthly_da(3.5, name="pr")]},
    }
    bounds = build_monthly_cell_bounds("pr", exp2member2data)
    assert bounds is not None
    assert float(bounds["min"].item()) == 0.0
    assert float(bounds["max"].item()) == 13.5


def test_validate_monthly_cell_bounds_passes_in_range() -> None:
    bounds = {"min": _make_monthly_da(10.0), "max": _make_monthly_da(20.0)}
    data = _make_monthly_da(15.0)
    validate_monthly_cell_bounds(
        data=data, var_name="tas", exp="1.5", member="001", bounds=bounds
    )


def test_validate_monthly_cell_bounds_fails_high_on_baseline() -> None:
    """Upper bound vs SSP245 stack is only enforced for ``baseline``."""
    bounds = {"min": _make_monthly_da(10.0), "max": _make_monthly_da(20.0)}
    data = _make_monthly_da(25.0)
    with pytest.raises(ValueError, match="Monthly grid-cell bound check failed"):
        validate_monthly_cell_bounds(
            data=data, var_name="tas", exp="baseline", member="001", bounds=bounds
        )


def test_validate_monthly_cell_bounds_allows_high_for_sai() -> None:
    bounds = {"min": _make_monthly_da(10.0), "max": _make_monthly_da(20.0)}
    data = _make_monthly_da(25.0)
    validate_monthly_cell_bounds(
        data=data, var_name="tas", exp="1.5", member="001", bounds=bounds
    )


def test_validate_monthly_cell_bounds_allows_low_for_sai() -> None:
    bounds = {"min": _make_monthly_da(10.0), "max": _make_monthly_da(20.0)}
    data = _make_monthly_da(5.0)
    validate_monthly_cell_bounds(
        data=data, var_name="tasmin", exp="0.5", member="002", bounds=bounds
    )


@pytest.mark.parametrize("exp", ["g6-1.5", "g6-0.5"])
def test_validate_monthly_cell_bounds_allows_low_for_g6_sai(exp: str) -> None:
    """MIROC's G6 runs are SAI experiments too — exempt from the lower bound."""
    bounds = {"min": _make_monthly_da(10.0), "max": _make_monthly_da(20.0)}
    data = _make_monthly_da(5.0)
    validate_monthly_cell_bounds(
        data=data, var_name="tas", exp=exp, member="r01", bounds=bounds
    )


def test_validate_monthly_cell_bounds_fails_low_historical() -> None:
    bounds = {"min": _make_monthly_da(10.0), "max": _make_monthly_da(20.0)}
    data = _make_monthly_da(5.0)
    with pytest.raises(ValueError, match="Monthly grid-cell bound check failed"):
        validate_monthly_cell_bounds(
            data=data, var_name="tas", exp="historical", member="001", bounds=bounds
        )


def test_validate_monthly_cell_bounds_historical_ignores_upper() -> None:
    bounds = {"min": _make_monthly_da(10.0), "max": _make_monthly_da(20.0)}
    data = _make_monthly_da(25.0)
    validate_monthly_cell_bounds(
        data=data, var_name="tas", exp="historical", member="001", bounds=bounds
    )


def test_validate_variable_specific_ranges_passes_for_icefrac() -> None:
    data = xr.DataArray([0.0, 0.5, 1.0], dims=["time"], name="icefrac")
    validate_variable_specific_ranges(
        data=data, var_name="icefrac", exp="baseline", member="001", stage="test"
    )


def test_validate_variable_specific_ranges_fails_for_icefrac_above_one() -> None:
    data = xr.DataArray([0.1, 1.2], dims=["time"], name="icefrac")
    with pytest.raises(ValueError, match="Range sanity check failed"):
        validate_variable_specific_ranges(
            data=data, var_name="icefrac", exp="baseline", member="001", stage="test"
        )


def test_validate_variable_specific_ranges_passes_for_exposure() -> None:
    data = xr.DataArray([0.0, 120.0, 365.0, 372.0], dims=["time"], name="tas_above_35")
    validate_variable_specific_ranges(
        data=data, var_name="tas_above_35", exp="1.5", member="001", stage="test"
    )


def test_validate_variable_specific_ranges_fails_for_exposure_out_of_bounds() -> None:
    data = xr.DataArray([-1.0, 373.0], dims=["time"], name="pr_above_10")
    with pytest.raises(ValueError, match="Range sanity check failed"):
        validate_variable_specific_ranges(
            data=data, var_name="pr_above_10", exp="1.0", member="002", stage="test"
        )


def test_validate_variable_specific_ranges_passes_for_realistic_pr() -> None:
    data = xr.DataArray([0.0, 2.5, 30.0], dims=["time"], name="pr")
    validate_variable_specific_ranges(
        data=data, var_name="pr", exp="baseline", member="001", stage="test"
    )


def test_validate_variable_specific_ranges_fails_for_pr_unit_error() -> None:
    """A kg/m²/s-vs-m/s 1000x inflation (mean ~2600 mm/day) must be rejected."""
    data = xr.DataArray([2600.0, 29629.0], dims=["time"], name="pr")
    with pytest.raises(SanityCheckError, match="unit/scale error"):
        validate_variable_specific_ranges(
            data=data, var_name="pr", exp="baseline", member="001", stage="test"
        )


def test_validate_year_continuity_strict_passes() -> None:
    validate_year_continuity_strict(
        years=np.array([2015, 2016, 2017]),
        var_name="tas",
        exp="baseline",
        member="001",
        stage="test",
    )


def test_validate_year_continuity_strict_fails_on_gap() -> None:
    with pytest.raises(ValueError, match="not strictly continuous"):
        validate_year_continuity_strict(
            years=np.array([2015, 2017, 2018]),
            var_name="tas",
            exp="1.5",
            member="001",
            stage="test",
        )


def test_validate_year_continuity_with_max_gap_passes_at_limit() -> None:
    validate_year_continuity_with_max_gap(
        years=np.array([2015, 2020, 2021]),
        var_name="tas",
        exp="baseline",
        member="001",
        stage="test",
        max_gap_years=5,
    )


def test_validate_year_continuity_with_max_gap_fails_above_limit() -> None:
    with pytest.raises(ValueError, match="maximum year gap is 6"):
        validate_year_continuity_with_max_gap(
            years=np.array([2015, 2021]),
            var_name="tas",
            exp="baseline",
            member="001",
            stage="test",
            max_gap_years=5,
        )


def test_validate_year_continuity_with_max_gap_allows_hist_baseline_seam() -> None:
    """Stitched yearly series can jump ~7y after incomplete-year drops (e.g. pr member 003)."""
    validate_year_continuity_with_max_gap(
        years=np.array([2007, 2014, 2015]),
        var_name="pr",
        exp="baseline",
        member="003",
        stage="merged continuity",
        max_gap_years=10,
    )
