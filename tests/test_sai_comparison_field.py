"""Unit tests for backend.get_outputs._with_sai_comparison_field — the with-SAI field the
detectability test compares against the 1850-1900 reference, shared by the live path and
scripts/cache.py."""
import numpy as np
import xarray as xr

from backend.get_outputs import _VariableFields, _with_sai_comparison_field

PROJ = np.arange(2015, 2021)


def _field(value, time=PROJ):
    return xr.DataArray(
        np.full((len(time), 1, 1), value, dtype=float),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [70.0], "lon": [0.0]},
    )


def _fields(regional_map, rebase):
    return _VariableFields(
        regional_map=regional_map,
        regional_no_sai_p_values=None,
        historical_model=_field(0.0, np.arange(1850, 1901)),
        historical_rebase=_field(rebase, [0]).isel(time=0, drop=True),
        # the test's reference and the map's display shift; the comparison field uses neither
        significance_reference=None,
        display_offset=None,
        area=None,
    )


def test_count_variables_are_absolute_and_not_rebased_again():
    # tas_above_35 maps are absolute day counts: adding the 1850-1900 level would double it.
    fields = _fields(_field(20.0), rebase=5.0)
    out = _with_sai_comparison_field(
        "tas_above_35", fields=fields, regional_delta=_field(-3.0), icefrac_with_sai=None
    )
    np.testing.assert_allclose(out.values, 17.0)


def test_rebased_variables_are_put_back_on_the_absolute_level():
    fields = _fields(_field(2.0), rebase=287.0)  # tas anomaly map
    out = _with_sai_comparison_field(
        "tas", fields=fields, regional_delta=_field(-1.0), icefrac_with_sai=None
    )
    np.testing.assert_allclose(out.values, 288.0)


def test_icefrac_uses_the_with_sai_fraction_itself():
    fields = _fields(_field(0.2), rebase=0.9)
    with_sai = _field(0.6, PROJ[2:])  # SAI from 2017
    out = _with_sai_comparison_field(
        "icefrac", fields=fields, regional_delta=_field(0.4, PROJ[2:]), icefrac_with_sai=with_sai
    )
    # before the start the with-SAI world is the no-SAI world
    np.testing.assert_allclose(out.sel(time=[2015, 2016]).values, 0.2)
    np.testing.assert_allclose(out.sel(time=slice(2017, None)).values, 0.6)


def test_delta_is_zero_before_the_sai_start():
    fields = _fields(_field(20.0), rebase=5.0)
    out = _with_sai_comparison_field(
        "pr_above_10", fields=fields, regional_delta=_field(-3.0, PROJ[3:]), icefrac_with_sai=None
    )
    assert list(out.time.values) == list(PROJ)
    np.testing.assert_allclose(out.sel(time=PROJ[:3]).values, 20.0)
    np.testing.assert_allclose(out.sel(time=PROJ[3:]).values, 17.0)
