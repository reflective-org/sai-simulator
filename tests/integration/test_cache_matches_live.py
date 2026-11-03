"""scripts/cache.py precomputes what the dashboard serves at the default deployment; it must be
exactly what a live request computes (review on PR #56), and it must not leave a ramped delta
where live requests expect the unramped one."""
import numpy as np
import pytest
import xarray as xr

from backend.baseline import get_global_temp, get_regional_map
from backend.constants import DEFAULT_RAMP_UP, DEFAULT_SAI_START_YEAR
from backend.get_outputs import _load_variable_fields, get_outputs, get_temp_diff
from backend.utils import get_interpolator
from scripts.cache import cache_no_sai, cache_sai_p_values

SSP, TARGET = "SSP2-4.5", 1.0


def _cache_one(var, data_dir, model_dir, cache_dir):
    """Run the cache steps for one variable, one target, into a fresh ``cache_dir``."""
    cache_dir.mkdir()
    global_temp = get_global_temp(SSP)
    cache_no_sai("tas", SSP, global_temp, data_dir, model_dir, cache_dir)
    rt, rp, rpm = get_regional_map("tas", data_dir, model_dir, cache_dir, SSP)
    temp = dict(regional_temp=rt, regional_temp_no_sai_p_values=rp, regional_temp_per_model=rpm)
    cache_no_sai(var, SSP, global_temp, data_dir, model_dir, cache_dir, **temp)
    fields = _load_variable_fields(
        var, ssp_scenario=SSP, data_dir=data_dir, model_dir=model_dir, cache_dir=cache_dir, **temp
    )
    temp_diff = get_temp_diff(data_dir, model_dir, cache_dir, SSP, TARGET)
    p = cache_sai_p_values(
        var, fields, SSP, TARGET, temp_diff, data_dir, model_dir, cache_dir, **temp
    )
    return p, temp_diff


@pytest.mark.parametrize("var", ["exposure_above_35", "icefrac", "tas"])
def test_cached_sai_p_values_equal_live_at_default_deployment(var, data_dir, model_dir, tmp_path):
    """An exposure count (absolute, not rebased), icefrac (end-to-end sigmoids) and a rebased one."""
    cached, _ = _cache_one(var, data_dir, model_dir, tmp_path / "cache")

    live_dir = tmp_path / "live"  # no SAI p-value file here, so get_outputs computes it
    live_dir.mkdir()
    out = get_outputs(
        SSP, TARGET, None, None, 2091, 2100, DEFAULT_SAI_START_YEAR, DEFAULT_RAMP_UP,
        data_dir, model_dir, live_dir, var=var,
    )
    live = out[var]["regional_sai_p_values"]
    xr.testing.assert_allclose(cached.sel(time=2100), live)


def test_cache_leaves_the_unramped_delta_on_disk(data_dir, model_dir, tmp_path):
    """Live requests ramp the cached delta themselves; a ramped file would ramp twice."""
    cache_dir = tmp_path / "cache"
    _, temp_diff = _cache_one("tas", data_dir, model_dir, cache_dir)
    on_disk = xr.open_dataarray(cache_dir / "ssp245_1.0_regional_delta_tas.nc").load()
    slope = get_interpolator(model_dir / "tas" / "interpolator.nc").sel(features="slope")["tas"]
    expected = (slope * temp_diff).transpose(*on_disk.dims)

    assert int(on_disk.time.min()) == int(temp_diff.time.min())  # whole period, not start-sliced
    for year in (DEFAULT_SAI_START_YEAR, DEFAULT_SAI_START_YEAR + 5, 2090):
        np.testing.assert_allclose(
            on_disk.sel(time=year).values, expected.sel(time=year).values, rtol=1e-5
        )
