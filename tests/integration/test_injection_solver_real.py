"""A solved hold-target injection, run through ``get_outputs`` as a custom injection, puts the
dashboard's with-SAI line on the target for every latitude pattern the dashboard offers."""
import numpy as np
import pytest

from backend.get_outputs import get_outputs
from backend.injection_solver import lowest_reachable_target_for_scenario, solve_hold_target_injection

SSP, TARGET, START, RAMP = "SSP2-4.5", 1.0, 2035, 10

# The frontend's location presets (rows 60N -> 60S); the backend only ever sees the weights.
LOCATIONS = {
    "gauss": [0.0, 0.016, 0.3, 0.0, 0.555, 0.129, 0.0],
    "global": [1 / 7] * 7,
    "uniform": [0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
    "arctic-polar-only": [0.527, 0.0, 0.0, 0.0, 0.0, 0.0, 0.473],
    "arctic-polar-focused": [0.413, 0.174, 0.0, 0.0, 0.0, 0.159, 0.254],
    "nh": [0.492, 0.317, 0.19, 0.0, 0.0, 0.0, 0.0],
}


@pytest.mark.parametrize("location", list(LOCATIONS))
def test_solved_injection_holds_the_chart_at_the_target(location, data_dir, model_dir, scratch_cache):
    sol = solve_hold_target_injection(
        SSP, TARGET, LOCATIONS[location], data_dir, model_dir, scratch_cache, start_year=START, ramp_up=RAMP,
    )
    assert sol.reached
    assert sol.peak_tg <= 28.0

    out = get_outputs(
        SSP, None, None, None, 2091, 2100, START, RAMP, data_dir, model_dir, scratch_cache,
        var="tas", variable_injection=sol.injection,
    )["tas"]
    with_sai = out["mean_over_time"]["with_sai"].sel(time=slice(START, 2100)).values
    # the solver's line is the chart's line, and it sits on the target once the ramp is done
    np.testing.assert_allclose(with_sai, sol.with_sai, atol=1e-6)
    np.testing.assert_allclose(with_sai[RAMP:], TARGET, atol=0.01)
    assert with_sai[-10:].mean() == pytest.approx(TARGET, abs=1e-3)


@pytest.mark.parametrize("location", ["arctic-polar-only", "nh"])
def test_the_lowest_reachable_target_is_reached_on_the_dashboard_line(location, data_dir, model_dir, scratch_cache):
    floor = lowest_reachable_target_for_scenario(
        "SSP5-Baseline", LOCATIONS[location], data_dir, model_dir, scratch_cache, start_year=START, ramp_up=RAMP,
    )
    assert 1.5 < floor < 4.4  # SSP5-Baseline cannot be held near 1 K with 28 Tg/yr
    at = solve_hold_target_injection("SSP5-Baseline", floor, LOCATIONS[location], data_dir, model_dir, scratch_cache)
    below = solve_hold_target_injection("SSP5-Baseline", floor - 0.1, LOCATIONS[location], data_dir, model_dir, scratch_cache)
    assert at.reached and not below.reached
