"""Unit tests for ``backend.injection_solver``: the 1-D global-mean CIDER model and the solver
that inverts a temperature target into an injection.

The CIDER parameter file and ``correct_lat.npy`` are tracked in git (``data/processed``), so the
model runs on the real parameters. The no-SAI lines here are synthetic; the real one is covered
by ``tests/integration/test_injection_solver_real.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend import injection_solver as js
from backend.variable import AOD_from_injection_vectorized, get_variable_regional_delta

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data" / "processed"

# the dashboard's "GAUSS" latitude split (rows 60N -> 60S)
GAUSS_WEIGHTS = [0.0, 0.016, 0.3, 0.0, 0.555, 0.129, 0.0]
POLAR_WEIGHTS = [0.527, 0.0, 0.0, 0.0, 0.0, 0.0, 0.473]


@pytest.fixture(scope="module")
def cider() -> js.GlobalMeanCider:
    return js.load_global_mean_cider(str(DATA_DIR))


def _uneven_injection() -> np.ndarray:
    """Different Tg/yr at every latitude and year, zero before 2040."""
    rng = np.random.default_rng(0)
    injection = rng.uniform(0.0, 4.0, size=(js.N_LATS, js.N_YEARS))
    injection[:, :5] = 0.0
    injection[3] = 0.0  # one latitude unused
    return injection


def _warming_line(start: float = 1.6, end: float = 2.8) -> np.ndarray:
    """A smooth no-SAI global-mean line over 2035-2100 (K vs 1850-1900)."""
    return np.linspace(start, end, js.N_YEARS)


# --- the 1-D model ------------------------------------------------------------------------------


def test_global_mean_matches_the_map_path(cider):
    """The 7-latitude 1-D model equals the cos-lat global mean of the full custom-injection map."""
    injection = _uneven_injection()
    delta_map = get_variable_regional_delta("tas", DATA_DIR, DATA_DIR, injection)
    expected = delta_map.weighted(np.cos(np.deg2rad(delta_map.lat))).mean(("lat", "lon")).values
    np.testing.assert_allclose(js.global_mean_tas_delta(cider, injection), expected, rtol=0, atol=1e-10)


def test_closed_form_aod_matches_the_reference_sum(cider):
    yearly = _uneven_injection()[0]
    monthly = np.repeat(yearly, js.MONTHS_PER_YEAR) / js.MONTHS_PER_YEAR
    for i in range(js.N_LATS):
        np.testing.assert_allclose(
            js._aod_from_yearly_injection(cider.param_aod[i], yearly),
            AOD_from_injection_vectorized(cider.param_aod[i], monthly),
            rtol=1e-12, atol=1e-15,
        )


def test_each_source_year_contributes_independently(cider):
    """Column s of the contribution matrix depends on year s's injection only (what the solver's
    one-shot Jacobian relies on)."""
    injection = _uneven_injection()
    full = js._contributions(cider, injection)
    for s in (5, 20, 65):
        alone = np.zeros_like(injection)
        alone[:, s] = injection[:, s]
        np.testing.assert_allclose(js._contributions(cider, alone)[:, s], full[:, s], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(full.sum(axis=1), js.global_mean_tas_delta(cider, injection))


def test_injection_cools_and_nothing_happens_before_it_starts(cider):
    delta = js.global_mean_tas_delta(cider, _uneven_injection())
    assert np.all(delta[:5] == 0.0)
    assert np.all(delta[5:] < 0.0)


def test_global_mean_rejects_a_wrong_shape(cider):
    with pytest.raises(ValueError, match="injection must be"):
        js.global_mean_tas_delta(cider, np.zeros((7, 65)))


# --- the solver ---------------------------------------------------------------------------------


@pytest.mark.parametrize("weights", [GAUSS_WEIGHTS, POLAR_WEIGHTS, [1 / 7] * 7])
def test_solver_holds_the_target(cider, weights):
    no_sai = _warming_line()
    sol = js.solve_hold_target(no_sai, 1.0, weights, cider)
    assert sol.reached
    assert sol.max_hold_error < 0.01
    np.testing.assert_allclose(sol.with_sai[10:], 1.0, atol=0.01)
    # the reported line is the no-SAI line plus the model's response to the returned injection
    np.testing.assert_allclose(sol.with_sai, no_sai + js.global_mean_tas_delta(cider, sol.injection))


def test_solver_keeps_the_latitude_split(cider):
    sol = js.solve_hold_target(_warming_line(), 1.0, GAUSS_WEIGHTS, cider)
    ratio = np.asarray(GAUSS_WEIGHTS) / np.sum(GAUSS_WEIGHTS)
    np.testing.assert_allclose(sol.injection, np.outer(ratio, sol.total_tg))
    np.testing.assert_allclose(sol.injection.sum(axis=0), sol.total_tg)
    assert np.all(sol.injection[np.asarray(GAUSS_WEIGHTS) == 0] == 0.0)


def test_nothing_is_injected_before_the_start_and_the_ramp_is_followed(cider):
    no_sai = _warming_line()
    sol = js.solve_hold_target(no_sai, 1.0, GAUSS_WEIGHTS, cider, start_year=2045, ramp_up=10)
    assert np.all(sol.total_tg[:10] == 0.0)
    np.testing.assert_array_equal(sol.with_sai[:10], no_sai[:10])
    # over the ramp the cooling grows linearly to the full gap, then the target is held
    frac = np.arange(1, 11) / 10
    expected_ramp = no_sai[10:20] - frac * (no_sai[10:20] - 1.0)
    np.testing.assert_allclose(sol.with_sai[10:20], expected_ramp, atol=0.02)
    np.testing.assert_allclose(sol.with_sai[20:], 1.0, atol=0.01)


def test_injection_stays_within_the_bounds(cider):
    sol = js.solve_hold_target(_warming_line(), 1.0, POLAR_WEIGHTS, cider, max_total_tg=28.0)
    assert sol.total_tg.min() >= 0.0
    assert sol.peak_tg <= 28.0 + 1e-9


def test_a_target_beyond_the_ceiling_is_reported_unreached(cider):
    no_sai = np.full(js.N_YEARS, 4.5)
    sol = js.solve_hold_target(no_sai, 1.0, GAUSS_WEIGHTS, cider, max_total_tg=28.0)
    assert not sol.reached
    assert sol.peak_tg == pytest.approx(28.0)
    assert sol.max_hold_error > js.HOLD_TOLERANCE_K
    # capped, it still cools as much as it can
    assert np.all(sol.with_sai[10:] < no_sai[10:])


def test_when_the_ceiling_binds_the_target_is_held_until_then_and_never_undercut(cider):
    """A line that warms past what 28 Tg/yr can offset: the target is held while it can be, the
    injection then stays at the ceiling, and the earlier years are not cooled below the target to
    make up for the later shortfall."""
    no_sai = _warming_line(1.7, 4.7)
    sol = js.solve_hold_target(no_sai, 1.0, GAUSS_WEIGHTS, cider, max_total_tg=28.0)
    assert not sol.reached
    at_cap = np.flatnonzero(sol.total_tg >= 28.0 - 1e-6)
    first_capped = int(at_cap[0])
    assert 10 < first_capped < 50
    np.testing.assert_allclose(sol.total_tg[first_capped:], 28.0, atol=1e-6)  # stays at the ceiling
    np.testing.assert_allclose(sol.with_sai[10:first_capped], 1.0, atol=0.02)  # held until then
    assert np.all(sol.with_sai >= np.minimum(no_sai, 1.0) - 0.02)  # never colder than the target
    assert sol.with_sai[-1] > 1.5  # and the line rises once capped


def test_the_injection_does_not_zig_zag(cider):
    """Each year's injection cools the next year more than its own, so an exact inversion
    alternates; the smoothness penalty keeps the curve monotone apart from small wiggles."""
    sol = js.solve_hold_target(_warming_line(), 1.0, GAUSS_WEIGHTS, cider, start_year=2050, ramp_up=5)
    step = np.diff(sol.total_tg[15:])
    reversals = (step[1:] * step[:-1] < 0) & (np.minimum(abs(step[1:]), abs(step[:-1])) > 1.0)
    assert reversals.sum() <= 1


def test_a_lower_ceiling_binds(cider):
    sol = js.solve_hold_target(_warming_line(), 1.0, GAUSS_WEIGHTS, cider, max_total_tg=5.0)
    assert sol.peak_tg <= 5.0 + 1e-9
    assert not sol.reached


def test_no_injection_when_already_below_the_target(cider):
    no_sai = _warming_line(0.6, 0.9)
    sol = js.solve_hold_target(no_sai, 1.0, GAUSS_WEIGHTS, cider)
    assert sol.reached
    assert sol.peak_tg == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(sol.with_sai, no_sai, atol=1e-9)


def test_cooling_only_starts_where_the_line_crosses_the_target(cider):
    """Years whose no-SAI line is below the target are left alone (no pre-cooling below it)."""
    no_sai = _warming_line(0.7, 2.0)  # crosses 1.0 around 2050
    sol = js.solve_hold_target(no_sai, 1.0, GAUSS_WEIGHTS, cider)
    assert sol.reached
    assert np.all(sol.with_sai >= np.minimum(no_sai, 1.0) - 0.01)
    np.testing.assert_allclose(sol.with_sai[no_sai >= 1.05], 1.0, atol=0.01)


@pytest.mark.parametrize(
    "weights, match",
    [([1.0] * 6, "7 entries"), ([1.0, -1.0, 0, 0, 0, 0, 0], "non-negative"), ([0.0] * 7, "all be zero")],
)
def test_bad_weights_are_rejected(cider, weights, match):
    with pytest.raises(ValueError, match=match):
        js.solve_hold_target(_warming_line(), 1.0, weights, cider)


def test_bad_inputs_are_rejected(cider):
    with pytest.raises(ValueError, match="must cover"):
        js.solve_hold_target(np.zeros(65), 1.0, GAUSS_WEIGHTS, cider)
    with pytest.raises(ValueError, match="start_year"):
        js.solve_hold_target(_warming_line(), 1.0, GAUSS_WEIGHTS, cider, start_year=2101)


# --- the lowest reachable target ----------------------------------------------------------------


@pytest.mark.parametrize("weights", [GAUSS_WEIGHTS, POLAR_WEIGHTS])
def test_the_lowest_reachable_target_is_reached_and_lower_ones_are_not(cider, weights):
    no_sai = _warming_line(1.7, 4.7)
    floor = js.lowest_reachable_target(no_sai, weights, cider)
    assert js.solve_hold_target(no_sai, floor, weights, cider).reached
    assert not js.solve_hold_target(no_sai, floor - 0.1, weights, cider).reached


def test_the_floor_lies_between_its_bounds(cider):
    no_sai = _warming_line(1.7, 4.7)
    capped = no_sai + js.global_mean_tas_delta(cider, np.outer(np.asarray(GAUSS_WEIGHTS) / sum(GAUSS_WEIGHTS), np.full(js.N_YEARS, 28.0)))
    floor = js.lowest_reachable_target(no_sai, GAUSS_WEIGHTS, cider)
    # the ceiling from the start cools every year the most, but a flat target cannot use it early
    assert capped[10:].max() < floor < no_sai[10:].max()


def test_a_higher_ceiling_lowers_the_floor(cider):
    no_sai = _warming_line(1.7, 4.7)
    assert js.lowest_reachable_target(no_sai, GAUSS_WEIGHTS, cider, max_total_tg=40.0) < js.lowest_reachable_target(
        no_sai, GAUSS_WEIGHTS, cider, max_total_tg=28.0
    )


def test_first_needed_year_is_where_the_line_passes_the_target(cider):
    no_sai = _warming_line(1.6, 2.9)  # passes 2.0 a third of the way, 2.5 later
    assert js.solve_hold_target(no_sai, 1.0, GAUSS_WEIGHTS, cider).first_needed_year == 2035
    year = js.solve_hold_target(no_sai, 2.0, GAUSS_WEIGHTS, cider).first_needed_year
    assert year == int(js.YEARS[np.flatnonzero(no_sai > 2.0)[0]])
    assert js.solve_hold_target(no_sai, 2.5, GAUSS_WEIGHTS, cider).first_needed_year > year
    # counted from 2035 whatever the start year (a later start could move earlier), and None
    # when the line never passes the target
    assert js.solve_hold_target(no_sai, 1.0, GAUSS_WEIGHTS, cider, start_year=2050).first_needed_year == 2035
    assert js.solve_hold_target(no_sai, 3.0, GAUSS_WEIGHTS, cider).first_needed_year is None
