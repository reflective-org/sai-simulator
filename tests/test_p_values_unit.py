"""Unit tests for backend.p_values — the detectability test and the FDR adjustment, on synthetic arrays."""
import numpy as np
import pytest

from backend.p_values import DECADES, RHO_CLIP, benjamini_hochberg, detectability_pvalues


def test_decades_cover_2041_to_2100_in_six_windows():
    assert DECADES[0] == (2041, 2050) and DECADES[-1] == (2091, 2100) and len(DECADES) == 6


def test_zero_change_gives_p_one_and_huge_change_gives_p_zero():
    sigma = np.ones((2, 2))
    p = detectability_pvalues(np.zeros((2, 2)), sigma, 0.0, k=10)
    np.testing.assert_allclose(p, 1.0)
    p = detectability_pvalues(np.full((2, 2), 100.0), sigma, 0.0, k=10)
    np.testing.assert_allclose(p, 0.0, atol=1e-12)


def test_standard_error_is_sigma_over_sqrt_k_when_uncorrelated():
    # delta equal to 1.96 * sigma / sqrt(k) must give p ~ 0.05
    k = 10
    delta = np.array([[1.959964 / np.sqrt(k)]])
    p = detectability_pvalues(delta, np.ones((1, 1)), 0.0, k=k)
    assert p[0, 0] == pytest.approx(0.05, abs=1e-5)


def test_positive_autocorrelation_widens_the_error_and_raises_p():
    delta = np.array([[0.5]])
    p0 = detectability_pvalues(delta, np.ones((1, 1)), 0.0, k=10)
    p_rho = detectability_pvalues(delta, np.ones((1, 1)), 0.5, k=10)
    assert p_rho[0, 0] > p0[0, 0]
    # rho = 0.5 -> c_rho = 3 -> SE grows by sqrt(3)
    z0 = 0.5 * np.sqrt(10)
    z_rho = z0 / np.sqrt(3)
    from scipy.stats import norm

    assert p_rho[0, 0] == pytest.approx(2 * norm.sf(z_rho))


def test_more_years_in_the_window_lowers_p():
    delta = np.array([[0.5]])
    p10 = detectability_pvalues(delta, np.ones((1, 1)), 0.0, k=10)
    p1 = detectability_pvalues(delta, np.ones((1, 1)), 0.0, k=1)
    assert p10[0, 0] < p1[0, 0]


def test_zero_sigma_keeps_the_legacy_rule_and_undefined_inputs_give_nan():
    # sigma == 0: p = 1 where nothing changed, p = 0 where something changed (as before).
    delta = np.array([[0.0, 1.0, 1.0, np.nan]])
    sigma = np.array([[0.0, 0.0, np.nan, 1.0]])
    p = detectability_pvalues(delta, sigma, 0.0, k=10)
    assert p[0, 0] == 1.0 and p[0, 1] == 0.0
    assert np.isnan(p[0, 2]) and np.isnan(p[0, 3])


def test_rho_is_clipped_so_the_factor_stays_finite():
    p = detectability_pvalues(np.array([[1.0]]), np.ones((1, 1)), 0.999, k=10)
    assert np.isfinite(p).all()
    p_clip = detectability_pvalues(np.array([[1.0]]), np.ones((1, 1)), RHO_CLIP, k=10)
    np.testing.assert_allclose(p, p_clip)


def test_k_must_be_positive():
    with pytest.raises(ValueError):
        detectability_pvalues(np.zeros((1, 1)), np.ones((1, 1)), 0.0, k=0)


def test_benjamini_hochberg_matches_hand_computation_and_preserves_nan():
    p = np.array([[0.01, 0.04, np.nan], [0.03, 0.20, 0.50]])
    q = benjamini_hochberg(p)
    # sorted p: .01 .03 .04 .20 .50 (m=5); raw m*p/i: .05 .075 .0667 .25 .5; step-up mins from
    # the right: .05 .0667 .0667 .25 .5
    assert np.isnan(q[0, 2])
    np.testing.assert_allclose(q[0, 0], 0.05)
    np.testing.assert_allclose(q[1, 0], 0.0667, atol=1e-4)
    np.testing.assert_allclose(q[0, 1], 0.0667, atol=1e-4)
    np.testing.assert_allclose(q[1, 1], 0.25)
    np.testing.assert_allclose(q[1, 2], 0.5)


def test_benjamini_hochberg_is_monotone_never_below_p_and_capped_at_one():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=(30, 40))
    q = benjamini_hochberg(p)
    assert np.all(q >= p - 1e-12) and np.all(q <= 1.0)
    order = np.argsort(p.ravel())
    assert np.all(np.diff(q.ravel()[order]) >= -1e-12)


def test_benjamini_hochberg_all_nan_returns_all_nan():
    assert np.isnan(benjamini_hochberg(np.full((3, 3), np.nan))).all()


def test_benjamini_hochberg_removes_chance_speckle_on_pure_noise():
    # Under the null, ~5% of raw p-values pass 0.05, but almost none survive BH at q=0.10.
    rng = np.random.default_rng(1)
    p = rng.uniform(size=(192, 288))
    raw = (p < 0.05).mean()
    adjusted = (benjamini_hochberg(p) < 0.10).mean()
    assert 0.04 < raw < 0.06
    assert adjusted < 0.005


# --- Welch test with pluggable noise sources ---------------------------------------------

from pathlib import Path  # noqa: E402

import xarray as xr  # noqa: E402
from scipy import stats  # noqa: E402

from backend import p_values as pv  # noqa: E402


def test_welch_reduces_to_the_known_sigma_z_test_without_reference_noise():
    delta = np.array([0.0, 0.1, 0.3, -0.5])
    decade = pv.MeanNoise(variance=np.full(4, 0.04), dof=np.inf)
    p = pv.welch_pvalues(delta, decade, pv.MeanNoise(0.0, np.inf))
    np.testing.assert_allclose(p, 2 * stats.norm.sf(np.abs(delta) / 0.2))
    np.testing.assert_allclose(p, pv.detectability_pvalues(delta, np.full(4, np.sqrt(0.4)), 0.0, 10))


def test_welch_adds_the_reference_noise_and_uses_satterthwaite_dof():
    delta = np.array([0.5])
    v_d, v_r, d_d, d_r = 0.04, 0.01, 78.0, 108.0
    p = pv.welch_pvalues(delta, pv.MeanNoise(np.array([v_d]), d_d), pv.MeanNoise(np.array([v_r]), d_r))
    df = (v_d + v_r) ** 2 / (v_d**2 / d_d + v_r**2 / d_r)
    np.testing.assert_allclose(p, 2 * stats.t.sf(0.5 / np.sqrt(v_d + v_r), df))
    # more noise and finite dof -> never more significant than the decade-only z-test
    assert p[0] > pv.welch_pvalues(delta, pv.MeanNoise(np.array([v_d]), np.inf), pv.MeanNoise(0.0, np.inf))[0]


def test_welch_zero_noise_rule_and_undefined_inputs():
    zero = pv.MeanNoise(np.zeros(3), 10.0)
    p = pv.welch_pvalues(np.array([0.0, 1.0, np.nan]), zero, pv.MeanNoise(np.zeros(3), 10.0))
    assert p[0] == 1.0 and p[1] == 0.0 and np.isnan(p[2])


def _write(dirpath: Path, stem: str, model: str, value: float, **attrs) -> None:
    da = xr.DataArray(np.full((2, 3), value), dims=("lat", "lon"),
                      coords={"lat": [60.0, 70.0], "lon": [0.0, 1.0, 2.0]}, attrs=attrs)
    da.to_dataset(name="tas").to_netcdf(dirpath / f"{stem}_{model}.nc")


def _artifacts(tmp_path: Path) -> Path:
    var_dir = tmp_path / "tas"
    var_dir.mkdir()
    for model, sigma, scatter, members in (("A", 0.4, 0.4, 3), ("B", 0.6, 0.5, 3), ("C", 0.5, 0.3, 1)):
        _write(var_dir, "grid_level_model_internal_variability", model, sigma, dof=44)
        _write(var_dir, "grid_level_lag1_autocorr", model, 0.0)
        _write(var_dir, "grid_level_reference_variability", model, scatter,
               n_members=members, n_years=20, dof=members * 18)
    return tmp_path


def test_member_reference_noise_is_the_variance_of_the_multi_model_ensemble_mean(tmp_path):
    data_dir = _artifacts(tmp_path)
    noise = pv.member_reference_noise("tas", data_dir, models=["A", "B", "C"])
    terms = np.array([0.4**2 / (20 * 3), 0.5**2 / (20 * 3), 0.3**2 / (20 * 1)]) / 9
    np.testing.assert_allclose(noise.variance, terms.sum())
    dofs = np.array([54.0, 54.0, 18.0])
    np.testing.assert_allclose(noise.dof, terms.sum() ** 2 / np.sum(terms**2 / dofs))


def test_member_decade_noise_pools_sigma_and_sums_dof(tmp_path):
    data_dir = _artifacts(tmp_path)
    noise = pv.member_decade_noise("tas", data_dir, 10, models=["A", "B", "C"])
    np.testing.assert_allclose(noise.variance, np.mean([0.16, 0.36, 0.25]) / 10)
    assert noise.dof == 3 * 44


def test_a_registered_decade_noise_source_replaces_the_members(tmp_path, monkeypatch):
    calls = []

    def stochastic_emulator(var, data_dir, k):
        calls.append((var, k))
        return pv.MeanNoise(variance=np.full((2, 3), 1e-6), dof=np.inf)

    monkeypatch.setitem(pv.DECADE_NOISE_SOURCES, "tas", stochastic_emulator)
    source = pv.decade_noise_for("tas")
    noise = source("tas", tmp_path, 10)
    assert calls == [("tas", 10)] and float(noise.variance.max()) == 1e-6
    # other variables keep the members' σ
    assert pv.decade_noise_for("pr") is not stochastic_emulator
