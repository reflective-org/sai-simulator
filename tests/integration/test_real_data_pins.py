"""Pins from the regenerated CESM + MIROC pipeline (see the review doc for provenance).

Tolerances are loose enough to survive re-fitting noise (the emulators are bootstrapped) but
tight enough to catch a change of meaning — a different model set, a different σ source, a
different reference window.
"""
import numpy as np
import pytest
import xarray as xr

from backend.constants import REGIONAL_MODEL_NAMES
from backend.historical import get_historical_model, get_historical_model_per_model
from backend.icefrac import get_area
from backend.p_values import get_grid_level_p_values, load_mean_rho, load_pooled_sigma
from backend.uncertainty import combine_series
from scripts.compare.leverage import anchor_leverages

MODELS = ["CESM2-WACCM", "MIROC-ES2H", "UKESM1-1-LL"]  # the models these pins were measured for


def test_the_pinned_models_are_in_the_roster():
    assert set(MODELS) <= set(REGIONAL_MODEL_NAMES)


@pytest.mark.parametrize(
    "var, expected",
    [
        ("tas", {"CESM2-WACCM": 0.1190, "MIROC-ES2H": 0.1812, "UKESM1-1-LL": 0.1460}),
        ("tasmax", {"CESM2-WACCM": 0.1606, "MIROC-ES2H": 0.2617, "UKESM1-1-LL": 0.1841}),
        # P-E, not evaporation (U7); UKESM's from its one member with historical evaporation
        ("p-e", {"CESM2-WACCM": 0.0015, "MIROC-ES2H": 0.0200, "UKESM1-1-LL": 0.0018}),
        # extent, million km^2: MIROC from the 2026-09-22 rebuilt (seam-free) ICEFRAC stores,
        # all three from the sigmoids refit under #55 (ice-free years kept); UKESM one member
        ("icefrac", {"CESM2-WACCM": 0.5830, "MIROC-ES2H": 0.3081, "UKESM1-1-LL": 0.3025}),
    ],
)
def test_per_model_global_sigma_pins(data_dir, var, expected):
    for model, value in expected.items():
        sigma = float(np.load(data_dir / var / f"model_internal_variability_{model}.npy"))
        assert sigma == pytest.approx(value, rel=0.05), (var, model, sigma)


def test_pe_sigma_is_not_evaporation_sigma(data_dir):
    """The legacy CESM p-e scalar was evaporation-only (0.0129); P-E's own is ~9x smaller."""
    sigma_pe = float(np.load(data_dir / "p-e" / "model_internal_variability_CESM2-WACCM.npy"))
    assert sigma_pe < 0.005


def test_pooled_sigma_and_rho_artifacts_load_on_the_common_grid(data_dir):
    sigma = load_pooled_sigma("tas", data_dir)
    rho = load_mean_rho("tas", data_dir)
    assert sigma.shape == rho.shape == (192, 288)
    assert np.nanmedian(sigma) == pytest.approx(0.45, abs=0.2)
    assert -0.9 <= np.nanmin(rho) and np.nanmax(rho) <= 0.9
    assert 0.0 <= np.nanmedian(rho) <= 0.3  # measured 0.08-0.19 for tas


def test_anchor_leverage_split(data_dir):
    rows, per_model = anchor_leverages(data_dir)
    assert len(rows) == 6  # 3 CESM GAUSS, 2 MIROC G6, 1 UKESM G6 (#55)
    assert per_model["CESM2-WACCM"] == pytest.approx(0.511, abs=0.01)
    assert per_model["MIROC-ES2H"] == pytest.approx(0.336, abs=0.01)
    assert per_model["UKESM1-1-LL"] == pytest.approx(0.154, abs=0.01)
    coolings = {r["anchor"]: r["cooling_K"] for r in rows}
    assert coolings["gauss-0.5"] == pytest.approx(-1.84, abs=0.05)
    assert coolings["g6-0.5-miroc"] == pytest.approx(-1.845, abs=0.05)
    assert coolings["g6-1.5-ukesm"] == pytest.approx(-1.367, abs=0.05)


def test_icefrac_historical_is_the_mean_of_the_models(data_dir, scratch_cache):
    """X1: 1985-2014 September extent CESM 5.55, MIROC 6.11, UKESM 4.89 -> combined 5.52
    million km^2 (MIROC from the rebuilt ICEFRAC stores; two-model 5.83 before UKESM)."""
    area = get_area(data_dir)
    combined = get_historical_model("icefrac", data_dir, scratch_cache)
    per_model = get_historical_model_per_model("icefrac", data_dir, scratch_cache)

    def extent(da):
        return float((da.sel(time=slice(1985, 2014)).mean("time").fillna(0) * area).sum())

    assert list(per_model.model.values) == MODELS
    assert extent(per_model.sel(model="CESM2-WACCM")) == pytest.approx(5.55, abs=0.1)
    assert extent(per_model.sel(model="MIROC-ES2H")) == pytest.approx(6.11, abs=0.1)
    assert extent(per_model.sel(model="UKESM1-1-LL")) == pytest.approx(4.89, abs=0.1)
    assert extent(combined) == pytest.approx(5.52, abs=0.1)


def test_per_model_historical_windows_differ_by_one_year(data_dir, scratch_cache):
    """X5: CESM's annual series starts 1851, MIROC's and UKESM's 1850."""
    per_model = get_historical_model_per_model("tas", data_dir, scratch_cache)
    years = {m: int(per_model.sel(model=m).dropna("time", how="all").time.min()) for m in MODELS}
    assert years == {"CESM2-WACCM": 1851, "MIROC-ES2H": 1850, "UKESM1-1-LL": 1850}


def test_combined_band_on_real_icefrac_series(data_dir, scratch_cache):
    """Three model series with the real per-model sigmas: the spread is reported (ddof=1 from
    three models) but the band is the internal variability of the mean only."""
    sig = {m: float(np.load(data_dir / "icefrac" / f"model_internal_variability_{m}.npy")) for m in MODELS}
    per_model = xr.DataArray(
        [[5.83, 5.83], [2.58, 2.58], [3.40, 3.40]], dims=("model", "time"),
        coords={"model": MODELS, "time": [2099, 2100]},
    )
    res = combine_series(per_model, sig)
    assert res.ddof == 1 and res.n_models == 3
    np.testing.assert_allclose(res.sigma_model.values, np.std([5.83, 2.58, 3.40], ddof=1), atol=1e-6)
    # 2 * sqrt(0.583^2 + 0.308^2 + 0.3025^2) / 3: what the dashboard draws for sea ice
    assert float(res.internal[0]) == pytest.approx(0.484, abs=0.005)


def test_pvalues_on_real_reference_are_in_range_and_masked(data_dir, scratch_cache):
    """Detectability test on a synthetic +1 K shift over the real tas reference: significant
    almost everywhere on land, NaN over ocean, and each decade its own map."""
    hist = get_historical_model("tas", data_dir, scratch_cache)
    reference = hist.sel(time=slice(1850, 1900)).mean("time")
    comparison = xr.concat(
        [(reference + 1.0).expand_dims(time=[y]) for y in range(2041, 2101)], dim="time"
    )
    p = get_grid_level_p_values("tas", data_dir, reference, comparison)
    assert list(p.time.values) == [2050, 2060, 2070, 2080, 2090, 2100]
    valid = p.sel(time=2100).notnull()
    assert 15000 < int(valid.sum()) < 20000  # land cells
    assert float((p.sel(time=2100) < 0.05).sum() / valid.sum()) > 0.95
    assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0


def test_decade_before_sai_start_is_the_no_sai_world(data_dir, model_dir, scratch_cache):
    """P6 live path: a 2041-2050 request with a 2060 start has no SAI yet, so its with-SAI
    p-values, histogram and decade-mean delta equal the no-SAI ones (and it must not crash)."""
    from backend.get_outputs import get_outputs

    for var in ("tas", "icefrac"):
        out = get_outputs("SSP2-4.5", 1.0, None, None, 2041, 2050, 2060, 10,
                          data_dir, model_dir, scratch_cache, var=var)
        p_sai = out[var]["regional_sai_p_values"]
        p_no_sai = out[var]["regional_no_sai_p_values"]
        assert p_sai.notnull().any()
        xr.testing.assert_allclose(p_sai, p_no_sai)
        dist = out[var]["distribution"]
        assert sum(dist["with_sai"]["counts"]) == pytest.approx(1.0)
        np.testing.assert_allclose(dist["with_sai"]["counts"], dist["no_sai"]["counts"])
        assert float(abs(out[var]["regional_delta_mean"]).max()) == 0.0
