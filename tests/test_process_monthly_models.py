"""Unit tests for the multi-model config in :mod:`scripts.process_monthly_gauss`.

Covers the ``MODELS`` table (CESM2-WACCM-MA + MIROC-ES2H, with MIROC's
no-variability / ``-miroc`` suffix quirks), the variable→Zarr group routing in
``_group_for_var``, the ``var2esm_var`` name map, and the ``process_monthly``
argument guards. Importing this module must not require a compiled ESMF
library — ``xesmf`` is imported lazily in the regridding functions — so these
run in a plain CI environment.
"""
import pytest

import scripts.process_monthly_gauss as pmg


def test_models_are_cesm_and_miroc() -> None:
    assert set(pmg.MODELS) == {"CESM2-WACCM-MA", "MIROC-ES2H"}


def test_cesm_config() -> None:
    cfg = pmg.MODELS["CESM2-WACCM-MA"]
    assert cfg["model_coord"] == "CESM2-WACCM"
    assert cfg["output_suffix"] == ""
    assert cfg["has_historical"] is True
    assert cfg["compute_variability"] is True
    labels = [label for _, label in cfg["experiments"]]
    assert "historical" in labels
    assert "baseline" in labels
    # One run serves as both the GAUSS branch point and the projection baseline.
    assert cfg["no_sai_experiments"] == ("baseline",)
    assert cfg["experiment_missing_vars"] == {}


def test_miroc_config_isolates_its_quirks() -> None:
    cfg = pmg.MODELS["MIROC-ES2H"]
    assert cfg["model_coord"] == "MIROC-ES2H"
    assert cfg["output_suffix"] == "-miroc"
    assert cfg["has_historical"] is True
    assert cfg["compute_variability"] is False
    labels = [label for _, label in cfg["experiments"]]
    assert labels == ["baseline", "ssp245", "historical", "g6-1.5", "g6-0.5"]


def test_miroc_publishes_two_no_sai_runs() -> None:
    """The GeoMIP branch point and the ScenarioMIP run are both no-SAI references."""
    cfg = pmg.MODELS["MIROC-ES2H"]
    assert cfg["no_sai_experiments"] == ("baseline", "ssp245")
    # Historical is stitched into both so each carries the full 1850- series.
    assert "historical" not in cfg["no_sai_experiments"]


def test_miroc_ssp245_has_no_missing_variables() -> None:
    """The ssp245 store is complete: QFLX filled from the G6 baseline hfls
    (2020-2084) and ICEFRAC regridded from the run's own CMOR siconc (2015-2100)."""
    cfg = pmg.MODELS["MIROC-ES2H"]
    assert cfg["experiment_missing_vars"] == {}


def test_experiment_uris_use_zarr_root() -> None:
    for cfg in pmg.MODELS.values():
        for uri, _label in cfg["experiments"]:
            assert uri.startswith(pmg.ZARR_STORES_URI)
            assert uri.endswith(".zarr")


@pytest.mark.parametrize(
    ("var", "group"),
    [
        ("tas", "Mon/atmos_2d"),
        ("pr", "Mon/atmos_2d"),
        ("p-e", "Mon/atmos_2d"),
        ("icefrac", "Mon/atmos_2d"),
        ("tasmin", "Mon/atmos_2d_tasminmax"),
        ("tasmax", "Mon/atmos_2d_tasminmax"),
        ("tas_above_35", "Mon/atmos_2d_derived"),
        ("tas_above_40", "Mon/atmos_2d_derived"),
        ("tas_below_0", "Mon/atmos_2d_derived"),
        ("pr_above_10", "Mon/atmos_2d_derived"),
        ("pr_above_20", "Mon/atmos_2d_derived"),
    ],
)
def test_group_for_var(var: str, group: str) -> None:
    assert pmg._group_for_var(var) == group


@pytest.mark.parametrize(
    ("var", "esm_var"),
    [
        ("tas", "TREFHT"),
        ("pr", "PRECT"),
        ("tasmin", "TREFHTMN"),
        ("tasmax", "TREFHTMX"),
        ("p-e", "QFLX"),
        ("icefrac", "ICEFRAC"),
        ("pr_above_20", "PRECT_above_20"),
    ],
)
def test_var2esm_var_mapping(var: str, esm_var: str) -> None:
    assert pmg.var2esm_var[var] == esm_var


def test_prect_unit_scaler_is_m_per_s_to_mm_per_day() -> None:
    # CESM PRECT is m/s; MIROC is normalised to m/s in the producer so this
    # single scaler is correct for both. 86400 * 1000 = m/s -> mm/day.
    assert pmg.UNIT_SCALERS["PRECT"] == 86400 * 1000


def test_process_monthly_rejects_unknown_var() -> None:
    with pytest.raises(ValueError, match="var must be one of"):
        pmg.process_monthly(var="not_a_var", output_dir="/tmp/should-not-be-created")


def test_process_monthly_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="model must be 'all' or one of"):
        pmg.process_monthly(
            var="tas", model="NOPE", output_dir="/tmp/should-not-be-created"
        )
