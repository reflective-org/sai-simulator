"""Unit tests for projection-baseline file resolution.

A model can publish two no-SAI runs serving different roles: the run its SAI experiments
branched from (the delta fit's rebase reference, pinned in ``scripts.fit_delta.ANCHORS``)
and the standard ScenarioMIP SSP2-4.5 run the climate-change-only emulator trains on
(pinned in :data:`backend.constants.REGIONAL_MODEL2PROJECTION_BASELINE_FILES`). These
tests cover the preference order, the fallback for variables the ScenarioMIP store lacks,
and the invariant that the two roles stay distinct for MIROC.
"""
import scripts.fit_delta as fd
from backend.constants import (
    REGIONAL_MODEL2PROJECTION_BASELINE_FILES,
    REGIONAL_MODEL_NAMES,
    resolve_projection_baseline_file,
)


def _touch(directory, name):
    path = directory / name
    path.write_bytes(b"")
    return path


def test_every_regional_model_has_projection_baseline_files() -> None:
    assert set(REGIONAL_MODEL2PROJECTION_BASELINE_FILES) == set(REGIONAL_MODEL_NAMES)
    for candidates in REGIONAL_MODEL2PROJECTION_BASELINE_FILES.values():
        assert candidates, "each model needs at least one candidate baseline file"


def test_cesm_single_run_serves_both_roles() -> None:
    """CESM's one baseline run is both the GAUSS branch point and the projection baseline."""
    assert REGIONAL_MODEL2PROJECTION_BASELINE_FILES["CESM2-WACCM"] == (
        "output_gauss-baseline.nc",
    )
    gauss_baselines = {a.baseline_file for a in fd.ANCHORS if a.name.startswith("gauss-")}
    assert gauss_baselines == {"output_gauss-baseline.nc"}


def test_miroc_prefers_scenariomip_over_the_geomip_branch_point() -> None:
    assert REGIONAL_MODEL2PROJECTION_BASELINE_FILES["MIROC-ES2H"] == (
        "output_gauss-ssp245-miroc.nc",
        "output_gauss-baseline-miroc.nc",
    )


def test_miroc_delta_anchors_keep_the_branch_point_run() -> None:
    """The SAI delta must rebase against the run the G6 experiments branched from,
    not the ScenarioMIP run the projection emulator uses."""
    miroc_baselines = {a.baseline_file for a in fd.ANCHORS if a.name.endswith("-miroc")}
    assert miroc_baselines == {"output_gauss-baseline-miroc.nc"}
    assert "output_gauss-ssp245-miroc.nc" not in miroc_baselines


def test_resolve_prefers_scenariomip_when_both_present(tmp_path) -> None:
    _touch(tmp_path, "output_gauss-ssp245-miroc.nc")
    _touch(tmp_path, "output_gauss-baseline-miroc.nc")
    resolved = resolve_projection_baseline_file(tmp_path, "MIROC-ES2H")
    assert resolved.name == "output_gauss-ssp245-miroc.nc"


def test_resolve_falls_back_when_scenariomip_output_absent(tmp_path) -> None:
    """p-e and icefrac have no ssp245 output (all-NaN source), so they fall back."""
    _touch(tmp_path, "output_gauss-baseline-miroc.nc")
    resolved = resolve_projection_baseline_file(tmp_path, "MIROC-ES2H")
    assert resolved.name == "output_gauss-baseline-miroc.nc"


def test_resolve_returns_none_when_nothing_processed(tmp_path) -> None:
    assert resolve_projection_baseline_file(tmp_path, "MIROC-ES2H") is None
    assert resolve_projection_baseline_file(tmp_path, "CESM2-WACCM") is None


def test_resolve_cesm_ignores_miroc_files(tmp_path) -> None:
    _touch(tmp_path, "output_gauss-ssp245-miroc.nc")
    assert resolve_projection_baseline_file(tmp_path, "CESM2-WACCM") is None
    _touch(tmp_path, "output_gauss-baseline.nc")
    assert resolve_projection_baseline_file(tmp_path, "CESM2-WACCM").name == (
        "output_gauss-baseline.nc"
    )
