"""The FaIR climate-response calibration switch.

``fair_wrap.fair_utils`` selects a climate-response calibration by name. The default is
the AR6 calibrated-constrained ensemble, which keeps each member's own climate parameters
so the run spans the assessed uncertainty; ``"cesm"`` collapses every member onto one
CESM2-WACCM-tuned set and is retained for reproducing pre-multi-model output.
"""
import pandas as pd
import pytest

from fair_wrap.fair_utils import (
    CLIMATE_CALIBRATIONS,
    DEFAULT_CALIBRATION,
    apply_climate_calibration,
)

CLIMATE_COLUMNS = [
    "clim_c1", "clim_c2", "clim_c3",
    "clim_kappa1", "clim_kappa2", "clim_kappa3",
    "clim_epsilon", "clim_gamma", "clim_sigma_eta", "clim_sigma_xi", "clim_F_4xCO2",
]


def _ensemble() -> pd.DataFrame:
    """Two members with deliberately different climate parameters."""
    return pd.DataFrame({col: [i + 1.0, i + 100.0] for i, col in enumerate(CLIMATE_COLUMNS)})


def test_default_is_the_constrained_ensemble():
    assert DEFAULT_CALIBRATION == "constrained"


def test_constrained_keeps_per_member_spread():
    df = _ensemble()
    out = apply_climate_calibration(df, "constrained")
    pd.testing.assert_frame_equal(out, df)
    # The point of the constrained ensemble is that members differ.
    assert out["clim_kappa1"].nunique() == 2


def test_input_frame_is_never_mutated():
    df = _ensemble()
    pristine = df.copy()
    apply_climate_calibration(df, "cesm")
    pd.testing.assert_frame_equal(df, pristine)


def test_cesm_then_constrained_on_same_frame_gives_distinct_calibrations():
    # The A/B reproduce-old-output workflow: both calibrations from one ensemble frame.
    df = _ensemble()
    cesm = apply_climate_calibration(df, "cesm")
    constrained = apply_climate_calibration(df, "constrained")
    assert cesm["clim_kappa1"].nunique() == 1
    assert constrained["clim_kappa1"].nunique() == 2


def test_override_of_missing_column_raises():
    # A typo'd/renamed key must not silently append a dead column.
    df = _ensemble().drop(columns=["clim_kappa1"])
    with pytest.raises(ValueError, match="not present"):
        apply_climate_calibration(df, "cesm")


def test_cesm_collapses_every_member_onto_one_set():
    out = apply_climate_calibration(_ensemble(), "cesm")
    for column, value in CLIMATE_CALIBRATIONS["cesm"].items():
        assert out[column].nunique() == 1
        assert out[column].iloc[0] == pytest.approx(value)


def test_cesm_calibration_covers_every_climate_column():
    # A partial override would silently mix CESM and ensemble parameters.
    assert set(CLIMATE_CALIBRATIONS["cesm"]) == set(CLIMATE_COLUMNS)


def test_unknown_calibration_name_raises():
    with pytest.raises(ValueError, match="Unknown calibrate_to"):
        apply_climate_calibration(_ensemble(), "cmip6-mmm")


def test_cesm_kappa1_is_more_sensitive_than_the_constrained_median():
    # Guards the documented reason for the switch: the CESM tuning sits at the
    # sensitive end of the assessed range (ensemble median clim_kappa1 ~1.31).
    assert CLIMATE_CALIBRATIONS["cesm"]["clim_kappa1"] < 0.75
