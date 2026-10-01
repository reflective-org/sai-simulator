"""Unit tests for the line-plot envelope contract: mean_over_time["band"] = per-year low/high
offsets, from a registered source when there is one, else the members' internal variability."""
import numpy as np
import pytest
import xarray as xr

import importlib

from backend.uncertainty import Z

go = importlib.import_module("backend.get_outputs")  # the module, not the get_outputs function


def _mot():
    s = lambda t: xr.DataArray(np.ones(len(t)), dims=("time",), coords={"time": np.asarray(t)})  # noqa: E731
    return {"historical_model": s(range(1850, 2015)), "no_sai": s(range(2035, 2101)), "with_sai": s(range(2035, 2101))}


BANDS = {"sigma_internal_mean": 0.1, "band_labels": {"internal": "±2σ internal variability of the multi-model mean, M=3"}}


def test_member_band_is_constant_plus_minus_z_sigma_over_every_series_year():
    band = go._line_band("tas", _mot(), BANDS)
    assert band["source"] == "members" and band["label"].endswith("M=3")
    assert int(band["low"].time.min()) == 1850 and int(band["high"].time.max()) == 2100
    np.testing.assert_allclose(band["high"].values, Z * 0.1)
    np.testing.assert_allclose(band["low"].values, -Z * 0.1)


def test_a_registered_source_replaces_the_member_band(monkeypatch):
    seen = {}

    def emulator_envelope(*, var, time, data_dir, **context):
        seen.update(var=var, n=len(time), data_dir=data_dir)
        low = xr.DataArray(np.linspace(-0.3, -0.1, len(time)), dims=("time",), coords={"time": time})
        return go.make_band(low, -low * 1.5, label="20-member range", source="emulator")

    monkeypatch.setitem(go.BAND_SOURCES, "tas", emulator_envelope)
    band = go._line_band("tas", _mot(), BANDS, data_dir="/data")
    assert band["source"] == "emulator" and seen == {"var": "tas", "n": len(go._series_time(_mot())), "data_dir": "/data"}
    # asymmetric offsets pass through untouched
    assert float(band["high"][0]) == pytest.approx(0.45)


def test_a_source_returning_none_falls_back_to_the_members(monkeypatch):
    monkeypatch.setitem(go.BAND_SOURCES, "tas", lambda **_: None)
    assert go._line_band("tas", _mot(), BANDS)["source"] == "members"


def test_no_band_without_members_or_source():
    assert go._line_band("tas", _mot(), None) is None
