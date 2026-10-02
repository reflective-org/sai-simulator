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


# --- the InVERT source registered by backend.get_outputs ---------------------------------

INVERT_VARS = ("tas", "tasmin", "tasmax", "pr", "p-e")


def _template():
    lat, lon = np.linspace(-89, 89, 4), np.linspace(0, 360, 6, endpoint=False)
    return xr.DataArray(np.zeros((1, 4, 6)), dims=("time", "lat", "lon"), coords={"time": [2015], "lat": lat, "lon": lon})


def test_invert_source_is_registered_for_its_variables_only():
    for var in INVERT_VARS:
        assert go.BAND_SOURCES[var] is go._invert_band
    for var in ("icefrac", "exposure_above_35", "tas_above_35"):
        assert var not in go.BAND_SOURCES


def test_invert_band_is_the_members_range_labelled_natural_variability(monkeypatch):
    time = go._series_time(_mot())
    members = xr.DataArray(
        np.array([[-1.0, 0.5], [2.0, -0.5], [0.0, 1.5]]),
        dims=("ensemble", "time"),
        coords={"ensemble": [0, 1, 2], "time": [1851, 2100]},
    )
    calls = {}

    def fake_area_mean(var, data_dir, cache_dir, *, template, region_mask=None):
        calls.update(var=var, cache_dir=cache_dir, regional=region_mask is not None)
        return members

    monkeypatch.setattr(go, "get_area_mean_variability", fake_area_mean)
    band = go._line_band(
        "p-e", _mot(), BANDS,
        data_dir="/data", cache_dir="/cache", regional_map=_template(), spatial_gdf=None, spatial_item=None,
    )
    # p-e borrows the pr envelope; the request path got the cache dir; global mode -> no mask.
    assert calls == {"var": "pr", "cache_dir": "/cache", "regional": False}
    assert band["source"] == "invert" and band["label"].startswith("Natural variability")
    # Lowest / highest member per year, reindexed (nearest) onto every series year.
    assert float(band["low"].sel(time=1850)) == -1.0 and float(band["high"].sel(time=1850)) == 2.0
    assert float(band["low"].sel(time=2100)) == -0.5 and float(band["high"].sel(time=2100)) == 1.5
    assert set(band["low"].time.values) == set(time)
    # 1850 (before InVERT's 1851 start) reuses 1851's offsets.
    assert float(band["low"].sel(time=1850)) == float(band["low"].sel(time=1851))


def test_invert_band_falls_back_to_the_members_without_a_cache(monkeypatch):
    monkeypatch.setattr(go, "get_area_mean_variability", lambda *a, **k: None)
    band = go._line_band(
        "tas", _mot(), BANDS,
        data_dir="/data", cache_dir="/cache", regional_map=_template(), spatial_gdf=None, spatial_item=None,
    )
    assert band["source"] == "members"


def test_invert_band_without_context_or_caches_falls_back_to_the_members(monkeypatch):
    # No data_dir / cache_dir / template in the call: nothing to look up.
    assert go._line_band("tas", _mot(), BANDS)["source"] == "members"
    # Context present but no valid cache.
    monkeypatch.setattr(go, "get_area_mean_variability", lambda *a, **k: None)
    band = go._line_band(
        "tas", _mot(), BANDS,
        data_dir="/d", cache_dir="/c", regional_map=_template(), spatial_gdf=None, spatial_item=None,
    )
    assert band["source"] == "members"


def test_invert_band_regional_mode_passes_the_region_mask(monkeypatch):
    seen = {}

    def fake(var, data_dir, cache_dir, *, template, region_mask=None):
        seen["mask"] = region_mask
        members = xr.DataArray(
            np.zeros((2, 2)), dims=("ensemble", "time"), coords={"ensemble": [0, 1], "time": [1851, 2100]}
        )
        return members

    monkeypatch.setattr(go, "get_area_mean_variability", fake)
    monkeypatch.setattr(go, "create_mask", lambda da, region: xr.ones_like(da, dtype=bool))
    gdf = type("GDF", (), {"name": None, "__getitem__": lambda self, key: "region"})()
    band = go._line_band(
        "tas", _mot(), BANDS,
        data_dir="/d", cache_dir="/c", regional_map=_template(), spatial_gdf=gdf, spatial_item="N.Europe",
    )
    assert band["source"] == "invert"
    assert seen["mask"] is not None and seen["mask"].dims == ("lat", "lon")
