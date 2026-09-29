"""Unit tests for the anchor machinery in :mod:`scripts.fit_delta`.

Covers the ``ANCHORS`` table (the three GAUSS cooling targets plus the MIROC
G6-1.5K / G6-0.5K runs, each paired with its own model's baseline), the
cos-weighted global mean, the singleton model/ssp squeeze, and — most
importantly — the ``_load_anchor_2050_2069`` behaviour that silently drops an
anchor whose files are absent (the mechanism that lets a variable fall back to
a GAUSS-only fit when another model doesn't provide it).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import scripts.fit_delta as fd


def _save_yearly_nc(path: Path, var: str, value: float) -> None:
    """Write a tiny ``(time, lat, lon)`` NetCDF of constant ``value`` spanning 2048-2071 (annual)."""
    time = pd.date_range("2048-01-01", "2071-01-01", freq="YS")
    data = np.full((len(time), 1, 1), value, dtype=float)
    ds = xr.Dataset(
        {var: (["time", "lat", "lon"], data)},
        coords={"time": time, "lat": [0.0], "lon": [0.0]},
    )
    ds.to_netcdf(path)


def test_anchors_cover_gauss_and_miroc() -> None:
    names = [a.name for a in fd.ANCHORS]
    assert names[:3] == ["gauss-1.5", "gauss-1.0", "gauss-0.5"]
    assert "g6-1.5-miroc" in names
    assert "g6-0.5-miroc" in names


def test_miroc_anchors_point_at_miroc_files() -> None:
    miroc_anchors = [a for a in fd.ANCHORS if a.name.endswith("-miroc")]
    assert len(miroc_anchors) == 2
    for anchor in miroc_anchors:
        assert anchor.exp_file.endswith("-miroc.nc")
        assert anchor.baseline_file.endswith("-miroc.nc")


def test_gauss_anchors_share_one_baseline() -> None:
    gauss = [a for a in fd.ANCHORS if a.name.startswith("gauss-")]
    assert {a.baseline_file for a in gauss} == {"output_gauss-baseline.nc"}


def test_miroc_anchors_share_one_miroc_baseline() -> None:
    miroc = [a for a in fd.ANCHORS if a.name.endswith("-miroc")]
    assert {a.baseline_file for a in miroc} == {"output_gauss-baseline-miroc.nc"}


def test_anchor_models_match_their_files() -> None:
    for anchor in fd.ANCHORS:
        if anchor.name.startswith("gauss-"):
            assert anchor.model == "CESM2-WACCM"
        elif anchor.name.endswith("-miroc"):
            assert anchor.model == "MIROC-ES2H"


def _write_anchor_pair(var_dir, tas_dir, anchor, tas_exp, tas_base, var_exp, var_base, var):
    _save_yearly_nc(tas_dir / anchor.exp_file, "tas", tas_exp)
    _save_yearly_nc(tas_dir / anchor.baseline_file, "tas", tas_base)
    _save_yearly_nc(var_dir / anchor.exp_file, var, var_exp)
    _save_yearly_nc(var_dir / anchor.baseline_file, var, var_base)


def test_gridded_fit_pools_all_anchors_across_models(tmp_path, monkeypatch) -> None:
    """One pooled through-origin slope over every anchor, regardless of model.

    Anchors: (x=-1, y=+2), (x=-2, y=+2) from MODEL-A and (x=-1, y=-1) from
    MODEL-B. Pooled slope = sum(x*y)/sum(x^2) = (-2 - 4 + 1)/(1 + 4 + 1) = -5/6.
    """
    anchors = [
        fd.Anchor("a-deep",    "output_a-deep.nc",    "output_a-base.nc", "MODEL-A"),
        fd.Anchor("a-shallow", "output_a-shallow.nc", "output_a-base.nc", "MODEL-A"),
        fd.Anchor("b-only",    "output_b-only.nc",    "output_b-base.nc", "MODEL-B"),
    ]
    monkeypatch.setattr(fd, "ANCHORS", anchors)

    var_dir = tmp_path / "pr"
    var_dir.mkdir()
    tas_dir = tmp_path / "tas"
    tas_dir.mkdir()
    _write_anchor_pair(var_dir, tas_dir, anchors[0], 2.0, 3.0, 5.0, 3.0, "pr")
    _save_yearly_nc(tas_dir / anchors[1].exp_file, "tas", 1.0)
    _save_yearly_nc(var_dir / anchors[1].exp_file, "pr", 5.0)
    _write_anchor_pair(var_dir, tas_dir, anchors[2], 2.0, 3.0, 2.0, 3.0, "pr")

    out = tmp_path / "interpolator.nc"
    fd._fit_delta_gridded("pr", var_dir, tas_dir, out)

    import xarray as xr
    ds = xr.open_dataset(out)
    assert "model" not in ds.dims  # pooled: a single slope, no model dim
    slope = ds["pr"].sel(features="slope")
    assert float(slope.item()) == pytest.approx(-5.0 / 6.0)


def test_gridded_fit_drops_missing_anchors_but_keeps_pooling(tmp_path, monkeypatch) -> None:
    """Anchors without files drop out; the rest still pool into one slope."""
    anchors = [
        fd.Anchor("a-1", "output_a-1.nc", "output_a-base.nc", "MODEL-A"),
        fd.Anchor("a-2", "output_a-2.nc", "output_a-base.nc", "MODEL-A"),
        fd.Anchor("b-only", "output_b-only.nc", "output_b-base.nc", "MODEL-B"),
    ]
    monkeypatch.setattr(fd, "ANCHORS", anchors)

    var_dir = tmp_path / "pr"
    var_dir.mkdir()
    tas_dir = tmp_path / "tas"
    tas_dir.mkdir()
    # Only MODEL-A's two anchors have files; MODEL-B's are absent.
    _write_anchor_pair(var_dir, tas_dir, anchors[0], 2.0, 3.0, 5.0, 3.0, "pr")
    _save_yearly_nc(tas_dir / anchors[1].exp_file, "tas", 1.0)
    _save_yearly_nc(var_dir / anchors[1].exp_file, "pr", 5.0)

    out = tmp_path / "interpolator.nc"
    fd._fit_delta_gridded("pr", var_dir, tas_dir, out)

    import xarray as xr
    slope = xr.open_dataset(out)["pr"].sel(features="slope")
    assert float(slope.item()) == pytest.approx(-1.2)  # (-2 - 4)/(1 + 4)


def test_gridded_fit_requires_two_anchors(tmp_path, monkeypatch) -> None:
    anchors = [
        fd.Anchor("a", "output_a.nc", "output_a-base.nc", "MODEL-A"),
        fd.Anchor("b", "missing-exp.nc", "missing-base.nc", "MODEL-B"),
    ]
    monkeypatch.setattr(fd, "ANCHORS", anchors)
    var_dir = tmp_path / "pr"
    var_dir.mkdir()
    tas_dir = tmp_path / "tas"
    tas_dir.mkdir()
    _write_anchor_pair(var_dir, tas_dir, anchors[0], 2.0, 3.0, 5.0, 3.0, "pr")
    with pytest.raises(ValueError, match="Need >=2 anchors"):
        fd._fit_delta_gridded("pr", var_dir, tas_dir, tmp_path / "interpolator.nc")


def test_squeeze_model_ssp_drops_singletons() -> None:
    da = xr.DataArray(
        np.ones((1, 1, 2, 2)), dims=["model", "ssp", "lat", "lon"]
    )
    out = fd._squeeze_model_ssp(da)
    assert "model" not in out.dims
    assert "ssp" not in out.dims
    assert out.dims == ("lat", "lon")


def test_squeeze_model_ssp_noop_without_those_dims() -> None:
    da = xr.DataArray(np.ones((2, 2)), dims=["lat", "lon"])
    out = fd._squeeze_model_ssp(da)
    assert out.dims == ("lat", "lon")


def test_global_mean_constant_field() -> None:
    ds = xr.Dataset(
        {"tas": (["lat", "lon"], np.full((3, 4), 5.0))},
        coords={"lat": [-30.0, 0.0, 30.0], "lon": [0.0, 90.0, 180.0, 270.0]},
    )
    assert float(fd._global_mean(ds)["tas"]) == pytest.approx(5.0)


def test_global_mean_is_cos_weighted() -> None:
    """cos(lat) weighting: a 0/10/0 lat profile averages to 10/2 = 5, not 10/3."""
    vals = np.array([[0.0, 0.0], [10.0, 10.0], [0.0, 0.0]])
    ds = xr.Dataset(
        {"tas": (["lat", "lon"], vals)},
        coords={"lat": [-60.0, 0.0, 60.0], "lon": [0.0, 180.0]},
    )
    assert float(fd._global_mean(ds)["tas"]) == pytest.approx(5.0)


def test_temporal_average_slices_then_means() -> None:
    time = pd.date_range("2048-01-01", "2071-01-01", freq="YS")
    da = xr.DataArray(
        np.arange(len(time), dtype=float), dims=["time"],
        coords={"time": time}, name="tas",
    )
    out = fd.temporal_average(da, "2050", "2069")
    expected = da.sel(time=slice("2050", "2069")).mean()
    assert float(out) == pytest.approx(float(expected))


def test_load_anchor_returns_none_when_files_missing(tmp_path) -> None:
    anchor = fd.Anchor("g", "output_gauss-1.5.nc", "output_gauss-baseline.nc", "CESM2-WACCM")
    assert fd._load_anchor_2050_2069(tmp_path, anchor) is None


def test_load_anchor_returns_means_when_present(tmp_path) -> None:
    _save_yearly_nc(tmp_path / "output_gauss-1.5.nc", "tas", 2.0)
    _save_yearly_nc(tmp_path / "output_gauss-baseline.nc", "tas", 1.0)
    anchor = fd.Anchor("g", "output_gauss-1.5.nc", "output_gauss-baseline.nc", "CESM2-WACCM")

    result = fd._load_anchor_2050_2069(tmp_path, anchor)
    assert result is not None
    exp, base = result
    assert float(exp["tas"].mean()) == pytest.approx(2.0)
    assert float(base["tas"].mean()) == pytest.approx(1.0)
    assert "time" not in exp.dims  # 2050-2069 collapsed to a time mean


def test_load_anchor_none_when_only_baseline_present(tmp_path) -> None:
    """Both files are required; a lone baseline (no experiment) still drops the anchor."""
    _save_yearly_nc(tmp_path / "output_gauss-baseline.nc", "tas", 1.0)
    anchor = fd.Anchor("g", "output_gauss-1.5.nc", "output_gauss-baseline.nc", "CESM2-WACCM")
    assert fd._load_anchor_2050_2069(tmp_path, anchor) is None
