"""Unit tests for the Zarr-vs-NetCDF dispatch in :func:`backend.utils.open_xarray_datasets`.

The migration routes a path to :func:`xarray.open_zarr` when it ends in ``.zarr``
(optionally with a trailing slash), contains ``.zarr/``, or starts with a remote
URL scheme (``s3://`` / ``gs://`` / ``http://`` / ``https://``); everything else
goes to :func:`xarray.open_dataset`. These tests stub both openers so the routing
and kwarg-forwarding are exercised without touching the network or real files.
"""
import xarray as xr
import pytest

import backend.utils as utils


@pytest.fixture
def record_open(monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    """Stub ``xr.open_zarr`` / ``xr.open_dataset`` to record calls and return a tiny Dataset.

    Returns a dict with ``"zarr"`` and ``"netcdf"`` lists of ``(path, kwargs)`` tuples
    so a test can assert which branch ran and what was forwarded.
    """
    calls: dict[str, list] = {"zarr": [], "netcdf": []}

    def fake_open_zarr(path: str, **kwargs: object) -> xr.Dataset:
        calls["zarr"].append((path, kwargs))
        return xr.Dataset({"tas": ("x", [1.0, 2.0])})

    def fake_open_dataset(path: object, **kwargs: object) -> xr.Dataset:
        calls["netcdf"].append((str(path), kwargs))
        return xr.Dataset({"tas": ("x", [1.0, 2.0])})

    monkeypatch.setattr(utils.xr, "open_zarr", fake_open_zarr)
    monkeypatch.setattr(utils.xr, "open_dataset", fake_open_dataset)
    return calls


@pytest.mark.parametrize(
    "path",
    [
        "store.zarr",
        "store.zarr/",
        "local/x.zarr/Mon/atmos_2d",
        "s3://bucket/store.zarr",
        "gs://bucket/store.zarr",
        "https://host/simulator-inputs/x.zarr",
        "https://host/simulator-inputs/store",  # remote scheme, no .zarr suffix
    ],
)
def test_dispatches_to_zarr(record_open: dict[str, list], path: str) -> None:
    utils.open_xarray_datasets(path)
    assert len(record_open["zarr"]) == 1
    assert record_open["netcdf"] == []


def test_dispatches_to_netcdf_for_existing_file(
    record_open: dict[str, list], tmp_path
) -> None:
    nc_path = tmp_path / "local.nc"
    nc_path.touch()  # only existence is checked before the (stubbed) open
    utils.open_xarray_datasets(str(nc_path))
    assert len(record_open["netcdf"]) == 1
    assert record_open["zarr"] == []


def test_missing_netcdf_raises_file_not_found(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        utils.open_xarray_datasets(str(tmp_path / "does_not_exist.nc"))


def test_zarr_forwards_optional_kwargs(record_open: dict[str, list]) -> None:
    utils.open_xarray_datasets(
        "x.zarr",
        group="Mon/atmos_2d",
        consolidated=True,
        storage_options={"anon": True},
    )
    _, kwargs = record_open["zarr"][0]
    assert kwargs["group"] == "Mon/atmos_2d"
    assert kwargs["consolidated"] is True
    assert kwargs["storage_options"] == {"anon": True}


def test_zarr_omits_optional_kwargs_when_absent(record_open: dict[str, list]) -> None:
    utils.open_xarray_datasets("x.zarr")
    _, kwargs = record_open["zarr"][0]
    assert "group" not in kwargs
    assert "consolidated" not in kwargs
    assert "storage_options" not in kwargs


def test_decode_times_false_is_forwarded(record_open: dict[str, list]) -> None:
    utils.open_xarray_datasets("x.zarr", decode_times=False)
    _, kwargs = record_open["zarr"][0]
    assert kwargs["decode_times"] is False
