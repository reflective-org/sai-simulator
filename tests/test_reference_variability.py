"""Unit tests for scripts.variability's reference-period scatter (the Welch test's reference noise)."""
import numpy as np
import xarray as xr

from scripts.variability import REFERENCE_YEARS, field_grid_variability, field_reference_variability


def _members(years, n_members=3, slope=0.05, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.asarray(years, dtype=float)
    data = slope * (t - t[0])[None, :, None, None] + rng.normal(0, noise, (n_members, len(t), 4, 5))
    return xr.DataArray(
        data, dims=("member", "time", "lat", "lon"),
        coords={"member": [f"r{i}" for i in range(n_members)], "time": np.asarray(years),
                "lat": np.linspace(-60, 60, 4), "lon": np.linspace(0, 288, 5)},
    )


def test_reference_scatter_removes_the_trend_and_recovers_the_noise():
    members = _members(range(1950, 2030), n_members=50, slope=0.05, noise=0.3)
    ds = field_reference_variability(members, "tas")
    assert ds["tas"].attrs["n_years"] == REFERENCE_YEARS[1] - REFERENCE_YEARS[0] + 1
    assert ds["tas"].attrs["n_members"] == 50
    assert ds["tas"].attrs["dof"] == 50 * 20 - 2 * 50
    # a 0.05/yr trend over 20 years would add ~0.29 of spread if it were not removed
    np.testing.assert_allclose(ds["tas"].values, 0.3, rtol=0.05)


def test_members_not_covering_the_reference_period_give_none():
    assert field_reference_variability(_members(range(2015, 2100)), "tas") is None


def test_grid_sigma_records_its_sample_count():
    ds = field_grid_variability(_members(range(2015, 2100)), "tas")
    assert ds["tas"].attrs["n_members"] == 3 and ds["tas"].attrs["n_years"] == 15
    assert ds["tas"].attrs["dof"] == 44


def _split_members(n_each=3, noise=0.3):
    """CESM's tasmax layout: three historical-only members and three projection-only ones."""
    hist = _members(range(1950, 2015), n_members=n_each, noise=noise, seed=1)
    proj = _members(range(2015, 2100), n_members=n_each, noise=noise, seed=2)
    hist = hist.assign_coords(member=[f"h{i}" for i in range(n_each)])
    proj = proj.assign_coords(member=[f"p{i}" for i in range(n_each)])
    return xr.concat([hist, proj], dim="member", join="outer")


def test_reference_scatter_counts_only_members_with_data_in_the_window():
    ds = field_reference_variability(_split_members(n_each=40), "tas")
    assert ds["tas"].attrs["n_members"] == 40           # the historical-only members
    assert ds["tas"].attrs["dof"] == 40 * 20 - 2 * 40
    np.testing.assert_allclose(ds["tas"].values, 0.3, rtol=0.06)  # not low by sqrt(2)


def test_grid_sigma_dof_counts_only_members_with_data_in_the_window():
    ds = field_grid_variability(_split_members(n_each=3), "tas")
    assert ds["tas"].attrs["n_members"] == 3            # the projection-only members
    assert ds["tas"].attrs["dof"] == 3 * 15 - 1
