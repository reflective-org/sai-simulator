"""Unit tests for ``scripts/zarrify_common.py``: alignment guards and store I/O.

These pin the failure modes the adversarial review found in the first
producer version — silent outer-join NaN padding, 30-day blocks mislabelled as
months when a daily axis starts mid-month, and inherited/incorrect metadata
reaching the store. The write→read round trip runs on an in-memory zarr store.
"""

import numpy as np
import pytest
import xarray as xr

import scripts.zarrify_common as zc


def daily_axis(start, ndays):
    return xr.date_range(start, periods=ndays, freq="D", calendar="360_day", use_cftime=True)


def field(time, value=1.0, lat=(0.0, 1.0), lon=(0.0, 1.0), name="x"):
    data = np.full((len(time), len(lat), len(lon)), value, dtype="float32")
    return xr.DataArray(data, dims=("time", "lat", "lon"),
                        coords={"time": time, "lat": list(lat), "lon": list(lon)}, name=name)


class TestDailyAxisGuard:
    def test_mid_month_start_rejected(self):
        with pytest.raises(ValueError, match="day 1 of a month"):
            zc.month_starts(field(daily_axis("2015-01-16", 60)).time)

    def test_gap_rejected(self):
        t = daily_axis("2015-01-01", 60)
        gapped = xr.CFTimeIndex(list(t[:29]) + list(t[30:60]) + [t[59] + (t[1] - t[0])])
        with pytest.raises(ValueError, match="not contiguous"):
            zc.month_starts(xr.DataArray(gapped, dims="time").time)

    def test_clean_axis_labels_calendar_months(self):
        starts = zc.month_starts(field(daily_axis("2034-01-01", 90)).time)
        assert [(s.year, s.month, s.day) for s in starts] == [(2034, 1, 1), (2034, 2, 1), (2034, 3, 1)]

    def test_midday_stamps_on_day_one_are_accepted(self):
        # UM daily means are stamped 12:00; only the DAY matters for 30-day blocks.
        starts = zc.month_starts(field(daily_axis("2015-01-01 12:00", 60)).time)
        assert [(s.month, s.day, s.hour) for s in starts] == [(1, 1, 0), (2, 1, 0)]


class TestExactAlignment:
    def test_stack_members_rejects_shifted_member(self):
        a = field(daily_axis("2015-01-01", 30))
        b = field(daily_axis("2015-01-02", 30))
        with pytest.raises(ValueError):
            zc.stack_members({"r2": a, "r3": b})

    def test_stack_members_rejects_grid_mismatch(self):
        a = field(daily_axis("2015-01-01", 30), lat=(0.0, 1.0))
        b = field(daily_axis("2015-01-01", 30), lat=(0.0, 1.0000001))
        with pytest.raises(ValueError):
            zc.stack_members({"r2": a, "r3": b})

    def test_exact_dataset_rejects_misaligned_variables(self):
        a = field(daily_axis("2015-01-01", 30))
        b = field(daily_axis("2015-02-01", 30))
        with pytest.raises(ValueError):
            zc.exact_dataset({"A": a, "B": b})

    def test_exact_dataset_keeps_aligned_variables(self):
        t = daily_axis("2015-01-01", 30)
        ds = zc.exact_dataset({"A": field(t, 1.0), "B": field(t, 2.0)})
        assert set(ds.data_vars) == {"A", "B"}
        assert ds.sizes["time"] == 30


class TestGroupAxes:
    def _monthly(self, start, months):
        t = [s.replace(day=1) for s in xr.date_range(start, periods=months, freq="30D",
                                                     calendar="360_day", use_cftime=True)]
        return xr.Dataset({"TREFHT": field(xr.CFTimeIndex(t), 280.0, lat=range(144), lon=range(192))})

    def test_span_and_grid_enforced(self):
        zc.assert_group_axes(self._monthly("2034-01-01", 612), (2034, 2084), 144, 192)
        with pytest.raises(ValueError, match="time="):
            zc.assert_group_axes(self._monthly("2034-01-01", 600), (2034, 2084), 144, 192)
        with pytest.raises(ValueError, match="starts"):
            zc.assert_group_axes(self._monthly("2035-01-01", 612), (2034, 2084), 144, 192)
        with pytest.raises(ValueError, match="grid"):
            zc.assert_group_axes(self._monthly("2034-01-01", 612), (2034, 2084), 128, 256)


class TestStoreRoundTrip:
    @staticmethod
    def _group():
        t = xr.CFTimeIndex([s.replace(day=1) for s in xr.date_range(
            "2015-01-01", periods=24, freq="30D", calendar="360_day", use_cftime=True)])
        lat, lon = np.linspace(-60, 60, 4), np.linspace(0, 315, 8)
        base = field(t, 280.0, lat=lat, lon=lon)
        counts = xr.DataArray(np.full(base.shape, 3, dtype="int64"), dims=base.dims, coords=base.coords)
        # Inherited junk that must NOT reach the store: source units on a
        # converted field, a stray UM coord, a cell_methods on a count.
        base.attrs = {"units": "kg m-2 s-1", "standard_name": "air_temperature", "note": "keep me"}
        base = base.assign_coords(height=1.5)
        counts.attrs = {"cell_methods": "time: mean (interval: 1 hour)", "units": "K"}
        ds = xr.Dataset({"TREFHT": zc.stack_members({"r2": base, "r3": base}),
                         "TREFHT_above_35": zc.stack_members({"r2": counts, "r3": counts})})
        return ds

    def test_write_verify_round_trip_on_memory_store(self):
        from zarr.storage import MemoryStore

        target = MemoryStore()
        groups = {"Mon/atmos_2d": ["TREFHT"], "Mon/atmos_2d_derived": ["TREFHT_above_35"]}
        ds = self._group()
        zc.write_group("t.zarr", "Mon/atmos_2d", ds[["TREFHT"]],
                       {"member": 1, "time": 12, "lat": 4, "lon": 8}, target=target)
        zc.write_group("t.zarr", "Mon/atmos_2d_derived", ds[["TREFHT_above_35"]],
                       {"member": 1, "time": 12, "lat": 4, "lon": 8}, target=target)
        zc.finalize_store(None, "t.zarr", target=target)
        failures = zc.verify_store("t.zarr", groups, 2, (2015, 2016), 4, 8, target=target)
        assert failures == []
        back = xr.open_zarr(target, group="Mon/atmos_2d", consolidated=True)
        assert back["TREFHT"].attrs["units"] == "K"          # store attrs, not source attrs
        assert back["TREFHT"].attrs["note"] == "keep me"      # provenance survives
        assert "standard_name" not in back["TREFHT"].attrs
        assert "height" not in back.coords                    # stray UM coord dropped
        assert back.time.values[0].calendar == "360_day"
        # Counts must read back as plain integers with xarray defaults — a CF
        # time unit like "days" would make them decode as timedelta64.
        counts = xr.open_zarr(target, group="Mon/atmos_2d_derived", consolidated=True)
        assert counts["TREFHT_above_35"].dtype == np.int64
        assert "units" not in counts["TREFHT_above_35"].attrs

    def test_verify_catches_nan_and_wrong_span(self):
        from zarr.storage import MemoryStore

        target = MemoryStore()
        ds = self._group()[["TREFHT"]]
        ds["TREFHT"][0, 3, 0, 0] = np.nan  # a padded/unfilled cell in a non-placeholder
        zc.write_group("t.zarr", "Mon/atmos_2d", ds, {"member": 1, "time": 12, "lat": 4, "lon": 8},
                       target=target)
        zc.finalize_store(None, "t.zarr", target=target)
        failures = zc.verify_store("t.zarr", {"Mon/atmos_2d": ["TREFHT"]}, 2, (2015, 2016), 4, 8,
                                   target=target)
        assert any("NaN" in f for f in failures)
        failures = zc.verify_store("t.zarr", {"Mon/atmos_2d": ["TREFHT"]}, 2, (2015, 2017), 4, 8,
                                   target=target)
        assert any("sizes" in f for f in failures)

    def test_placeholder_nan_is_allowed(self):
        from zarr.storage import MemoryStore

        target = MemoryStore()
        ds = self._group()[["TREFHT"]].rename({"TREFHT": "QFLX"})
        ds["QFLX"][:] = np.nan
        ds["QFLX"].attrs["note"] = "PLACEHOLDER — all NaN: no source"
        zc.write_group("t.zarr", "Mon/atmos_2d", ds, {"member": 1, "time": 12, "lat": 4, "lon": 8},
                       target=target)
        zc.finalize_store(None, "t.zarr", target=target)
        assert zc.verify_store("t.zarr", {"Mon/atmos_2d": ["QFLX"]}, 2, (2015, 2016), 4, 8,
                               target=target) == []


class TestFillMasking:
    def test_default_netcdf_fill_and_um_rmdi_become_nan(self):
        da = xr.DataArray([0.5, 9.969e36, zc.UM_RMDI, 0.2])
        out = zc.mask_fill_values(da)
        assert np.isnan(out.values[1]) and np.isnan(out.values[2])
        assert out.values[0] == 0.5


class TestPartialAndPlaceholderNotes:
    """PLACEHOLDER and PARTIAL are provenance claims, so verify_store checks the
    array actually matches the claim — a note that has gone stale is a failure."""

    @staticmethod
    def _store(note, fill):
        from zarr.storage import MemoryStore
        t = xr.CFTimeIndex([s.replace(day=1) for s in xr.date_range(
            "2015-01-01", periods=24, freq="30D", calendar="360_day", use_cftime=True)])
        lat, lon = np.linspace(-60, 60, 4), np.linspace(0, 315, 8)
        base = field(t, 280.0, lat=lat, lon=lon)
        ds = xr.Dataset({"ICEFRAC": zc.stack_members({"r2": base, "r3": base})})
        ds["ICEFRAC"] = fill(ds["ICEFRAC"])
        ds["ICEFRAC"].attrs["note"] = note
        target = MemoryStore()
        zc.write_group("t.zarr", "Mon/atmos_2d", ds,
                       {"member": 1, "time": 12, "lat": 4, "lon": 8}, target=target)
        zc.finalize_store(None, "t.zarr", target=target)
        return zc.verify_store("t.zarr", {"Mon/atmos_2d": ["ICEFRAC"]}, 2, (2015, 2016),
                               4, 8, target=target)

    @staticmethod
    def _one_member_nan(da):
        return da.where(da.member != "r3")

    def test_partial_accepts_one_member_present_and_one_missing(self):
        assert self._store("PARTIAL — real for r2, all-NaN for r3", self._one_member_nan) == []

    def test_partial_rejects_a_fully_populated_array(self):
        out = self._store("PARTIAL — real for r2, all-NaN for r3", lambda da: da)
        assert any("nothing is NaN" in f for f in out)

    def test_partial_rejects_an_entirely_empty_array(self):
        out = self._store("PARTIAL — real for r2", lambda da: da.where(False))
        assert any("everything is NaN" in f for f in out)

    def test_placeholder_rejects_an_array_that_has_real_data(self):
        out = self._store("PLACEHOLDER — all NaN: no source", self._one_member_nan)
        assert any("declared PLACEHOLDER" in f for f in out)
