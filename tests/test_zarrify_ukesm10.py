"""Unit tests for the pure aggregation helpers in ``scripts/zarrify_ukesm10.py``.

Synthetic 360-day-calendar data only — no network, no zarr writes. These pin
the store semantics that are easy to get silently wrong: absolute monthly
extremes (not means), threshold counts in native units, and start-of-month
time stamps.
"""

import numpy as np
import pytest
import xarray as xr

import scripts.zarrify_ukesm10 as zu


def daily(values_per_day, start="2015-01-01"):
    """(time, lat, lon) daily DataArray on a 360-day calendar, 1x1 grid."""
    n = len(values_per_day)
    time = xr.date_range(start, periods=n, freq="D", calendar="360_day", use_cftime=True)
    data = np.asarray(values_per_day, dtype="float64").reshape(n, 1, 1)
    return xr.DataArray(
        data, dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [0.0], "lon": [0.0]},
    )


class TestCalendarGuards:
    def test_accepts_360_day(self):
        zu.assert_360_day(daily([0.0] * 30).time)

    def test_rejects_gregorian(self):
        time = xr.date_range("2015-01-01", periods=30, freq="D", calendar="standard", use_cftime=True)
        da = xr.DataArray(np.zeros(30), dims="time", coords={"time": time})
        with pytest.raises(ValueError, match="360_day"):
            zu.assert_360_day(da.time)

    def test_month_starts_requires_whole_months(self):
        with pytest.raises(ValueError, match="whole number of months"):
            zu.month_starts(daily([0.0] * 31).time)

    def test_month_starts_are_day_one(self):
        starts = zu.month_starts(daily([0.0] * 60).time)
        assert len(starts) == 2
        assert all(t.day == 1 for t in starts)
        assert starts[0].month == 1 and starts[1].month == 2


class TestMonthlyCount:
    def test_counts_days_above_threshold_per_month(self):
        # month 1: 3 days above; month 2: 1 day above
        vals = [310.0] * 3 + [300.0] * 27 + [320.0] * 1 + [300.0] * 29
        counts = zu.monthly_count(daily(vals), 308.15, "above")
        assert counts.dtype == np.int64
        assert counts.values.ravel().tolist() == [3, 1]

    def test_counts_days_below_threshold(self):
        vals = [270.0] * 5 + [280.0] * 25 + [280.0] * 30
        counts = zu.monthly_count(daily(vals), 273.15, "below")
        assert counts.values.ravel().tolist() == [5, 0]

    def test_pr_threshold_is_in_native_units(self):
        # 10 mm/day in kg m-2 s-1; one wet day per month
        wet = 12.0 / 86400.0
        dry = 1.0 / 86400.0
        vals = [wet] + [dry] * 29 + [dry] * 30
        counts = zu.monthly_count(daily(vals), zu.PR_COUNTS["PRECT_above_10"][0], "above")
        assert counts.values.ravel().tolist() == [1, 0]

    def test_timestamps_are_month_starts(self):
        counts = zu.monthly_count(daily([0.0] * 60), 1.0, "above")
        assert all(t.day == 1 for t in counts.time.values)


class TestMonthlyExtreme:
    def test_max_is_absolute_not_mean(self):
        vals = [300.0] * 29 + [315.0] + [301.0] * 30
        mx = zu.monthly_extreme(daily(vals), "max")
        assert mx.values.ravel().tolist() == [315.0, 301.0]

    def test_min_is_absolute(self):
        vals = [280.0] * 15 + [260.0] + [280.0] * 14 + [275.0] * 30
        mn = zu.monthly_extreme(daily(vals), "min")
        assert mn.values.ravel().tolist() == [260.0, 275.0]


class TestConversionsAndTime:
    def test_to_icefrac_scales_fills_and_clips(self):
        pct = daily([50.0, np.nan, 120.0] + [0.0] * 27)
        frac = zu.to_icefrac(pct, "%")
        assert frac.dtype == np.float32
        assert frac.values.ravel()[:3].tolist() == [0.5, 0.0, 1.0]

    def test_normalize_monthly_time_floors_to_month_start(self):
        time = xr.date_range("2015-01-16", periods=3, freq="30D", calendar="360_day",
                             use_cftime=True)
        ds = xr.Dataset({"x": ("time", [1.0, 2.0, 3.0])}, coords={"time": time})
        out = zu.normalize_monthly_time(ds)
        assert all(t.day == 1 for t in out.time.values)
        assert [t.month for t in out.time.values] == [1, 2, 3]

    def test_expected_months(self):
        assert zu.expected_months("historical") == 165 * 12
        assert zu.expected_months("ssp245") == 86 * 12


class TestStackMembers:
    def test_stacks_in_declared_member_order(self):
        a = daily([1.0] * 30)
        b = daily([2.0] * 30)
        stacked = zu.stack_members({"r2i1p1f2": a, "r12i1p1f2": b})
        assert list(stacked.member.values) == ["r2i1p1f2", "r12i1p1f2"]
        assert stacked.dims == ("member", "time", "lat", "lon")
