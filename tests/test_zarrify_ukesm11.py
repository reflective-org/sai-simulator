"""Unit tests for the pure parts of ``scripts/zarrify_ukesm11.py``.

Pins the u-suite handling that is easy to get silently wrong: file picking
with the u-dp903 duplicate-evap trap, cell_methods-keyed (never positional)
splitting of the daily tas triple, the historical member-map contract, the
PARTIAL provenance note both ESGF-backed historical arrays carry, and the
cross-store member-label consistency the consumer's stitching relies on.
"""

import numpy as np
import pytest
import xarray as xr

import scripts.zarrify_ukesm11 as z11
from scripts.zarrify_common import to_icefrac


class TestPickFile:
    def test_single_candidate_needs_no_token(self):
        assert z11.pick_file(["a/tas_mon_x.nc"], "_SSP245_") == "a/tas_mon_x.nc"

    def test_token_resolves_the_dp903_duplicate_evap(self):
        paths = [
            "SSP245/r3i1p1f1/evap_mon_UKESM1-1-LL_SSP245_u-dp903_201501-210012.nc",
            "SSP245/r3i1p1f1/evap_mon_UKESM1-1-LL_u-dp903_200001-210012.nc",
        ]
        assert z11.pick_file(paths, "_SSP245_") == paths[0]

    def test_ambiguity_is_an_error_not_a_guess(self):
        with pytest.raises(ValueError, match="cannot pick uniquely"):
            z11.pick_file(["a_SSP245_1.nc", "b_SSP245_2.nc"], "_SSP245_")


class TestSplitTasTriple:
    @staticmethod
    def _triple(order):
        time = xr.date_range("2015-01-01", periods=2, freq="D",
                             calendar="360_day", use_cftime=True)
        ds = xr.Dataset(coords={"time": time})
        for var, method in order:
            ds[var] = xr.DataArray([0.0, 1.0], dims="time",
                                   attrs={"cell_methods": f"time: {method} (interval: 1 hour)"})
        return ds

    def test_keys_by_cell_methods_not_position(self):
        # historical files put maximum first; ssp245 files differ — both must work
        for order in (
            [("air_temperature", "maximum"), ("air_temperature_0", "minimum"),
             ("air_temperature_1", "mean")],
            [("air_temperature", "mean"), ("air_temperature_0", "minimum"),
             ("air_temperature_1", "maximum")],
        ):
            triple = z11.split_tas_triple(self._triple(order))
            by_var = {k: v.name for k, v in triple.items()}
            expected = {{"maximum": "max", "minimum": "min", "mean": "mean"}[m]: v
                        for v, m in order}
            assert by_var == expected

    def test_missing_field_is_an_error(self):
        ds = self._triple([("air_temperature", "mean"), ("air_temperature_0", "minimum")])
        with pytest.raises(ValueError, match="missing.*max"):
            z11.split_tas_triple(ds)

    def test_duplicate_method_is_an_error(self):
        ds = self._triple([("a", "mean"), ("b", "mean"), ("c", "maximum")])
        with pytest.raises(ValueError, match="two fields claim"):
            z11.split_tas_triple(ds)


class TestHistoricalMap:
    def test_parses_full_map(self):
        # The pairing the shipped store was built with; kept in sync with the
        # docstring and --historical-map help so nobody copies a swapped example.
        m = z11.parse_historical_map("r2=u-bz502,r3=u-by791,r12=u-bz897")
        assert m == {"r2": "historical/u-bz502", "r3": "historical/u-by791",
                     "r12": "historical/u-bz897"}

    def test_partial_or_wrong_members_rejected(self):
        with pytest.raises(ValueError, match="must cover exactly"):
            z11.parse_historical_map("r2=u-by791,r3=u-bz502")
        with pytest.raises(ValueError, match="must cover exactly"):
            z11.parse_historical_map("r1=u-by791,r3=u-bz502,r12=u-bz897")


class TestExperimentConfig:
    def test_store_names_match_the_repo_stubs(self):
        stores = {cfg["store"] for cfg in z11.EXPERIMENTS.values()}
        assert stores == {"UKESM1-1-LL_baseline.zarr", "UKESM1-1-LL_G6-1.5K-SAI.zarr",
                          "UKESM1-1-LL_historical.zarr"}

    def test_spans(self):
        assert z11.expected_months("baseline") == 86 * 12
        assert z11.expected_months("g6-1.5k-sai") == 51 * 12
        assert z11.expected_months("historical") == 165 * 12

    def test_member_labels_identical_across_experiments(self):
        # The consumer stitches historical into the baseline by member label.
        assert z11.MEMBERS == ["r2", "r3", "r12"]
        for name, cfg in z11.EXPERIMENTS.items():
            if cfg["dirs"] is not None:
                assert set(cfg["dirs"]) == set(z11.MEMBERS), name

    def test_sai_dirs_point_at_the_hub_layout(self):
        assert z11.EXPERIMENTS["g6-1.5k-sai"]["dirs"]["r12"] == "G6-1p5K-SAI/r12i1p1f1"


class TestToIcefracUnits:
    @staticmethod
    def _da(values):
        time = xr.date_range("2015-01-01", periods=len(values), freq="D",
                             calendar="360_day", use_cftime=True)
        return xr.DataArray(values, dims="time", coords={"time": time})

    def test_fraction_units_are_not_rescaled(self):
        # MACLOUD siconca is already [0,1] (units '1').
        out = to_icefrac(self._da([0.5, np.nan, 1.2]), "1")
        assert out.values.tolist() == [0.5, 0.0, 1.0]

    def test_percent_units_are_scaled(self):
        assert to_icefrac(self._da([50.0, np.nan, 120.0]), "%").values.tolist() == [0.5, 0.0, 1.0]

    def test_unmasked_land_fill_becomes_zero_not_percent_detection(self):
        # The bug that shipped a land mask as ICEFRAC: land held 9.97e36 with
        # no _FillValue attr, and a max-based %-vs-fraction guess saw the fill.
        out = to_icefrac(self._da([0.9, 9.969e36, 0.3]), "1")
        assert out.values.tolist() == [pytest.approx(0.9, abs=1e-6), 0.0, pytest.approx(0.3, abs=1e-6)]

    def test_units_and_data_must_agree(self):
        with pytest.raises(ValueError, match="inconsistent with units"):
            to_icefrac(self._da([50.0, 80.0]), "1")
        with pytest.raises(ValueError, match="must be"):
            to_icefrac(self._da([0.5]), "kg")

    def test_all_fill_input_is_an_error(self):
        with pytest.raises(ValueError, match="entirely NaN"):
            to_icefrac(self._da([9.969e36, np.nan]), "1")


class TestHistoricalMapStrictness:
    def test_duplicate_suite_rejected(self):
        with pytest.raises(ValueError, match="one suite to several members"):
            z11.parse_historical_map("r2=u-by791,r3=u-by791,r12=u-bz897")

    def test_duplicate_member_rejected(self):
        with pytest.raises(ValueError, match="listed twice"):
            z11.parse_historical_map("r2=u-a,r2=u-b,r3=u-c,r12=u-d")

    def test_whitespace_tolerated(self):
        m = z11.parse_historical_map(" r2=u-bz502, r3=u-by791 ,r12=u-bz897")
        assert m["r3"] == "historical/u-by791"


class TestDailySpan:
    def test_accepts_exact_span(self):
        time = xr.date_range("2034-01-01", periods=51 * 360, freq="D",
                             calendar="360_day", use_cftime=True)
        z11.assert_daily_span(xr.DataArray(time, dims="time").time, (2034, 2084), "x")

    def test_rejects_filename_claimed_span(self):
        # SAI files are NAMED 2015-2100 but hold 2034-2084; the axis must win.
        time = xr.date_range("2034-01-01", periods=51 * 360, freq="D",
                             calendar="360_day", use_cftime=True)
        with pytest.raises(ValueError, match="starts 2034"):
            z11.assert_daily_span(xr.DataArray(time, dims="time").time, (2015, 2100), "x")


class TestTripleSamplingInterval:
    def test_six_hourly_mean_is_rejected(self):
        time = xr.date_range("2015-01-01", periods=2, freq="D", calendar="360_day", use_cftime=True)
        ds = xr.Dataset(coords={"time": time})
        for var, cm in (("a", "time: maximum (interval: 1 hour)"),
                        ("b", "time: minimum (interval: 1 hour)"),
                        ("c", "time: mean (interval: 6 hour)")):
            ds[var] = xr.DataArray([0.0, 1.0], dims="time", attrs={"cell_methods": cm})
        with pytest.raises(ValueError, match="not 1-hourly"):
            z11.split_tas_triple(ds)


class TestPartialNote:
    """The note is the provenance claim ``verify_store`` enforces, so it has to
    name the members truthfully — an array real on r3 alone must not read as
    though it covers the store."""

    def test_names_both_the_covered_and_the_missing_members(self):
        note = z11.partial_note(["r3"], "Body.")
        assert note.startswith("PARTIAL — real for member(s) r3; all-NaN for r2, r12.")
        assert "Body." in note

    def test_full_coverage_says_none_rather_than_an_empty_list(self):
        assert "; all-NaN for none." in z11.partial_note(list(z11.MEMBERS), "Body.")

    def test_missing_members_follow_store_order_not_the_caller_s(self):
        # MEMBERS order is the store's member axis; r12 sorts before r2
        # alphabetically, and mislabelling which member is NaN is exactly the
        # kind of silent provenance error the note exists to prevent.
        assert "all-NaN for r2, r12" in z11.partial_note(["r3"], "Body.")
        assert "all-NaN for r3" in z11.partial_note(["r2", "r12"], "Body.")
