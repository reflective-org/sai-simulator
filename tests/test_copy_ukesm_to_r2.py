"""Unit tests for the planning core of ``scripts/copy_ukesm_to_r2.py``.

Only the pure diff/filter logic is exercised — no boto3. That logic is where a
wrong decision either silently re-transfers terabytes (S3 egress dollars) or
skips a file the zarrification step needs, so it gets deterministic coverage.
"""

import scripts.copy_ukesm_to_r2 as cur


class TestEnsureDirPrefix:
    def test_adds_trailing_slash(self):
        assert cur.ensure_dir_prefix("UKESM1-1") == "UKESM1-1/"

    def test_keeps_existing_slash(self):
        assert cur.ensure_dir_prefix("UKESM1-1/") == "UKESM1-1/"

    def test_collapses_duplicate_slashes(self):
        assert cur.ensure_dir_prefix("UKESM1-1//") == "UKESM1-1/"

    def test_empty_prefix_means_whole_bucket(self):
        assert cur.ensure_dir_prefix("") == ""


class TestMatchesFilters:
    def test_no_filters_matches_everything(self):
        assert cur.matches_filters("SSP245/r12i1p1f1/tas_mon.nc", (), ())

    def test_only_requires_a_substring_hit(self):
        assert cur.matches_filters("SSP245/r12i1p1f1/tas_mon.nc", ("SSP245",), ())
        assert not cur.matches_filters("G6-1p5K-SAI/r12i1p1f2/tas_mon.nc", ("SSP245",), ())

    def test_only_is_an_or_over_substrings(self):
        assert cur.matches_filters("G6-1p5K-SAI/tas.nc", ("SSP245", "G6-"), ())

    def test_exclude_drops_matches(self):
        assert not cur.matches_filters("SSP245/o3_mmr_mon.nc", (), ("o3_",))

    def test_exclude_wins_over_only(self):
        assert not cur.matches_filters("SSP245/o3_mmr_mon.nc", ("SSP245",), ("o3_",))

    def test_ta_exclude_does_not_hit_tas(self):
        # The 3D fields (ta_all_*, ta_85lev_*) are excluded with "ta_", which must
        # not also drop the 2m air temperature files (tas_*).
        assert cur.matches_filters("SSP245/r12i1p1f1/tas_mon_UKESM1-1-LL.nc", (), ("ta_",))
        assert not cur.matches_filters("SSP245/r12i1p1f1/ta_all_mon_UKESM1-1-LL.nc", (), ("ta_",))


class TestPlanTransfers:
    def test_missing_keys_are_copied_and_same_size_skipped(self):
        src = {"a.nc": 10, "b.nc": 20}
        dest = {"a.nc": 10}
        plan = cur.plan_transfers(src, dest)
        assert plan.to_copy == ["b.nc"]
        assert plan.mismatched == []
        assert plan.up_to_date == 1
        assert plan.bytes_to_copy == 20

    def test_size_mismatch_is_recopied_and_reported(self):
        src = {"a.nc": 10}
        dest = {"a.nc": 7}
        plan = cur.plan_transfers(src, dest)
        assert plan.to_copy == ["a.nc"]
        assert plan.mismatched == ["a.nc"]
        assert plan.up_to_date == 0
        assert plan.bytes_to_copy == 10

    def test_placeholder_directory_keys_are_ignored(self):
        src = {"SSP245/": 0, "SSP245/a.nc": 5}
        plan = cur.plan_transfers(src, {})
        assert plan.to_copy == ["SSP245/a.nc"]
        assert plan.filtered_out == 0

    def test_filters_are_counted_not_copied(self):
        src = {"SSP245/tas.nc": 5, "SSP245/o3_mmr.nc": 500}
        plan = cur.plan_transfers(src, {}, exclude=("o3_",))
        assert plan.to_copy == ["SSP245/tas.nc"]
        assert plan.filtered_out == 1
        assert plan.bytes_to_copy == 5

    def test_copy_order_is_sorted_and_deterministic(self):
        src = {"z.nc": 1, "a.nc": 1, "m.nc": 1}
        plan = cur.plan_transfers(src, {})
        assert plan.to_copy == ["a.nc", "m.nc", "z.nc"]

    def test_bytes_to_copy_sums_missing_and_mismatched(self):
        src = {"a.nc": 10, "b.nc": 20, "c.nc": 30}
        dest = {"a.nc": 10, "b.nc": 19}
        plan = cur.plan_transfers(src, dest)
        assert plan.to_copy == ["b.nc", "c.nc"]
        assert plan.bytes_to_copy == 50
