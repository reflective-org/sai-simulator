"""Unit tests for the pure parts of ``scripts/mirror_macloud_to_r2.py``.

The listing parser, the filename filter, and the suite map — the places where
a silent mistake either downloads gigabytes of chemistry fields or drops a
variable a store needs.
"""

import pytest

import scripts.mirror_macloud_to_r2 as mm

LISTING = """
<html><body>
<a href="?C=M;O=A">sort</a>
<a href="/public/macloud/">parent</a>
<a href="u-dp583/">u-dp583/</a>
<a href="tas_mon_UKESM1-1-LL_G6-1p5K-SAI_u-dp583_203401-208512.nc">x</a>
<a href="injection_data_UKESM_dp583_g6sai_1.log">log</a>
<a href="seaice_day_UKESM1-1-LL_u-dp583_201501-210012.nc">x</a>
</body></html>
"""


class TestExtractNcNames:
    def test_keeps_only_nc_files(self):
        names = mm.extract_nc_names(LISTING)
        assert names == [
            "tas_mon_UKESM1-1-LL_G6-1p5K-SAI_u-dp583_203401-208512.nc",
            "seaice_day_UKESM1-1-LL_u-dp583_201501-210012.nc",
        ]

    def test_absolute_hrefs_are_not_treated_as_files(self):
        # Listings link files relatively; an absolute href is navigation, not a
        # file. fetch_manifest turns an all-empty listing into an error rather
        # than a quiet "nothing to do" (see TestEmptyListingGuard).
        html = '<a href="/public/macloud/G6/u-x/tas_mon_a.nc">a</a>'
        assert mm.extract_nc_names(html) == []


class TestEmptyListingGuard:
    class _Resp:
        def __init__(self, body):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def test_suite_with_no_wanted_files_raises(self, monkeypatch):
        # A changed listing format must never look like "R2 is up to date".
        monkeypatch.setattr(mm, "_get", lambda url, timeout=60: TestEmptyListingGuard._Resp(
            b'<a href="?C=M">sort</a><a href="o3_mmr_mon_x.nc">o3</a>'))
        with pytest.raises(RuntimeError, match="no wanted .nc files"):
            mm.fetch_manifest()


class TestWanted:
    def test_accepts_every_store_input_spelling(self):
        for name in (
            "tas_mon_UKESM1-1-LL_G6-1p5K-SAI_u-dp583_203401-208512.nc",
            "pr_mon_UKESM1-1-LL_SSP245_u-dp902_201501-210012.nc",
            "evap_mon_UKESM1-1-LL_SSP245_u-dp902_201501-210012.nc",
            "siconca_mon_UKESM1-1-LL_G6-1p5K-SAI_u-dp583_203401-208512.nc",
            "tas_day_UKESM1-1-LL_u-dp583_201501-210012.nc",
            "tas_mean_min_max_day_UKESM1-1-LL_SSP245_u-dp902_201501-210012.nc",
            "tas-min-mn-max_daily_UKESM1-1-LL_historical_u-by791_185001-201512.nc",
            "pr_day_UKESM1-1-LL_u-dp583_201501-210012.nc",
            "pr_daily_UKESM1-1-LL_historical_u-by791_185001-201512.nc",
            "seaice_day_UKESM1-1-LL_u-dp583_201501-210012.nc",
        ):
            assert mm.wanted(name), name

    def test_rejects_heavy_and_unrelated_files(self):
        for name in (
            "ta_all_mon_UKESM1-1-LL_u-dp904_201501-210012.nc",       # 3D temp
            "o3_mmr_mon_UKESM1-1-LL_SSP245_u-dp902_201501-210012.nc",  # 3D ozone
            "aero_num_mixratio_acc_mon_UKESM1-1-LL_x.nc",
            "so41_mon_UKESM1-1-LL_x.nc",
            "hurs_day_UKESM1-1_G6-1p5K-SAI_r2i1p1f2_gn_20350101-20491230.nc",
            "ps_CFday_UKESM1-1_G6-1p5K-SAI_r2i1p1f2_gn_20350101-20491230.nc",
            "hurs_daily_UKESM1-1-LL_historical_u-by791_185001-201512.nc",
            "rsds_daily_UKESM1-1-LL_historical_u-by791_185001-201512.nc",
            "strataod1_mon_UKESM1-1-LL_u-ds287_201501-210012.nc",
        ):
            assert not mm.wanted(name), name


class TestSuites:
    def test_nine_suites_with_unique_destinations(self):
        assert len(mm.SUITES) == 9
        dests = list(mm.SUITES.values())
        assert len(set(dests)) == 9

    def test_sai_and_ssp245_match_hub_layout(self):
        assert mm.SUITES[("G6-1p5k-SAI", "u-dp583")] == "G6-1p5K-SAI/r2i1p1f1"
        assert mm.SUITES[("SSP245_ukesm1p1", "u-dp904")] == "SSP245/r12i1p1f1"

    def test_historical_is_keyed_by_suite_not_guessed_member(self):
        # The suite->member mapping is unconfirmed; keys must not bake in a guess.
        for (exp_dir, suite), dest in mm.SUITES.items():
            if exp_dir == "historical_ukesm1p1":
                assert dest == f"historical/{suite}"

    def test_mcb_family_is_excluded(self):
        suites = {s for _, s in mm.SUITES}
        assert not suites & {"u-dh561", "u-di183", "u-di184"}
        assert not any(exp == "G6-1p5k-MCB" for exp, _ in mm.SUITES)
