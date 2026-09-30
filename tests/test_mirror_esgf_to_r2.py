"""Unit tests for the pure parts of ``scripts/mirror_esgf_to_r2.py``.

Covers the Solr-doc parsing, the streaming hash wrapper, and — because they
encode a deliberate cost decision — the shape of the ``SELECTIONS`` spec.
No network, no boto3 objects.
"""

import hashlib
import io

import pytest

import scripts.mirror_esgf_to_r2 as mer


def _doc(**over):
    doc = {
        "title": ["tas_Amon_UKESM1-0-LL_ssp245_r2i1p1f2_gn_201501-204912.nc"],
        "url": [
            "https://host/thredds/dodsC/x.nc.html|application/opendap-html|OPENDAP",
            "https://host/thredds/fileServer/x.nc|application/netcdf|HTTPServer",
        ],
        "size": [12345],
        "checksum": ["ab" * 32],
        "checksum_type": ["SHA256"],
        "variable_id": ["tas"],
        "table_id": ["Amon"],
        "experiment_id": ["ssp245"],
        "variant_label": ["r2i1p1f2"],
    }
    doc.update(over)
    return doc


class TestPickHttpUrl:
    def test_picks_httpserver_and_strips_service_suffix(self):
        assert mer.pick_http_url(_doc()["url"]) == "https://host/thredds/fileServer/x.nc"

    def test_ignores_opendap_only(self):
        assert mer.pick_http_url([_doc()["url"][0]]) == ""


class TestDocToRecord:
    def test_builds_rel_key_and_unwraps_lists(self):
        rec = mer.doc_to_record(_doc())
        assert rec.rel_key == (
            "ssp245/r2i1p1f2/Amon/tas/"
            "tas_Amon_UKESM1-0-LL_ssp245_r2i1p1f2_gn_201501-204912.nc"
        )
        assert rec.size == 12345
        assert rec.sha256 == "ab" * 32
        assert rec.url == "https://host/thredds/fileServer/x.nc"

    def test_rejects_missing_httpserver_url(self):
        with pytest.raises(ValueError, match="no HTTPServer url"):
            mer.doc_to_record(_doc(url=[_doc()["url"][0]]))

    def test_rejects_non_sha256_checksums(self):
        with pytest.raises(ValueError, match="checksum type"):
            mer.doc_to_record(_doc(checksum_type=["MD5"]))


class TestPartRanges:
    def test_single_part_when_size_fits(self):
        assert mer.part_ranges(100, 1000) == [(0, 99)]

    def test_uniform_parts_with_smaller_tail(self):
        ranges = mer.part_ranges(250, 100)
        assert ranges == [(0, 99), (100, 199), (200, 249)]

    def test_ranges_cover_every_byte_exactly_once(self):
        ranges = mer.part_ranges(mer.PART_SIZE * 3 + 17, mer.PART_SIZE)
        assert ranges[0][0] == 0
        assert all(a[1] + 1 == b[0] for a, b in zip(ranges, ranges[1:]))
        assert sum(end - start + 1 for start, end in ranges) == mer.PART_SIZE * 3 + 17

    def test_exact_multiple_has_no_empty_tail(self):
        ranges = mer.part_ranges(200, 100)
        assert ranges == [(0, 99), (100, 199)]

    def test_rejects_nonpositive_sizes(self):
        with pytest.raises(ValueError):
            mer.part_ranges(0, 100)


class TestRangedGet:
    class _Resp:
        def __init__(self, status, body):
            self.status = status
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def test_a_200_means_range_ignored_and_is_never_retried(self, monkeypatch):
        calls = []

        def fake_urlopen(req, timeout):
            calls.append(req)
            return TestRangedGet._Resp(200, b"x" * 1000)  # whole file, not the range

        monkeypatch.setattr(mer.urllib.request, "urlopen", fake_urlopen)
        with pytest.raises(mer.RangeUnsupported):
            mer._ranged_get("https://h/f.nc", 0, 99)
        assert len(calls) == 1  # no retries: every retry would re-download the whole file

    def test_short_range_is_retried_then_raised(self, monkeypatch):
        calls = []

        def fake_urlopen(req, timeout):
            calls.append(req)
            return TestRangedGet._Resp(206, b"x" * 10)

        monkeypatch.setattr(mer.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(mer.time, "sleep", lambda s: None)
        with pytest.raises(OSError, match="short range read"):
            mer._ranged_get("https://h/f.nc", 0, 99, attempts=3)
        assert len(calls) == 3

    def test_a_good_range_returns_bytes(self, monkeypatch):
        monkeypatch.setattr(mer.urllib.request, "urlopen",
                            lambda req, timeout: TestRangedGet._Resp(206, b"y" * 100))
        assert mer._ranged_get("https://h/f.nc", 0, 99) == b"y" * 100


class TestHashingReader:
    def test_hashes_and_counts_across_chunked_reads(self):
        payload = b"x" * 10000 + b"y" * 3
        reader = mer.HashingReader(io.BytesIO(payload))
        out = b""
        while True:
            chunk = reader.read(4096)
            if not chunk:
                break
            out += chunk
        assert out == payload
        assert reader.count == len(payload)
        assert reader.hexdigest() == hashlib.sha256(payload).hexdigest()


class TestSelections:
    """The selections encode what we pay CEDA bandwidth for — pin the intent."""

    def test_daily_hfls_is_never_requested(self):
        for sel in mer.SELECTIONS:
            if "day" in sel["table_id"]:
                assert "hfls" not in sel["variable_id"].split(",")

    def test_daily_extremes_are_requested_for_every_member_and_experiment(self):
        # TREFHTMX/MN in the simulator stores are ABSOLUTE monthly extremes of
        # daily tasmax/tasmin (see the shipped MIROC store attrs), so the daily
        # fields are needed everywhere.
        day = next(s for s in mer.SELECTIONS if s["table_id"] == "day")
        assert {"tasmax", "tasmin"} <= set(day["variable_id"].split(","))
        assert set(day["experiment_id"].split(",")) == {"historical", "ssp245"}
        assert set(day["variant_label"].split(",")) == {"r2i1p1f2", "r3i1p1f2", "r12i1p1f2"}

    def test_monthly_mean_tasminmax_is_not_mirrored(self):
        # Amon tasmax/tasmin are monthly MEANS of the daily extremes — the
        # wrong quantity for the stores' absolute-extreme convention.
        for sel in mer.SELECTIONS:
            if sel["table_id"] == "Amon":
                assert not {"tasmax", "tasmin"} & set(sel["variable_id"].split(","))

    def test_monthly_selection_covers_all_simulator_sources(self):
        amon = next(s for s in mer.SELECTIONS if s["table_id"] == "Amon")
        assert set(amon["variable_id"].split(",")) == {"tas", "pr", "hfls"}
        assert set(amon["experiment_id"].split(",")) == {"historical", "ssp245"}
        assert set(amon["variant_label"].split(",")) == {"r2i1p1f2", "r3i1p1f2", "r12i1p1f2"}

    def test_seaice_requests_both_grid_variants(self):
        simon = next(s for s in mer.SELECTIONS if s["table_id"] == "SImon")
        assert set(simon["variable_id"].split(",")) == {"siconc", "siconca"}
