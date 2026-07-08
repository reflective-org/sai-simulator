"""Unit tests for :mod:`scripts.data_sanity_checks`.

Covers the pure helpers (S3 URI parsing, expected-filename builders, duplicate detection)
plus end-to-end ``run_download_sanity_checks`` paths exercised against ``tmp_path`` trees.
The corrupted-NetCDF path is tested with a stubbed validator so the suite stays fast and
doesn't need real CESM files.
"""
from pathlib import Path

import pytest

from version import DATA_BUCKET_URI, SIMULATOR_VERSION

from scripts.data_sanity_checks import (
    build_variable_to_members_map,
    extract_member_and_variable,
    find_corrupted_netcdf_files,
    find_duplicate_basenames,
    list_netcdf_paths_relative,
    parse_aws_s3_ls_recursive_output,
    parse_required_variables,
    parse_s3_uri,
    run_download_sanity_checks,
    strip_s3_key_prefix_for_local_compare,
    validate_expected_files,
    validate_required_variables_and_members,
)


def test_validate_expected_files_returns_missing() -> None:
    expected = ["a/file1.nc", "a/file2.nc", "b/file3.nc"]
    actual = ["a/file1.nc", "b/file3.nc"]
    assert validate_expected_files(expected, actual) == ["a/file2.nc"]


def test_find_duplicate_basenames_detects_duplicates() -> None:
    files = ["a/same.nc", "b/same.nc", "c/other.nc"]
    duplicates = find_duplicate_basenames(files)
    assert duplicates == {"same.nc": 2}


def test_list_netcdf_paths_relative_ignores_non_netcdf(tmp_path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.nc").write_text("x", encoding="utf-8")
    (tmp_path / "a" / "README.md").write_text("doc", encoding="utf-8")
    assert list_netcdf_paths_relative(tmp_path) == ["a/x.nc"]


def test_duplicate_check_uses_only_netcdf_paths(tmp_path, monkeypatch) -> None:
    """Same basename for non-.nc must not affect duplicate detection."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "dup.nc").write_text("x", encoding="utf-8")
    (tmp_path / "b" / "dup.nc").write_text("y", encoding="utf-8")
    (tmp_path / "a" / "note.txt").write_text("n", encoding="utf-8")
    (tmp_path / "b" / "note.txt").write_text("n", encoding="utf-8")

    monkeypatch.setattr("scripts.data_sanity_checks.find_corrupted_netcdf_files", lambda _d: [])

    with pytest.raises(ValueError, match="Duplicate NetCDF basenames"):
        run_download_sanity_checks(tmp_path)

    # Two identical .txt basenames alone should not raise (no duplicate .nc basenames)
    (tmp_path / "a" / "dup.nc").unlink()
    (tmp_path / "b" / "dup.nc").unlink()
    (tmp_path / "a" / "only.nc").write_text("z", encoding="utf-8")
    run_download_sanity_checks(tmp_path)


def test_parse_aws_listing_output() -> None:
    listing = (
        "2026-01-01 00:00:01       1234 path/to/file1.nc\n"
        "2026-01-01 00:00:02       9999 path/to/file2.nc\n"
    )
    assert parse_aws_s3_ls_recursive_output(listing) == [
        "path/to/file1.nc",
        "path/to/file2.nc",
    ]


def test_parse_s3_uri_with_version_prefix() -> None:
    bucket, prefix = parse_s3_uri(DATA_BUCKET_URI)
    assert bucket == "reflective-simulator-bucket"
    assert prefix == SIMULATOR_VERSION


def test_parse_s3_uri_bucket_only() -> None:
    bucket, prefix = parse_s3_uri("s3://my-bucket")
    assert bucket == "my-bucket"
    assert prefix == ""


def test_strip_s3_key_prefix_strips_version_folder() -> None:
    assert (
        strip_s3_key_prefix_for_local_compare(
            f"{SIMULATOR_VERSION}/b.e21.BWHIST.f09_g17.foo.nc", SIMULATOR_VERSION
        )
        == "b.e21.BWHIST.f09_g17.foo.nc"
    )


def test_strip_s3_key_prefix_returns_none_for_prefix_only_object() -> None:
    assert strip_s3_key_prefix_for_local_compare(SIMULATOR_VERSION, SIMULATOR_VERSION) is None


def test_parse_required_variables() -> None:
    assert parse_required_variables("TREFHT, PRECT ,ICEFRAC") == ["TREFHT", "PRECT", "ICEFRAC"]


def test_extract_member_and_variable() -> None:
    filename = "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM.001.cam.h0.TREFHT.185001-201412.nc"
    assert extract_member_and_variable(filename) == ("001", "TREFHT")


def test_build_variable_to_members_map() -> None:
    files = [
        "a/b.001.cam.h0.TREFHT.185001-201412.nc",
        "a/b.002.cam.h0.TREFHT.185001-201412.nc",
        "a/b.001.cam.h0.PRECT.185001-201412.nc",
    ]
    mapping = build_variable_to_members_map(files)
    assert mapping["TREFHT"] == {"001", "002"}
    assert mapping["PRECT"] == {"001"}


def test_validate_required_variables_and_members_fails_on_missing_variable() -> None:
    files = ["a/b.001.cam.h0.TREFHT.185001-201412.nc"]
    with pytest.raises(ValueError, match="Missing required variables"):
        validate_required_variables_and_members(
            files, required_variables=["TREFHT", "PRECT"], min_ensemble_members=1
        )


def test_validate_required_variables_and_members_fails_on_insufficient_members() -> None:
    files = [
        "a/b.001.cam.h0.TREFHT.185001-201412.nc",
        "a/b.002.cam.h0.TREFHT.185001-201412.nc",
    ]
    with pytest.raises(ValueError, match="fewer than 3 ensemble members"):
        validate_required_variables_and_members(
            files, required_variables=["TREFHT"], min_ensemble_members=3
        )


def test_run_download_sanity_checks_passes_when_expected_present(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path
    (root / "a").mkdir(parents=True, exist_ok=True)
    (root / "a" / "x.001.cam.h0.TREFHT.185001-201412.nc").write_text("x", encoding="utf-8")
    (root / "a" / "x.002.cam.h0.TREFHT.185001-201412.nc").write_text("x", encoding="utf-8")
    (root / "a" / "x.003.cam.h0.TREFHT.185001-201412.nc").write_text("x", encoding="utf-8")
    monkeypatch.setattr("scripts.data_sanity_checks.find_corrupted_netcdf_files", lambda _d: [])

    # Should not raise.
    run_download_sanity_checks(
        root,
        required_variables=["TREFHT"],
        min_ensemble_members=3,
    )


def test_run_download_sanity_checks_fails_on_duplicate_basenames(tmp_path: Path) -> None:
    root = tmp_path
    (root / "a").mkdir(parents=True, exist_ok=True)
    (root / "b").mkdir(parents=True, exist_ok=True)
    (root / "a" / "dup.nc").write_text("x", encoding="utf-8")
    (root / "b" / "dup.nc").write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate NetCDF basenames"):
        run_download_sanity_checks(root)


def test_run_download_sanity_checks_fails_when_required_var_has_too_few_members(
    tmp_path: Path,
) -> None:
    root = tmp_path
    (root / "a").mkdir(parents=True, exist_ok=True)
    (root / "a" / "x.001.cam.h0.PRECT.185001-201412.nc").write_text("x", encoding="utf-8")
    (root / "a" / "x.002.cam.h0.PRECT.185001-201412.nc").write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="fewer than 3 ensemble members"):
        run_download_sanity_checks(
            root,
            required_variables=["PRECT"],
            min_ensemble_members=3,
        )


def test_find_corrupted_netcdf_files_detects_failures(tmp_path: Path) -> None:
    root = tmp_path
    (root / "ok.nc").write_text("ok", encoding="utf-8")
    (root / "bad.nc").write_text("bad", encoding="utf-8")
    (root / "skip.txt").write_text("not netcdf", encoding="utf-8")

    def fake_validator(path: Path) -> None:
        if path.name == "bad.nc":
            raise ValueError("corrupt")

    corrupted = find_corrupted_netcdf_files(root, validator=fake_validator)
    assert corrupted == ["bad.nc"]


def test_run_download_sanity_checks_fails_on_corrupted_netcdf(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    (root / "x.001.cam.h0.TREFHT.185001-201412.nc").write_text("x", encoding="utf-8")
    (root / "x.002.cam.h0.TREFHT.185001-201412.nc").write_text("x", encoding="utf-8")
    (root / "x.003.cam.h0.TREFHT.185001-201412.nc").write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.data_sanity_checks.find_corrupted_netcdf_files",
        lambda _data_dir: ["x.001.cam.h0.TREFHT.185001-201412.nc"],
    )

    with pytest.raises(ValueError, match="corrupted/unreadable NetCDF files"):
        run_download_sanity_checks(
            root,
            required_variables=["TREFHT"],
            min_ensemble_members=3,
        )
