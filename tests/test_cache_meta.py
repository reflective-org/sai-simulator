"""The cache-dir (simulator version, FaIR calibration) fingerprint.

Every FaIR-dependent artifact in a cache dir is only valid for the (code version,
climate calibration) pair that produced it, and neither is recoverable from the files
themselves — so the pair is stamped into ``cache_meta.json`` when a dir is first
populated and checked before any cached file is trusted. A populated dir without a
stamp predates the mechanism (and the AR6-constrained FaIR default) and must be
refused rather than silently served.
"""
import json

import pytest

from backend.baseline import (
    CACHE_META_FILENAME,
    FAIR_CALIBRATION,
    SIMULATOR_VERSION,
    stamp_cache_dir,
    validate_cache_dir,
)


def test_fresh_dir_is_stamped_and_passes(tmp_path):
    cache_dir = tmp_path / "cache"
    validate_cache_dir(cache_dir)
    meta = json.loads((cache_dir / CACHE_META_FILENAME).read_text())
    from backend.constants import REGIONAL_MODEL_NAMES, REGIONAL_MODEL_VAR_EXCLUSIONS
    assert meta == {
        "simulator_version": SIMULATOR_VERSION,
        "fair_calibration": FAIR_CALIBRATION,
        "regional_models": list(REGIONAL_MODEL_NAMES),
        "var_exclusions": {k: list(v) for k, v in sorted(REGIONAL_MODEL_VAR_EXCLUSIONS.items())},
    }
    # Idempotent: a matching stamped dir keeps passing.
    validate_cache_dir(cache_dir)


def test_other_model_roster_stamp_is_refused(tmp_path):
    """A cache built under a different regional-model roster (e.g. the two-model
    v1.4 caches) must be refused: every multi-model mean and the pooled SAI delta
    depend on the roster, and nothing in the artifacts records it otherwise."""
    validate_cache_dir(tmp_path)
    meta_path = tmp_path / CACHE_META_FILENAME
    meta = json.loads(meta_path.read_text())
    meta["regional_models"] = ["CESM2-WACCM", "MIROC-ES2H"]
    meta_path.write_text(json.dumps(meta))
    (tmp_path / "historical_tas.nc").write_bytes(b"x")
    with pytest.raises(Exception):
        validate_cache_dir(tmp_path)


def test_populated_unstamped_dir_is_refused(tmp_path):
    # Pre-stamp caches were all generated under the old CESM calibration.
    (tmp_path / "ssp245_regional_tas.nc").touch()
    with pytest.raises(ValueError, match="predates the cache fingerprint"):
        validate_cache_dir(tmp_path)


def test_other_calibration_stamp_is_refused(tmp_path):
    stamp_cache_dir(tmp_path)
    meta_path = tmp_path / CACHE_META_FILENAME
    meta = json.loads(meta_path.read_text())
    meta["fair_calibration"] = "cesm" if FAIR_CALIBRATION != "cesm" else "constrained"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="was generated under"):
        validate_cache_dir(tmp_path)


def test_other_version_stamp_is_refused(tmp_path):
    stamp_cache_dir(tmp_path)
    meta_path = tmp_path / CACHE_META_FILENAME
    meta = json.loads(meta_path.read_text())
    meta["simulator_version"] = "v0.0.0"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="was generated under"):
        validate_cache_dir(tmp_path)


def test_empty_but_existing_dir_is_stamped(tmp_path):
    # An empty dir (e.g. freshly mkdir'd by a launcher) is this process's to claim.
    non_artifact = tmp_path / "notes.txt"
    non_artifact.touch()  # only *.nc artifacts mark a dir as populated
    validate_cache_dir(tmp_path)
    assert (tmp_path / CACHE_META_FILENAME).exists()


def test_corrupt_stamp_is_refused_with_guidance(tmp_path):
    (tmp_path / CACHE_META_FILENAME).write_text("{truncated")
    with pytest.raises(ValueError, match="Unreadable"):
        validate_cache_dir(tmp_path)


def test_delta_reader_enforces_the_fingerprint(tmp_path):
    # The enforcement points must actually call the validator, not just define it.
    from backend.delta import get_regional_delta_without_rampup

    (tmp_path / "ssp245_1.5_regional_delta_tas.nc").touch()
    with pytest.raises(ValueError, match="predates the cache fingerprint"):
        get_regional_delta_without_rampup(
            "tas", tmp_path, tmp_path, tmp_path, "SSP2-4.5", 1.5, None
        )


def test_historical_reader_enforces_the_fingerprint(tmp_path):
    from backend.historical import get_historical_model

    (tmp_path / "historical_tas.nc").touch()
    with pytest.raises(ValueError, match="predates the cache fingerprint"):
        get_historical_model("tas", tmp_path, tmp_path)
