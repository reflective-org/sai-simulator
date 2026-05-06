"""Fast tests that do not need S3 or local data."""

from __future__ import annotations

from tests.verification.validate_outputs import default_s3_key, geotiff_params_hash


def test_geotiff_params_hash_length_and_stability() -> None:
    """GeoTIFF parameter hash is 10 hex chars and stable for identical inputs."""
    h = geotiff_params_hash("SSP2-4.5", 1.0, 2091, 2100, 2040, 10)
    assert len(h) == 10
    assert h == geotiff_params_hash("SSP2-4.5", 1.0, 2091, 2100, 2040, 10)


def test_default_s3_key_format() -> None:
    """Default S3 keys include prefix, filetype folder, and hashed filename."""
    case = {
        "ssp_scenario": "SSP2-4.5",
        "temp_target": 1.0,
        "start_year": 2040,
    }
    h = geotiff_params_hash("SSP2-4.5", 1.0, 2091, 2100, 2040, 10)
    key = default_s3_key(
        s3_prefix="ref/v1",
        case=case,
        decade_visualization="2091-2100",
        ramp_up=10,
        var="tas",
        suffix="no_sai",
        filetype="geotiffs",
    )
    assert key == f"ref/v1/geotiffs/tas_no_sai_{h}.tif"

    plot_key = default_s3_key(
        s3_prefix="ref/v1",
        case=case,
        decade_visualization="2091-2100",
        ramp_up=10,
        var="tas",
        suffix="with_sai",
        filetype="plot_data",
    )
    assert plot_key == f"ref/v1/plot_data/tas_with_sai_{h}.tif"
