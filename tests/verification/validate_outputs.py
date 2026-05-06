"""
Run ``get_outputs`` for each verification case and compare GeoTIFFs to S3 references.

Reference objects in S3 must use the same filenames as ``create_geotiff_files`` in ``backend/get_outputs.py``
(``{var}_{no_sai|with_sai}_{hash}.tif``) and the same 10-character MD5 hash of the parameter tuple:

``ssp_scenario``, ``temp_target``, ``decade_start_year``, ``decade_end_year``, ``start_year``, ``ramp_up``.

Default S3 object key layout (bucket is separate; this is the key inside the bucket)::

    {VALIDATION_S3_PREFIX}/{filetype}/{filename}

where ``filetype`` is ``geotiffs`` or ``plot_data``, and ``filename`` is
``{var}_{no_sai|with_sai}_{hash}.tif`` (hash matches ``create_geotiff_files``).
"""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol

from tests.verification.repo_root import ensure_repo_on_sys_path

ensure_repo_on_sys_path()

# Imports below intentionally come after repo-root path setup.
import numpy as np  # noqa: E402
from botocore.exceptions import ClientError  # noqa: E402

from .relative_error import compute_relative_error, load_dataset  # noqa: E402


class S3KeyFn(Protocol):
    """Builds an S3 object key for a GeoTIFF (same contract as :func:`default_s3_key`)."""

    def __call__(
        self,
        *,
        s3_prefix: str,
        case: dict[str, Any],
        decade_visualization: str,
        ramp_up: int,
        var: str,
        suffix: str,
        filetype: str,
    ) -> str:
        ...


VALIDATION_VARIABLE_POOL: tuple[str, ...] = (
    "tas",
    "tasmax",
    "tasmin",
    "exposure_above_40",
    "exposure_above_35",
    "exposure_below_0",
    "p-e",
    "exposure_above_20",
    "exposure_above_10",
    "icefrac",
)


def geotiff_params_hash(
    ssp_scenario: str,
    temp_target: float,
    decade_start_year: int,
    decade_end_year: int,
    start_year: int,
    ramp_up: int,
) -> str:
    """Same 10-char hash as ``create_geotiff_files`` in ``backend/get_outputs.py``."""
    geotiff_params = [
        ssp_scenario,
        temp_target,
        decade_start_year,
        decade_end_year,
        start_year,
        ramp_up,
    ]
    geotiff_params_str = "_".join(map(str, geotiff_params))
    return hashlib.md5(geotiff_params_str.encode()).hexdigest()[:10]


def parse_decade(decade_visualization: str) -> tuple[int, int]:
    """Split a ``YYYY1-YYYY2`` decade label into start and end years (inclusive)."""
    a, b = decade_visualization.split("-")
    return int(a), int(b)


def default_s3_key(
    *,
    s3_prefix: str,
    case: dict[str, Any],
    decade_visualization: str,
    ramp_up: int,
    var: str,
    suffix: str,
    filetype: str = "geotiffs",
) -> str:
    """Build S3 object key: ``{prefix}/{filetype}/{filename}`` (``filename`` encodes case params via hash)."""
    if filetype not in ("geotiffs", "plot_data"):
        raise ValueError(
            f"filetype must be 'geotiffs' or 'plot_data', got {filetype!r}"
        )
    d0, d1 = parse_decade(decade_visualization)
    h = geotiff_params_hash(
        case["ssp_scenario"],
        float(case["temp_target"]),
        d0,
        d1,
        int(case["start_year"]),
        ramp_up,
    )
    fname = f"{var}_{suffix}_{h}.tif"
    prefix = s3_prefix.strip("/")
    parts = [p for p in (prefix, filetype.strip("/"), fname) if p]
    return "/".join(parts)


def compare_geotiff_files(
    reference_path: Path,
    candidate_path: Path,
    *,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    """Pixel-wise relative error stats (same definition as ``relative_error`` CLI)."""
    baseline = load_dataset(reference_path)
    comparison = load_dataset(candidate_path)
    rel = compute_relative_error(baseline, comparison, epsilon)
    return {
        "mean_rel_err": float(np.mean(rel)),
        "median_rel_err": float(np.median(rel)),
        "max_rel_err": float(np.max(rel)),
        "p95_rel_err": float(np.percentile(rel, 95)),
    }


@dataclass
class VerificationConfig:
    data_dir: Path
    model_dir: Path
    cache_dir: Path
    s3_bucket: str
    """S3 bucket name (no ``s3://``)."""

    s3_prefix: str = ""
    """Key prefix inside the bucket (from ``VALIDATION_S3_PREFIX``); see :func:`default_s3_key`."""

    s3_filetype: str = "geotiffs"
    """Subfolder under prefix: ``geotiffs`` or ``plot_data``."""

    decade_visualization: str = "2091-2100"
    ramp_up: int = 10
    spatial_gdf: Any = None
    spatial_item: Any = None
    variables: list[str] | None = None
    """If None, defaults to ``[\"tas\"]`` (single-variable runs keep GeoTIFF paths correct)."""

    epsilon: float = 1e-12
    output_config_extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeotiffComparisonResult:
    variable: str
    suffix: str
    s3_key: str
    local_path: Path
    metrics: dict[str, float]


@dataclass
class CaseVerificationResult:
    case: dict[str, Any]
    comparisons: list[GeotiffComparisonResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when this case has no recorded errors."""
        return not self.errors


def geotiff_comparison_to_dict(
    c: GeotiffComparisonResult,
    *,
    include_local_path: bool = False,
) -> dict[str, Any]:
    """Serialize a comparison row for reports or Celery JSON results."""
    row: dict[str, Any] = {
        "variable": c.variable,
        "suffix": c.suffix,
        "s3_key": c.s3_key,
        "metrics": c.metrics,
    }
    if include_local_path:
        row["local_path"] = str(c.local_path)
    return row


def _download_s3_object(bucket: str, key: str, dest: Path) -> None:
    """Download ``bucket``/``key`` from S3 to ``dest`` using boto3."""
    import boto3

    dest.parent.mkdir(parents=True, exist_ok=True)
    client = boto3.client("s3")
    client.download_file(bucket, key, str(dest))


def _run_get_outputs_for_var(
    case: dict[str, Any],
    var: str,
    config: VerificationConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Run :func:`backend.get_outputs.get_outputs` for one variable with verification-friendly options."""
    from backend.get_outputs import get_outputs

    decade_start_year, decade_end_year = parse_decade(config.decade_visualization)
    oc = {
        "create_geotiffs": True,
        "output_dir": str(output_dir),
        # Skip mean-over-time, PDFs, SO2 block, etc. (see get_outputs).
        "verification_minimal": True,
    }
    oc.update(config.output_config_extra)
    return get_outputs(
        case["ssp_scenario"],
        float(case["temp_target"]),
        config.spatial_gdf,
        config.spatial_item,
        decade_start_year,
        decade_end_year,
        int(case["start_year"]),
        config.ramp_up,
        Path(config.data_dir),
        Path(config.model_dir),
        Path(config.cache_dir),
        var=var,
        output_config=oc,
    )


def validate_case(
    case: dict[str, Any],
    config: VerificationConfig,
    *,
    s3_key_fn: S3KeyFn = default_s3_key,
) -> CaseVerificationResult:
    """
    For one parameter dict, run ``get_outputs`` per variable, compare each
    ``no_sai`` / ``with_sai`` GeoTIFF to the object at the S3 key from ``s3_key_fn``.

    Parameters
    ----------
    s3_key_fn
        Callable matching :class:`S3KeyFn` (same keyword contract as :func:`default_s3_key`).
    """
    variables = config.variables if config.variables is not None else ["tas"]
    result = CaseVerificationResult(case=dict(case))

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for var in variables:
            try:
                output_data = _run_get_outputs_for_var(case, var, config, tmp_path)
            except Exception as e:
                result.errors.append(f"{var}: get_outputs failed: {e!r}")
                continue

            geotiff_paths = output_data.get("geotiff_paths") or {}
            for suffix in ("no_sai", "with_sai"):
                key = f"{var}_{suffix}"
                if key not in geotiff_paths:
                    result.errors.append(f"{var}: missing local GeoTIFF key {key!r}")
                    continue

                local_path = Path(geotiff_paths[key])
                s3_key = s3_key_fn(
                    s3_prefix=config.s3_prefix,
                    case=case,
                    decade_visualization=config.decade_visualization,
                    ramp_up=config.ramp_up,
                    var=var,
                    suffix=suffix,
                    filetype=config.s3_filetype,
                )
                ref_path = tmp_path / "reference" / f"{var}_{suffix}.tif"

                try:
                    _download_s3_object(config.s3_bucket, s3_key, ref_path)
                except ClientError as e:
                    result.errors.append(f"{config.s3_bucket}/{var}/{suffix}: S3 download {s3_key!r}: {e}")
                    continue
                except Exception as e:
                    result.errors.append(f"{var}/{suffix}: S3 error: {e!r}")
                    continue

                try:
                    metrics = compare_geotiff_files(
                        ref_path, local_path, epsilon=config.epsilon
                    )
                except Exception as e:
                    result.errors.append(f"{var}/{suffix}: compare failed: {e!r}")
                    continue

                result.comparisons.append(
                    GeotiffComparisonResult(
                        variable=var,
                        suffix=suffix,
                        s3_key=s3_key,
                        local_path=local_path,
                        metrics=metrics,
                    )
                )

    return result


def validate_cases(
    cases: list[dict[str, Any]] | Iterator[dict[str, Any]],
    config: VerificationConfig,
    *,
    s3_key_fn: S3KeyFn = default_s3_key,
) -> Iterator[CaseVerificationResult]:
    """Yield a :class:`CaseVerificationResult` for each case."""
    for case in cases:
        yield validate_case(case, config, s3_key_fn=s3_key_fn)
