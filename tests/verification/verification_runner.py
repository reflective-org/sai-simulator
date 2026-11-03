"""
Shared logic for running the S3 GeoTIFF verification suite (used by pytest).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from tests.verification.generate_test_cases import generate_test_cases
from tests.verification.validate_outputs import (
    VerificationConfig,
    geotiff_comparison_to_dict,
    validate_cases,
)


def load_cases_json(path: Path) -> list[dict[str, Any]]:
    """Load a JSON array of case dicts from ``path``."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array of objects in {path}")
    return data


def resolve_cases(
    *,
    cases_path: Path | None = None,
    generate_count: int | None = None,
    limit: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Build the case list for verification.

    - If ``cases_path`` is set, load that JSON file and optionally slice to ``limit``.
    - Otherwise call :func:`generate_test_cases` in memory (no JSON file).

    How many random cases to generate: ``generate_count`` if set, else ``limit`` if set,
    else ``1``.
    """
    if cases_path is not None:
        cases = load_cases_json(cases_path)
        if limit is not None:
            return cases[:limit]
        return cases

    n = generate_count if generate_count is not None else (limit if limit is not None else 1)
    if n < 1:
        raise ValueError("Case count must be at least 1.")
    return generate_test_cases(n, seed=seed, allow_repeats=False)


def run_verification_report(
    cases: list[dict[str, Any]],
    config: VerificationConfig,
) -> dict[str, Any]:
    """Run verification for all cases; return a summary dict (JSON-serializable)."""
    report_cases: list[dict[str, Any]] = []
    failed = 0

    t0 = perf_counter()
    for result in validate_cases(cases, config):
        case_t0 = perf_counter()
        report_cases.append(
            {
                "case": result.case,
                "ok": result.ok,
                "errors": result.errors,
                "comparisons": [
                    geotiff_comparison_to_dict(c) for c in result.comparisons
                ],
            }
        )
        report_cases[-1]["seconds"] = perf_counter() - case_t0
        if not result.ok:
            failed += 1
    total_seconds = perf_counter() - t0

    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "seconds": total_seconds,
        "total_cases": len(cases),
        "cases_run": len(report_cases),
        "failed_cases": failed,
        "config": {
            "data_dir": str(config.data_dir),
            "model_dir": str(config.model_dir),
            "cache_dir": str(config.cache_dir),
            "s3_bucket": config.s3_bucket,
            "s3_prefix": config.s3_prefix,
            "s3_filetype": config.s3_filetype,
            "decade_visualization": config.decade_visualization,
            "ramp_up": config.ramp_up,
            "variables": config.variables,
            "epsilon": config.epsilon,
        },
        "results": report_cases,
    }
