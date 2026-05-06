"""
S3 reference verification: run ``get_outputs``, compare GeoTIFFs to S3.

Skipped unless ``VALIDATION_S3_BUCKET`` is set. See ``conftest.py`` for env vars.

**Parallelism:** use **Celery** (see ``tests/verification/run_verification_celery.py`` and
``tests/verification/celery_app.py``). Pytest runs cases **sequentially**.

Example::

    # Parallel (recommended for many cases):
    celery -A tests.verification.celery_app worker --loglevel=INFO --concurrency=2
    python tests/verification/run_verification_celery.py --bucket my-bucket --limit 20

    # Sequential in pytest (single parametrized items or batch):
    VALIDATION_S3_BUCKET=my-bucket pytest tests/verification/test_verification_s3.py -m s3_verification
    pytest tests/verification/test_verification_s3.py -m s3_verification_batch

Use ``--validation-print-timing`` and/or ``VALIDATION_PRINT_TIMING=1`` for timing lines (stderr).
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from tests.verification.timing import (
    print_verification_timing_lines,
    verification_print_timing_enabled,
)
from tests.verification.validate_outputs import VerificationConfig
from tests.verification.verification_runner import run_verification_report


@pytest.mark.s3_verification
def test_get_outputs_matches_s3_reference_single_case(
    request: pytest.FixtureRequest,
    verification_config: VerificationConfig,
    verification_case: dict[str, Any],
) -> None:
    """One case per parametrized item; runs sequentially in pytest (use Celery for parallel)."""
    from tests.verification.validate_outputs import validate_case

    t0 = time.perf_counter()
    result = validate_case(verification_case, verification_config)
    elapsed = time.perf_counter() - t0

    assert result.ok, result.errors

    if verification_print_timing_enabled(request.config):
        lines = [
            f"[verification] case={verification_case} seconds={elapsed:.3f} ok={result.ok}",
        ]
        for c in result.comparisons:
            m = c.metrics
            lines.append(
                f"  {c.variable}_{c.suffix}: mean_rel={m['mean_rel_err']:.3e} "
                f"max_rel={m['max_rel_err']:.3e}"
            )
        print_verification_timing_lines(lines)


@pytest.mark.s3_verification_batch
def test_get_outputs_matches_s3_references_batch(
    request: pytest.FixtureRequest,
    verification_config: VerificationConfig,
    verification_cases: list[dict[str, Any]],
) -> None:
    """All cases in one test (sequential loop)."""
    report = run_verification_report(verification_cases, verification_config)
    assert report["cases_run"] == len(verification_cases)
    assert report["failed_cases"] == 0, (
        f"{report['failed_cases']} case(s) failed: "
        f"{[r for r in report['results'] if not r['ok']]}"
    )

    assert isinstance(report.get("seconds"), (int, float))
    assert report["seconds"] >= 0

    if verification_print_timing_enabled(request.config):
        lines = [
            f"[verification-batch] cases={report['cases_run']} "
            f"total_s={report['seconds']:.3f} "
            f"mean_case_s={(report['seconds'] / max(report['cases_run'], 1)):.3f}",
        ]
        for i, row in enumerate(report["results"], start=1):
            lines.append(
                f"  - case {i}/{report['cases_run']}: {row['case']} "
                f"ok={row['ok']} seconds={row.get('seconds', float('nan')):.3f}"
            )
        print_verification_timing_lines(lines)
