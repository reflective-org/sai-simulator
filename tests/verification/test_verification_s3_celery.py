"""
S3 verification through the **Celery** task ``verification.verify_case_task``.

Pytest fails when the task raises (e.g. ``VerificationCaseFailed`` or import errors),
because results are collected with ``AsyncResult.get(propagate=True)``.

**Without a running worker** (typical CI): run tasks in-process::

    export VALIDATION_CELERY_EAGER=1
    export VALIDATION_S3_BUCKET=your-bucket
    pytest tests/verification/test_verification_s3_celery.py -m s3_verification_celery -v

**With a running worker**: omit ``VALIDATION_CELERY_EAGER``; start Redis + worker first.

See ``conftest.py`` for other ``VALIDATION_*`` variables and pytest flags.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import pytest

pytest.importorskip("celery")

from tests.verification.celery_tasks import config_to_payload, validate_case_task
from tests.verification.env import env_truthy
from tests.verification.timing import (
    print_verification_timing_lines,
    verification_print_timing_enabled,
)
from tests.verification.validate_outputs import VerificationConfig


@pytest.fixture(scope="module", autouse=True)
def _configure_celery_eager_for_pytest() -> Iterator[None]:
    """When VALIDATION_CELERY_EAGER=1, run tasks in-process so pytest does not need a worker."""
    from tests.verification.celery_app import celery_app

    if not env_truthy("VALIDATION_CELERY_EAGER"):
        yield
        return

    prev = {
        "task_always_eager": celery_app.conf.task_always_eager,
        "task_eager_propagates": getattr(
            celery_app.conf, "task_eager_propagates", False
        ),
    }
    celery_app.conf.update(
        task_always_eager=True,
        task_eager_propagates=True,
    )
    yield
    celery_app.conf.update(
        task_always_eager=prev["task_always_eager"],
        task_eager_propagates=prev["task_eager_propagates"],
    )


@pytest.mark.s3_verification_celery
def test_get_outputs_matches_s3_reference_celery_single_case(
    request: pytest.FixtureRequest,
    verification_config: VerificationConfig,
    verification_case: dict[str, Any],
) -> None:
    """One Celery task per parametrized case; failure raises → pytest fails."""
    payload = config_to_payload(verification_config)
    t0 = time.perf_counter()
    async_result = validate_case_task.delay(verification_case, payload)
    row = async_result.get(propagate=True, timeout=None)
    elapsed = time.perf_counter() - t0

    assert row["ok"] is True, row.get("errors", [])
    if verification_print_timing_enabled(request.config):
        lines = [
            f"[verification-celery] case={verification_case} seconds={elapsed:.3f} ok={row['ok']}",
        ]
        for c in row.get("comparisons", []):
            m = c["metrics"]
            lines.append(
                f"  {c['variable']}_{c['suffix']}: mean_rel={m['mean_rel_err']:.3e} "
                f"max_rel={m['max_rel_err']:.3e}"
            )
        print_verification_timing_lines(lines)


@pytest.mark.s3_verification_celery_batch
def test_get_outputs_matches_s3_references_celery_batch(
    request: pytest.FixtureRequest,
    verification_config: VerificationConfig,
    verification_cases: list[dict[str, Any]],
) -> None:
    """All cases via Celery; any task exception fails the test."""
    payload = config_to_payload(verification_config)
    t0 = time.perf_counter()
    for i, case in enumerate(verification_cases, start=1):
        async_result = validate_case_task.delay(case, payload)
        row = async_result.get(propagate=True, timeout=None)
        assert row["ok"] is True, f"case {i}/{len(verification_cases)}: {row.get('errors')}"
    total = time.perf_counter() - t0
    if verification_print_timing_enabled(request.config):
        print_verification_timing_lines(
            [
                f"[verification-celery-batch] cases={len(verification_cases)} "
                f"total_s={total:.3f} "
                f"mean_case_s={total / max(len(verification_cases), 1):.3f}",
            ]
        )
