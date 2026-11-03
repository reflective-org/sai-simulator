"""Celery tasks and JSON-safe :class:`~tests.verification.validate_outputs.VerificationConfig` payloads."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from tests.verification.celery_app import celery_app
from tests.verification.repo_root import ensure_repo_on_sys_path
from tests.verification.validate_outputs import (
    VerificationConfig,
    geotiff_comparison_to_dict,
    validate_case,
)


class VerificationCaseFailed(RuntimeError):
    """Raised when a verification case finishes with logical failures."""


def _config_from_payload(payload: dict[str, Any]) -> VerificationConfig:
    """Rebuild :class:`~tests.verification.validate_outputs.VerificationConfig` from a JSON payload."""
    payload = dict(payload)
    payload["data_dir"] = Path(payload["data_dir"])
    payload["model_dir"] = Path(payload["model_dir"])
    payload["cache_dir"] = Path(payload["cache_dir"])
    return VerificationConfig(**payload)


def config_to_payload(config: VerificationConfig) -> dict[str, Any]:
    """Serialize ``config`` for Celery JSON; directory fields become strings."""
    data = asdict(config)
    data["data_dir"] = str(config.data_dir)
    data["model_dir"] = str(config.model_dir)
    data["cache_dir"] = str(config.cache_dir)
    return data


@celery_app.task(name="verification.ping_task")
def ping_task() -> str:
    """Minimal task to confirm workers execute messages (used in CI preflight)."""
    return "pong"


@celery_app.task(name="verification.verify_case_task")
def validate_case_task(case: dict[str, Any], config_payload: dict[str, Any]) -> dict[str, Any]:
    """
    Run :func:`~tests.verification.validate_outputs.validate_case` in a worker.

    Ensures the repo root is on ``sys.path``, then returns a JSON-serializable result dict.
    Raises :exc:`VerificationCaseFailed` when verification fails logically.
    """
    ensure_repo_on_sys_path()
    config = _config_from_payload(config_payload)
    result = validate_case(case, config)
    payload = {
        "case": result.case,
        "ok": result.ok,
        "errors": result.errors,
        "comparisons": [
            geotiff_comparison_to_dict(c, include_local_path=True)
            for c in result.comparisons
        ],
    }
    if not result.ok:
        raise VerificationCaseFailed(
            f"Verification failed for case={result.case}. Errors={result.errors}"
        )
    return payload
