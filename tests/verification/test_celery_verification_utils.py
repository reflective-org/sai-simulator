"""Unit tests for Celery verification helpers (skips when Celery is not installed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.verification.validate_outputs import VerificationConfig


def test_config_to_payload_serializes_paths() -> None:
    """``config_to_payload`` stringifies ``data_dir``, ``model_dir``, and ``cache_dir``."""
    pytest.importorskip("celery")
    from tests.verification.celery_tasks import config_to_payload

    cfg = VerificationConfig(
        data_dir=Path("data/processed"),
        model_dir=Path("data/models"),
        cache_dir=Path("data/cache"),
        s3_bucket="bucket",
    )
    payload = config_to_payload(cfg)
    assert payload["data_dir"] == "data/processed"
    assert payload["model_dir"] == "data/models"
    assert payload["cache_dir"] == "data/cache"
