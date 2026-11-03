"""Shared parsing of ``VALIDATION_*`` environment variables."""

from __future__ import annotations

import os
import secrets
from pathlib import Path


def env_str(name: str, default: str) -> str:
    """Return ``name`` from the environment, stripped, or ``default`` if unset or blank."""
    return (os.environ.get(name) or default).strip() or default


def env_int(name: str, default: int) -> int:
    """Parse ``name`` as int via :func:`env_str`, using ``str(default)`` as fallback text."""
    return int(env_str(name, str(default)))


def env_float(name: str, default: float) -> float:
    """Parse ``name`` as float via :func:`env_str`, using ``str(default)`` as fallback text."""
    return float(env_str(name, str(default)))


def env_path(name: str, default: Path) -> Path:
    """Return ``Path`` from ``name`` when set and non-blank; otherwise ``default``."""
    raw = (os.environ.get(name) or "").strip()
    return Path(raw) if raw else default


def env_optional_int(name: str) -> int | None:
    """Parse ``name`` as int when set and non-blank; otherwise ``None``."""
    raw = (os.environ.get(name) or "").strip()
    return int(raw) if raw else None


def env_truthy(name: str) -> bool:
    """True when ``name`` is one of ``1``, ``true``, or ``yes`` (case-insensitive)."""
    return env_str(name, "").lower() in ("1", "true", "yes")


def parse_verification_seed(raw: str) -> int:
    """Integer seed for case generation; ``random`` / ``rand`` → cryptographically strong seed."""
    s = str(raw).strip().lower()
    if s in ("random", "rand"):
        return secrets.randbits(31)
    return int(s or "42")
