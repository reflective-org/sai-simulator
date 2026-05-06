"""Optional stderr timing lines for verification pytest modules."""

from __future__ import annotations

import sys

import pytest

from tests.verification.env import env_truthy


def verification_print_timing_enabled(pytest_config: pytest.Config) -> bool:
    """True when timing lines should be printed (CLI flag or ``VALIDATION_PRINT_TIMING``)."""
    if pytest_config.getoption("--validation-print-timing"):
        return True
    return env_truthy("VALIDATION_PRINT_TIMING")


def print_verification_timing_lines(lines: list[str]) -> None:
    """Write ``lines`` to stderr with flush (reliable under pytest capture)."""
    for line in lines:
        sys.stderr.write(line + "\n")
    sys.stderr.flush()
