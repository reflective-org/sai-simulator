"""End-to-end verification harness comparing :func:`backend.get_outputs.get_outputs` to S3 references.

Re-exports :func:`generate_test_cases` so callers can drive the harness from a notebook or
script without reaching into submodules.
"""
from tests.verification.generate_test_cases import generate_test_cases

__all__ = ["generate_test_cases"]
