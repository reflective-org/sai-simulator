"""Real-data integration tests.

These run against the locally regenerated pipeline outputs (``data/processed``,
``data/models``) and pin numbers measured during the multi-model uncertainty review
(``docs/pvalues-uncertainty-histograms-review.md``). They are skipped when the data is
absent, so CI's ``pytest -q tests/test_*.py`` never sees them and a bare ``pytest`` on a
machine without data skips them cleanly.
"""
