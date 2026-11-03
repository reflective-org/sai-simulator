"""Before/after comparison harness for the p-value, uncertainty-band and histogram audit.

Read-only with respect to ``backend/``: the harness reconstructs model-subset configurations
from the per-model artifacts the pipeline already writes, persists the expensive
intermediates once per config, and recomputes the three audited quantities live from them so
a code change is visible without regenerating the production cache. See
``docs/pvalues-uncertainty-histograms-review.md``.
"""
