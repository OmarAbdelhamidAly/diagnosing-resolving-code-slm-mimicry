"""Metric functions for the Reduction Ladder framework.

.. deprecated::
    This module is a **backward-compatibility shim**.
    The canonical implementation lives in ``src/evaluation/metrics.py``.
    All new code should import from there directly.

    The re-exports below ensure that existing notebooks (nb_02, nb_05)
    and ``shared/engine.py`` continue working without changes.

Usage (preferred)::

    from src.evaluation.metrics import compute_ladder_auc, compute_degradation_slope

Usage (legacy, still works)::

    from src.shared.metrics import compute_ladder_auc
"""

# ── Single source of truth ────────────────────────────────────────────────────
from src.evaluation.metrics import (
    compute_pass_at_k,
    compute_ladder_auc,
    compute_collapse_point,
    compute_consistency_delta,
    compute_mri_v2 as compute_mri,        # backward-compat alias
    compute_overthinking_tax as compute_token_efficiency,  # backward-compat alias
    compute_degradation_slope,
    compute_ood_score,
    compute_relative_gain,
)

__all__ = [
    "compute_pass_at_k",
    "compute_ladder_auc",
    "compute_collapse_point",
    "compute_consistency_delta",
    "compute_mri",
    "compute_token_efficiency",
    "compute_degradation_slope",
    "compute_ood_score",
    "compute_relative_gain",
]
