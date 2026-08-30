"""Evaluation subpackage providing the unified EvaluationEngine, metrics, and visualization tools."""

from src.evaluation.engine import EvaluationEngine
from src.evaluation.metrics import (
    compute_pass_at_k,
    compute_ladder_auc,
    compute_collapse_point,
    compute_consistency_delta,
    compute_mri,
    compute_token_efficiency,
)
from src.evaluation.plots import (
    plot_single_model_degradation,
    plot_error_taxonomy,
    plot_multi_model_comparison,
)

__all__ = [
    "EvaluationEngine",
    "compute_pass_at_k",
    "compute_ladder_auc",
    "compute_collapse_point",
    "compute_consistency_delta",
    "compute_mri",
    "compute_token_efficiency",
    "plot_single_model_degradation",
    "plot_error_taxonomy",
    "plot_multi_model_comparison",
]
