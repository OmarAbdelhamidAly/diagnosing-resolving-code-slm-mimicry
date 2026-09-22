"""Shared module: data ingestion, evaluation engine, metrics, and plots.

Used across multiple stages:
  - data_service  → Stage 1 (nb_01) + Stage 3 (nb_04)
  - engine        → Stage 2 (nb_02) + Stage 5 (nb_05)
  - metrics       → Stage 2 (nb_02) + Stage 5 (nb_05)
  - plots         → Stage 2 (nb_02) + Stage 5 (nb_05)
"""

from src.shared.data_service import DataService
from src.shared.engine import EvaluationEngine
from src.shared.metrics import (
    compute_ladder_auc,
    compute_collapse_point,
    compute_consistency_delta,
    compute_mri,
    compute_token_efficiency,
)
from src.shared.plots import (
    plot_single_model_degradation,
    plot_error_taxonomy,
    plot_multi_model_comparison,
)

__all__ = [
    "DataService",
    "EvaluationEngine",
    "compute_ladder_auc",
    "compute_collapse_point",
    "compute_consistency_delta",
    "compute_mri",
    "compute_token_efficiency",
    "plot_single_model_degradation",
    "plot_error_taxonomy",
    "plot_multi_model_comparison",
]
