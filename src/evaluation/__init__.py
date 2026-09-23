"""Isolated Evaluation Framework for the Reduction Ladder for Code project.

This module is completely decoupled from training. It evaluates any trained
checkpoint identically on a fixed, immutable benchmark pool.

Public API:
    EvaluationSuite   — main orchestrator, evaluate_model(model_id, ...)
    BenchmarkRegistry — maps level keys to JSONL task files
    EvaluationReporter — generates publication-quality comparison plots
    metrics           — full metric stack (9 functions)
"""

from .suite import EvaluationSuite
from .registry import BenchmarkRegistry
from .reporter import EvaluationReporter
from . import metrics

__all__ = [
    "EvaluationSuite",
    "BenchmarkRegistry",
    "EvaluationReporter",
    "metrics",
]
