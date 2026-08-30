"""Mathematical evaluation metrics for the Reduction Ladder framework.

Computes:
- Pass@1 (Greedy deterministic accuracy)
- Pass@5 (Unbiased nucleus sampling accuracy)
- Collapse Point (l* - lowest level where Pass@1 drops below threshold)
- Ladder AUC (Area under the degradation curve)
- Consistency Delta (Sensitivity to neutral perturbations)
- Memorization Risk Index (MRI)
- Token Efficiency and Overthinking Tax
"""

from typing import Dict, List, Any, Optional
import math


def compute_pass_at_k(n: int, c: int, k: int = 5) -> float:
    """Calculate unbiased pass@k metric based on Chen et al. (HumanEval 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def compute_ladder_auc(level_pass_rates: Dict[str, float]) -> float:
    """Compute the Area Under the Ladder Curve (AUC) across L0-L5.
    
    A score of 1.0 indicates perfect algorithmic invariance.
    """
    levels = ["L0", "L1", "L2", "L3", "L4", "L5"]
    scores = [level_pass_rates.get(lvl, 0.0) for lvl in levels if lvl in level_pass_rates]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def compute_collapse_point(level_pass_rates: Dict[str, float], threshold: float = 0.50) -> str:
    """Find the earliest ladder level where Pass@1 drops below the threshold (default 50%)."""
    for lvl in ["L0", "L1", "L2", "L3", "L4", "L5"]:
        if lvl in level_pass_rates:
            if level_pass_rates[lvl] < threshold:
                return lvl
    return "None (Maintained > 50%)"


def compute_consistency_delta(p1_l1: float, p1_l2: float) -> float:
    """Compute performance difference under neutral surface variations."""
    return abs(p1_l1 - p1_l2)


def compute_mri(p1_l0: float, p1_transformed: float, template_similarity: float = 1.0) -> float:
    """Compute Memorization Risk Index (MRI) based on Yang et al. (2025)."""
    return template_similarity * max(0.0, p1_l0 - p1_transformed)


def compute_token_efficiency(accuracy: float, avg_reasoning_tokens: float) -> float:
    """Compute Per-Token Intelligence ratio (accuracy achieved per 100 reasoning tokens)."""
    if avg_reasoning_tokens <= 0:
        return 0.0
    return (accuracy / avg_reasoning_tokens) * 100.0
