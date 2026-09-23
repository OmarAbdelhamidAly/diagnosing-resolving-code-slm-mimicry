"""Complete metric stack for the Reduction Ladder evaluation framework.

All functions are pure (no side-effects) and framework-independent.

References
----------
- Pass@k unbiased estimator  : Chen et al., "Evaluating Large Language Models
  Trained on Code", arXiv:2107.03374 (HumanEval, 2021)
- Degradation Slope          : Apple Research, "GSM-Symbolic: Understanding the
  Limitations of Mathematical Reasoning in LLMs", arXiv:2410.05229 (2024)
- Overthinking Tax           : Muennighoff et al., "s1: Simple Test-Time Scaling",
  arXiv:2501.12599 (2025)
- OOD Generalization (Ctrl)  : Jain et al., "LiveCodeBench: Holistic and
  Contamination-Free Evaluation of LLMs for Code", arXiv:2403.07974 (2024)
- MRI / Consistency Delta    : Reduction Ladder for Code proposal (our work)
"""

from __future__ import annotations

import math
from difflib import SequenceMatcher
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# 1. Pass@k  (Chen et al., 2021)
# ---------------------------------------------------------------------------

def compute_pass_at_k(n: int, c: int, k: int = 1) -> float:
    """Unbiased Pass@k estimator.

    Parameters
    ----------
    n : total samples generated per task
    c : number of samples that pass all unit tests
    k : target k (1 or 5)

    Returns
    -------
    float in [0, 1]
    """
    if n == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def compute_pass_at_1_greedy(passed: bool) -> float:
    """Deterministic greedy Pass@1 (single sample, temperature=0)."""
    return 1.0 if passed else 0.0


# ---------------------------------------------------------------------------
# 2. Ladder AUC  (our proposal)
# ---------------------------------------------------------------------------

def compute_ladder_auc(level_pass_rates: Dict[str, float]) -> float:
    """Area under the Pass@1 degradation curve across L0–L5.

    Uses the trapezoidal rule over the 6 levels.  A perfect model scores 1.0.
    A model that collapses immediately at L1 scores ≈ 0.08.

    Parameters
    ----------
    level_pass_rates : dict mapping level key → Pass@1 float, e.g.
        {"L0": 0.90, "L1": 0.82, ..., "L5": 0.40}
    """
    levels = ["L0", "L1", "L2", "L3", "L4", "L5"]
    scores = [level_pass_rates.get(lvl, 0.0) for lvl in levels]
    if not any(lvl in level_pass_rates for lvl in levels):
        return 0.0
    # Trapezoidal rule: n=5 intervals over 6 points, normalise to [0,1]
    auc = 0.0
    for i in range(len(scores) - 1):
        auc += 0.5 * (scores[i] + scores[i + 1])
    return auc / (len(scores) - 1)


# ---------------------------------------------------------------------------
# 3. Degradation Slope  (inspired by GSM-Symbolic, Apple 2024)
# ---------------------------------------------------------------------------

def compute_degradation_slope(level_pass_rates: Dict[str, float]) -> float:
    """Linear regression slope of Pass@1 across L0→L5.

    A steeper negative slope indicates worse robustness to transformations.
    Returns slope in units of Δ(Pass@1) per level step.

    References
    ----------
    Apple Research (2024) use a similar linear-fit analysis to show that
    symbolic perturbations cause monotonic degradation in mathematical
    reasoning models (GSM-Symbolic, §3).
    """
    levels = ["L0", "L1", "L2", "L3", "L4", "L5"]
    xy = [(i, level_pass_rates[lvl]) for i, lvl in enumerate(levels) if lvl in level_pass_rates]
    if len(xy) < 2:
        return 0.0
    xs = [p[0] for p in xy]
    ys = [p[1] for p in xy]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return num / den


# ---------------------------------------------------------------------------
# 4. Collapse Point  (our proposal)
# ---------------------------------------------------------------------------

def compute_collapse_point(
    level_pass_rates: Dict[str, float],
    threshold: float = 0.50,
) -> str:
    """Earliest ladder level where Pass@1 drops below *threshold* (default 50%).

    Returns the level key string, or "None" if the model never collapses.
    """
    for lvl in ["L0", "L1", "L2", "L3", "L4", "L5"]:
        if lvl in level_pass_rates and level_pass_rates[lvl] < threshold:
            return lvl
    return "None (Maintained ≥50%)"


# ---------------------------------------------------------------------------
# 5. MRI-v2 with automatic template similarity  (our proposal)
# ---------------------------------------------------------------------------

def compute_mri_v2(
    p1_l0: float,
    p1_transformed: float,
    original_prompt: str = "",
    transformed_prompt: str = "",
) -> float:
    """Memorization Risk Index v2.

    MRI = similarity(orig, transformed) × max(0, Pass@1_L0 − Pass@1_transformed)

    When template_similarity = 1.0 (prompts are identical), this reduces to
    the raw performance drop.  As prompts diverge, the penalty is discounted
    because the model *should* behave differently on different problems.

    Parameters
    ----------
    p1_l0          : Pass@1 at the original L0 (HumanEval) level
    p1_transformed : Pass@1 at the transformed level (e.g. L1)
    original_prompt, transformed_prompt : text strings used to auto-compute
        similarity via SequenceMatcher.  Pass empty strings to use similarity=1.
    """
    if original_prompt and transformed_prompt:
        similarity = SequenceMatcher(
            None, original_prompt, transformed_prompt
        ).ratio()
    else:
        similarity = 1.0
    return similarity * max(0.0, p1_l0 - p1_transformed)


# ---------------------------------------------------------------------------
# 6. Consistency Delta  (our proposal)
# ---------------------------------------------------------------------------

def compute_consistency_delta(p1_l1: float, p1_l2: float) -> float:
    """Sensitivity to neutral surface-level perturbations.

    Measures |Pass@1_L1 - Pass@1_L2|.  High values indicate the model is
    brittle to cosmetic changes unrelated to algorithmic difficulty.
    """
    return abs(p1_l1 - p1_l2)


# ---------------------------------------------------------------------------
# 7. OOD Generalization Score  (LiveCodeBench Ctrl level, Jain et al. 2024)
# ---------------------------------------------------------------------------

def compute_ood_score(ctrl_pass_at_1: float) -> float:
    """Out-of-distribution generalization score on the Ctrl (LiveCodeBench Lite) level.

    Returns the raw Pass@1 on the temporal control set.  This level uses
    problems published *after* the training cutoff to prevent contamination.

    References
    ----------
    Jain et al., "LiveCodeBench: Holistic and Contamination-Free Evaluation
    of LLMs for Code", arXiv:2403.07974 (2024).
    """
    return float(ctrl_pass_at_1)


# ---------------------------------------------------------------------------
# 8. Overthinking Tax  (Muennighoff et al., 2025)
# ---------------------------------------------------------------------------

def compute_overthinking_tax(
    accuracy: float,
    avg_reasoning_tokens: float,
) -> float:
    """Per-Token Intelligence ratio (PTI).

    PTI = (accuracy / avg_reasoning_tokens) × 100

    Higher is better: a high-PTI model achieves the same accuracy with fewer
    thinking tokens.  A model that generates 3× more tokens for the same
    accuracy has 1/3 the PTI.

    References
    ----------
    Muennighoff et al., "s1: Simple Test-Time Scaling", arXiv:2501.12599 (2025),
    §4: "thinking token budget" analysis.
    """
    if avg_reasoning_tokens <= 0:
        return 0.0
    return (accuracy / avg_reasoning_tokens) * 100.0


# ---------------------------------------------------------------------------
# 9. Relative Gain vs. Baseline  (standard in RL papers)
# ---------------------------------------------------------------------------

def compute_relative_gain(
    model_metric: float,
    baseline_metric: float,
) -> float:
    """Relative improvement of a model over the M1 baseline (percent points).

    Returns model_metric − baseline_metric.
    Positive values indicate improvement; negative indicates regression.
    """
    return model_metric - baseline_metric
