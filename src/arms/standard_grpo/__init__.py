"""Standard GRPO (Vanilla RLVR) Baseline Module.

Implements standard Group Relative Policy Optimization (DeepSeekMath) on isolated prompts,
serving as the direct ablation baseline (Model M4) against Invariance-Regularized GRPO (Model M6).
"""

from src.arms.standard_grpo.trainer import StandardGRPOTrainer

__all__ = ["StandardGRPOTrainer"]
