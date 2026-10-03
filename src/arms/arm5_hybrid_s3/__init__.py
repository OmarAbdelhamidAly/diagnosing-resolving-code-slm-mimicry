"""Arm 5: S³-GRPO (Structural, Stepwise & Invariant Group Relative Policy Optimization).

Flagship hybrid mitigation arm bridging process verification, syntactic tree regularization,
and cross-prompt invariance to eliminate reasoning bloat and supervised mimicry.
"""

from src.arms.arm5_hybrid_s3.reward_engine import S3RewardEngine
from src.arms.arm5_hybrid_s3.trainer import S3GRPOTrainer

__all__ = ["S3RewardEngine", "S3GRPOTrainer"]
