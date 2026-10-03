"""Arm 5: S³-GRPO (Structural, Stepwise & Invariant Group Relative Policy Optimization).

Flagship hybrid mitigation arm bridging process verification, syntactic tree regularization,
and cross-prompt invariance to eliminate reasoning bloat and supervised mimicry.
"""

from src.arms.arm5_hybrid_s3.reward_engine import S3RewardEngine


def __getattr__(name: str):
    if name == "S3GRPOTrainer":
        from src.arms.arm5_hybrid_s3.trainer import S3GRPOTrainer
        return S3GRPOTrainer
    raise AttributeError(f"module 'src.arms.arm5_hybrid_s3' has no attribute '{name}'")


__all__ = ["S3RewardEngine", "S3GRPOTrainer"]
