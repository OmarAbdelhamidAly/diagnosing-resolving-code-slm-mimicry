"""Arm 1: Invariance-Regularized Policy Optimization (Inv-GRPO).

Primary scientific mitigation arm for Orange Innovation Labs research proposal.
"""

from src.arms.arm1_inv_grpo.dataset import PairedTask, InvGRPODatasetLoader
from src.arms.arm1_inv_grpo.reward_engine import InvGRPORewardEngine
from src.arms.arm1_inv_grpo.trainer import InvGRPOTrainer
from src.arms.arm1_inv_grpo.evaluator import InvGRPOEvaluator

__all__ = [
    "PairedTask",
    "InvGRPODatasetLoader",
    "InvGRPORewardEngine",
    "InvGRPOTrainer",
    "InvGRPOEvaluator",
]
