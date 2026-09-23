"""Multi-Arm Mitigation Package for Small Language Model Mimicry.

Contains modular implementations of the 5 mitigation arms:
  - arm1_inv_grpo:       Invariance-Regularized Policy Optimization (Primary Innovation)
  - standard_grpo:       Standard Binary RLVR (Group Relative Policy Optimization baseline)
  - arm2_contrastive_dpo: Contrastive Thought-Template SFT / DPO
  - arm3_ast_rl:          AST-Guided Policy Optimization (TreeDiff/VeriSeek)
  - arm4_step_rlvr:       Stepwise Execution-Gated Process RLVR (CodePRM)
"""

from src.arms.arm2_contrastive_dpo import (
    ContrastiveDatasetParser,
    to_dpo_triplet,
    format_dpo_dataset,
)
from src.arms.arm3_ast_rl import (
    ASTNormalizer,
    get_ast_signature,
    simAST,
    ast_reward,
    ASTRewardEngine,
)
from src.arms.arm4_step_rlvr import (
    StepContract,
    StepwiseContractVerifier,
    StepwiseRewardEngine,
)

__all__ = [
    "ContrastiveDatasetParser",
    "to_dpo_triplet",
    "format_dpo_dataset",
    "ASTNormalizer",
    "get_ast_signature",
    "simAST",
    "ast_reward",
    "ASTRewardEngine",
    "StepContract",
    "StepwiseContractVerifier",
    "StepwiseRewardEngine",
]
