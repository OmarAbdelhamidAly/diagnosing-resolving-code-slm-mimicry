from src.arms.arm3_ast_rl.ast_engine import (
    ASTNormalizer,
    get_ast_signature,
    simAST,
    ast_reward,
)
from src.arms.arm3_ast_rl.reward_engine import ASTRewardEngine


def __getattr__(name: str):
    if name == "ASTRLTrainer":
        from src.arms.arm3_ast_rl.trainer import ASTRLTrainer
        return ASTRLTrainer
    raise AttributeError(f"module 'src.arms.arm3_ast_rl' has no attribute '{name}'")


__all__ = [
    "ASTNormalizer",
    "get_ast_signature",
    "simAST",
    "ast_reward",
    "ASTRewardEngine",
    "ASTRLTrainer",
]

