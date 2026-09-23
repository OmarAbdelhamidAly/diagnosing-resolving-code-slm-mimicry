"""Reward engine for Arm 3: AST-RL.

Combines binary sandbox execution reward with structural AST similarity reward:
    R_total(y, y*) = R_exec(y) + beta * simAST(y, y*)
"""

from typing import Optional, Dict, Any
from src.arms.arm3_ast_rl.ast_engine import simAST, ast_reward


class ASTRewardEngine:
    """Computes hybrid execution + AST structural guidance reward."""

    def __init__(self, beta: float = 0.3, alpha_tree: float = 0.05):
        """
        Args:
            beta: Weight for the AST similarity bonus (default 0.3).
            alpha_tree: Exponential decay factor for tree distance.
        """
        self.beta = beta
        self.alpha_tree = alpha_tree

    def compute_reward(
        self,
        code_gen: str,
        code_ref: str,
        exec_passed: bool,
    ) -> Dict[str, float]:
        """Compute the total composite reward and diagnostic components.

        Args:
            code_gen: Generated Python code.
            code_ref: Reference ground-truth canonical solution.
            exec_passed: Whether the code passed all unit tests in sandbox.

        Returns:
            Dict containing 'total_reward', 'exec_reward', 'ast_similarity', and 'ast_bonus'.
        """
        r_exec = 1.0 if exec_passed else 0.0
        sim = simAST(code_gen, code_ref)
        ast_bonus = self.beta * sim

        total = round(r_exec + ast_bonus, 4)
        return {
            "total_reward": total,
            "exec_reward": r_exec,
            "ast_similarity": sim,
            "ast_bonus": round(ast_bonus, 4),
        }
