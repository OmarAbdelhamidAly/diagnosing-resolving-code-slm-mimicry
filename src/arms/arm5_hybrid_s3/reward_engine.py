"""S³-GRPO Composite Multi-Objective Reward Engine.

Integrates:
1. Dense Stepwise Process Verification (CodePRM / ExecVerify)
2. Normalized Abstract Syntax Tree Structural Fidelity (TreeDiff / VeriSeek)
3. Cross-Prompt Invariance & Anti-Shortcut Regularization
4. Anti-Overthinking Parsimony Regularization (Information-Theoretic Penalty)

Mathematical Formulation:
    R_total(y, x) = w_step * R_step(y)
                  + w_ast  * simAST(y, y*)
                  - w_inv  * L_inv(y, y')
                  - w_tax  * Omega_parsimony(y, y*)
"""

import math
import re
from typing import Dict, Any, List, Optional, Tuple

from src.core.entities import ExecutionResult
from src.infrastructure.sandbox import SubprocessSandbox
from src.arms.arm3_ast_rl.ast_engine import simAST
from src.arms.arm4_step_rlvr.verifier import StepwiseContractVerifier, instrument_stepwise_test


def normalize_code_tokens(code: str) -> str:
    """Strips comments, thought tags, and collapses whitespace."""
    clean = re.sub(r"<thought>.*?</thought>", "", code, flags=re.DOTALL)
    clean = re.sub(r"#.*", "", clean)
    tokens = clean.split()
    return " ".join(tokens)


class S3RewardEngine:
    """Computes the unified S³-GRPO composite reward signal for code generation rollouts."""

    def __init__(
        self,
        sandbox: Optional[SubprocessSandbox] = None,
        w_step: float = 0.50,
        w_ast: float = 0.30,
        w_inv: float = 0.20,
        w_parsimony: float = 0.15,
        sandbox_timeout: float = 3.0,
    ):
        self.sandbox = sandbox or SubprocessSandbox(default_timeout=sandbox_timeout)
        self.step_verifier = StepwiseContractVerifier(sandbox=self.sandbox)
        self.w_step = w_step
        self.w_ast = w_ast
        self.w_inv = w_inv
        self.w_parsimony = w_parsimony

    def compute_stepwise_reward(
        self, prompt: str, solution: str, test: str, entry_point: str
    ) -> Tuple[float, bool]:
        """Calculates dense process credit by evaluating independent test assertions."""
        task_dict = {
            "prompt": prompt,
            "test": test,
            "entry_point": entry_point,
        }
        res = self.step_verifier.evaluate_task(solution, task_dict)
        r_step = float(res.get("step_reward", 0.0))
        passed = bool(res.get("all_passed", False))
        return r_step, passed

    def compute_ast_similarity(self, solution: str, canonical_ref: str) -> float:
        """Calculates normalized AST tree similarity between solution and canonical reference."""
        if not canonical_ref or not canonical_ref.strip():
            return 0.5  # Neutral prior if canonical solution unavailable
        return float(simAST(solution, canonical_ref))

    def compute_parsimony_penalty(self, solution: str, canonical_ref: str) -> float:
        """Penalizes runaway reasoning tokens and boilerplate hallucination.

        Omega_parsimony = max(0, (len(sol) - len(ref)) / max(len(ref), 1))
        Capped at 1.0 to avoid unbounded negative penalties.
        """
        if not canonical_ref or not canonical_ref.strip():
            return 0.0

        sol_len = len(solution.split())
        ref_len = len(canonical_ref.split())

        if sol_len <= ref_len:
            return 0.0

        excess_ratio = (sol_len - ref_len) / max(ref_len, 1)
        return min(float(excess_ratio), 1.0)

    def compute_reward(
        self,
        prompt: str,
        solution: str,
        test: str,
        entry_point: str,
        canonical_solution: str = "",
        paired_prompt: str = "",
        paired_test: str = "",
        paired_entry_point: str = "",
        solution_pert: str = "",
    ) -> Dict[str, Any]:
        """Computes the full composite S³-GRPO reward for a generated rollout.

        Returns a detailed breakdown dictionary for transparent logging.
        """
        # 1. Dense Stepwise Process Verification
        r_step, all_passed = self.compute_stepwise_reward(prompt, solution, test, entry_point)

        # 2. Structural AST Fidelity
        r_ast = self.compute_ast_similarity(solution, canonical_solution)

        # 3. Anti-Overthinking Parsimony Penalty
        p_tax = self.compute_parsimony_penalty(solution, canonical_solution)

        # 4. Cross-Prompt Invariance Regularization (if paired task is provided)
        l_inv = 0.0
        r_pert_step = 0.0
        if paired_prompt and paired_test and solution_pert:
            r_pert_step, pert_passed = self.compute_stepwise_reward(
                paired_prompt, solution_pert, paired_test, paired_entry_point or entry_point
            )
            # Invariance loss: difference in execution performance under surface shift
            l_inv = abs(r_step - r_pert_step)

        # 5. Composite S³-GRPO Reward
        # Base components: Stepwise pass rate + Structural AST guidance
        r_composite = (self.w_step * r_step) + (self.w_ast * r_ast)

        # Regularization deductions: Invariance divergence + Reasoning bloat
        r_composite -= (self.w_inv * l_inv)
        r_composite -= (self.w_parsimony * p_tax)

        # If completely passed, guarantee a clean bonus floor
        if all_passed:
            r_composite += 0.20

        # Bound reward to [0.0, 1.5]
        r_final = max(0.0, min(float(r_composite), 1.5))

        return {
            "r_total": round(r_final, 4),
            "r_step": round(r_step, 4),
            "r_ast": round(r_ast, 4),
            "l_inv": round(l_inv, 4),
            "p_parsimony": round(p_tax, 4),
            "all_passed": all_passed,
        }

    def compute_group_advantages(self, rewards: List[float], eps: float = 1e-6) -> List[float]:
        """Computes GRPO relative advantages across group rollouts G.

        A_i = (R_i - mean(R)) / (std(R) + eps)
        """
        if not rewards:
            return []
        if len(rewards) == 1:
            return [0.0]

        mean_r = sum(rewards) / len(rewards)
        variance = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
        std_r = math.sqrt(variance)

        if std_r < eps:
            return [0.0 for _ in rewards]

        return [round((r - mean_r) / (std_r + eps), 4) for r in rewards]
