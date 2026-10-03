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
from src.arms.arm3_ast_rl.ast_engine import simAST, get_ast_signature
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
        r_step = float(res.get("total_stepwise_reward", res.get("step_reward", 0.0)))
        passed = bool(res.get("passed", res.get("all_passed", False)))
        return r_step, passed

    def compute_ast_similarity(self, solution: str, canonical_ref: str) -> float:
        """Calculates normalized AST tree similarity between solution and canonical reference."""
        if not canonical_ref or not canonical_ref.strip():
            return 0.5  # Neutral prior if canonical solution unavailable
        return float(simAST(solution, canonical_ref))

    def compute_ast_parsimony(self, solution: str, canonical_ref: str) -> Tuple[float, int, int]:
        """Calculates syntactic tree bloat at the AST node level (rather than surface tokens).

        Omega_ast = max(0.0, (|AST(sol)| - |AST(ref)|) / max(|AST(ref)|, 1))
        Capped at 1.0. If code has syntax errors, returns 1.0 bloat penalty.
        """
        if not canonical_ref or not canonical_ref.strip():
            return 0.0, 0, 0

        sig_sol = get_ast_signature(solution)
        sig_ref = get_ast_signature(canonical_ref)

        if "SyntaxError" in sig_sol:
            return 1.0, 0, len(sig_ref)

        len_sol = len(sig_sol)
        len_ref = len(sig_ref)

        if len_sol <= len_ref:
            return 0.0, len_sol, len_ref

        excess = (len_sol - len_ref) / max(len_ref, 1)
        return min(float(excess), 1.0), len_sol, len_ref

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
        """Computes the unified SEGO (Syntactic-Execution Gated Optimization) reward.

        Literature Basis:
        - Yeo et al. (2025) 'Demystifying Long CoT': Naive length subtraction causes
          'premature disengagement' (fast & wrong answers). Regularization MUST be
          execution-gated!
        - TreeDiff / VeriSeek (2025): AST node metrics reflect true algorithmic complexity.

        Formulation:
            If r_step == 0:
                R_SEGO = 0.0  (Free exploration; zero length penalty on failing attempts)
            If r_step > 0:
                R_SEGO = r_step * [1.0 + alpha * sim_AST - gamma * Omega_AST] - lambda * L_inv
        """
        # 1. Dense Stepwise Process Verification
        r_step, all_passed = self.compute_stepwise_reward(prompt, solution, test, entry_point)

        # 2. Structural AST Fidelity
        r_ast = self.compute_ast_similarity(solution, canonical_solution)

        # 3. AST Syntactic Tree Bloat Penalty (Omega_AST)
        omega_ast, nodes_sol, nodes_ref = self.compute_ast_parsimony(solution, canonical_solution)

        # 4. Cross-Prompt Invariance Regularization
        l_inv = 0.0
        if paired_prompt and paired_test and solution_pert:
            r_pert_step, _ = self.compute_stepwise_reward(
                paired_prompt, solution_pert, paired_test, paired_entry_point or entry_point
            )
            l_inv = abs(r_step - r_pert_step)

        # 5. Execution-Gated Multiplicative Synthesis (SEGO)
        alpha = self.w_ast         # Default 0.30
        gamma = self.w_parsimony   # Default 0.20
        lambda_inv = self.w_inv    # Default 0.15

        if r_step <= 0.0:
            # Execution Gating: When failing, NO length penalty is applied to prevent
            # premature disengagement (Yeo et al., 2025)
            r_final = 0.0
        else:
            # Modulate passing credit by tree structural fidelity and syntactic parsimony
            structural_multiplier = 1.0 + (alpha * r_ast) - (gamma * omega_ast)
            # Bound multiplier to [0.5, 1.5]
            structural_multiplier = max(0.5, min(structural_multiplier, 1.5))
            r_base = r_step * structural_multiplier

            # Subtract cross-prompt representation divergence
            r_composite = r_base - (lambda_inv * l_inv)

            # Bonus floor for pristine, canonical solutions passing 100% of assertions with low bloat
            if all_passed and omega_ast <= 0.10:
                r_composite += 0.20

            r_final = max(0.0, min(float(r_composite), 1.5))

        return {
            "r_total": round(r_final, 4),
            "r_step": round(r_step, 4),
            "r_ast": round(r_ast, 4),
            "l_inv": round(l_inv, 4),
            "omega_ast": round(omega_ast, 4),
            "nodes_sol": nodes_sol,
            "nodes_ref": nodes_ref,
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
