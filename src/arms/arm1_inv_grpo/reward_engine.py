"""Inv-GRPO Multi-Objective Invariance Reward Engine.

Implements the Orange Innovation Labs mathematical reward formulation from proposal Section 8.1:
    R_total(y, y') = R_exec(y) + R_exec(y') + lambda * R_consistency(y, y') - gamma * P_template(y')

Where:
    - R_exec(y) in {0, 1}       : Binary sandbox unit-test pass/fail for prompt x
    - R_exec(y') in {0, 1}      : Binary sandbox unit-test pass/fail for prompt x'
    - R_consistency(y, y')      : 1.0 if BOTH (y, y') pass, 0.0 otherwise
    - P_template(y')            : 1.0 if y' is a verbatim or near-verbatim mimic of the L0 decoy and fails
    - A_hat_i                   : Group-normalized relative advantage for policy gradient updates
"""

import re
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from src.core.entities import ExecutionResult
from src.infrastructure.sandbox import SubprocessSandbox
from src.arms.arm1_inv_grpo.dataset import PairedTask


def _normalize_code(code: str) -> str:
    """Strips comments, docstrings, thought tags, and collapses whitespace."""
    # Remove thought blocks
    clean = re.sub(r"<thought>.*?</thought>", "", code, flags=re.DOTALL)
    # Remove python comments
    clean = re.sub(r"#.*", "", clean)
    # Remove extra spaces/newlines
    tokens = clean.split()
    return " ".join(tokens)


class InvGRPORewardEngine:
    """Calculates paired multi-view invariance rewards and GRPO group advantages."""

    def __init__(
        self,
        lambda_consistency: float = 0.5,
        gamma_template_penalty: float = 0.5,
        sandbox: Optional[SubprocessSandbox] = None,
        timeout: float = 3.0,
    ):
        self.lambda_cons = lambda_consistency
        self.gamma_pen = gamma_template_penalty
        self.sandbox = sandbox or SubprocessSandbox(default_timeout=timeout)

    def execute_solution(self, prompt: str, solution: str, test: str, entry_point: str) -> ExecutionResult:
        """Executes a single solution inside the isolated Python sandbox."""
        return self.sandbox.execute(prompt, solution, test, entry_point)

    def compute_paired_reward(
        self,
        solution_orig: str,
        solution_pert: str,
        task: PairedTask,
    ) -> Dict[str, Any]:
        """Evaluates a single pair of solutions against the paired task contracts."""
        # 1. Sandbox execution on original problem (L0)
        res_orig = self.execute_solution(
            task.prompt_orig, solution_orig, task.test_orig, task.entry_orig
        )
        r_orig = 1.0 if res_orig.passed else 0.0

        # 2. Sandbox execution on perturbed problem (L2)
        res_pert = self.execute_solution(
            task.prompt_pert, solution_pert, task.test_pert, task.entry_pert
        )
        r_pert = 1.0 if res_pert.passed else 0.0

        # 3. Cross-view Consistency Bonus
        r_cons = 1.0 if (r_orig == 1.0 and r_pert == 1.0) else 0.0

        # 4. Shortcut Template Penalty: penalize if perturbed solution mimicked decoy and failed
        p_tmpl = 0.0
        if r_pert == 0.0 and task.decoy_code.strip():
            norm_decoy = _normalize_code(task.decoy_code)
            norm_sol = _normalize_code(solution_pert)
            # Catch verbatim substring or exact signature reproduction of canonical
            if norm_decoy and norm_decoy in norm_sol:
                p_tmpl = 1.0
            elif task.entry_orig != task.entry_pert and f"def {task.entry_orig}" in solution_pert:
                # Defined the wrong function name from L0 instead of perturbed entry point
                p_tmpl = 1.0

        # 5. Total Invariance-Regularized Reward
        r_total = r_orig + r_pert + (self.lambda_cons * r_cons) - (self.gamma_pen * p_tmpl)

        return {
            "r_total": round(float(r_total), 4),
            "r_exec_orig": r_orig,
            "r_exec_pert": r_pert,
            "r_consistency": r_cons,
            "p_template": p_tmpl,
            "passed_orig": res_orig.passed,
            "passed_pert": res_pert.passed,
        }

    def compute_group_advantages(
        self,
        paired_rollouts: List[Dict[str, str]],
        task: PairedTask,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Evaluates a group of G paired rollouts and normalizes advantages.

        Args:
            paired_rollouts: List of dicts with {"solution_orig": str, "solution_pert": str}
            task: The target PairedTask instance.

        Returns:
            Tuple of (advantages_array, detailed_eval_records).
        """
        eval_records = []
        raw_rewards = []

        for r in paired_rollouts:
            score_dict = self.compute_paired_reward(
                solution_orig=r["solution_orig"],
                solution_pert=r["solution_pert"],
                task=task,
            )
            eval_records.append(score_dict)
            raw_rewards.append(score_dict["r_total"])

        # Group normalization: A_hat_i = (R_i - mean(R)) / (std(R) + epsilon)
        rewards_arr = np.array(raw_rewards, dtype=np.float32)
        mean_r = float(rewards_arr.mean())
        std_r = float(rewards_arr.std())
        advantages = (rewards_arr - mean_r) / (std_r + 1e-8)

        return advantages, eval_records
