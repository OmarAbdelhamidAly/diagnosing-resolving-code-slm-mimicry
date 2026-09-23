"""Stepwise contract verifier and process reward engine for Arm 4: Step-RLVR.

Literature Basis:
- CodePRM: Process Reward Models for Code Reasoning (ACL 2025)
- ExecVerify: Stepwise Execution-Gated Verification (ICSE 2026)

On complex multi-goal algorithmic levels (L4: Difficult, L5: Combine), standard
binary RLVR is sparse (R=0 if any edge case fails, discarding all correct sub-steps).
Step-RLVR evaluates intermediate sub-function contracts to award dense, stepwise partial credits:
    R_stepwise(y) = sum_{s=1}^S w_s * I(Contract_s(y) == Valid)
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from src.core.interfaces import ICodeExecutor


@dataclass
class StepContract:
    """Represents an executable sub-function contract test specification."""
    name: str
    entry_point: str
    weight: float
    test: str


class StepwiseContractVerifier:
    """Evaluates multi-step code by executing independent sub-function contracts."""

    def __init__(self, sandbox: ICodeExecutor):
        self.sandbox = sandbox

    def evaluate_steps(
        self,
        full_code: str,
        step_specs: List[Union[StepContract, Dict[str, Any]]],
        prompt: str = "",
    ) -> Dict[str, Any]:
        """Runs each sub-function contract test and accumulates weighted partial credits.

        Args:
            full_code: Full generated Python code containing one or more subroutines.
            step_specs: List of StepContract objects or equivalent dictionaries.
            prompt: Optional problem prompt context.

        Returns:
            Dict containing:
                - total_stepwise_reward: Sum of earned credits (float in [0, 1]).
                - steps: Detailed per-step execution breakdown list.
        """
        step_results = []
        total_reward = 0.0

        for item in step_specs:
            if isinstance(item, StepContract):
                name = item.name
                entry_point = item.entry_point
                weight = item.weight
                test_code = item.test
            else:
                name = item.get("name", "Unknown Step")
                entry_point = item.get("entry_point", "")
                weight = float(item.get("weight", 0.0))
                test_code = item.get("test", "")

            res = self.sandbox.execute(
                prompt=prompt,
                generation=full_code,
                test_code=test_code,
                entry_point=entry_point,
            )
            passed = res.passed
            credits = weight if passed else 0.0
            total_reward += credits

            step_results.append({
                "name": name,
                "entry_point": entry_point,
                "passed": passed,
                "weight": weight,
                "credits": round(credits, 3),
                "error_message": res.error_message if not passed else None,
            })

        return {
            "total_stepwise_reward": round(total_reward, 3),
            "steps": step_results,
        }


class StepwiseRewardEngine:
    """Computes comparison metrics between sparse terminal RLVR and dense Step-RLVR."""

    @staticmethod
    def compare_rewards(
        stepwise_result: Dict[str, Any],
        terminal_passed: bool,
    ) -> Dict[str, float]:
        """Compares sparse binary RLVR reward against dense Step-RLVR reward.

        Args:
            stepwise_result: Output from StepwiseContractVerifier.evaluate_steps().
            terminal_passed: Whether the complete program passed all tests end-to-end.

        Returns:
            Dict containing 'binary_reward', 'stepwise_reward', and 'dense_signal_gain'.
        """
        binary = 1.0 if terminal_passed else 0.0
        stepwise = float(stepwise_result.get("total_stepwise_reward", 0.0))
        gain = round(stepwise - binary, 3)

        return {
            "binary_reward": binary,
            "stepwise_reward": stepwise,
            "dense_signal_gain": gain,
        }
