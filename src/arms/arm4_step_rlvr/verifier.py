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


import re


def instrument_stepwise_test(test_src: str) -> str:
    """Instruments benchmark test code to track stepwise assertion success rate.

    Literature Basis:
    - CodePRM: Process Reward Models for Code Reasoning (ACL 2025)
    - ExecVerify: Stepwise Execution-Gated Verification (ICSE 2026)

    Evaluates each test assertion independently so partial success receives dense credit.
    """
    lines = test_src.splitlines()
    header = [
        "import sys",
        "_passed_contracts = 0",
        "_total_contracts = 0",
    ]
    new_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def check("):
            new_lines.append(line)
            indent = " " * (len(line) - len(line.lstrip()) + 4)
            new_lines.append(f"{indent}global _passed_contracts, _total_contracts")
            continue

        if stripped.startswith("assert "):
            indent = " " * (len(line) - len(line.lstrip()))
            new_lines.append(f"{indent}_total_contracts += 1")
            new_lines.append(f"{indent}try:")
            new_lines.append(f"{indent}    {stripped}")
            new_lines.append(f"{indent}    _passed_contracts += 1")
            new_lines.append(f"{indent}except Exception:")
            new_lines.append(f"{indent}    pass")
        else:
            new_lines.append(line)

    footer = [
        "",
        "if '_total_contracts' in globals() and _total_contracts > 0:",
        "    if _passed_contracts < _total_contracts:",
        "        raise AssertionError(f'__STEPWISE_RESULT__:{_passed_contracts}:{_total_contracts}')",
        "    else:",
        "        pass  # All passed cleanly",
    ]
    return "\n".join(header + new_lines + footer)


class StepwiseContractVerifier:
    """Evaluates multi-step code by executing independent sub-function contracts."""

    def __init__(self, sandbox: ICodeExecutor):
        self.sandbox = sandbox

    def evaluate_task(self, full_code: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluates code on a task using explicit contracts or instrumented test assertions.

        Implements genuine process verification (CodePRM ACL 2025 / ExecVerify ICSE 2026).
        """
        prompt = task.get("prompt", "")
        entry_point = task.get("entry_point", "")
        contracts = task.get("contracts")

        if contracts:
            return self.evaluate_steps(full_code, contracts, prompt=prompt)

        test_code = task.get("test", "")
        instrumented_test = instrument_stepwise_test(test_code)

        res = self.sandbox.execute(
            prompt=prompt,
            solution=full_code,
            test=instrumented_test,
            entry_point=entry_point,
        )

        output_text = (res.error_message or "") + "\n" + (res.status or "")
        match = re.search(r"__STEPWISE_RESULT__:(\d+):(\d+)", output_text)

        if res.passed:
            passed = 1
            total = 1
            reward = 1.0
        elif match:
            passed = int(match.group(1))
            total = max(int(match.group(2)), 1)
            reward = round(float(passed / total), 4)
        else:
            passed = 0
            total = 1
            reward = 0.0

        return {
            "total_stepwise_reward": reward,
            "passed_contracts": passed,
            "total_contracts": total,
            "passed": res.passed,
            "steps": [{
                "name": f"Stepwise Verification ({passed}/{total})",
                "Step": f"Stepwise Verification ({passed}/{total})",
                "entry_point": entry_point,
                "passed": res.passed,
                "Passed": res.passed,
                "weight": 1.0,
                "Weight": 1.0,
                "credits": reward,
                "Credits": reward,
                "error_message": res.error_message if not res.passed else None,
            }],
        }

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
                solution=full_code,
                test=test_code,
                entry_point=entry_point,
            )
            passed = res.passed
            credits = weight if passed else 0.0
            total_reward += credits

            step_results.append({
                "name": name,
                "Step": name,
                "entry_point": entry_point,
                "passed": passed,
                "Passed": passed,
                "weight": weight,
                "Weight": weight,
                "credits": round(credits, 3),
                "Credits": round(credits, 3),
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
