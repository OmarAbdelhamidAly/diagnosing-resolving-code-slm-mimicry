"""Error classifier diagnosing failure modes into domain error categories."""

import re
from src.core.interfaces import IErrorClassifier
from src.core.entities import BenchmarkTask, ExecutionResult, ErrorCategory


class RuleBasedErrorClassifier(IErrorClassifier):
    """Diagnoses model generation errors into On-Path, Off-Path, Wrong-Template, Syntax, or Runtime errors."""

    # Common canonical HumanEval problem signature markers
    CANONICAL_PROBLEM_SIGNATURES = {
        "two_sum": [r"two_sum", r"seen\s*=\s*\{\}", r"target\s*-\s*num"],
        "fibonacci": [r"fib", r"fibonacci", r"a,\s*b\s*=\s*0,\s*1"],
        "prime_check": [r"is_prime", r"i\s*\*\s*i\s*<=\s*n"],
        "palindrome": [r"is_palindrome", r"s\[::-1\]"],
        "binary_search": [r"binary_search", r"left\s*<=\s*right", r"mid\s*=\s*\(left"],
    }

    def classify(
        self,
        task: BenchmarkTask,
        generated_code: str,
        execution_result: ExecutionResult
    ) -> str:
        if execution_result.passed:
            return ErrorCategory.PASS.value

        if execution_result.status == "TIMEOUT":
            return ErrorCategory.TIMEOUT.value

        if execution_result.status == "CRASH":
            return ErrorCategory.CRASH.value

        err_msg = execution_result.error_message

        # 1. Syntax & Indentation Errors
        if "SyntaxError" in err_msg or "IndentationError" in err_msg or "TabError" in err_msg:
            return ErrorCategory.SYNTAX_ERROR.value

        # 2. Wrong Template Heuristic (Template Memorization Collapse)
        # Check if the generated code attempts to solve a different standard textbook problem
        if self._is_wrong_template_collapse(task, generated_code):
            return ErrorCategory.WRONG_TEMPLATE.value

        # 3. Assertion Failures -> Check if on-path or off-path
        if "AssertionError" in err_msg:
            if self._is_on_path_logic(task, generated_code):
                return ErrorCategory.ON_PATH.value
            else:
                return ErrorCategory.OFF_PATH.value

        # 4. Standard Runtime Errors (IndexError, TypeError, KeyError, etc.)
        return ErrorCategory.RUNTIME_ERROR.value

    def _is_wrong_template_collapse(self, task: BenchmarkTask, code: str) -> bool:
        """Check if code regurgitated an unrelated textbook solution pattern."""
        code_lower = code.lower()
        prompt_lower = task.prompt.lower()

        # If problem is not about fibonacci, but model wrote a standalone fibonacci generator
        for concept, patterns in self.CANONICAL_PROBLEM_SIGNATURES.items():
            if concept not in prompt_lower:
                matches = sum(1 for p in patterns if re.search(p, code_lower))
                if matches >= 2:
                    return True
        return False

    def _is_on_path_logic(self, task: BenchmarkTask, code: str) -> bool:
        """Check if code contains core algorithmic logic but failed an edge case or off-by-one."""
        if not task.entry_point:
            return False

        # If it defines the correct function and contains meaningful control flow
        has_entry = task.entry_point in code
        has_control = any(k in code for k in ["for ", "while ", "if ", "return "])
        return has_entry and has_control
