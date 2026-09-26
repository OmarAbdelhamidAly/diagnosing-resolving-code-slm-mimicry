"""Concrete implementation of ICodeExecutor using isolated native subprocesses with UTF-8 support."""

import os
import sys
import time
import subprocess
import re as _re
from typing import Tuple, Optional
from src.core.interfaces import ICodeExecutor
from src.core.entities import ExecutionResult


def _fix_check_body(test: str, entry_point: str) -> str:
    """Fix LiveCodeBench-style test stubs where ``check()`` body is comment-only.

    LiveCodeBench tasks store I/O examples as inline comments inside the
    ``check(candidate)`` function:

        def check(candidate):
            # test 1
            # input : '3\\nfoo\\n'
            # expect: '42\\n'

    Because the body contains **only** comments Python raises::

        IndentationError: expected an indented block after function definition

    This helper parses the `# input : ...` and `# expect: ...` cases from the comments,
    synthesizes a robust execution block that passes the arguments to `candidate`,
    and asserts expected outputs. It also gracefully resolves `candidate` whether defined
    as a standalone function or inside `class Solution`.
    """
    if not test.strip():
        return test

    lines = test.splitlines()
    in_check = False
    has_real_body = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('def check('):
            in_check = True
            continue
        if in_check:
            if not stripped:          # blank line — keep scanning
                continue
            if stripped.startswith('#'):
                continue              # comment-only line — still might be empty body
            # Found a real executable statement inside check()
            has_real_body = True
            break

    if in_check and not has_real_body:
        # Extract test cases from comments
        pattern = r"#\s*input\s*:\s*(.+?)\s*\n\s*#\s*expect\s*:\s*(.+?)(?=\s*\n\s*#\s*test|\Z)"
        matches = _re.findall(pattern, test, _re.DOTALL)

        runner = f"""
{test.rstrip()}
    # Auto-synthesized test execution from LiveCodeBench specification
    import json, ast

    def _parse_val(s):
        s = s.strip()
        if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
            try:
                s = ast.literal_eval(s)
            except Exception:
                pass
        if isinstance(s, str):
            try:
                return json.loads(s)
            except Exception:
                try:
                    return ast.literal_eval(s)
                except Exception:
                    return s
        return s

    raw_cases = {repr(matches)}
    if not raw_cases:
        return

    for inp_raw, exp_raw in raw_cases:
        inp_val = _parse_val(inp_raw)
        exp_val = _parse_val(exp_raw)

        if callable(candidate):
            if isinstance(inp_val, str) and "\\n" in inp_val:
                args = [_parse_val(line) for line in inp_val.splitlines() if line.strip()]
            else:
                args = [inp_val]
            
            try:
                res = candidate(*args)
            except TypeError:
                res = candidate(inp_val)

            if str(res).strip() != str(exp_val).strip() and res != exp_val:
                raise AssertionError(f"Expected {{exp_val}}, got {{res}}")

# Dynamic invocation of check() with entry_point resolution
_cand = None
if '{entry_point}' in globals():
    _cand = globals()['{entry_point}']
elif 'Solution' in globals():
    try:
        _sol = Solution()
        _cand = getattr(_sol, '{entry_point}', None) or getattr(Solution, '{entry_point}', None)
    except Exception:
        _cand = getattr(Solution, '{entry_point}', None)

if _cand is not None:
    check(_cand)
"""
        return runner

    return test


class SubprocessSandbox(ICodeExecutor):
    """Executes generated code in an isolated native subprocess with strict time guarding and UTF-8 encoding."""

    def __init__(self, default_timeout: float = 5.0, python_executable: Optional[str] = None):
        self.default_timeout = default_timeout
        self.python_executable = python_executable or sys.executable

    @staticmethod
    def _fix_body_indent(prompt: str, body: str) -> str:
        """Ensure body is properly indented relative to the enclosing def in prompt.

        Handles both top-level functions (def base=0 -> body=4) and class methods
        (def base=4 -> body=8, e.g. LeetCode class Solution).
        Properly handles single-line and multi-line generated bodies without
        leaving any statement at column 0.
        """
        if not body.strip():
            return body

        lines_p = prompt.strip().splitlines()
        def_line = ""
        for ln in reversed(lines_p):
            if ln.strip().startswith("def ") or "def " in ln:
                def_line = ln
                break

        base_indent = len(def_line) - len(def_line.lstrip()) if def_line else 0
        target_indent = base_indent + 4

        body_lines = body.splitlines()
        non_empty = [ln for ln in body_lines if ln.strip()]
        if not non_empty:
            return body

        first_indent = len(non_empty[0]) - len(non_empty[0].lstrip())
        shift = target_indent - first_indent

        if shift == 0:
            return body
        elif shift > 0:
            shift_str = " " * shift
            return "\n".join(shift_str + ln if ln.strip() else ln for ln in body_lines)
        else:
            abs_shift = abs(shift)
            can_unindent = all((len(ln) - len(ln.lstrip())) >= abs_shift for ln in non_empty)
            if can_unindent:
                return "\n".join(ln[abs_shift:] if ln.strip() else ln for ln in body_lines)
            return body

    def execute(
        self,
        prompt: str,
        solution: str,
        test: str,
        entry_point: str,
        timeout_seconds: float = 5.0
    ) -> ExecutionResult:
        timeout = timeout_seconds or self.default_timeout

        # Strip any CoT reasoning block (<thought>...</thought>) that SFT/RL
        # models may emit before the actual code block.
        clean_solution = _re.sub(r"<thought>.*?</thought>", "", solution, flags=_re.DOTALL).strip()

        # Sanitize prompt signatures with invalid ellipsis (e.g. def foo(...):)
        sanitized_prompt = _re.sub(r'def\s+(\w+)\s*\(\s*\.\.\.\s*\)\s*:', r'def \1(*args, **kwargs):', prompt)

        # The SFT-trained adapter (M2/M3) generates only the function body
        # (everything after the def/docstring). Prepend the original prompt so
        # Python always sees a syntactically complete, executable function.
        # Guard against signature duplication for models that output full functions.
        if entry_point and f"def {entry_point}" not in clean_solution and f"def {entry_point}" in sanitized_prompt:
            indented_body = self._fix_body_indent(sanitized_prompt, clean_solution)
            full_solution = f"{sanitized_prompt.rstrip()}\n{indented_body}"
        else:
            full_solution = clean_solution

        # Build complete executable script:
        # - Standard typing and algorithmic imports (List, Dict, Optional, math, etc.)
        # - Prompt as reference comment
        # - Full solution defining function(s) / class
        # - Fixed test with assertions
        standard_imports = "from typing import *\nimport math, collections, itertools, functools, re, sys, io\n\n"
        safe_prompt = "\n".join(f"# {line}" for line in sanitized_prompt.splitlines())
        fixed_test = _fix_check_body(test, entry_point)
        full_code = f"{standard_imports}{safe_prompt}\n\n{full_solution}\n\n{fixed_test}\n"

        start_time = time.perf_counter()
        try:
            # Stream code via stdin to handle arbitrary sizes and bypass Windows CLI argument limits
            proc = subprocess.run(
                [self.python_executable, "-X", "utf8", "-c", "import sys; exec(sys.stdin.read())"],
                input=full_code,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout
            )
            elapsed = time.perf_counter() - start_time

            if proc.returncode == 0:
                return ExecutionResult(
                    passed=True,
                    status="PASS",
                    error_message="Execution passed all assertions",
                    execution_time_seconds=elapsed
                )
            else:
                stderr_msg = proc.stderr.strip()
                status = "FAIL" if "AssertionError" in stderr_msg else "ERROR"
                return ExecutionResult(
                    passed=False,
                    status=status,
                    error_message=stderr_msg,
                    execution_time_seconds=elapsed
                )

        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - start_time
            return ExecutionResult(
                passed=False,
                status="TIMEOUT",
                error_message=f"Execution exceeded timeout limit ({timeout}s)",
                execution_time_seconds=elapsed
            )
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            return ExecutionResult(
                passed=False,
                status="CRASH",
                error_message=f"Sandbox process failed: {type(e).__name__}: {e}",
                execution_time_seconds=elapsed
            )


# Backward compatibility alias
MultiprocessSandbox = SubprocessSandbox
