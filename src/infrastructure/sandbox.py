"""Concrete implementation of ICodeExecutor using isolated native subprocesses with UTF-8 support."""

import os
import sys
import time
import subprocess
import re as _re
from typing import Tuple, Optional
from src.core.interfaces import ICodeExecutor
from src.core.entities import ExecutionResult


class SubprocessSandbox(ICodeExecutor):
    """Executes generated code in an isolated native subprocess with strict time guarding and UTF-8 encoding."""

    def __init__(self, default_timeout: float = 5.0, python_executable: Optional[str] = None):
        self.default_timeout = default_timeout
        self.python_executable = python_executable or sys.executable

    @staticmethod
    def _fix_body_indent(body: str) -> str:
        """Ensure every line of a function body is indented by at least 4 spaces.

        SFT-trained models (M2 / M3 / M6) are trained on data where the
        canonical solution already has 4-space indentation (since it lives
        inside a ``def`` block).  However the model sometimes emits the very
        first statement at column-0 while keeping the relative indentation of
        nested lines intact, e.g.::

            for i in range(n):     # ← 0-indent  (should be 4)
                    result += i    # ← 8-indent  (relative: OK)
            return result          # ← 0-indent  (should be 4)

        After prepending this body to the ``def`` prompt the interpreter sees
        statements outside the function, raising an IndentationError or
        ``SyntaxError: 'return' outside function``.

        This helper detects the pattern (first non-empty line at col-0, next
        non-empty line at col ≥ 4) and adds 4 spaces to every line whose
        current indentation is 0.
        """
        lines = body.split("\n")
        non_empty = [ln for ln in lines if ln.strip()]
        if not non_empty:
            return body

        first_indent = len(non_empty[0]) - len(non_empty[0].lstrip())
        # If the first real line already has indentation, leave everything alone.
        if first_indent >= 4:
            return body

        # Check second non-empty line for relative indentation.
        second_indent = (
            len(non_empty[1]) - len(non_empty[1].lstrip()) if len(non_empty) > 1 else 0
        )
        # Only apply the fix when the pattern "0-indent first line, ≥4 indent
        # second line" is detected – this is the classic SFT body artefact.
        if second_indent < 4:
            return body

        fixed = []
        for ln in lines:
            if not ln.strip():
                fixed.append("")
            elif len(ln) - len(ln.lstrip()) == 0:
                # Top-level body statement with missing 4-space prefix.
                fixed.append("    " + ln)
            else:
                fixed.append(ln)
        return "\n".join(fixed)

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

        # The SFT-trained adapter (M2/M3) generates only the function body
        # (everything after the def/docstring).  Prepend the original prompt so
        # Python always sees a syntactically complete, executable function.
        # For the zero-shot baseline the model typically generates the full
        # function, so prepending the prompt would duplicate the signature; we
        # guard against that by only prepending when the solution does NOT
        # already contain the entry_point definition.
        if entry_point and f"def {entry_point}" not in clean_solution and f"def {entry_point}" in prompt:
            # SFT/RL models output only the body.  Restore 4-space indentation
            # if the model dropped it from the outermost statement.
            indented_body = self._fix_body_indent(clean_solution)
            full_solution = f"{prompt.rstrip()}\n{indented_body}"
        else:
            full_solution = clean_solution

        # Build complete executable script:
        # - Prompt is a comment only (not executable code)
        # - Solution defines the function(s)
        # - Test contains the assertions to run
        safe_prompt = "\n".join(f"# {line}" for line in prompt.splitlines())
        full_code = f"{safe_prompt}\n\n{full_solution}\n\n{test}\n"

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
