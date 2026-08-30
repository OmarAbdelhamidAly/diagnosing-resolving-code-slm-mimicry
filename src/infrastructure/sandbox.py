"""Concrete implementation of ICodeExecutor using isolated native subprocesses with UTF-8 support."""

import os
import sys
import time
import subprocess
from typing import Tuple, Optional
from src.core.interfaces import ICodeExecutor
from src.core.entities import ExecutionResult


class SubprocessSandbox(ICodeExecutor):
    """Executes generated code in an isolated native subprocess with strict time guarding and UTF-8 encoding."""

    def __init__(self, default_timeout: float = 5.0, python_executable: Optional[str] = None):
        self.default_timeout = default_timeout
        self.python_executable = python_executable or sys.executable

    def execute(
        self,
        prompt: str,
        solution: str,
        test: str,
        entry_point: str,
        timeout_seconds: float = 5.0
    ) -> ExecutionResult:
        timeout = timeout_seconds or self.default_timeout
        
        # Build complete executable script
        full_code = f"{prompt}\n{solution}\n{test}\n"
        if "check(" not in test and entry_point:
            full_code += f"\ncheck({entry_point})\n"

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
