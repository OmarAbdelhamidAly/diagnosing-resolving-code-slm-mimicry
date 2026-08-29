"""Concrete implementation of ICodeExecutor using isolated worker processes with timeouts."""

import time
import multiprocessing
from typing import Tuple
from src.core.interfaces import ICodeExecutor
from src.core.entities import ExecutionResult


def _worker_exec(code_str: str, entry_point: str, queue: multiprocessing.Queue):
    """Isolated subprocess worker executing code against test assertions."""
    try:
        scope = {}
        exec(code_str, scope)
        if "check" in scope and entry_point in scope:
            scope["check"](scope[entry_point])
        queue.put(("PASS", "Execution passed all assertions"))
    except AssertionError as e:
        queue.put(("FAIL", f"AssertionError: {e}"))
    except Exception as e:
        queue.put(("ERROR", f"{type(e).__name__}: {e}"))


class MultiprocessSandbox(ICodeExecutor):
    """Executes generated code in an isolated subprocess with strict time guarding."""

    def __init__(self, default_timeout: float = 5.0):
        self.default_timeout = default_timeout

    def execute(
        self,
        prompt: str,
        solution: str,
        test: str,
        entry_point: str,
        timeout_seconds: float = 5.0
    ) -> ExecutionResult:
        timeout = timeout_seconds or self.default_timeout
        full_code = f"{prompt}\n{solution}\n{test}\n"
        if entry_point:
            full_code += f"\ncheck({entry_point})\n"

        queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=_worker_exec,
            args=(full_code, entry_point, queue)
        )

        start_time = time.perf_counter()
        process.start()
        process.join(timeout=timeout)
        elapsed = time.perf_counter() - start_time

        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
            return ExecutionResult(
                passed=False,
                status="TIMEOUT",
                error_message=f"Execution exceeded timeout limit ({timeout}s)",
                execution_time_seconds=elapsed
            )

        if not queue.empty():
            status, msg = queue.get()
            return ExecutionResult(
                passed=(status == "PASS"),
                status=status,
                error_message=msg,
                execution_time_seconds=elapsed
            )
        else:
            return ExecutionResult(
                passed=False,
                status="CRASH",
                error_message="Subprocess crashed or exited without return value",
                execution_time_seconds=elapsed
            )
