"""Sandbox execution and verification module for ground-truth benchmark tasks.

Executes `prompt + canonical_solution + test + check(entry_point)` in isolated
worker processes with strict timeouts to verify 100% test-suite correctness.
"""

import sys
import multiprocessing
from typing import Dict, List, Any, Tuple
from tqdm import tqdm


def _execute_code_snippet(code_str: str, entry_point: str, result_queue: multiprocessing.Queue):
    """Worker function executed in an isolated process."""
    try:
        global_scope = {}
        exec(code_str, global_scope)
        # Call the check function if available
        if "check" in global_scope and entry_point in global_scope:
            global_scope["check"](global_scope[entry_point])
        result_queue.put(("PASS", "Execution completed successfully"))
    except AssertionError as e:
        result_queue.put(("FAIL", f"AssertionError: {e}"))
    except Exception as e:
        result_queue.put(("ERROR", f"{type(e).__name__}: {e}"))


def run_sandbox_execution(
    prompt: str,
    solution: str,
    test: str,
    entry_point: str,
    timeout_seconds: float = 5.0
) -> Tuple[bool, str, str]:
    """Run code execution inside an isolated multiprocessing worker with timeout."""
    full_code = f"{prompt}\n{solution}\n{test}\n"
    if entry_point:
        full_code += f"\ncheck({entry_point})\n"

    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_execute_code_snippet,
        args=(full_code, entry_point, result_queue)
    )

    process.start()
    process.join(timeout=timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        return False, "TIMEOUT", f"Execution exceeded {timeout_seconds}s limit"

    if not result_queue.empty():
        status, message = result_queue.get()
        return (status == "PASS"), status, message
    else:
        return False, "CRASH", "Subprocess crashed or exited without return"


def verify_dataset_tasks(tasks: List[Dict[str, Any]], max_workers: int = 4) -> Dict[str, Any]:
    """Verify a list of normalized benchmark tasks."""
    print(f"\n[VERIFY] Verifying {len(tasks)} tasks in sandbox...")
    passed_count = 0
    failed_tasks = []

    for task in tqdm(tasks, desc="Verifying Ground Truth"):
        passed, status, msg = run_sandbox_execution(
            prompt=task["prompt"],
            solution=task["canonical_solution"],
            test=task["test"],
            entry_point=task.get("entry_point", "")
        )

        if passed:
            passed_count += 1
        else:
            failed_tasks.append({
                "task_id": task["task_id"],
                "ladder_level": task.get("ladder_level", ""),
                "status": status,
                "error": msg
            })

    pass_rate = (passed_count / len(tasks)) * 100 if tasks else 0.0
    print(f"[RESULT] Verification Result: {passed_count}/{len(tasks)} ({pass_rate:.2f}%) PASSED.")

    if failed_tasks:
        print(f"[WARN] {len(failed_tasks)} tasks failed ground-truth verification:")
        for ft in failed_tasks[:5]:
            print(f"   - {ft['task_id']} ({ft['ladder_level']}): {ft['status']} -> {ft['error']}")

    return {
        "total": len(tasks),
        "passed": passed_count,
        "failed": len(failed_tasks),
        "pass_rate": pass_rate,
        "failed_tasks": failed_tasks
    }
