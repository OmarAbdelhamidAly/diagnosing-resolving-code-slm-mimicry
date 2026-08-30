"""DataService orchestrating benchmark ingestion and ground truth verification."""

from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from src.core.interfaces import IBenchmarkLoader, ICodeExecutor
from src.core.entities import BenchmarkTask, ExecutionResult
from src.infrastructure.hf_loader import HuggingFaceBenchmarkLoader
from src.infrastructure.sandbox import SubprocessSandbox, MultiprocessSandbox


class DataService:
    """Use-case service for preparing, caching, and verifying Reduction Ladder datasets."""

    def __init__(
        self,
        loader: IBenchmarkLoader = None,
        executor: ICodeExecutor = None
    ):
        self.loader = loader or HuggingFaceBenchmarkLoader()
        self.executor = executor or SubprocessSandbox()

    def prepare_all_benchmarks(self, force_download: bool = False) -> Dict[str, List[BenchmarkTask]]:
        """Fetch and cache all L0-L5 benchmark levels."""
        return self.loader.load_all_levels(force_download=force_download)

    def verify_ground_truth(self, tasks: List[BenchmarkTask], max_workers: int = 8) -> Dict[str, Any]:
        """Execute ground-truth verification inside the isolated sandbox with parallel workers."""
        if not tasks:
            return {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0, "failed_tasks": []}

        def _verify_single(task: BenchmarkTask) -> Tuple_Result:
            res: ExecutionResult = self.executor.execute(
                prompt=task.prompt,
                solution=task.canonical_solution,
                test=task.test,
                entry_point=task.entry_point,
                timeout_seconds=5.0
            )
            return task, res

        passed_count = 0
        failed_tasks = []

        desc = f"Verifying {tasks[0].ladder_level}" if tasks else "Verifying"
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_verify_single, t) for t in tasks]
            for fut in tqdm(futures, desc=desc):
                task, res = fut.result()
                if res.passed:
                    passed_count += 1
                else:
                    failed_tasks.append({
                        "task_id": task.task_id,
                        "level": task.ladder_level,
                        "status": res.status,
                        "error": res.error_message[:150]
                    })

        total = len(tasks)
        pass_rate = (passed_count / total * 100.0) if total > 0 else 0.0
        return {
            "total": total,
            "passed": passed_count,
            "failed": len(failed_tasks),
            "pass_rate": pass_rate,
            "failed_tasks": failed_tasks
        }


# Type helper
Tuple_Result = Any
