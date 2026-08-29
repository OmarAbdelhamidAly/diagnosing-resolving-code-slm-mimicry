"""DataService orchestrating benchmark ingestion and ground truth verification."""

from typing import Dict, List, Any
from tqdm import tqdm
from src.core.interfaces import IBenchmarkLoader, ICodeExecutor
from src.core.entities import BenchmarkTask, ExecutionResult
from src.infrastructure.hf_loader import HuggingFaceBenchmarkLoader
from src.infrastructure.sandbox import MultiprocessSandbox


class DataService:
    """Use-case service for preparing, caching, and verifying Reduction Ladder datasets."""

    def __init__(
        self,
        loader: IBenchmarkLoader = None,
        executor: ICodeExecutor = None
    ):
        self.loader = loader or HuggingFaceBenchmarkLoader()
        self.executor = executor or MultiprocessSandbox()

    def prepare_all_benchmarks(self, force_download: bool = False) -> Dict[str, List[BenchmarkTask]]:
        """Fetch and cache all L0-L5 benchmark levels."""
        return self.loader.load_all_levels(force_download=force_download)

    def verify_ground_truth(self, tasks: List[BenchmarkTask]) -> Dict[str, Any]:
        """Execute ground-truth verification inside the isolated sandbox."""
        passed_count = 0
        failed_tasks = []

        for task in tqdm(tasks, desc=f"Verifying {tasks[0].ladder_level if tasks else ''}"):
            res: ExecutionResult = self.executor.execute(
                prompt=task.prompt,
                solution=task.canonical_solution,
                test=task.test,
                entry_point=task.entry_point,
                timeout_seconds=5.0
            )
            if res.passed:
                passed_count += 1
            else:
                failed_tasks.append({
                    "task_id": task.task_id,
                    "level": task.ladder_level,
                    "status": res.status,
                    "error": res.error_message
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
