"""Concrete implementation of IBenchmarkLoader for Hugging Face datasets."""

import os
from typing import Dict, List, Any, Optional
from datasets import load_dataset
from src.core.interfaces import IBenchmarkLoader
from src.core.entities import BenchmarkTask
from src.core.exceptions import DatasetIngestionError
from src.infrastructure.persistence import save_jsonl, load_jsonl


LADDER_DATASET_CONFIGS = {
    "L0": {
        "name": "HumanEval_Standard",
        "dataset_path": "openai/openai_humaneval",
        "split": "test",
        "trust_remote_code": False
    },
    "L1": {
        "name": "EvoEval_Subtle",
        "dataset_path": "evoeval/EvoEval_subtle",
        "split": "test",
        "trust_remote_code": False
    },
    "L2": {
        "name": "EvoEval_ToolUse",
        "dataset_path": "evoeval/EvoEval_tool_use",
        "split": "test",
        "trust_remote_code": False
    },
    "L3": {
        "name": "EvoEval_Creative",
        "dataset_path": "evoeval/EvoEval_creative",
        "split": "test",
        "trust_remote_code": False
    },
    "L4": {
        "name": "EvoEval_Difficult",
        "dataset_path": "evoeval/EvoEval_difficult",
        "split": "test",
        "trust_remote_code": False
    },
    "L5": {
        "name": "EvoEval_Combine",
        "dataset_path": "evoeval/EvoEval_combine",
        "split": "test",
        "trust_remote_code": False
    },
    "Ctrl": {
        "name": "LiveCodeBench_Lite",
        "dataset_path": "livecodebench/code_generation_lite",
        "split": "test",
        "trust_remote_code": True
    }
}


class HuggingFaceBenchmarkLoader(IBenchmarkLoader):
    """Loads and normalizes coding benchmarks from Hugging Face into domain BenchmarkTasks."""

    def __init__(self, cache_dir: str = "data/ladder"):
        self.cache_dir = cache_dir

    def load_level(self, level_key: str, force_download: bool = False) -> List[BenchmarkTask]:
        if level_key not in LADDER_DATASET_CONFIGS:
            raise DatasetIngestionError(f"Unknown ladder level: {level_key}. Valid: {list(LADDER_DATASET_CONFIGS.keys())}")

        cfg = LADDER_DATASET_CONFIGS[level_key]
        cached_file = os.path.join(self.cache_dir, f"{level_key}_{cfg['name'].lower()}.jsonl")

        # Load from disk cache if exists and not forced
        if os.path.exists(cached_file) and not force_download:
            raw_records = load_jsonl(cached_file)
            return [BenchmarkTask.from_dict(r) for r in raw_records]

        # Otherwise download from Hugging Face
        try:
            ds = load_dataset(
                cfg["dataset_path"],
                split=cfg["split"],
                trust_remote_code=cfg.get("trust_remote_code", False)
            )
        except Exception as e:
            raise DatasetIngestionError(f"Failed to fetch {level_key} ({cfg['dataset_path']}): {e}") from e

        tasks: List[BenchmarkTask] = []
        for item in ds:
            task = BenchmarkTask(
                task_id=str(item.get("task_id", "")),
                ladder_level=level_key,
                benchmark=cfg["name"],
                prompt=str(item.get("prompt", "")),
                canonical_solution=str(item.get("canonical_solution", "")),
                test=str(item.get("test", "")),
                entry_point=str(item.get("entry_point", ""))
            )
            tasks.append(task)

        # Save to disk cache
        save_jsonl([t.to_dict() for t in tasks], cached_file)
        return tasks

    def load_all_levels(self, force_download: bool = False) -> Dict[str, List[BenchmarkTask]]:
        all_tasks = {}
        for level_key in ["L0", "L1", "L2", "L3", "L4", "L5"]:
            all_tasks[level_key] = self.load_level(level_key, force_download=force_download)
        return all_tasks
