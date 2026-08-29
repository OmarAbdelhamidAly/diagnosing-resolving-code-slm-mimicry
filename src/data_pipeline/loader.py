"""Data loader and benchmark ingestion module for Reduction Ladder for Code.

Loads verified datasets from Hugging Face:
- L0: openai/openai_humaneval
- L1: evoeval/EvoEval_subtle
- L2: evoeval/EvoEval_verbose
- L3: evoeval/EvoEval_creative
- L4: evoeval/EvoEval_difficult
- L5: evoeval/EvoEval_combine
- Control: livecodebench/code_generation_lite
"""

import json
import os
from typing import Dict, List, Any, Optional
from datasets import load_dataset
from tqdm import tqdm


LADDER_BENCHMARKS = {
    "L0": {
        "name": "HumanEval_Standard",
        "dataset_path": "openai/openai_humaneval",
        "split": "test",
        "description": "Verbatim classic HumanEval benchmark problems"
    },
    "L1": {
        "name": "EvoEval_Subtle",
        "dataset_path": "evoeval/EvoEval_subtle",
        "split": "test",
        "description": "Subtle wording and input specification shift"
    },
    "L2": {
        "name": "EvoEval_ToolUse",
        "dataset_path": "evoeval/EvoEval_tool_use",
        "split": "test",
        "description": "Structural API abstraction requiring helper function composition"
    },
    "L3": {
        "name": "EvoEval_Creative",
        "dataset_path": "evoeval/EvoEval_creative",
        "split": "test",
        "description": "Novel narrative context for the identical algorithmic logic"
    },
    "L4": {
        "name": "EvoEval_Difficult",
        "dataset_path": "evoeval/EvoEval_difficult",
        "split": "test",
        "description": "Core algorithm with added boundary conditions and constraints"
    },
    "L5": {
        "name": "EvoEval_Combine",
        "dataset_path": "evoeval/EvoEval_combine",
        "split": "test",
        "description": "Multi-algorithmic composition and concept integration"
    }
}


def normalize_task(raw_item: Dict[str, Any], level_key: str, benchmark_name: str) -> Dict[str, Any]:
    """Normalize a raw dataset record into the standard schema."""
    task_id = str(raw_item.get("task_id", ""))
    prompt = raw_item.get("prompt", "")
    canonical_solution = raw_item.get("canonical_solution", "")
    test = raw_item.get("test", "")
    entry_point = raw_item.get("entry_point", "")

    return {
        "task_id": task_id,
        "ladder_level": level_key,
        "benchmark": benchmark_name,
        "prompt": prompt,
        "canonical_solution": canonical_solution,
        "test": test,
        "entry_point": entry_point
    }


def download_level(level_key: str, split_override: Optional[str] = None) -> List[Dict[str, Any]]:
    """Download and normalize a specific ladder benchmark level."""
    if level_key not in LADDER_BENCHMARKS:
        raise ValueError(f"Unknown ladder level: {level_key}. Valid levels: {list(LADDER_BENCHMARKS.keys())}")

    config = LADDER_BENCHMARKS[level_key]
    dataset_path = config["dataset_path"]
    split = split_override or config["split"]

    print(f"[LOAD] Loading {level_key} ({config['name']}) from '{dataset_path}'...")
    try:
        ds = load_dataset(dataset_path, split=split)
    except Exception as e:
        # Fallback to loading default split if 'test' fails
        print(f"[WARN] Split '{split}' failed ({e}). Attempting default split loading...")
        ds_dict = load_dataset(dataset_path)
        ds = ds_dict[list(ds_dict.keys())[0]]

    normalized_tasks = []
    for item in ds:
        normalized_tasks.append(normalize_task(item, level_key, config["name"]))

    print(f"[OK] Loaded {len(normalized_tasks)} tasks for {level_key}.")
    return normalized_tasks


def save_tasks_to_jsonl(tasks: List[Dict[str, Any]], output_path: str):
    """Save normalized tasks to a JSON Lines file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")
    print(f"[SAVE] Saved {len(tasks)} tasks to '{output_path}'.")


def load_tasks_from_jsonl(input_path: str) -> List[Dict[str, Any]]:
    """Load normalized tasks from a JSON Lines file."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    tasks = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


def fetch_all_ladder_levels(output_dir: str = "data/ladder") -> Dict[str, List[Dict[str, Any]]]:
    """Fetch and serialize all L0-L5 ladder benchmark levels."""
    os.makedirs(output_dir, exist_ok=True)
    all_data = {}

    for level_key in LADDER_BENCHMARKS:
        tasks = download_level(level_key)
        filename = f"{level_key}_{LADDER_BENCHMARKS[level_key]['name'].lower()}.jsonl"
        filepath = os.path.join(output_dir, filename)
        save_tasks_to_jsonl(tasks, filepath)
        all_data[level_key] = tasks

    return all_data


if __name__ == "__main__":
    fetch_all_ladder_levels()
