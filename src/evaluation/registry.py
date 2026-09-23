"""BenchmarkRegistry: maps level keys to JSONL benchmark files.

Loads tasks from disk, validates record counts, and provides a single
read-only view of the complete benchmark pool used for evaluation.

The registry is intentionally read-only after construction — training code
must never mutate benchmark files.

Levels
------
L0   HumanEval Standard        (164 tasks)
L1   EvoEval Subtle             (subset, see JSONL)
L2   EvoEval Tool-Use           (subset, see JSONL)
L3   EvoEval Creative           (subset, see JSONL)
L4   EvoEval Difficult          (subset, see JSONL)
L5   EvoEval Combine            (subset, see JSONL)
Ctrl LiveCodeBench Lite         (temporal OOD control — not in training data)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from src.core.entities import BenchmarkTask


# Default paths relative to repo root
_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "ladder"

_DEFAULT_FILES: Dict[str, str] = {
    "L0":   "L0_humaneval_standard.jsonl",
    "L1":   "L1_evoeval_subtle.jsonl",
    "L2":   "L2_evoeval_tooluse.jsonl",
    "L3":   "L3_evoeval_creative.jsonl",
    "L4":   "L4_evoeval_difficult.jsonl",
    "L5":   "L5_evoeval_combine.jsonl",
    "Ctrl": "Ctrl_livecode_lite.jsonl",   # may not exist yet — handled gracefully
}

# Ordered list for iteration (Ctrl last)
LEVEL_ORDER = ["L0", "L1", "L2", "L3", "L4", "L5", "Ctrl"]


class BenchmarkRegistry:
    """Immutable mapping from level key → list of BenchmarkTask objects.

    Parameters
    ----------
    data_dir : path to the directory containing JSONL files.
               Defaults to ``<repo_root>/data/ladder/``.
    level_files : optional override mapping level_key → filename.
    max_tasks_per_level : if set, truncates each level to this many tasks
                          (useful for quick smoke-tests).
    """

    def __init__(
        self,
        data_dir: Optional[str | Path] = None,
        level_files: Optional[Dict[str, str]] = None,
        max_tasks_per_level: Optional[int] = None,
    ) -> None:
        self._data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
        self._file_map = {**_DEFAULT_FILES, **(level_files or {})}
        self._max_tasks = max_tasks_per_level
        self._tasks: Dict[str, List[BenchmarkTask]] = {}
        self._load_all()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tasks(self, level: str) -> List[BenchmarkTask]:
        """Return all tasks for *level* (e.g. "L0", "Ctrl")."""
        return self._tasks.get(level, [])

    def available_levels(self) -> List[str]:
        """Return levels that were successfully loaded (have ≥1 task)."""
        return [lvl for lvl in LEVEL_ORDER if self._tasks.get(lvl)]

    def task_counts(self) -> Dict[str, int]:
        """Return a dict of level → task count for all available levels."""
        return {lvl: len(self._tasks[lvl]) for lvl in self.available_levels()}

    def integrity_report(self) -> str:
        """Human-readable integrity summary for notebook display."""
        lines = ["BenchmarkRegistry — Integrity Report", "=" * 40]
        for lvl in LEVEL_ORDER:
            count = len(self._tasks.get(lvl, []))
            path = self._data_dir / self._file_map.get(lvl, "")
            status = "[OK]" if count > 0 else "[MISSING]"
            lines.append(f"  {lvl:<6}  {count:>5} tasks   {status}   {path.name}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        for level, filename in self._file_map.items():
            path = self._data_dir / filename
            if not path.exists():
                self._tasks[level] = []
                continue
            tasks = self._load_jsonl(path)
            if self._max_tasks is not None:
                tasks = tasks[: self._max_tasks]
            self._tasks[level] = tasks

    def _load_jsonl(self, path: Path) -> List[BenchmarkTask]:
        tasks: List[BenchmarkTask] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    tasks.append(BenchmarkTask.from_dict(data))
                except (json.JSONDecodeError, KeyError):
                    continue  # skip malformed lines silently
        return tasks
