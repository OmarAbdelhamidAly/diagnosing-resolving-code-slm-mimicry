"""Dataset loader and pair synthesizer for Arm 1: Inv-GRPO.

Inv-GRPO requires paired prompts (x, x') representing the same underlying algorithmic
problem under two different surface representations:
  - x  : Canonical L0 task (HumanEval)
  - x' : Perturbed counterpart (e.g., L2 EvoEval ToolUse or L1 Subtle)
  - decoy : The memorized textbook solution to x (to penalize if blindly pasted into x')

This module aligns L0 and L1-L5 tasks by index and yields structured PairedTask objects.
"""

import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

from src.core.config import get_settings


@dataclass
class PairedTask:
    """A pair of logically equivalent benchmark problems for invariance training."""
    pair_id: str
    l0_task_id: str
    pert_task_id: str
    ladder_level: str
    # Original canonical problem (L0)
    prompt_orig: str
    test_orig: str
    entry_orig: str
    canonical_orig: str
    # Perturbed problem (e.g., L2)
    prompt_pert: str
    test_pert: str
    entry_pert: str
    canonical_pert: str
    # Shortcut template to penalize if mimicked blindly on perturbed
    decoy_code: str


from typing import List, Tuple, Optional, Dict, Any, Union

class InvGRPODatasetLoader:
    """Loads and pairs canonical L0 problems with their perturbed counterparts across L1-L5."""

    def __init__(
        self,
        ladder_dir: Optional[str] = None,
        perturbed_level: Union[str, List[str]] = "ALL",
    ):
        settings = get_settings()
        self.ladder_dir = ladder_dir or settings.storage.ladder_cache_dir

        self.level_files = {
            "L1": "L1_evoeval_subtle.jsonl",
            "L2": "L2_evoeval_tooluse.jsonl",
            "L3": "L3_evoeval_creative.jsonl",
            "L4": "L4_evoeval_difficult.jsonl",
            "L5": "L5_evoeval_combine.jsonl",
        }

        if isinstance(perturbed_level, str):
            if perturbed_level.upper() == "ALL":
                self.target_levels = ["L1", "L2", "L3", "L4", "L5"]
            else:
                lvl = perturbed_level.upper()
                if lvl not in self.level_files:
                    raise ValueError(f"Unsupported level: {lvl}. Choose from {list(self.level_files.keys())} or 'ALL'")
                self.target_levels = [lvl]
        elif isinstance(perturbed_level, list):
            self.target_levels = [lvl.upper() for lvl in perturbed_level]
        else:
            self.target_levels = ["L1", "L2", "L3", "L4", "L5"]

        self.l0_path = os.path.join(self.ladder_dir, "L0_humaneval_standard.jsonl")

    def _read_jsonl(self, path: str) -> List[Dict[str, Any]]:
        records = []
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def load_pairs(
        self,
        max_pairs: Optional[int] = None,
        train_ratio: float = 0.8,
        seed: int = 42
    ) -> Tuple[List[PairedTask], List[PairedTask]]:
        """Loads matched pairs across all targeted levels and splits into train/test.

        Args:
            max_pairs: Optional limit on the total number of pairs to load.
            train_ratio: Fraction of pairs to allocate to training.
            seed: Random seed for deterministic splitting.

        Returns:
            Tuple of (train_pairs, test_pairs).
        """
        import random

        l0_data = self._read_jsonl(self.l0_path)
        all_paired_tasks: List[PairedTask] = []

        for lvl in self.target_levels:
            pert_path = os.path.join(self.ladder_dir, self.level_files[lvl])
            if not os.path.exists(pert_path):
                continue
            pert_data = self._read_jsonl(pert_path)
            num_items = min(len(l0_data), len(pert_data))

            for i in range(num_items):
                t0 = l0_data[i]
                t_pert = pert_data[i]

                pair = PairedTask(
                    pair_id=f"pair_{lvl}_{i}",
                    l0_task_id=t0["task_id"],
                    pert_task_id=t_pert["task_id"],
                    ladder_level=lvl,
                    prompt_orig=t0["prompt"],
                    test_orig=t0["test"],
                    entry_orig=t0["entry_point"],
                    canonical_orig=t0.get("canonical_solution", ""),
                    prompt_pert=t_pert["prompt"],
                    test_pert=t_pert["test"],
                    entry_pert=t_pert["entry_point"],
                    canonical_pert=t_pert.get("canonical_solution", ""),
                    decoy_code=t0.get("canonical_solution", ""),
                )
                all_paired_tasks.append(pair)

        # Deterministic shuffle across all ladder levels
        rng = random.Random(seed)
        shuffled = list(all_paired_tasks)
        rng.shuffle(shuffled)

        if max_pairs:
            shuffled = shuffled[:max_pairs]

        split_idx = int(len(shuffled) * train_ratio)
        train_pairs = shuffled[:split_idx]
        test_pairs = shuffled[split_idx:]

        return train_pairs, test_pairs
