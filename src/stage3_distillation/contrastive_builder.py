"""Contrastive Thought-Template SFT Dataset Builder.

Synthesizes paired (problem, contrastive_reasoning) examples that explicitly
teach the model to:
  1. Identify the "decoy" shortcut template it would normally memorize.
  2. Verify that the template is INVALID for this perturbed problem.
  3. Derive the correct algorithm from first principles.

Output schema (per JSONL line):
{
    "task_id":         "L1/EvoEval/5",
    "source":          "contrastive_synthetic",
    "l0_task_id":      "HumanEval/5",           # Canonical ancestor
    "ladder_level":    "L1",                    # Perturbation level of this example
    "prompt":          "...",                   # Perturbed problem statement
    "decoy_template":  "...",                   # The shortcut the model would recall
    "response":        "...",                   # Correct contrastive <thought> + solution
    "token_count":     512
}
"""

import os
import json
import random
from typing import List, Dict, Any, Optional

from tqdm import tqdm
from src.core.entities import BenchmarkTask
from src.core.config import settings


# ---------------------------------------------------------------------------
# Contrastive thought template (the core scientific contribution).
# ---------------------------------------------------------------------------
_CONTRASTIVE_THOUGHT_TEMPLATE = """\
<thought>
Initial intuition: apply the memorized template from the canonical HumanEval/{canonical_id} problem.

Decoy check: The specification in this version has been modified. Specifically, {constraint_difference}.
Applying the original template without modification would produce incorrect behavior because {why_decoy_fails}.

Correct strategy: {correct_strategy_description}
Therefore, I must {correct_action} to satisfy the modified invariant.
</thought>

```python
{canonical_solution}
```"""


def _build_constraint_difference(l0_task: BenchmarkTask, perturbed_task: BenchmarkTask) -> str:
    """
    Heuristic: generate a human-readable difference description between L0 and its perturbation.
    Uses prompt length delta and keyword detection as a proxy.
    """
    delta = len(perturbed_task.prompt) - len(l0_task.prompt)
    if delta > 200:
        return ("the problem statement has been significantly expanded with additional narrative "
                "context and modified input/output specifications")
    elif delta > 50:
        return ("the variable names and problem framing have been altered, even though "
                "the underlying algorithmic logic remains the same")
    else:
        return ("subtle wording and constraint details have changed, invalidating the exact "
                "original implementation without careful re-verification")


def _build_why_decoy_fails(l0_task: BenchmarkTask, perturbed_task: BenchmarkTask) -> str:
    """Heuristic explanation of why the decoy template is wrong."""
    if "creative" in perturbed_task.benchmark.lower():
        return ("the function signature and domain context are different — direct copy-paste "
                "would miss the type conversions or boundary conditions introduced by the new narrative")
    elif "difficult" in perturbed_task.benchmark.lower():
        return ("there are additional edge-case constraints and stricter boundary requirements "
                "that the original template does not handle")
    elif "combine" in perturbed_task.benchmark.lower():
        return ("this problem requires composing multiple algorithmic patterns, whereas the "
                "original template only handled a single pattern in isolation")
    else:
        return ("the semantics of one or more function arguments have been redefined in "
                "ways that the original template's assumptions do not cover")


def _build_correct_strategy(l0_task: BenchmarkTask, perturbed_task: BenchmarkTask) -> str:
    """Describe the correct reasoning approach."""
    return ("re-derive the algorithm from the modified problem specification, "
            "verify all data types and edge cases explicitly, "
            "and implement a solution that satisfies the new constraints without relying on memorized patterns")


def _build_correct_action(l0_task: BenchmarkTask, perturbed_task: BenchmarkTask) -> str:
    """Describe the specific action to take."""
    return ("parse the new constraints carefully, adapt the core algorithm to match "
            "the redefined invariants, and validate boundary conditions from the updated problem statement")


class ContrastiveDatasetBuilder:
    """Builds Contrastive SFT pairs from the already-cached Reduction Ladder benchmarks.

    For each perturbed task at level L1-L5, we pair it with its canonical L0 ancestor
    (matched by index, since EvoEval generates perturbations in order from HumanEval)
    and synthesize a contrastive reasoning trace that explicitly rejects the L0 shortcut.

    The output is directly usable as training data for Arm 2B (M3: Contrastive SFT).
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        levels_to_use: Optional[List[str]] = None,
        max_pairs_per_level: int = 164,
        seed: Optional[int] = None,
    ):
        self.output_dir = output_dir or settings.storage.distillation_cache_dir
        self.levels_to_use = levels_to_use or ["L1", "L2", "L3", "L4", "L5"]
        self.max_pairs_per_level = max_pairs_per_level
        self.seed = seed if seed is not None else settings.project.seed
        random.seed(self.seed)
        os.makedirs(self.output_dir, exist_ok=True)

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4

    def _synthesize_contrastive_pair(
        self,
        l0_task: BenchmarkTask,
        perturbed_task: BenchmarkTask,
    ) -> Dict[str, Any]:
        """Generate a single contrastive training example."""
        # Extract a short canonical id (e.g., "5" from "HumanEval/5")
        canonical_id_short = l0_task.task_id.split("/")[-1]

        contrastive_response = _CONTRASTIVE_THOUGHT_TEMPLATE.format(
            canonical_id=canonical_id_short,
            constraint_difference=_build_constraint_difference(l0_task, perturbed_task),
            why_decoy_fails=_build_why_decoy_fails(l0_task, perturbed_task),
            correct_strategy_description=_build_correct_strategy(l0_task, perturbed_task),
            correct_action=_build_correct_action(l0_task, perturbed_task),
            canonical_solution=perturbed_task.canonical_solution,
        )

        token_count = self._estimate_tokens(perturbed_task.prompt + contrastive_response)

        return {
            "task_id": perturbed_task.task_id,
            "source": "contrastive_synthetic",
            "l0_task_id": l0_task.task_id,
            "ladder_level": perturbed_task.ladder_level,
            "prompt": perturbed_task.prompt,
            "decoy_template": (
                f"The straightforward template from HumanEval/{canonical_id_short} is:\n"
                f"```python\n{l0_task.canonical_solution}\n```"
            ),
            "response": contrastive_response,
            "token_count": token_count,
        }

    def build(
        self,
        ladder_data: Dict[str, List[BenchmarkTask]],
        output_filename: str = "sft_contrastive_pairs.jsonl",
        force: bool = False,
    ) -> str:
        """Build the contrastive dataset from pre-loaded ladder benchmark tasks.

        Args:
            ladder_data: Dict mapping level keys (e.g. "L0", "L1", ...) to task lists.
            output_filename: Name of the output JSONL file.
            force: Whether to overwrite existing file.

        Returns:
            Path to the saved JSONL file.
        """
        output_path = os.path.join(self.output_dir, output_filename)

        if os.path.exists(output_path) and not force:
            print(f"[ContrastiveBuilder] [OK] Found existing dataset at '{output_path}'. Skipping synthesis.")
            return output_path

        l0_tasks = ladder_data.get("L0", [])
        if not l0_tasks:
            raise ValueError("[ContrastiveBuilder] L0 (HumanEval) tasks are required but not found in ladder_data.")

        print(f"[ContrastiveBuilder] L0 pool: {len(l0_tasks)} canonical tasks.")

        all_pairs: List[Dict[str, Any]] = []

        for level_key in self.levels_to_use:
            perturbed_tasks = ladder_data.get(level_key, [])
            if not perturbed_tasks:
                print(f"[ContrastiveBuilder] WARNING: No tasks found for level '{level_key}', skipping.")
                continue

            # Match perturbed tasks with their L0 counterparts by list index
            # (EvoEval generates perturbations in the same order as HumanEval)
            max_pairs = min(self.max_pairs_per_level, len(perturbed_tasks), len(l0_tasks))
            level_pairs = []

            for i in tqdm(range(max_pairs), desc=f"Synthesizing contrastive pairs for {level_key}"):
                l0_task = l0_tasks[i % len(l0_tasks)]
                perturbed_task = perturbed_tasks[i]
                pair = self._synthesize_contrastive_pair(l0_task, perturbed_task)
                level_pairs.append(pair)

            print(f"[ContrastiveBuilder] Level {level_key}: {len(level_pairs)} contrastive pairs synthesized.")
            all_pairs.extend(level_pairs)

        # Shuffle the final dataset
        random.shuffle(all_pairs)

        # Save to JSONL
        with open(output_path, "w", encoding="utf-8") as f:
            for rec in all_pairs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"[ContrastiveBuilder] [OK] Total: {len(all_pairs)} contrastive pairs saved to '{output_path}'")
        return output_path

    @staticmethod
    def load_jsonl(path: str) -> List[Dict[str, Any]]:
        """Load a saved contrastive JSONL file."""
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    @staticmethod
    def get_stats(path: str) -> Dict[str, Any]:
        """Return summary statistics about a saved contrastive JSONL corpus."""
        records = ContrastiveDatasetBuilder.load_jsonl(path)
        level_counts: Dict[str, int] = {}
        token_counts = []
        for r in records:
            lvl = r.get("ladder_level", "unknown")
            level_counts[lvl] = level_counts.get(lvl, 0) + 1
            token_counts.append(r.get("token_count", 0))
        return {
            "total_pairs": len(records),
            "level_breakdown": level_counts,
            "avg_tokens": int(sum(token_counts) / max(len(token_counts), 1)),
            "min_tokens": min(token_counts, default=0),
            "max_tokens": max(token_counts, default=0),
        }
