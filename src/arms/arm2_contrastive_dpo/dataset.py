"""Dataset loader and DPO preference pair formatter for Arm 2: Contrastive SFT / DPO.

Literature Basis:
- SuperCorrect (ICLR 2025 / NeurIPS 2024)
- ReCode (2025)
- Contrastive CoT (Chia et al., 2023)

In Contrastive SFT / DPO:
- prompt (x): Perturbed problem statement
- chosen (y+): Reasoning trace that explicitly rejects the decoy shortcut and derives the correct algorithm
- rejected (y-): Memorized shortcut template from canonical HumanEval that fails on the perturbed task
"""

import json
import os
from typing import Dict, List, Any, Optional


class ContrastiveDatasetParser:
    """Loads and inspects contrastive pair datasets for SFT and DPO training."""

    @staticmethod
    def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
        """Load JSONL dataset of contrastive pairs."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Contrastive dataset not found at: {filepath}")
        records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    @staticmethod
    def get_stats(filepath: str) -> Dict[str, Any]:
        """Compute summary statistics across contrastive records."""
        records = ContrastiveDatasetParser.load_jsonl(filepath)
        if not records:
            return {"total_records": 0, "avg_tokens": 0, "level_breakdown": {}}

        total_tokens = 0
        level_counts: Dict[str, int] = {}

        for r in records:
            # Approximate token count (char count / 4)
            resp = r.get("response", "")
            total_tokens += len(resp.split())
            lvl = r.get("ladder_level", "Unknown")
            level_counts[lvl] = level_counts.get(lvl, 0) + 1

        avg_tok = round(total_tokens / max(len(records), 1))
        return {
            "total_records": len(records),
            "avg_tokens": avg_tok,
            "level_breakdown": dict(sorted(level_counts.items())),
        }


def to_dpo_triplet(record: Dict[str, Any]) -> Dict[str, str]:
    """Convert a raw contrastive record to a standard HuggingFace/TRL DPOTrainer triplet.

    Args:
        record: Raw dict containing 'prompt', 'response', and 'decoy_template'.

    Returns:
        Dict with keys:
            - prompt: Problem statement.
            - chosen: Ground-truth/contrastive reasoning trace (y+).
            - rejected: Memorized decoy shortcut template (y-).
    """
    return {
        "prompt": record.get("prompt", ""),
        "chosen": record.get("response", ""),
        "rejected": record.get("decoy_template", ""),
    }


def format_dpo_dataset(records: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Transform an entire list of contrastive records into DPO preference triplets."""
    return [to_dpo_triplet(rec) for rec in records]
