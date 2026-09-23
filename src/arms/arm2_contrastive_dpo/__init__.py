"""Arm 2: Contrastive Thought-Template SFT / DPO package."""

from src.arms.arm2_contrastive_dpo.dataset import (
    ContrastiveDatasetParser,
    to_dpo_triplet,
    format_dpo_dataset,
)

__all__ = [
    "ContrastiveDatasetParser",
    "to_dpo_triplet",
    "format_dpo_dataset",
]
