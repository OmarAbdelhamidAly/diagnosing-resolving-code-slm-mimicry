from src.arms.arm2_contrastive_dpo.dataset import (
    ContrastiveDatasetParser,
    to_dpo_triplet,
    format_dpo_dataset,
)


def __getattr__(name: str):
    if name in ("ContrastiveSFTTrainer", "ContrastiveDPOTrainer"):
        from src.arms.arm2_contrastive_dpo.trainer import (
            ContrastiveSFTTrainer,
            ContrastiveDPOTrainer,
        )
        return ContrastiveSFTTrainer if name == "ContrastiveSFTTrainer" else ContrastiveDPOTrainer
    raise AttributeError(f"module 'src.arms.arm2_contrastive_dpo' has no attribute '{name}'")


__all__ = [
    "ContrastiveDatasetParser",
    "to_dpo_triplet",
    "format_dpo_dataset",
    "ContrastiveSFTTrainer",
    "ContrastiveDPOTrainer",
]

