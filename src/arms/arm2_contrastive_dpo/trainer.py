"""Contrastive SFT / DPO Trainer for Arm 2 (Model M3).

Literature Basis:
- SuperCorrect (ICLR 2025 / NeurIPS 2024)
- ReCode (2025)

Fine-tunes Qwen2.5-Coder-1.5B-Instruct on paired contrastive examples
explicitly conditioning on rejecting memorized decoy templates:
    Prompt: Task statement + modified constraints
    Known shortcut (to REJECT): Decoy template from canonical L0
    Target Response: Invariant reasoning trace + correct code
"""

import os
import sys
from typing import Dict, Any, Optional
from datasets import Dataset

from src.core.config import get_settings
from src.stage4_training.qlora_finetune import QLoRAFineTuner


class ContrastiveSFTTrainer:
    """Production 500-step/pair QLoRA trainer for Arm 2 (Contrastive SFT - Model M3)."""

    def __init__(
        self,
        data_path: Optional[str] = None,
        model_name: Optional[str] = None,
        output_dir: str = "checkpoints/qlora_contrastive_adapter",
        num_epochs: int = 1,
        learning_rate: float = 2e-4,
    ):
        self.settings = get_settings()
        self.data_path = data_path or os.path.join(
            self.settings.storage.distillation_cache_dir,
            self.settings.distillation.contrastive_file,
        )
        self.model_name = model_name or self.settings.models.student_model
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        os.makedirs(self.output_dir, exist_ok=True)

    def train(
        self,
        max_steps: Optional[int] = 500,
    ) -> str:
        """Executes full QLoRA fine-tuning for Model M3.

        Args:
            max_steps: Maximum training optimization steps (default 500).

        Returns:
            Path string to saved adapter checkpoint directory.
        """
        print(f"\n[ContrastiveSFTTrainer] [>>] Initializing Contrastive QLoRA Fine-Tuner...")
        print(f"   Model       : {self.model_name}")
        print(f"   Corpus      : {self.data_path}")
        print(f"   Output      : {self.output_dir}")
        print(f"   Target Steps: {max_steps}")

        training_overrides = {
            "learning_rate": self.learning_rate,
            "output_dir": self.output_dir,
        }
        if max_steps is not None and max_steps > 0:
            training_overrides["max_steps"] = max_steps

        tuner = QLoRAFineTuner(
            variant="contrastive",
            data_path=self.data_path,
            model_name=self.model_name,
            training_overrides=training_overrides,
        )

        saved_path = tuner.train(output_dir=self.output_dir)
        print(f"[ContrastiveSFTTrainer] [OK] Model M3 adapter successfully saved to: {saved_path}")
        return saved_path
