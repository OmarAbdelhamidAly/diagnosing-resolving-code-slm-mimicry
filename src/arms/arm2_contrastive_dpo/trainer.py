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


class ContrastiveDPOTrainer:
    """Production Direct Preference Optimization (DPO) Trainer for Arm 2 (Rafailov et al., NeurIPS 2023).

    Directly optimizes the policy using the closed-form DPO objective against the implicit reference model:
        L_DPO(theta; pi_ref) = -E_{(x, y+, y-) ~ D}[ log sigma( beta * log(pi_theta(y+|x) / pi_ref(y+|x))
                                                               - beta * log(pi_theta(y-|x) / pi_ref(y-|x)) ) ]
    Where:
        x   : Problem prompt under perturbation (e.g. L2/L4)
        y+  : Invariant reasoning trace and valid algorithm (chosen)
        y-  : Memorized canonical HumanEval L0 shortcut that fails under perturbation (rejected decoy)
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        model_name: Optional[str] = None,
        output_dir: str = "checkpoints/qlora_contrastive_adapter",
        beta: float = 0.1,
        learning_rate: float = 5e-5,
        max_steps: int = 500,
        per_device_batch_size: int = 2,
        gradient_accumulation_steps: int = 8,
        max_prompt_length: int = 512,
        max_length: int = 1024,
    ):
        self.settings = get_settings()
        self.data_path = data_path or os.path.join(
            self.settings.storage.distillation_cache_dir,
            self.settings.distillation.contrastive_file,
        )
        self.model_name = model_name or self.settings.models.student_model
        self.output_dir = output_dir
        self.beta = beta
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.per_device_batch_size = per_device_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_prompt_length = max_prompt_length
        self.max_length = max_length

        os.makedirs(self.output_dir, exist_ok=True)

    def _prepare_dataset(self) -> Dataset:
        """Loads and formats the contrastive corpus into standard HuggingFace DPO triplets."""
        from src.arms.arm2_contrastive_dpo.dataset import ContrastiveDatasetParser, format_dpo_dataset
        records = ContrastiveDatasetParser.load_jsonl(self.data_path)
        triplets = format_dpo_dataset(records)
        print(f"[ContrastiveDPOTrainer] Formatted {len(triplets)} DPO triplets (prompt, chosen, rejected).")
        return Dataset.from_list(triplets)

    def train(self) -> str:
        """Executes full DPO training loop with LoRA and 4-bit NF4 base quantization."""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig, prepare_model_for_kbit_training
        from trl import DPOTrainer, DPOConfig

        print(f"\n[ContrastiveDPOTrainer] [>>] Initializing Direct Preference Optimization (Rafailov et al., 2023)...")
        print(f"   Model       : {self.model_name}")
        print(f"   Corpus      : {self.data_path}")
        print(f"   Output      : {self.output_dir}")
        print(f"   Beta (β)    : {self.beta}")
        print(f"   Target Steps: {self.max_steps}")

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

        peft_config = LoraConfig(
            r=self.settings.qlora.r,
            lora_alpha=self.settings.qlora.alpha,
            lora_dropout=self.settings.qlora.dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(self.settings.qlora.target_modules),
        )

        train_dataset = self._prepare_dataset()

        use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
        dpo_config = DPOConfig(
            output_dir=self.output_dir,
            beta=self.beta,
            learning_rate=self.learning_rate,
            max_steps=self.max_steps,
            per_device_train_batch_size=self.per_device_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            max_prompt_length=self.max_prompt_length,
            max_length=self.max_length,
            bf16=use_bf16,
            fp16=not use_bf16,
            remove_unused_columns=False,
            report_to="none",
        )

        dpo_trainer = DPOTrainer(
            model=base_model,
            ref_model=None,  # PEFT disables adapter automatically for implicit ref_model
            peft_config=peft_config,
            args=dpo_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )

        print("[ContrastiveDPOTrainer] [>>] Launching DPO preference optimization...")
        dpo_trainer.train()

        print(f"[ContrastiveDPOTrainer] [Checkpoint] Saving DPO LoRA adapter to: {self.output_dir}")
        dpo_trainer.save_model(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        print(f"[ContrastiveDPOTrainer] [OK] DPO Model M3 successfully saved to: {self.output_dir}")
        return self.output_dir

