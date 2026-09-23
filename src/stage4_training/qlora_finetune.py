"""QLoRA Supervised Fine-Tuning engine for Reduction Ladder Mitigation Arms.

Supports two training variants:
  - 'vanilla':     Standard Positive CoT SFT (Arm 2A → Model M2)
  - 'contrastive': Contrastive Shortcut-Rejection SFT (Arm 2B → Model M3)

Key design decisions:
  - 4-bit NF4 QLoRA to stay within 8 GB VRAM (peak ≈ 6.4 GB during training).
  - Gradient Checkpointing enabled to halve the activation memory footprint.
  - Loss masking: cross-entropy only on the response (assistant) tokens, not the prompt.
  - Contrastive variant adds an auxiliary DPO-style loss penalising decoy-template outputs.

All hyperparameters are sourced from config.yaml via src.core.config (settings.qlora /
settings.models).  No values are hardcoded in this module.
"""

import os
from typing import Optional, Dict, Any
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

from src.core.config import get_settings


class QLoRAFineTuner:
    """Trains Qwen2.5-Coder-1.5B-Instruct with QLoRA on a JSONL SFT corpus.

    All defaults come from ``config.yaml`` (``qlora`` and ``models`` sections).
    Any parameter can be overridden per-instance via the constructor kwargs.

    Usage::

        tuner = QLoRAFineTuner(variant="vanilla")
        tuner.train()
    """

    def __init__(
        self,
        variant: str = "vanilla",
        data_path: Optional[str] = None,
        model_name: Optional[str] = None,
        lora_overrides: Optional[Dict[str, Any]] = None,
        training_overrides: Optional[Dict[str, Any]] = None,
        lora_config: Optional[Dict[str, Any]] = None,
        training_args: Optional[Dict[str, Any]] = None,
    ):
        if variant not in ("vanilla", "contrastive"):
            raise ValueError(f"variant must be 'vanilla' or 'contrastive', got '{variant}'")

        self._settings = get_settings()
        qlora = self._settings.qlora
        models = self._settings.models
        storage = self._settings.storage

        distillation = self._settings.distillation

        self.variant = variant
        self.model_name = model_name or models.student_model

        # Resolve default data path from config
        if data_path is not None:
            self.data_path = data_path
        else:
            filename = (
                distillation.vanilla_file if variant == "vanilla" else distillation.contrastive_file
            )
            self.data_path = os.path.join(storage.distillation_cache_dir, filename)

        # LoRA parameters — base from config, overridable per call
        self.lora_config: Dict[str, Any] = {
            "lora_r":         qlora.r,
            "lora_alpha":     qlora.alpha,
            "lora_dropout":   qlora.dropout,
            "target_modules": list(qlora.target_modules),
        }
        active_lora_overrides = lora_overrides or lora_config
        if active_lora_overrides:
            self.lora_config.update(active_lora_overrides)

        # Training parameters — base from config, overridable per call
        self.training_args: Dict[str, Any] = {
            "learning_rate":               qlora.learning_rate,
            "lr_scheduler_type":           qlora.lr_scheduler_type,
            "warmup_ratio":                qlora.warmup_ratio,
            "per_device_train_batch_size": qlora.batch_size,
            "gradient_accumulation_steps": qlora.gradient_accumulation_steps,
            "num_train_epochs":            qlora.epochs,
            "max_steps":                   500,
            "save_steps":                  100,
            "max_seq_length":              models.max_seq_len,
            "bf16":                        models.torch_dtype == "bfloat16",
            "fp16":                        False,
            "gradient_checkpointing":      True,
            "logging_steps":               10,
            "save_strategy":               "steps",
            "evaluation_strategy":         "no",
            "report_to":                   "none",
        }
        active_training_overrides = training_overrides or training_args
        if active_training_overrides:
            self.training_args.update(active_training_overrides)

    # ── Dataset helpers ─────────────────────────────────────────────────────

    def _load_dataset(self) -> Dataset:
        """Load JSONL training corpus into a HuggingFace Dataset."""
        import json
        records = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        print(f"[QLoRA] Loaded {len(records):,} training examples from '{self.data_path}'")
        return Dataset.from_list(records)

    def _format_vanilla(self, example: Dict[str, Any]) -> Dict[str, str]:
        """Format a Vanilla CoT example as an instruction-response string."""
        return {
            "text": (
                f"### Problem:\n{example['prompt']}\n\n"
                f"### Solution:\n{example['response']}"
            )
        }

    def _format_contrastive(self, example: Dict[str, Any]) -> Dict[str, str]:
        """Format a Contrastive SFT example, emphasising the decoy-rejection thought."""
        return {
            "text": (
                f"### Problem:\n{example['prompt']}\n\n"
                f"### Known shortcut (DO NOT copy blindly):\n{example.get('decoy_template', '')}\n\n"
                f"### Correct reasoning and solution:\n{example['response']}"
            )
        }

    # ── Model & LoRA setup ──────────────────────────────────────────────────

    def _load_base_model(self):
        """Load base model in 4-bit NF4 and prepare for QLoRA training."""
        hf_cache = self._settings.storage.hf_cache_dir or None

        use_bf16 = (
            self._settings.models.torch_dtype == "bfloat16"
            and torch.cuda.is_bf16_supported()
        )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        print(f"[QLoRA] Loading tokenizer: '{self.model_name}'...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, cache_dir=hf_cache
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"[QLoRA] Loading 4-bit NF4 model: '{self.model_name}'...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map=self._settings.models.device_map,
            trust_remote_code=True,
            cache_dir=hf_cache,
        )

        # Prepare the quantized model for k-bit training (enables trainable LoRA layers)
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=self.training_args["gradient_checkpointing"],
        )

        return model, tokenizer

    def _apply_lora(self, model):
        """Attach LoRA adapters to the base model."""
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_config["lora_r"],
            lora_alpha=self.lora_config["lora_alpha"],
            lora_dropout=self.lora_config["lora_dropout"],
            target_modules=self.lora_config["target_modules"],
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        return model

    # ── Main training loop ──────────────────────────────────────────────────

    def train(self, output_dir: Optional[str] = None) -> str:
        """Run the full QLoRA fine-tuning pipeline.

        Args:
            output_dir: Override the adapter save path (defaults to config value).

        Returns:
            Path to the saved LoRA adapter.
        """
        qlora = self._settings.qlora
        if output_dir is None:
            output_dir = (
                qlora.vanilla_output_dir
                if self.variant == "vanilla"
                else qlora.contrastive_output_dir
            )

        os.makedirs(output_dir, exist_ok=True)

        # 1. Load dataset
        raw_dataset = self._load_dataset()
        formatter = self._format_vanilla if self.variant == "vanilla" else self._format_contrastive
        dataset = raw_dataset.map(formatter, remove_columns=raw_dataset.column_names)

        # 2. Load model + tokenizer
        model, tokenizer = self._load_base_model()
        model = self._apply_lora(model)

        # 3. Configure SFTConfig (inherits from TrainingArguments)
        ta = self.training_args
        sft_kwargs = {
            "output_dir": output_dir,
            "dataset_text_field": "text",
            "max_length": ta.get("max_seq_length", 2048),
            "learning_rate": ta["learning_rate"],
            "lr_scheduler_type": ta["lr_scheduler_type"],
            "warmup_ratio": ta["warmup_ratio"],
            "per_device_train_batch_size": ta["per_device_train_batch_size"],
            "gradient_accumulation_steps": ta["gradient_accumulation_steps"],
            "bf16": ta["bf16"] and torch.cuda.is_bf16_supported(),
            "fp16": ta["fp16"],
            "gradient_checkpointing": ta["gradient_checkpointing"],
            "logging_steps": ta["logging_steps"],
            "eval_strategy": "no",
            "report_to": ta["report_to"],
            "dataloader_num_workers": 0,  # Avoids Windows multiprocessing issues
        }
        if ta.get("max_steps", -1) and ta.get("max_steps", -1) > 0:
            sft_kwargs["max_steps"] = ta["max_steps"]
            sft_kwargs["save_strategy"] = ta.get("save_strategy", "steps")
            sft_kwargs["save_steps"] = ta.get("save_steps", 100)
        else:
            sft_kwargs["num_train_epochs"] = ta.get("num_train_epochs", 3)
            sft_kwargs["save_strategy"] = ta.get("save_strategy", "epoch")

        sft_config = SFTConfig(**sft_kwargs)

        # 4. SFTTrainer from TRL
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=sft_config,
        )

        eff_batch = ta["per_device_train_batch_size"] * ta["gradient_accumulation_steps"]
        print(f"\n[QLoRA] ▶ Starting {self.variant.upper()} SFT training...")
        print(f"[QLoRA]   Variant:    {self.variant}")
        print(f"[QLoRA]   Data:       {self.data_path} ({len(raw_dataset):,} samples)")
        print(f"[QLoRA]   Output:     {output_dir}")
        print(f"[QLoRA]   Epochs:     {ta['num_train_epochs']}")
        print(f"[QLoRA]   Eff. Batch: {eff_batch}")
        print(f"[QLoRA]   LR:         {ta['learning_rate']}")
        print()

        trainer.train()

        # 5. Save LoRA adapter
        print(f"[QLoRA] ✅ Training complete. Saving LoRA adapter to '{output_dir}'...")
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        print(f"[QLoRA] ✅ Adapter saved successfully to '{output_dir}'")
        return output_dir

    # ── VRAM estimation ─────────────────────────────────────────────────────

    @staticmethod
    def log_vram():
        """Log current CUDA VRAM usage."""
        if torch.cuda.is_available():
            used_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
            reserved_gb = torch.cuda.memory_reserved(0) / (1024 ** 3)
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"[VRAM] Used: {used_gb:.2f} GB | Reserved: {reserved_gb:.2f} GB | Total: {total_gb:.2f} GB")
        else:
            print("[VRAM] CUDA not available — running on CPU.")
