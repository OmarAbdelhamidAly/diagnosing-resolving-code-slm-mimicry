"""Inv-GRPO Online Reinforcement Learning Trainer.

Orchestrates the full Group Relative Policy Optimization training loop with paired
invariance regularization on 4-bit NF4 quantized models via PEFT/LoRA.
"""

import os
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from src.core.config import get_settings
from src.arms.arm1_inv_grpo.dataset import PairedTask
from src.arms.arm1_inv_grpo.reward_engine import InvGRPORewardEngine
from src.infrastructure.model_loader import extract_code as _extract_code  # canonical source


class InvGRPOTrainer:
    """End-to-end Inv-GRPO Policy Optimization Trainer."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        output_dir: str = "checkpoints/inv_grpo_final",
        group_size: int = 4,
        learning_rate: float = 1e-5,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        lambda_consistency: float = 0.5,
        gamma_template_penalty: float = 0.5,
    ):
        self.settings = get_settings()
        self.model_name = model_name or self.settings.models.student_model
        self.output_dir = output_dir
        self.group_size = group_size
        self.learning_rate = learning_rate
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.reward_engine = InvGRPORewardEngine(
            lambda_consistency=lambda_consistency,
            gamma_template_penalty=gamma_template_penalty,
        )

        os.makedirs(self.output_dir, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._init_model_and_tokenizer()

    def _init_model_and_tokenizer(self):
        """Initializes tokenizer and 4-bit NF4 quantized base model with LoRA."""
        print(f"[InvGRPOTrainer] Loading tokenizer for: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("[InvGRPOTrainer] Loading 4-bit NF4 quantized model...")
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
        self.model = get_peft_model(base_model, peft_config)
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"[InvGRPOTrainer] LoRA initialized: {trainable:,} trainable params ({trainable/total*100:.2f}%).")

    def _format_prompt(self, prompt: str) -> str:
        """Formats code generation prompt with standard chat template."""
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _generate_group(self, formatted_prompt: str) -> Tuple[torch.Tensor, List[str], int]:
        """Samples G completions for a single prompt."""
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs.input_ids.shape[1]

        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.95,
                num_return_sequences=self.group_size,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        self.model.train()

        # Decode generated responses
        raw_texts = self.tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
        extracted = [_extract_code(t) for t in raw_texts]
        return outputs, extracted, prompt_len

    def _backward_group_loss(
        self,
        token_ids: torch.Tensor,
        prompt_len: int,
        adv_tensor: torch.Tensor,
        grad_accum_steps: int,
    ) -> float:
        """Computes completion log-probs and executes backward pass sample-by-sample.

        Using micro-batch size of 1 with fused cross-entropy avoids materializing
        giant (G, SeqLen, VocabSize) logit/log-prob tensors, reducing peak VRAM by >75%
        and preventing CUDA OOM on consumer GPUs (e.g. RTX 3070 Ti 8GB).
        """
        total_loss = 0.0
        comp_start = max(prompt_len - 1, 0)

        for i in range(self.group_size):
            sample_ids = token_ids[i : i + 1]
            attention_mask = (sample_ids != self.tokenizer.pad_token_id).long()
            self.model.config.use_cache = False

            logits = self.model(input_ids=sample_ids, attention_mask=attention_mask).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = sample_ids[:, 1:].contiguous()

            # Fused cross-entropy without allocating full (1, T, V) log-softmax tensor
            nll = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).view(1, -1)

            token_log_probs = -nll
            comp_log_prob = token_log_probs[:, comp_start:].mean()

            # Inv-GRPO sample policy loss: - (log_prob * A) / (G * grad_accum_steps)
            loss_i = - (comp_log_prob * adv_tensor[i]) / (self.group_size * grad_accum_steps)
            loss_i.backward()
            total_loss += loss_i.item()

            del logits, shift_logits, shift_labels, nll, token_log_probs, comp_log_prob, loss_i

        return total_loss

    def train(
        self,
        train_pairs: List[PairedTask],
        num_steps: int = 500,
        grad_accum_steps: int = 2,
    ) -> Dict[str, List[float]]:
        """Executes the Inv-GRPO policy optimization loop.

        Args:
            train_pairs: List of PairedTask instances to sample from.
            num_steps: Total number of optimization steps to run (default 500 for full production training).
            grad_accum_steps: Number of steps to accumulate gradients before optimizer step.

        Returns:
            Dictionary containing logged training history metrics.
        """
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=0.01,
        )

        history = {
            "step": [],
            "loss": [],
            "mean_reward": [],
            "consistency_rate": [],
            "template_penalty_rate": [],
        }

        self.model.train()
        print(f"\n[InvGRPOTrainer] [>>] Starting Inv-GRPO Training ({num_steps} steps, Group size G={self.group_size})...")

        pbar = tqdm(range(1, num_steps + 1), desc="Inv-GRPO Training")
        accum_loss = 0.0

        for step in pbar:
            # Pick a paired task (cycling through train_pairs)
            task = train_pairs[(step - 1) % len(train_pairs)]

            # 1. Rollout: Generate G completions for canonical x and G for perturbed x'
            fmt_orig = self._format_prompt(task.prompt_orig)
            fmt_pert = self._format_prompt(task.prompt_pert)

            tokens_orig, sols_orig, len_orig = self._generate_group(fmt_orig)
            tokens_pert, sols_pert, len_pert = self._generate_group(fmt_pert)

            # Build rollout pairs
            paired_rollouts = [
                {"solution_orig": sols_orig[i], "solution_pert": sols_pert[i]}
                for i in range(self.group_size)
            ]

            # 2. Compute Inv-GRPO Invariance Rewards & Group Advantages
            advantages, eval_records = self.reward_engine.compute_group_advantages(paired_rollouts, task)
            adv_tensor = torch.tensor(advantages, device=self.device, dtype=torch.float32)

            # 3. Compute policy loss & backward sample-by-sample (micro-batch=1)
            loss_orig = self._backward_group_loss(tokens_orig, len_orig, adv_tensor, grad_accum_steps)
            loss_pert = self._backward_group_loss(tokens_pert, len_pert, adv_tensor, grad_accum_steps)
            step_loss = loss_orig + loss_pert

            accum_loss += step_loss * grad_accum_steps

            if step % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            # Metric logging
            rewards = [r["r_total"] for r in eval_records]
            cons_count = sum(1 for r in eval_records if r["r_consistency"] > 0)
            pen_count = sum(1 for r in eval_records if r["p_template"] > 0)

            mean_rew = float(np.mean(rewards))
            cons_pct = float(cons_count / self.group_size * 100)
            pen_pct = float(pen_count / self.group_size * 100)

            history["step"].append(step)
            history["loss"].append(round(accum_loss, 4))
            history["mean_reward"].append(round(mean_rew, 4))
            history["consistency_rate"].append(round(cons_pct, 1))
            history["template_penalty_rate"].append(round(pen_pct, 1))

            pbar.set_postfix({
                "Reward": f"{mean_rew:.2f}",
                "Cons%": f"{cons_pct:.0f}%",
                "Pen%": f"{pen_pct:.0f}%",
                "Loss": f"{step_loss:.3f}"
            })
            accum_loss = 0.0

            # Periodic checkpoint save every 100 steps
            if step % 100 == 0 and step < num_steps:
                self.model.save_pretrained(self.output_dir)
                self.tokenizer.save_pretrained(self.output_dir)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"\n[InvGRPOTrainer] [Checkpoint] Saving Final Inv-GRPO Adapter Checkpoint to: {self.output_dir}")
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print("[InvGRPOTrainer] [OK] Checkpoint successfully saved!")

        return history
