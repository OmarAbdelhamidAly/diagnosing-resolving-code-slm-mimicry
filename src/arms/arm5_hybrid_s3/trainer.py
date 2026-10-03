"""S³-GRPO Trainer: Structural, Stepwise & Invariant Group Relative Policy Optimization.

Literature Basis:
- DeepSeekMath & DeepSeek-R1 GRPO (Shao et al. 2024, Guo et al. 2025)
- CodePRM: Process Reward Models for Code Reasoning (ACL 2025)
- TreeDiff & VeriSeek: AST Structural Code Verification (ASE/ICSE 2025)
- Invariance Regularization for SLM Mimicry (Aly et al. 2026)

Orchestrates multi-objective RL with:
1. Dense stepwise contract verification (solves reward sparsity)
2. Normalized AST structural fidelity (solves syntactic rambling)
3. Anti-overthinking parsimony penalty (eliminates reward hacking on length)
4. Group-normalized advantage policy updates (4-bit NF4 QLoRA)
"""

import os
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from src.core.config import get_settings
from src.infrastructure.sandbox import SubprocessSandbox
from src.infrastructure.code_utils import extract_code as _extract_code
from src.arms.arm5_hybrid_s3.reward_engine import S3RewardEngine


class S3GRPOTrainer:
    """Flagship multi-objective S³-GRPO Trainer for anti-mimicry reasoning in SLMs."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        output_dir: str = "checkpoints/s3_grpo_final",
        group_size: int = 4,
        learning_rate: float = 1e-5,
        max_new_tokens: int = 384,
        temperature: float = 0.8,
        sandbox_timeout: float = 3.0,
        w_step: float = 0.50,
        w_ast: float = 0.30,
        w_inv: float = 0.20,
        w_parsimony: float = 0.15,
    ):
        self.settings = get_settings()
        self.model_name = model_name or self.settings.models.student_model
        self.output_dir = output_dir
        self.group_size = group_size
        self.learning_rate = learning_rate
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.sandbox = SubprocessSandbox(default_timeout=sandbox_timeout)
        self.reward_engine = S3RewardEngine(
            sandbox=self.sandbox,
            w_step=w_step,
            w_ast=w_ast,
            w_inv=w_inv,
            w_parsimony=w_parsimony,
            sandbox_timeout=sandbox_timeout,
        )

        os.makedirs(self.output_dir, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._init_model_and_tokenizer()

    def _init_model_and_tokenizer(self):
        """Initializes tokenizer and 4-bit NF4 quantized base model with LoRA."""
        print(f"[S³-GRPO] Loading tokenizer for: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        use_quant = str(getattr(self.settings.models, "quantization", "none")).lower() not in (
            "none", "null", "false", "bfloat16", "float16", "fp16", "bf16", ""
        )
        torch_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else (torch.float16 if torch.cuda.is_available() else torch.float32)
        )

        if use_quant:
            print("[S³-GRPO] Loading 4-bit NF4 quantized model...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
        else:
            print(f"[S³-GRPO] Loading UNQUANTIZED native model ({torch_dtype})...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
            )

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
        print(f"[S³-GRPO] LoRA initialized: {trainable:,} trainable params ({trainable/total*100:.2f}%).")

    def _format_prompt(self, prompt: str) -> str:
        """Formats prompt using chat template with system guidance."""
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _generate_group(self, formatted_prompt: str) -> Tuple[torch.Tensor, List[str], int]:
        """Samples G completions for a single prompt."""
        target_device = getattr(self.model, "device", None) or next(self.model.parameters()).device
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(target_device)
        prompt_len = inputs.input_ids.shape[1]

        # Multi-token EOS termination to halt immediately upon generation end
        eos_token_ids = [self.tokenizer.eos_token_id]
        for special in ["<|im_end|>", "<|endoftext|>"]:
            try:
                tok_id = self.tokenizer.convert_tokens_to_ids(special)
                if tok_id is not None and isinstance(tok_id, int) and tok_id not in eos_token_ids:
                    eos_token_ids.append(tok_id)
            except Exception:
                pass

        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.95,
                num_return_sequences=self.group_size,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=eos_token_ids,
            )
        self.model.train()

        raw_texts = self.tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
        extracted = [_extract_code(t) for t in raw_texts]
        return outputs, extracted, prompt_len

    def _backward_group_loss(
        self,
        token_ids: torch.Tensor,
        comp_start: int,
        adv_tensor: torch.Tensor,
        grad_accum_steps: int,
    ) -> float:
        """Computes policy loss sample-by-sample with micro-batching to preserve VRAM."""
        total_loss = 0.0

        for i in range(self.group_size):
            sample_ids = token_ids[i : i + 1]
            attention_mask = (sample_ids != self.tokenizer.pad_token_id).long()
            self.model.config.use_cache = False

            logits = self.model(input_ids=sample_ids, attention_mask=attention_mask).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = sample_ids[:, 1:].contiguous()

            nll = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).view(1, -1)

            token_log_probs = -nll
            comp_log_prob = token_log_probs[:, comp_start:].mean()

            loss_i = - (comp_log_prob * adv_tensor[i]) / (self.group_size * grad_accum_steps)
            loss_i.backward()
            total_loss += loss_i.item()

            del logits, shift_logits, shift_labels, nll, token_log_probs, comp_log_prob, loss_i

        return total_loss

    def train(
        self,
        tasks: List[Dict[str, Any]],
        num_steps: int = 500,
        grad_accum_steps: int = 2,
    ) -> Dict[str, List[float]]:
        """Executes full S³-GRPO multi-objective training loop.

        Args:
            tasks: List of task dicts (prompts, tests, entry_points, and canonical solutions).
            num_steps: Total policy optimization steps.
            grad_accum_steps: Gradient accumulation steps.

        Returns:
            Dictionary containing logged training metrics.
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
            "mean_step_reward": [],
            "mean_ast_sim": [],
            "mean_parsimony_penalty": [],
            "pass_rate": [],
        }

        self.model.train()
        print(f"\n[S³-GRPO] 🚀 Starting S³-GRPO Multi-Objective Training ({num_steps} steps, G={self.group_size})...")

        pbar = tqdm(range(1, num_steps + 1), desc="S³-GRPO Training")
        accum_loss = 0.0

        for step in pbar:
            task = tasks[(step - 1) % len(tasks)]
            prompt = task.get("prompt", "")
            test = task.get("test", "")
            entry = task.get("entry_point", "")
            canonical = task.get("canonical_solution", "") or task.get("solution", "")

            # 1. Sample G rollouts from current policy
            fmt_prompt = self._format_prompt(prompt)
            tokens, solutions, prompt_len = self._generate_group(fmt_prompt)

            # 2. Score rollouts across all 4 S³ objectives
            rewards = []
            step_rewards = []
            ast_sims = []
            parsimony_pens = []
            pass_flags = []

            for sol in solutions:
                r_dict = self.reward_engine.compute_reward(
                    prompt=prompt,
                    solution=sol,
                    test=test,
                    entry_point=entry,
                    canonical_solution=canonical,
                )
                rewards.append(r_dict["r_total"])
                step_rewards.append(r_dict["r_step"])
                ast_sims.append(r_dict["r_ast"])
                parsimony_pens.append(r_dict["p_parsimony"])
                pass_flags.append(1.0 if r_dict["all_passed"] else 0.0)

            # 3. Compute GRPO Relative Advantages across group G
            advantages = self.reward_engine.compute_group_advantages(rewards)
            adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=tokens.device)

            # 4. Backward policy gradient loss with micro-batching
            step_loss = self._backward_group_loss(tokens, prompt_len, adv_tensor, grad_accum_steps)
            accum_loss += step_loss

            # 5. Optimizer step
            if step % grad_accum_steps == 0 or step == num_steps:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                accum_loss = 0.0

            # 6. Logging
            m_r = float(np.mean(rewards))
            m_step = float(np.mean(step_rewards))
            m_ast = float(np.mean(ast_sims))
            m_tax = float(np.mean(parsimony_pens))
            p_rate = float(np.mean(pass_flags))

            history["step"].append(step)
            history["loss"].append(step_loss)
            history["mean_reward"].append(m_r)
            history["mean_step_reward"].append(m_step)
            history["mean_ast_sim"].append(m_ast)
            history["mean_parsimony_penalty"].append(m_tax)
            history["pass_rate"].append(p_rate)

            pbar.set_postfix({
                "R": f"{m_r:.2f}",
                "StepR": f"{m_step:.2f}",
                "AST": f"{m_ast:.2f}",
                "Tax": f"{m_tax:.2f}",
                "Pass%": f"{p_rate*100:.0f}%",
                "Loss": f"{step_loss:.3f}",
            })

            # Checkpoint saving every 100 steps
            if step % 100 == 0 or step == num_steps:
                ckpt_dir = os.path.join(self.output_dir, f"checkpoint-{step}")
                self.model.save_pretrained(ckpt_dir)
                self.tokenizer.save_pretrained(ckpt_dir)

        # Save final checkpoint
        print(f"\n[S³-GRPO] [Checkpoint] Saving Final S³-GRPO Checkpoint to: {self.output_dir}")
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print("[S³-GRPO] [OK] Final Checkpoint successfully saved!")
        return history
