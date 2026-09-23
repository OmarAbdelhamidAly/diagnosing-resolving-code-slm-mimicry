"""AST-Guided Policy Optimization Trainer for Arm 3: AST-RL (Model M5).

Literature Basis:
- TreeDiff: Structural Code Comparison (ASE 2025)
- VeriSeek: Structure-Guided Code Synthesis Verification (ICSE 2025)

Reinforces policy generations using a composite reward:
    R_total(y, y*) = R_exec(y) + beta * simAST(AST(y), AST(y*))
Eliminates lexical overfitting on variable names and docstrings while guiding
the policy toward canonical control-flow algorithms.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from src.core.config import get_settings
from src.infrastructure.sandbox import SubprocessSandbox
from src.infrastructure.code_utils import extract_code as _extract_code
from src.arms.arm3_ast_rl.reward_engine import ASTRewardEngine


class ASTRLTrainer:
    """Production 500-step RLVR trainer guided by Abstract Syntax Tree similarity (Model M5)."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        output_dir: str = "checkpoints/rlvr_ast_final",
        group_size: int = 4,
        learning_rate: float = 1e-5,
        beta_ast: float = 0.3,
        alpha_tree: float = 0.05,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        sandbox_timeout: float = 3.0,
    ):
        self.settings = get_settings()
        self.model_name = model_name or self.settings.models.student_model
        self.output_dir = output_dir
        self.group_size = group_size
        self.learning_rate = learning_rate
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.reward_engine = ASTRewardEngine(beta=beta_ast, alpha_tree=alpha_tree)
        self.sandbox = SubprocessSandbox(default_timeout=sandbox_timeout)

        os.makedirs(self.output_dir, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._init_model_and_tokenizer()

    def _init_model_and_tokenizer(self):
        """Initializes tokenizer and 4-bit NF4 quantized base model with LoRA."""
        print(f"[ASTRLTrainer] Loading tokenizer for: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("[ASTRLTrainer] Loading 4-bit NF4 quantized model...")
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
        print(f"[ASTRLTrainer] LoRA initialized: {trainable:,} trainable params ({trainable/total*100:.2f}%).")

    def _format_prompt(self, prompt: str) -> str:
        """Formats prompt using chat template."""
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
        """Executes the AST-RL training loop for 500 steps.

        Args:
            tasks: List of task dicts containing 'prompt', 'test', 'entry_point', and 'canonical_solution'.
            num_steps: Total optimization steps (default 500).
            grad_accum_steps: Gradient accumulation steps.

        Returns:
            Dictionary containing logged training dynamics metrics.
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
            "mean_ast_sim": [],
            "pass_rate": [],
        }

        self.model.train()
        print(f"\n[ASTRLTrainer] [>>] Starting AST-RL Training ({num_steps} steps, G={self.group_size})...")

        pbar = tqdm(range(1, num_steps + 1), desc="AST-RL Training")
        accum_loss = 0.0

        for step in pbar:
            task = tasks[(step - 1) % len(tasks)]
            prompt = task.get("prompt", "")
            test = task.get("test", "")
            entry_point = task.get("entry_point", "")
            canonical_solution = task.get("canonical_solution", "")

            # 1. Rollout
            fmt_prompt = self._format_prompt(prompt)
            tokens, solutions, prompt_len = self._generate_group(fmt_prompt)

            # 2. Score completions via Execution + AST similarity
            rewards = []
            ast_sims = []
            for sol in solutions:
                res = self.sandbox.execute(prompt, sol, test, entry_point)
                rew_info = self.reward_engine.compute_reward(
                    code_gen=sol,
                    code_ref=canonical_solution,
                    exec_passed=res.passed,
                )
                rewards.append(rew_info["total_reward"])
                ast_sims.append(rew_info["ast_similarity"])

            # 3. Normalized Group Advantage
            mean_r = float(np.mean(rewards))
            std_r = float(np.std(rewards))
            advantages = [(r - mean_r) / (std_r + 1e-4) for r in rewards]
            adv_tensor = torch.tensor(advantages, device=self.device, dtype=torch.float32)

            # 4. Backward Pass (Micro-batched)
            step_loss = self._backward_group_loss(tokens, prompt_len, adv_tensor, grad_accum_steps)
            accum_loss += step_loss * grad_accum_steps

            if step % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            mean_sim = float(np.mean(ast_sims))
            pass_pct = float(sum(1 for r in rewards if r >= 1.0) / self.group_size * 100)

            history["step"].append(step)
            history["loss"].append(round(accum_loss, 4))
            history["mean_reward"].append(round(mean_r, 4))
            history["mean_ast_sim"].append(round(mean_sim, 4))
            history["pass_rate"].append(round(pass_pct, 1))

            pbar.set_postfix({
                "Reward": f"{mean_r:.2f}",
                "simAST": f"{mean_sim:.2f}",
                "Pass%": f"{pass_pct:.0f}%",
                "Loss": f"{step_loss:.3f}",
            })
            accum_loss = 0.0

            # Periodic checkpoint save every 100 steps
            if step % 100 == 0 and step < num_steps:
                self.model.save_pretrained(self.output_dir)
                self.tokenizer.save_pretrained(self.output_dir)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"\n[ASTRLTrainer] [Checkpoint] Saving Final AST-RL Checkpoint to: {self.output_dir}")
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print("[ASTRLTrainer] [OK] Checkpoint successfully saved!")

        return history
