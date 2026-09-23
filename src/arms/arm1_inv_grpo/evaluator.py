"""Evaluation and comparison harness for Arm 1: Inv-GRPO (Model M1 vs M6).

Compares the zero-shot base policy (M1) against the Invariance-Regularized Policy (M6)
across Canonical (L0) and Perturbed (L2) tasks to measure:
  1. Pass@1 degradation across levels
  2. Pairwise Consistency Rate (solving both representations)
  3. Reduction in Shortcut Mimicry Rate (Template Penalty)
"""

from typing import List, Dict, Any, Optional
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.infrastructure.sandbox import SubprocessSandbox
from src.arms.arm1_inv_grpo.dataset import PairedTask
from src.arms.arm1_inv_grpo.reward_engine import InvGRPORewardEngine, _normalize_code
from src.infrastructure.model_loader import extract_code as _extract_code  # canonical source


class InvGRPOEvaluator:
    """Evaluates policies on held-out paired tasks."""

    def __init__(self, sandbox: Optional[SubprocessSandbox] = None):
        self.reward_engine = InvGRPORewardEngine(sandbox=sandbox)

    def evaluate_model(
        self,
        model,
        tokenizer,
        test_pairs: List[PairedTask],
        model_label: str = "Model",
        max_new_tokens: int = 256,
    ) -> Dict[str, Any]:
        """Runs greedy deterministic evaluation on the test pairs."""
        results = []
        device = next(model.parameters()).device

        print(f"\n[InvGRPOEvaluator] [Eval] Evaluating {model_label} on {len(test_pairs)} held-out pairs (Greedy T=0.0)...")

        for task in tqdm(test_pairs, desc=f"Evaluating {model_label}"):
            # Format prompts
            fmt_orig = tokenizer.apply_chat_template(
                [{"role": "user", "content": task.prompt_orig}], tokenize=False, add_generation_prompt=True
            )
            fmt_pert = tokenizer.apply_chat_template(
                [{"role": "user", "content": task.prompt_pert}], tokenize=False, add_generation_prompt=True
            )

            # Generate greedily
            in_orig = tokenizer(fmt_orig, return_tensors="pt").to(device)
            in_pert = tokenizer(fmt_pert, return_tensors="pt").to(device)

            model.eval()
            with __import__("torch").no_grad():
                out_orig = model.generate(
                    **in_orig, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id
                )
                out_pert = model.generate(
                    **in_pert, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id
                )

            sol_orig = _extract_code(tokenizer.decode(out_orig[0][in_orig.input_ids.shape[1]:], skip_special_tokens=True))
            sol_pert = _extract_code(tokenizer.decode(out_pert[0][in_pert.input_ids.shape[1]:], skip_special_tokens=True))

            score = self.reward_engine.compute_paired_reward(sol_orig, sol_pert, task)
            results.append({
                "pair_id": task.pair_id,
                "l0_task_id": task.l0_task_id,
                "pert_task_id": task.pert_task_id,
                "passed_orig": score["passed_orig"],
                "passed_pert": score["passed_pert"],
                "consistency": score["r_consistency"] > 0,
                "template_penalty": score["p_template"] > 0,
                "r_total": score["r_total"],
            })

        df = pd.DataFrame(results)
        n = len(df)
        l0_pass = df["passed_orig"].mean() * 100
        pert_pass = df["passed_pert"].mean() * 100
        cons_rate = df["consistency"].mean() * 100
        mimic_rate = df["template_penalty"].mean() * 100

        metrics = {
            "model_label": model_label,
            "total_pairs": n,
            "l0_pass_rate": round(l0_pass, 1),
            "pert_pass_rate": round(pert_pass, 1),
            "consistency_rate": round(cons_rate, 1),
            "mimicry_rate": round(mimic_rate, 1),
            "df_results": df,
        }

        print(f"\n[InvGRPOEvaluator] {model_label} Results ({n} pairs):")
        print(f"   L0 (HumanEval) Pass Rate : {l0_pass:.1f}%")
        print(f"   L2 (EvoEval)   Pass Rate : {pert_pass:.1f}%")
        print(f"   Pair Consistency Bonus   : {cons_rate:.1f}%")
        print(f"   Shortcut Mimicry Rate    : {mimic_rate:.1f}%")

        return metrics

    @staticmethod
    def plot_comparison(
        m1_metrics: Dict[str, Any],
        m6_metrics: Dict[str, Any],
        save_path: str = "results/inv_grpo_m1_vs_m6_comparison.png",
    ):
        """Generates and saves a publication-ready comparative bar chart."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        categories = ["L0 Pass Rate", "L2 Pass Rate", "Consistency", "Mimicry (Penalty)"]
        m1_vals = [
            m1_metrics["l0_pass_rate"],
            m1_metrics["pert_pass_rate"],
            m1_metrics["consistency_rate"],
            m1_metrics["mimicry_rate"],
        ]
        m6_vals = [
            m6_metrics["l0_pass_rate"],
            m6_metrics["pert_pass_rate"],
            m6_metrics["consistency_rate"],
            m6_metrics["mimicry_rate"],
        ]

        x = [0, 1, 2, 3]
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
        bars1 = ax.bar([p - width/2 for p in x], m1_vals, width, label="M1 (Baseline)", color="#6C757D", edgecolor="black")
        bars2 = ax.bar([p + width/2 for p in x], m6_vals, width, label="M6 (Inv-GRPO)", color="#FF6400", edgecolor="black")

        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.0f}%", ha="center", va="bottom", fontsize=8, color="#495057")
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#C04000")

        ax.set_ylabel("Percentage (%)", fontsize=10)
        ax.set_title("Arm 1 Mitigation: M1 (Zero-Shot Baseline) vs. M6 (Inv-GRPO Trained)", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylim(0, max(max(m1_vals), max(m6_vals)) + 15)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(save_path, dpi=140)
        plt.close()
        print(f"[InvGRPOEvaluator] Comparison plot saved: {save_path}")
