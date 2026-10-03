"""Dedicated training script for Arm 5: S³-GRPO (Model M8).

Structural, Stepwise & Invariant Group Relative Policy Optimization.
Literature Basis:
- DeepSeekMath & DeepSeek-R1 GRPO (Shao et al. 2024, Guo et al. 2025)
- CodePRM: Process Reward Models for Code Reasoning (ACL 2025)
- TreeDiff & VeriSeek: AST Structural Code Verification (ASE/ICSE 2025)
- Invariance Regularization for SLM Mimicry (Aly et al. 2026)

Runs 500 optimization steps combining:
- Dense stepwise process verification
- Normalized AST structural similarity
- Anti-overthinking parsimony penalty
Final LoRA adapter saved to: checkpoints/s3_grpo_final
"""

import os
import sys
import time
from pathlib import Path

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from src.arms.arm5_hybrid_s3.trainer import S3GRPOTrainer
from src.evaluation.registry import BenchmarkRegistry


def main():
    data_dir = REPO_ROOT / "data" / "ladder"
    print(f"[S³-GRPO] Loading benchmark tasks from: {data_dir}")
    registry = BenchmarkRegistry(data_dir=data_dir)

    # Multi-rung training pool: L1 (Subtle), L3 (Creative), L4 (Difficult), L5 (Combine)
    tasks = [
        {
            "prompt": t.prompt,
            "test": t.test,
            "entry_point": t.entry_point,
            "canonical_solution": t.canonical_solution or "",
        }
        for t in (
            registry.get_tasks("L1")
            + registry.get_tasks("L3")
            + registry.get_tasks("L4")
            + registry.get_tasks("L5")
        )
    ]
    print(f"[S³-GRPO] Loaded {len(tasks)} multi-rung tasks with canonical solutions.")

    output_dir = "checkpoints/s3_grpo_final"
    trainer = S3GRPOTrainer(
        output_dir=output_dir,
        group_size=4,
        learning_rate=1e-5,
        max_new_tokens=384,
        w_step=0.50,
        w_ast=0.30,
        w_inv=0.20,
        w_parsimony=0.15,
    )

    NUM_TRAIN_STEPS = 500
    print(f"[S³-GRPO] Starting 500-step multi-objective policy optimization on GPU...")
    start_time = time.time()

    history = trainer.train(
        tasks=tasks,
        num_steps=NUM_TRAIN_STEPS,
        grad_accum_steps=2,
    )

    elapsed_min = (time.time() - start_time) / 60
    print(f"\n🎉 [OK] S³-GRPO (M8) Training Completed in {elapsed_min:.1f} minutes!")
    print(f"   Final LoRA adapter saved to: {output_dir}")


if __name__ == "__main__":
    main()
