"""Dedicated training script for Arm 4: Step-RLVR (Model M7).

Literature Basis:
- CodePRM: Process Reward Models for Code Reasoning (ACL 2025)
- ExecVerify: Stepwise Execution-Gated Verification (ICSE 2026)

Runs 500 optimization steps using 4-bit NF4 QLoRA and intermediate stepwise assertion scoring.
Checkpoints are saved periodically and at completion to: checkpoints/step_rlvr_final
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

from src.arms.arm4_step_rlvr.trainer import StepRLVRTrainer
from src.evaluation.registry import BenchmarkRegistry


def main():
    data_dir = REPO_ROOT / "data" / "ladder"
    print(f"[Arm 4] Loading benchmark tasks from: {data_dir}")
    registry = BenchmarkRegistry(data_dir=data_dir)

    # Multi-goal complex tasks from L4 (Difficult) and L5 (Combine)
    tasks = [
        {
            "prompt": t.prompt,
            "test": t.test,
            "entry_point": t.entry_point,
        }
        for t in registry.get_tasks("L4") + registry.get_tasks("L5")
    ]
    print(f"[Arm 4] Loaded {len(tasks)} multi-goal tasks for Step-RLVR contract verification.")

    output_dir = "checkpoints/step_rlvr_final"
    trainer = StepRLVRTrainer(
        output_dir=output_dir,
        group_size=4,
        learning_rate=1e-5,
    )

    NUM_TRAIN_STEPS = 500
    print(f"[Arm 4] Starting 500-step training loop on GPU...")
    start_time = time.time()

    history = trainer.train(
        tasks=tasks,
        num_steps=NUM_TRAIN_STEPS,
        grad_accum_steps=2,
    )

    elapsed_min = (time.time() - start_time) / 60
    print(f"\n🎉 [OK] Step-RLVR (M7) Training Completed in {elapsed_min:.1f} minutes!")
    print(f"   Final LoRA adapter saved to: {output_dir}")


if __name__ == "__main__":
    main()
