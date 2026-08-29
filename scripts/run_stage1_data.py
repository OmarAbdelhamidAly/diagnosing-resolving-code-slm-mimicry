"""Stage 1 CLI Runner — Ingest and verify all Reduction Ladder datasets (L0 to L5).

Usage:
    python scripts/run_stage1_data.py [--verify] [--output_dir data/ladder]
"""

import argparse
import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_pipeline.loader import fetch_all_ladder_levels
from src.data_pipeline.verifier import verify_dataset_tasks


def main():
    parser = argparse.ArgumentParser(description="Download and verify Reduction Ladder datasets (L0-L5).")
    parser.add_argument("--output_dir", type=str, default="data/ladder", help="Directory to save JSONL files.")
    parser.add_argument("--verify", action="store_true", default=True, help="Run sandbox verification on ground truth.")
    parser.add_argument("--skip_verify", action="store_false", dest="verify", help="Skip ground truth sandbox verification.")
    args = parser.parse_args()

    print("=" * 70)
    print("[STAGE 1] Reduction Ladder Benchmark Ingestion")
    print("   Orange Innovation Labs - Research & Advanced AI")
    print(f"   Output Directory: {args.output_dir}")
    print(f"   Sandbox Verification: {'ENABLED' if args.verify else 'DISABLED'}")
    print("=" * 70)

    # 1. Fetch & Normalize Datasets
    all_levels = fetch_all_ladder_levels(output_dir=args.output_dir)

    # 2. Verify Ground Truth
    if args.verify:
        print("\n" + "=" * 70)
        print("[VERIFY] Executing Ground-Truth Sandbox Verification Suite")
        print("=" * 70)
        for level_key, tasks in all_levels.items():
            print(f"\n--- Verifying Level {level_key} ---")
            verify_dataset_tasks(tasks)

    print("\n" + "=" * 70)
    print("[DONE] Stage 1 Data Pipeline Completed Successfully!")
    print(f"[FILES] Benchmark files ready in '{args.output_dir}'")
    print("=" * 70)


if __name__ == "__main__":
    main()
