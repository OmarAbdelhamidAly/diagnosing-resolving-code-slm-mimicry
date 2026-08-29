"""Stage 1 CLI Runner — Ingest and verify all Reduction Ladder datasets (L0 to L5).

Usage:
    python scripts/run_stage1_data.py [--verify] [--output_dir data/ladder]
"""

import argparse
import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.data_service import DataService
from src.infrastructure.hf_loader import HuggingFaceBenchmarkLoader
from src.infrastructure.sandbox import MultiprocessSandbox


def main():
    parser = argparse.ArgumentParser(description="Download and verify Reduction Ladder datasets (L0-L5).")
    parser.add_argument("--output_dir", type=str, default="data/ladder", help="Directory to save JSONL files.")
    parser.add_argument("--verify", action="store_true", default=True, help="Run sandbox verification on ground truth.")
    parser.add_argument("--skip_verify", action="store_false", dest="verify", help="Skip ground truth sandbox verification.")
    parser.add_argument("--force_download", action="store_true", default=False, help="Force fresh download ignoring disk cache.")
    args = parser.parse_args()

    print("=" * 70)
    print("[STAGE 1] Reduction Ladder Benchmark Ingestion (Clean Architecture)")
    print("   Orange Innovation Labs - Research & Advanced AI")
    print(f"   Output Directory: {args.output_dir}")
    print(f"   Sandbox Verification: {'ENABLED' if args.verify else 'DISABLED'}")
    print("=" * 70)

    loader = HuggingFaceBenchmarkLoader(cache_dir=args.output_dir)
    executor = MultiprocessSandbox()
    service = DataService(loader=loader, executor=executor)

    # 1. Fetch & Normalize Datasets into Domain Entities
    all_levels = service.prepare_all_benchmarks(force_download=args.force_download)

    # 2. Verify Ground Truth
    if args.verify:
        print("\n" + "=" * 70)
        print("[VERIFY] Executing Ground-Truth Sandbox Verification Suite")
        print("=" * 70)
        for level_key, tasks in all_levels.items():
            print(f"\n--- Verifying Level {level_key} ---")
            service.verify_ground_truth(tasks)

    print("\n" + "=" * 70)
    print("[DONE] Stage 1 Data Pipeline Completed Successfully!")
    print(f"[FILES] Benchmark files ready in '{args.output_dir}'")
    print("=" * 70)


if __name__ == "__main__":
    main()
