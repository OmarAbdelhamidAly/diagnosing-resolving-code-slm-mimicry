"""Packaging utility to prepare benchmark data and checkpoints for Kaggle upload.

Strips heavy training state files (e.g. optimizer states, checkpoint-X subdirs)
and creates lightweight zip archives ready for drag-and-drop Kaggle Dataset creation.
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = REPO_ROOT / "kaggle_upload"


def get_dir_size_mb(path: Path) -> float:
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


def package_benchmarks() -> Path:
    ladder_dir = REPO_ROOT / "data" / "ladder"
    if not ladder_dir.exists():
        print(f"❌ Ladder data directory not found at {ladder_dir}")
        return None

    jsonl_files = list(ladder_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"❌ No .jsonl files found in {ladder_dir}")
        return None

    out_zip = UPLOAD_DIR / "slm_ladder_benchmarks.zip"
    print(f"\n📦 [1/2] Packaging Benchmark Ladder ({len(jsonl_files)} files)...")

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in jsonl_files:
            # Store with relative path data/ladder/<filename> so it unzips nicely
            arcname = f"data/ladder/{f.name}"
            zf.write(f, arcname=arcname)
            print(f"   + {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")

    size_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"✅ Created benchmark package: {out_zip} ({size_mb:.2f} MB)")
    return out_zip


def package_checkpoints() -> Path:
    checkpoints_dir = REPO_ROOT / "checkpoints"
    if not checkpoints_dir.exists():
        print(f"❌ Checkpoints directory not found at {checkpoints_dir}")
        return None

    known_adapters = [
        "qlora_vanilla_adapter",
        "qlora_contrastive_adapter",
        "standard_grpo_final",
        "inv_grpo_final",
        "rlvr_inv_grpo_final",
        "rlvr_ast_final",
        "step_rlvr_final",
    ]

    out_zip = UPLOAD_DIR / "slm_checkpoints.zip"
    print(f"\n📦 [2/2] Packaging Clean Model Checkpoints for Inference...")
    print("   (Stripping heavy optimizer states & checkpoint-X training folders)")

    found_count = 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for adapter_name in known_adapters:
            adapter_path = checkpoints_dir / adapter_name
            if not adapter_path.exists():
                continue

            config_file = adapter_path / "adapter_config.json"
            if not config_file.exists():
                continue

            found_count += 1
            # Copy only top-level files (adapter_model.safetensors, configs, tokenizers)
            adapter_files = [f for f in adapter_path.iterdir() if f.is_file()]
            adapter_mb = sum(f.stat().st_size for f in adapter_files) / (1024 * 1024)
            print(f"   🎯 {adapter_name:<28}: {len(adapter_files)} files ({adapter_mb:.2f} MB)")

            for f in adapter_files:
                arcname = f"checkpoints/{adapter_name}/{f.name}"
                zf.write(f, arcname=arcname)

    if found_count == 0:
        print("⚠️ No valid adapter checkpoints found with adapter_config.json.")
        return None

    size_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"✅ Created clean checkpoints package: {out_zip} ({size_mb:.2f} MB for {found_count} models)")
    return out_zip


def main():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("🚀 REO - Kaggle Evaluation Upload Preparer")
    print("=" * 70)

    bench_zip = package_benchmarks()
    ckpt_zip = package_checkpoints()

    print("\n" + "=" * 70)
    print("🎉 Packaging Complete! Ready for Kaggle:")
    print("=" * 70)
    if bench_zip:
        print(f"1. Benchmarks : {bench_zip.resolve()} ({bench_zip.stat().st_size / (1024*1024):.1f} MB)")
    if ckpt_zip:
        print(f"2. Checkpoints: {ckpt_zip.resolve()} ({ckpt_zip.stat().st_size / (1024*1024):.1f} MB)")

    print("\n📋 Next Steps for Kaggle:")
    print("1. Go to https://www.kaggle.com/datasets/new and upload 'slm_ladder_benchmarks.zip'")
    print("   -> Name the dataset: 'slm-ladder-benchmarks'")
    print("2. Go to https://www.kaggle.com/datasets/new and upload 'slm_checkpoints.zip'")
    print("   -> Name the dataset: 'slm-checkpoints'")
    print("3. In your Kaggle notebook (nb_06_evaluation_suite.ipynb):")
    print("   -> Add both datasets as Input")
    print("   -> Set Accelerator: GPU T4 x 1 (or GPU T4 x 2)")
    print("   -> Turn INTERNET: ON in the notebook sidebar settings")
    print("   -> Run All Cells!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
