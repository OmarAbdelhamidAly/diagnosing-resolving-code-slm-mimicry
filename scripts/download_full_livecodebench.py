"""Downloads and formats the Full LiveCodeBench dataset into data/ladder/Ctrl_livecode_full.jsonl.

Fetches 1,055 competitive programming tasks from all LiveCodeBench release versions (v1-v6)
using selective column extraction over HTTP Parquet (bypassing multi-gigabyte private test blobs).
Saves in the exact same unified BenchmarkTask schema used across the Reduction Ladder.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _clean_name(title: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", title.lower()).strip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = "task_" + cleaned
    return cleaned


def _extract_entry_point(row: Dict[str, Any], meta: Dict[str, Any]) -> str:
    # 1. From metadata if available
    fn = meta.get("func_name")
    if fn and isinstance(fn, str) and fn.strip():
        return fn.strip()

    # 2. From starter code
    starter = (row.get("starter_code") or "").strip()
    if starter:
        m = re.search(r"def\s+([a-zA-Z0-9_]+)\s*\(", starter)
        if m:
            return m.group(1)

    # 3. From question title
    q_title = row.get("question_title") or ""
    return _clean_name(q_title)


def _format_test_cases(public_tcs_raw: Any) -> str:
    if isinstance(public_tcs_raw, str):
        try:
            tcs = json.loads(public_tcs_raw)
        except Exception:
            tcs = []
    elif isinstance(public_tcs_raw, list):
        tcs = public_tcs_raw
    else:
        tcs = []

    lines = ["def check(candidate):"]
    for idx, tc in enumerate(tcs, 1):
        if not isinstance(tc, dict):
            continue
        inp = tc.get("input", "")
        out = tc.get("output", "")
        lines.append(f"    # test {idx}")
        lines.append(f"    # input : {repr(inp)}")
        lines.append(f"    # expect: {repr(out)}")

    if len(lines) == 1:
        # Fallback dummy assertion if no public test cases
        lines.append("    pass")

    return "\n".join(lines)


def download_and_format_full_lcb(out_path: Optional[str] = None):
    print("=" * 70)
    print("🚀 LiveCodeBench Full Dataset Downloader (Reduction Ladder Format)")
    print("=" * 70)

    try:
        import fsspec
        import pyarrow.parquet as pq
    except ImportError:
        print("Installing required dependencies (pyarrow, fsspec)...")
        os.system("pip install pyarrow fsspec")
        import fsspec
        import pyarrow.parquet as pq

    base_url = "https://huggingface.co/datasets/marianna13/livecodebench_code_generation_lite_parquet/resolve/main/release_latest/"
    part_files = [f"test-{i:05d}-of-00009.parquet" for i in range(9)]

    if out_path is None:
        repo_root = Path(__file__).resolve().parents[1]
        out_file = repo_root / "data" / "ladder" / "Ctrl_livecode_full.jsonl"
    else:
        out_file = Path(out_path)

    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to Hugging Face CDN (9 Parquet chunks)...")
    print(f"Target destination: {out_file}")

    fs = fsspec.filesystem("http")
    saved_records = []
    seen_ids = set()

    t_start = time.time()

    for idx, part_name in enumerate(part_files, 1):
        part_url = base_url + part_name
        print(f"[{idx}/9] Fetching {part_name}...", end=" ", flush=True)
        t_chunk = time.time()

        with fs.open(part_url, "rb") as f:
            pf = pq.ParquetFile(f)
            # Exclude massive private test cases blob to download in seconds
            cols = [c for c in pf.schema.names if c != "private_test_cases"]
            table = pf.read(columns=cols)
            df = table.to_pandas()

        added_in_chunk = 0
        for _, row in df.iterrows():
            q_id = str(row.get("question_id", "")).strip()
            if not q_id or q_id in seen_ids:
                continue
            seen_ids.add(q_id)

            meta_raw = row.get("metadata")
            if isinstance(meta_raw, str):
                try:
                    meta = json.loads(meta_raw)
                except Exception:
                    meta = {}
            elif isinstance(meta_raw, dict):
                meta = meta_raw
            else:
                meta = {}

            q_title = str(row.get("question_title", "")).strip()
            q_content = str(row.get("question_content", "")).strip()
            starter = str(row.get("starter_code", "") or "").strip()
            difficulty = str(row.get("difficulty", "medium")).lower()
            platform = str(row.get("platform", "leetcode")).lower()
            contest_date = str(row.get("contest_date", "")).strip()

            entry_point = _extract_entry_point(row, meta)

            if starter:
                prompt = starter
            else:
                prompt = f'def {entry_point}(...):\n    """\n    {q_content}\n    """\n    pass'

            test_code = _format_test_cases(row.get("public_test_cases"))

            record = {
                "task_id": f"LCB_{q_id}",
                "ladder_level": "Ctrl_Full",
                "benchmark": "LiveCodeBench_Full",
                "prompt": prompt,
                "canonical_solution": "",
                "test": test_code,
                "entry_point": entry_point,
                "_title": q_title,
                "_difficulty": difficulty,
                "_contest_date": contest_date,
                "_platform": platform,
            }
            saved_records.append(record)
            added_in_chunk += 1

        print(f"Done ({added_in_chunk} tasks in {time.time() - t_chunk:.2f}s)")

    print("-" * 70)
    print(f"Writing {len(saved_records)} tasks to {out_file}...")
    with open(out_file, "w", encoding="utf-8") as f:
        for rec in saved_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    elapsed = time.time() - t_start
    size_mb = out_file.stat().st_size / (1024 * 1024)

    # Statistics
    diff_counts = Counter(r["_difficulty"] for r in saved_records)
    plat_counts = Counter(r["_platform"] for r in saved_records)

    print("=" * 70)
    print("✅ Full LiveCodeBench Dataset Ready!")
    print(f"Total Tasks Saved : {len(saved_records)}")
    print(f"Output File       : {out_file}")
    print(f"File Size         : {size_mb:.2f} MB")
    print(f"Time Taken        : {elapsed:.2f} seconds")
    print(f"Difficulty Breakdown: {dict(diff_counts)}")
    print(f"Platform Breakdown  : {dict(plat_counts)}")
    print("=" * 70)


if __name__ == "__main__":
    download_and_format_full_lcb()
