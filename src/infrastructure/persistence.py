"""File IO and persistence utilities with atomic writing and UTF-8 enforcement."""

import json
import os
import tempfile
import yaml
from typing import List, Dict, Any


def save_jsonl(records: List[Dict[str, Any]], filepath: str) -> None:
    """Save records to a JSONL file atomically."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    temp_dir = os.path.dirname(filepath)

    with tempfile.NamedTemporaryFile("w", dir=temp_dir, delete=False, encoding="utf-8") as tf:
        for r in records:
            tf.write(json.dumps(r, ensure_ascii=False) + "\n")
        temp_name = tf.name

    # Atomic rename/replace
    if os.path.exists(filepath):
        os.remove(filepath)
    os.rename(temp_name, filepath)


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load records from a JSONL file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JSONL file not found: {filepath}")

    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))
    return records


def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """Save structured data as formatted JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(filepath: str) -> Any:
    """Load JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(filepath: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"YAML config file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
