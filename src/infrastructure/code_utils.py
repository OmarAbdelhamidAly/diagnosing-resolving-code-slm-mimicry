"""Lightweight code-extraction utilities.

This module is intentionally **torch-free** so it can be imported at any
time without triggering CUDA initialisation or heavy framework loading.

``extract_code()`` is the canonical, single source of truth for stripping
markdown fences and model artefacts from completions.  All trainers,
evaluators, and notebooks MUST import from here.
"""

from __future__ import annotations

import re


def extract_code(raw_text: str) -> str:
    """Extract Python code from markdown or raw model output.

    Canonical single source of truth — all trainers/evaluators import this.

    Handles four output formats in priority order:

    1. ``<thought>...</thought>`` reasoning tags  (stripped first)
    2. ````python ... ```` fenced code blocks     (preferred)
    3. ```` ``` ... ```` generic fenced blocks
    4. ``### Solution:`` SFT prompt artefacts     (stripped last)
    """
    # 1. Strip reasoning / thinking tags
    raw_text = re.sub(r"<thought>.*?</thought>", "", raw_text, flags=re.DOTALL).strip()

    # 2. Fenced python block
    m = re.search(r"```python\s*(.*?)\s*```", raw_text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 3. Generic fenced block
    m = re.search(r"```\s*(.*?)\s*```", raw_text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 4. SFT prompt artefact — strip everything up to '### Solution:'
    marker = re.search(r"###\s*Solution\s*:\s*", raw_text, re.IGNORECASE)
    if marker:
        raw_text = raw_text[marker.end():]

    return raw_text.strip()
