# Arm 3 — AST-RL (Model M5)

> **Abstract Syntax Tree Guided Policy Optimization**  
> *Orange Innovation Labs AI R&D · 2026*

---

## Overview

Arm 3 addresses a fundamental limitation of binary RLVR: when code fails execution, the reward is `0` — even if the model produced the *correct algorithm* but made a trivial type error or off-by-one mistake. This **reward sparsity** blocks gradient flow and slows learning.

AST-RL augments the binary execution reward with a **continuous structural similarity bonus** computed by comparing the normalized Abstract Syntax Tree (AST) of the generated code against the canonical reference solution's AST. This provides dense intermediate signal for near-correct solutions.

---

## Papers

| Paper | Venue | What It Contributes |
|---|:---:|---|
| **TreeDiff: Structural Code Comparison via AST Differencing** | ASE 2025 | Tree-edit-distance metric on normalized ASTs to measure structural similarity, isolating control-flow equivalence from surface naming. Introduces the normalization convention: replace all variable names with `_v`, all args with `_a`. |
| **VeriSeek: Structure-Guided Code Synthesis Verification** | ICSE 2025 | Uses AST-based structural similarity as a verification signal in code synthesis pipelines. Shows that structure-rewarded models generalize better across problem reformulations. |

Links:
- TreeDiff (ASE 2025): https://dl.acm.org/doi/proceedings/10.1145/3691620  
- VeriSeek (ICSE 2025): https://conf.researchr.org/home/icse-2025

---

## The Core Insight: Structure over Surface

Two implementations of the same algorithm should be judged as structurally identical even if they use different variable names:

```python
# Implementation A:
def two_sum(nums, target):
    seen = {}
    for i, n in enumerate(nums):
        if target - n in seen: return [seen[target - n], i]
        seen[n] = i

# Implementation B (different names, same algorithm):
def find_pair(arr, goal):
    table = {}
    for idx, val in enumerate(arr):
        if goal - val in table: return [table[goal - val], idx]
        table[val] = idx
```

After AST normalization, both produce identical node-type sequences. The `simAST` score between them is `1.0`.

A model that correctly identifies the hash-map approach but uses wrong variable names gets `R_exec = 0` (execution fails due to other error) but `R_AST ≈ 0.9` — meaningful gradient signal.

---

## AST Normalization

Implemented in [`ast_engine.py`](ast_engine.py) via `ASTNormalizer(ast.NodeTransformer)`:

```python
# All Name nodes (variables) → "_v"
def visit_Name(self, node): return ast.Name(id="_v", ...)

# All arg nodes (function parameters) → "_a"  
def visit_arg(self, node): return ast.arg(arg="_a", ...)
```

Then the normalized AST is walked depth-first to produce a **node-type sequence**:
```
["Module", "FunctionDef", "arguments", "arg", "For", "Assign", "If", "Return", ...]
```

String literals and docstrings are excluded — only structural node types count.

---

## simAST Similarity Metric

Implemented in [`ast_engine.py`](ast_engine.py):

```
simAST(y, y*) = 0.6 · Jaccard(σ(y), σ(y*)) + 0.4 · LenRatio(σ(y), σ(y*))
```

Where `σ(c)` = normalized AST node-type sequence of code `c`:

| Component | Formula | Weight | Measures |
|---|---|:---:|---|
| **Jaccard similarity** | `|σ(y) ∩ σ(y*)| / |σ(y) ∪ σ(y*)|` | **60%** | Which code constructs are used (loops, conditionals, returns) |
| **Length ratio** | `min(|σ|, |σ*|) / max(|σ|, |σ*|, 1)` | **40%** | Completeness — penalizes truncated or stub solutions |

`simAST ∈ [0.0, 1.0]`

---

## Composite Reward Function

Implemented in [`reward_engine.py`](reward_engine.py):

```
R_total(y, y*) = R_exec(y) + β · simAST(y, y*)
```

| Term | Value | Description |
|---|---|---|
| `R_exec(y)` | ∈ {0, 1} | Binary sandbox pass/fail |
| `simAST(y, y*)` | ∈ [0.0, 1.0] | Structural AST similarity to canonical reference |
| `β` | **0.3** | AST bonus weight |

**Example gradient signals:**

| Scenario | R_exec | simAST | R_total |
|---|:---:|:---:|:---:|
| Correct solution | 1.0 | 0.95 | **1.285** |
| Right algorithm, trivial type error | 0.0 | 0.88 | **0.264** ← dense signal! |
| Wrong algorithm | 0.0 | 0.30 | 0.090 |
| Empty/syntax error | 0.0 | 0.0 | 0.000 |

---

## Training Algorithm (Step-by-Step)

```
For step = 1 to 500:
  1. Sample task from pool: L0 (164 tasks) + L1 (100 tasks) = 264 total
  2. Generate G=4 completions y^(g) [temperature=0.8, top_p=0.95]
  3. For each completion y^(g):
       a. Execute in subprocess sandbox → R_exec ∈ {0, 1}
       b. Compute simAST(y^(g), y*_canonical) ∈ [0.0, 1.0]
       c. R_total^(g) = R_exec + 0.3 · simAST
  4. Normalize: Â_g = (R_g - mean(R)) / (std(R) + 1e-8)
  5. Micro-batched backward (batch=1 per sample):
       loss_i = -(log_π_θ(y^(i)|x) · Â_i) / (G · grad_accum_steps)
  6. Accumulate gradients for 2 steps, then:
       clip_grad_norm_(1.0) → AdamW.step() → zero_grad()
  7. Checkpoint every 100 steps
```

---

## Implementation Files

| File | Purpose |
|---|---|
| [`trainer.py`](trainer.py) | `ASTRLTrainer` — full 500-step GRPO + AST reward loop |
| [`ast_engine.py`](ast_engine.py) | `ASTNormalizer`, `get_ast_signature()`, `simAST()`, `ast_reward()` |
| [`reward_engine.py`](reward_engine.py) | `ASTRewardEngine` — combines exec + AST + group normalization |

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| Quantization | 4-bit NF4, double quant |
| LoRA rank r | 16 |
| LoRA α | 32 |
| β_ast (AST bonus weight) | **0.3** |
| α_tree (exponential decay) | 0.05 |
| Learning rate | 1e-5 |
| Group size G | 4 |
| Gradient accumulation | 2 steps |
| Training steps | 500 |
| Task pool | L0 (164) + L1 (100) = 264 tasks |
| Sandbox timeout | 3.0s |

---

## Scientific Hypothesis

> If AST-based structural similarity provides dense gradient signal for near-correct solutions that binary RLVR leaves with zero reward, then **M5 should outperform M4 (Standard GRPO)** by learning to produce structurally sound code even under transformation.

**Predicted comparison:** `ΔLadder_AUC(M5 - M4) > 0`  
**Predicted mechanism:** The `β·simAST` term creates non-zero gradients for completions that fail execution but have correct control flow — reducing the effective reward sparsity on L2/L3 tasks.

---

## Checkpoint

Saved to: `checkpoints/rlvr_ast_final/`  
Notebook: [`notebooks/arm_03_ast_rl.ipynb`](../../notebooks/arm_03_ast_rl.ipynb)
