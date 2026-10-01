# Arm 1a — Invariant GRPO (Model M6)

> **Primary Contribution** — Invariance-Regularized Group Relative Policy Optimization  
> *Orange Innovation Labs AI R&D · 2026*

---

## Overview

Arm 1a is the **primary research contribution** of this project. It trains a code SLM to produce solutions that are invariant across semantically equivalent problem framings — forcing the model to reason about the underlying algorithm rather than pattern-match on surface-level tokens.

The model simultaneously generates solutions for a **canonical prompt** (x ∈ L0) and a **semantically equivalent perturbed prompt** (x' ∈ L2/L3), then receives a reward that explicitly requires correctness on **both**.

---

## Papers

| Paper | Venue | What It Contributes |
|---|:---:|---|
| **Invariant Risk Minimization (IRM)** — Arjovsky et al. | arXiv 2019 | The invariance principle: learning features that work across environments, not just within training distribution. Formalizes the IRM penalty. |
| **DeepSeek-R1: Incentivizing Reasoning via RL** — DeepSeek-AI | arXiv 2501.12948, 2025 | Introduces GRPO — group-relative advantage normalization that eliminates the value network. The base RL algorithm we build on. |
| **Let's Verify Step by Step** — Lightman et al. | ICLR 2024 | Validates verifiable reward signals (execution-grounded) as superior to learned reward models for code reasoning. |

Links:
- IRM: https://arxiv.org/abs/1907.02893  
- DeepSeek-R1 / GRPO: https://arxiv.org/abs/2501.12948  
- Let's Verify: https://arxiv.org/abs/2305.20050

---

## The Core Insight: Why Invariance?

Standard GRPO sees one prompt at a time. The model can maximize reward by learning **spurious correlations** — e.g., "if the function is named `two_sum`, output a hash-map solution" — without ever understanding the algorithm.

Inv-GRPO **breaks this shortcut** by simultaneously testing the policy on a semantically equivalent but surface-different prompt. A policy that pattern-matched on the function name will succeed on x but fail on x', getting zero consistency bonus and a template penalty.

```
Standard GRPO:     x ─────────────► R_exec(y)
                   one prompt, one reward

Inv-GRPO:          x ─────────────► R_exec(y)   ─┐
                                                   ├─► R_total = R_exec + R_exec' + λ·R_cons - γ·P_tmpl
                   x' ────────────► R_exec(y')  ─┘
                   paired prompt, invariance reward
```

---

## Reward Function

Implemented in [`src/arms/arm1_inv_grpo/reward_engine.py`](reward_engine.py):

```
R_total(y, y') = R_exec(y) + R_exec(y') + λ · R_consistency(y, y') - γ · P_template(y')
```

| Term | Value | Description |
|---|---|---|
| `R_exec(y)` | ∈ {0, 1} | Binary sandbox pass/fail on canonical prompt x |
| `R_exec(y')` | ∈ {0, 1} | Binary sandbox pass/fail on perturbed prompt x' |
| `R_consistency(y, y')` | 1.0 if both pass, else 0 | Explicit cross-view invariance reward |
| `P_template(y')` | 1.0 if y' mimics L0 decoy verbatim and fails | Template mimicry penalty |
| `λ` | 0.5 | Consistency bonus weight |
| `γ` | 0.5 | Template penalty weight |

**Maximum possible reward:** `1 + 1 + 0.5·1 - 0 = 2.5` (both pass, consistent, no mimicry)  
**Minimum possible reward:** `0 + 0 + 0 - 0.5·1 = -0.5` (both fail, template penalty applied)

### Group Advantage Normalization (GRPO)

```
Â_i = (R_i - mean(R)) / (std(R) + ε),   ε = 1e-8
```

Group size G = 4. Advantages are computed across all 4 paired rollouts per step.

---

## Training Algorithm (Step-by-Step)

```
For step = 1 to 500:
  1. Sample task (x, x') from PairedTask pool (L0 + L2 pairs)
  2. Generate G=4 completions y^(g) for x  [temperature=0.8, top_p=0.95]
  3. Generate G=4 completions y'^(g) for x' [same settings]
  4. For each pair (y^(g), y'^(g)):
       a. Execute y^(g) in subprocess sandbox → R_exec(y)
       b. Execute y'^(g) in subprocess sandbox → R_exec(y')
       c. Compute R_consistency, P_template
       d. R_total^(g) = R_exec + R_exec' + λ·R_cons - γ·P_tmpl
  5. Normalize: Â_g = (R_g - mean(R)) / (std(R) + 1e-8)
  6. Micro-batched backward (batch=1 per sample):
       loss_i = -(log_π_θ(y^(i)|x) · Â_i) / (G · grad_accum_steps)
       loss_i += -(log_π_θ(y'^(i)|x') · Â_i) / (G · grad_accum_steps)
  7. Accumulate gradients for 2 steps, then:
       clip_grad_norm_(1.0) → AdamW.step() → zero_grad()
  8. Checkpoint every 100 steps
```

---

## Implementation Files

| File | Purpose |
|---|---|
| [`trainer.py`](trainer.py) | `InvGRPOTrainer` — full 500-step training loop |
| [`reward_engine.py`](reward_engine.py) | `InvGRPORewardEngine` — paired reward computation + group normalization |
| [`dataset.py`](dataset.py) | `PairedTask` dataclass — (x, x', test, decoy_code) |

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| Quantization | 4-bit NF4, double quant |
| LoRA rank r | 16 |
| LoRA α | 32 |
| LoRA dropout | 0.05 |
| Learning rate | 1e-5 |
| Weight decay | 0.01 |
| Group size G | 4 |
| Gradient accumulation | 2 steps |
| λ (consistency) | 0.5 |
| γ (template penalty) | 0.5 |
| Training steps | 500 |
| Sandbox timeout | 3.0s |
| Max new tokens | 256 |
| Temperature | 0.8 |

---

## Scientific Hypothesis

> If Inv-GRPO's cross-view invariance regularization forces the model to learn algorithm-level features (control flow, data structure choice) rather than surface lexical cues, then **M6 should outperform M4 (Standard GRPO)** on the Reduction Ladder, with a measurably higher Ladder AUC and later Collapse Point.

**Predicted comparison:**  `ΔLadder_AUC(M6 - M4) > 0`  
**Predicted mechanism:** M6's consistency reward creates gradient signal that *penalizes* solutions that only work on the familiar L0 surface framing.

---

## Checkpoint

Saved to: `checkpoints/inv_grpo_final/`  
Notebook: [`notebooks/arm_01_inv_grpo.ipynb`](../../notebooks/arm_01_inv_grpo.ipynb)
