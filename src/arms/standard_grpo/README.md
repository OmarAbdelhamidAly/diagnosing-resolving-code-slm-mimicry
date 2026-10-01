# Arm 1b — Standard GRPO (Model M4)

> **RLVR Ablation Baseline — Outcome-Only Group Relative Policy Optimization**  
> *Orange Innovation Labs AI R&D · 2026*

---

## Overview

Arm 1b implements **standard GRPO without any additional structural or invariance constraints**. Its role in the experimental design is not to be the best model — its role is to serve as the **clean ablation baseline** that isolates the exact contribution of each additional component added in Arms 1a, 3, and 4.

**The key comparisons this model enables:**

| Comparison | What It Proves |
|---|---|
| `M4 vs M1 (Baseline)` | That online RL with execution feedback alone improves generalization |
| `M5 - M4` | The exact marginal gain from adding AST structural reward |
| `M6 - M4` | The exact marginal gain from invariance regularization |
| `M7 - M4` | The exact marginal gain from stepwise process rewards |

---

## Papers

| Paper | Venue | What It Contributes |
|---|:---:|---|
| **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL** — DeepSeek-AI | arXiv 2501.12948, 2025 | Introduces GRPO — group-relative advantage normalization that eliminates the value network. Demonstrates state-of-the-art reasoning gains without a critic model. |
| **RLVR: Reinforcement Learning with Verifiable Rewards** — Lightman et al. | NeurIPS 2024 | Grounds reward signals in objective execution results (pass/fail) rather than a learned reward model, eliminating reward hacking. |

Links:
- DeepSeek-R1 / GRPO: https://arxiv.org/abs/2501.12948
- Let's Verify Step by Step: https://arxiv.org/abs/2305.20050

---

## Reward Function

Standard binary execution reward — no additions:

```
R_exec(y) = 1.0  if all unit tests pass
R_exec(y) = 0.0  otherwise
```

Group advantage normalization:

```
Â_g = (R_g - mean(R)) / (std(R) + ε),   ε = 1e-8,   G = 4
```

Policy gradient loss:

```
L_GRPO = -(1/G) · Σ_g Â_g · log π_θ(y^(g) | x)
```

---

## Training Algorithm (Step-by-Step)

```
For step = 1 to 500:
  1. Sample task x from task pool (L0 + L1 = 264 tasks)
  2. Generate G=4 completions y^(g) [temperature=0.8, top_p=0.95]
  3. Execute each y^(g) in subprocess sandbox → R^(g) ∈ {0, 1}
  4. Normalize: Â_g = (R_g - mean(R)) / (std(R) + 1e-8)
  5. Micro-batched backward (batch=1 per sample):
       loss_i = -(log_π_θ(y^(i)|x) · Â_i) / (G · grad_accum_steps)
  6. Accumulate gradients for 2 steps → AdamW.step() → zero_grad()
  7. Checkpoint every 100 steps
```

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| Quantization | 4-bit NF4, double quant |
| LoRA rank r | 16 |
| LoRA α | 32 |
| Learning rate | 1e-5 |
| Group size G | 4 |
| Gradient accumulation | 2 steps |
| Training steps | 500 |
| Sandbox timeout | 3.0s |

---

## Checkpoint

Saved to: `checkpoints/standard_grpo_final/`  
Notebook: [`notebooks/arm_01b_standard_grpo.ipynb`](../../notebooks/arm_01b_standard_grpo.ipynb)
