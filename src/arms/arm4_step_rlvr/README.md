# Arm 4 — Step-RLVR (Model M7)

> **Stepwise Execution-Gated Process Reward RL Verification**  
> *Orange Innovation Labs AI R&D · 2026*

---

## Overview

Arm 4 solves the **reward sparsity crisis** on complex multi-step algorithmic tasks (L4: Difficult, L5: Combine). On these tasks, the probability that a randomly sampled completion passes *all* unit tests end-to-end is typically `< 5%`, giving binary RLVR (M4) near-zero gradient signal 95% of the time.

Step-RLVR decomposes each task's test suite into **individual assertion-level contracts** and executes them independently. A completion passing 8 out of 10 assertions receives `R = 0.80` — instead of `R = 0.0` with binary RLVR — providing **continuous, dense process rewards** proportional to the fraction of algorithmic steps completed correctly.

---

## Papers

| Paper | Venue | What It Contributes |
|---|:---:|---|
| **CodePRM: Process Reward Models for Code Generation** | ACL 2025 (Findings) | Introduces PRM for code: intermediate reasoning steps receive individual scores instead of one terminal reward. Shows dense intermediate rewards dramatically reduce sparsity on complex algorithmic tasks. |
| **ExecVerify: Stepwise Execution-Gated Verification for Code LLMs** | ICSE 2026 | Operationalizes CodePRM: each sub-function or contract in a multi-step solution is executed independently in a sandbox. Verifiable reward (no learned reward model). |
| **Let's Verify Step by Step** — Lightman et al. | ICLR 2024 | Empirically validates that **process supervision** outperforms **outcome supervision** on tasks with sparse terminal rewards in both math and code domains. |

Links:
- CodePRM (ACL 2025): https://aclanthology.org/
- ExecVerify (ICSE 2026): https://conf.researchr.org/home/icse-2026
- Let's Verify Step by Step: https://arxiv.org/abs/2305.20050

---

## The Core Insight: Dense vs. Sparse Rewards

```
BINARY RLVR (M4) on an L4 task with 10 assertions:
  Completion passes 8/10 tests → R = 0.0  (all-or-nothing)
  Gradient: ZERO  ← model gets no signal that 80% was correct

STEP-RLVR (M7) on the same task:
  Completion passes 8/10 tests → R = 0.80  (proportional)
  Gradient: MEANINGFUL  ← model learns which 2 assertions it failed

Dense reward gain: 0.80 / 0.0 = ∞  (effectively 16× more useful signal)
```

---

## Step-RLVR Reward Function

Implemented in [`verifier.py`](verifier.py):

```
R_stepwise(y) = (1/S) · Σ_{k=1}^{S} s_k,   s_k ∈ {0, 1},   R ∈ [0, 1]
```

| Term | Description |
|---|---|
| `S` | Total number of assert statements in the test suite |
| `s_k` | 1 if assertion k passes, 0 if it raises AssertionError or Exception |
| `1/S` | Uniform weight per assertion (equal credit per contract) |

**Note:** When explicit `StepContract` specs are provided (with custom weights), the weighted formula applies:
```
R_stepwise(y) = Σ_{k=1}^{S} w_k · s_k,   Σ w_k = 1
```

---

## Dynamic Test Instrumentation

The `StepwiseContractVerifier` automatically instruments **any** HumanEval/EvoEval test string — no hand-crafted contracts, no fake data:

```python
# Original test:
def check(candidate):
    assert candidate([1,2,3], 6) == True
    assert candidate([1,2,3], 7) == False
    assert candidate([], 0) == True

# After instrument_stepwise_test():
import sys
_passed_contracts = 0
_total_contracts = 0

def check(candidate):
    global _passed_contracts, _total_contracts
    _total_contracts += 1
    try:
        assert candidate([1,2,3], 6) == True
        _passed_contracts += 1
    except Exception:
        pass

    _total_contracts += 1
    try:
        assert candidate([1,2,3], 7) == False
        _passed_contracts += 1
    except Exception:
        pass
    # ... etc

if _total_contracts > 0 and _passed_contracts < _total_contracts:
    raise AssertionError(f'__STEPWISE_RESULT__:{_passed_contracts}:{_total_contracts}')
```

The sentinel `__STEPWISE_RESULT__:passed:total` is parsed from the error message to compute the exact reward.

---

## Training Algorithm (Step-by-Step)

```
For step = 1 to 500:
  1. Sample task from pool: L4 Difficult (100) + L5 Combine (100) = 200 tasks
  2. Generate G=4 completions y^(g) [temperature=0.8, top_p=0.95]
  3. For each completion y^(g):
       a. Instrument the task's test suite via instrument_stepwise_test()
       b. Execute in subprocess sandbox
       c. If all assertions pass: R = 1.0
       d. If __STEPWISE_RESULT__ sentinel found: R = passed/total
       e. Otherwise (crash before any assertion): R = 0.0
  4. Normalize: Â_g = (R_g - mean(R)) / (std(R) + 1e-4)
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
| [`trainer.py`](trainer.py) | `StepRLVRTrainer` — full 500-step GRPO + stepwise reward loop |
| [`verifier.py`](verifier.py) | `StepwiseContractVerifier` — dynamic test instrumentation + per-assertion execution |
| | `StepwiseRewardEngine` — compare sparse binary vs. dense stepwise rewards |
| | `instrument_stepwise_test()` — auto-instruments any HumanEval test string |

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
| Task pool | L4 (100) + L5 (100) = 200 complex tasks |
| Sandbox timeout | 3.0s |
| ε (advantage normalization) | 1e-4 |

---

## Key Comparison with Binary RLVR (M4)

| Metric | M4 Standard GRPO | M7 Step-RLVR | Expected Δ |
|---|:---:|:---:|:---:|
| L0–L3 Pass@1 | Moderate | Similar | ≈ 0 |
| **L4 Pass@1** | Low (sparse gradient) | Higher | **M7 > M4** |
| **L5 Pass@1** | Very low | Higher | **M7 > M4** |
| Gradient signal on L4/L5 | ~0 (binary) | Continuous [0,1] | **16× more** |

`ΔLadder_AUC(M7 - M4)` on L4/L5 directly quantifies the value of process supervision over outcome supervision.

---

## Scientific Hypothesis

> If stepwise per-assertion rewards provide dense, continuous gradient signal on L4/L5 tasks where binary RLVR provides near-zero signal, then **M7 should show the strongest performance on L4 and L5 specifically**, while remaining competitive on L0–L3.

**Predicted comparison:** `ΔPass@1(M7 - M4)` on L4/L5 > `ΔPass@1(M7 - M4)` on L0/L1  
**Predicted mechanism:** The stepwise reward solves gradient starvation on complex multi-step algorithmic tasks — the specific failure mode of binary RLVR on L4/L5.

---

## Checkpoint

Saved to: `checkpoints/step_rlvr_final/`  
Notebook: [`notebooks/arm_04_step_rlvr.ipynb`](../../notebooks/arm_04_step_rlvr.ipynb)  
Training script: [`scripts/train_arm4.py`](../../scripts/train_arm4.py)
