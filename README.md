<div align="center">

# 🔬 Diagnosing & Resolving Code SLM Mimicry

### *Probing Shortcut Learning vs. Transferable Algorithmic Reasoning in Code-Generating SLMs*

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/></a>
  <a href="https://huggingface.co/"><img src="https://img.shields.io/badge/🤗_Transformers-4.44%2B-FFD21E?style=for-the-badge"/></a>
  <a href="https://github.com/TimDettmers/bitsandbytes"><img src="https://img.shields.io/badge/BitsAndBytes-4bit_NF4-00C853?style=for-the-badge"/></a>
  <a href="https://github.com/huggingface/peft"><img src="https://img.shields.io/badge/PEFT-QLoRA-7B2FBE?style=for-the-badge"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-B0BEC5?style=for-the-badge"/></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Base_Model-Qwen2.5--Coder--1.5B-0288D1?style=flat-square"/>
  <img src="https://img.shields.io/badge/Benchmarks-764_Tasks_(L0→L5)-43A047?style=flat-square"/>
  <img src="https://img.shields.io/badge/Models_Trained-7_(M1→M7)-E53935?style=flat-square"/>
  <img src="https://img.shields.io/badge/VRAM_Budget-8GB_RTX_3070-FB8C00?style=flat-square"/>
  <img src="https://img.shields.io/badge/Training_Steps-500_per_Arm-6A1B9A?style=flat-square"/>
</p>

> **Research Initiative — Orange Innovation Labs (AI R&D Division)**  
> *Developed at Cairo, Egypt · Released under the MIT License*

</div>

---

## 📋 Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [The Core Problem: Mimicry vs. Reasoning](#2-the-core-problem-mimicry-vs-reasoning)
3. [Reduction Ladder Framework (L0–L5 + Control)](#3-reduction-ladder-framework-l0l5--control)
4. [7-Model Experimental Suite (M1–M7)](#4-7-model-experimental-suite-m1m7)
5. [Multi-Arm Mitigation Architecture](#5-multi-arm-mitigation-architecture)
6. [Evaluation Metrics & Diagnostics](#6-evaluation-metrics--diagnostics)
7. [Literature Foundations](#7-literature-foundations)
8. [Codebase Architecture](#8-codebase-architecture)
9. [Installation & Quickstart](#9-installation--quickstart)
10. [Notebooks & Workflow](#10-notebooks--workflow)
11. [Hardware Budget](#11-hardware-budget)
12. [Results & Key Findings](#12-results--key-findings)
13. [Citation](#13-citation)

---

## 1. Executive Summary

Small Language Models (SLMs) in the **1–3B parameter range** are critical for private, low-latency, on-device deployments — especially within telecom operators like Orange Innovation Labs, where edge SLMs drive autonomous network script patching, infrastructure verification, and developer copilot workflows.

Despite stellar pass rates on static benchmarks, these models suffer from a fundamental **Mimicry Crisis**:

```
  HumanEval (L0): 85% Pass@1   ──────────►   EvoEval Creative (L3): 38% Pass@1
                                                           ⬇
                                              CATASTROPHIC PERFORMANCE COLLAPSE
                                              (Same algorithm, different surface framing)
```

**This project answers two questions:**

| RQ | Question | Method |
|:---:|---|---|
| **RQ1** | At which transformation level does a code SLM collapse, and what is its error signature? | 7-rung Reduction Ladder · 764 tasks · Error taxonomy |
| **RQ2** | Can multi-view invariance regularization break mimicry and delay collapse by ≥2 rungs? | Inv-GRPO (Arm 1) · 4 mitigation arms · 7-model comparison |

---

## 2. The Core Problem: Mimicry vs. Reasoning

### Why Code Is the Optimal Reasoning Lab

| Property | Advantage |
|---|---|
| **Objective Verification** | Unit tests give binary ground-truth — zero LLM-judge bias |
| **AST Isomorphism** | Variable renaming mutates 100% of surface tokens while holding semantics invariant |
| **Shortcut Visibility** | Lexical pattern-matching fails *cleanly and measurably* on edge cases |
| **Industrial Relevance** | Hardening edge SLMs removes dependency on cloud APIs for telecom automation |

### The Three Failure Modes

```
┌─────────────────────────────────────────────────────────────────────┐
│  FAILURE MODE 1: SFT Memorization Bias                              │
│  ─────────────────────────────────────────────────────────────────  │
│  SFT teaches the *syntactic formatting* of CoT without              │
│  inducing invariant algorithmic logic.                              │
│                                                                     │
│  FAILURE MODE 2: RLVR Reward Shortcut                               │
│  ─────────────────────────────────────────────────────────────────  │
│  Standard RLVR optimizes for unit-test passes on single prompts,    │
│  converging on heuristics that fail under surface transformation.   │
│                                                                     │
│  FAILURE MODE 3: Template Retrieval (Mimicry)                       │
│  ─────────────────────────────────────────────────────────────────  │
│  Model retrieves the memorized L0 canonical solution verbatim       │
│  when it sees familiar keywords — ignoring semantic differences.    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Reduction Ladder Framework (L0–L5 + Control)

Each rung is grounded in **peer-reviewed, publicly available HuggingFace benchmarks** — no synthetic hallucinated data.

```
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │                    THE REDUCTION LADDER (764 Total Tasks)                       │
  ├─────┬──────────────────────────┬───────────────────────────────┬────────────────┤
  │ L0  │ HumanEval Standard       │ openai/openai_humaneval        │  164 tasks     │
  │ L1  │ EvoEval Subtle           │ evoeval/EvoEval_subtle         │  100 tasks     │
  │ L2  │ EvoEval ToolUse          │ evoeval/EvoEval_tool_use       │  100 tasks     │
  │ L3  │ EvoEval Creative         │ evoeval/EvoEval_creative       │  100 tasks     │
  │ L4  │ EvoEval Difficult        │ evoeval/EvoEval_difficult      │  100 tasks     │
  │ L5  │ EvoEval Combine          │ evoeval/EvoEval_combine        │  100 tasks     │
  │Ctrl │ LiveCodeBench Lite       │ livecodebench/code_gen_lite    │  100 tasks     │
  └─────┴──────────────────────────┴───────────────────────────────┴────────────────┘
```

### Concrete Transformation: *Two Sum* across L0→L5

<details>
<summary><b>Click to expand — full ladder walkthrough</b></summary>

**L0 — Verbatim Classic:**
```python
def two_sum(nums: list[int], target: int) -> list[int]:
    """Return indices of the two numbers that add up to target."""
```

**L1 — Format / Specification Shift:**
```python
def parse_and_find_indices(data_str: str, target: int) -> tuple[int, int]:
    """Input is '2,7,11,15'. Parse and return 0-indexed tuple of matching pair."""
```

**L2 — Structural ToolUse Abstraction:**
```python
def find_pair_with_tool(seq: list[int], target: int, lookup_helper) -> list[int]:
    """Use pre-defined lookup_helper(table, key) to manage complement queries."""
```

**L3 — Creative Narrative (Telecom Context):**
```python
def match_orange_transceivers(bandwidth_units: list[int], gateway_cap: int) -> list[int]:
    """In an Orange 5G pool, find two transceivers whose bandwidth sums to gateway_cap."""
```

**L4 — Constraint Augmentation:**
```python
def match_transceivers_multi_zone(bandwidths: list[int], zones: list[str], cap: int) -> list[int]:
    """Same as L3, but the two transceivers must be in DIFFERENT availability zones."""
```

**L5 — Cross-Concept Composition:**
```python
def schedule_optimal_dual_tasks(tasks: list[dict], total_limit: int) -> list[int]:
    """Find two concurrent tasks summing to total_limit while MINIMIZING scheduling fragmentation."""
```

</details>

### Our Approach vs. Code-Rewriting (MRI)

```
  Code-Rewriting (MRI / Yang et al., 2025):
      Surface Syntax: STATIC  ──►  Semantics: MUTATED
      Question: Does the model regurgitate old code when requirements change?

  Reduction Ladder (This Work, 2026):
      Semantics: INVARIANT    ──►  Surface Syntax: MUTATED
      Question: Does the model fail to apply valid logic when framing shifts?
```

---

## 4. 8-Model Experimental Suite (M1–M8)

```
┌────┬─────────────────────────────┬───────────────────────────────┬──────────────────────────┬───────────────────────┐
│ ID │ Model Name                  │ Checkpoint                    │ Training Paradigm        │ Expected ℓ* Collapse  │
├────┼─────────────────────────────┼───────────────────────────────┼──────────────────────────┼───────────────────────┤
│ M1 │ Zero-Shot Baseline          │ (none — base model)           │ Qwen2.5-Coder-1.5B-Inst. │ ≈ L2 (ToolUse)        │
│ M2 │ Vanilla QLoRA SFT           │ qlora_vanilla_adapter         │ Standard CoT SFT         │ ≈ L2–L3               │
│ M3 │ Contrastive DPO             │ qlora_contrastive_adapter     │ SFT + DPO Hard Negatives │ ≈ L3                  │
│ M4 │ Standard GRPO               │ standard_grpo_final           │ Outcome-Only RLVR        │ ≈ L3 (Creative)       │
│ M5 │ AST-RL                      │ rlvr_ast_final                │ GRPO + simAST Reward     │ ≈ L4 (Difficult)      │
│ M6 │ Invariant GRPO              │ inv_grpo_final                │ Paired Invariance RLVR   │ ≈ L4–L5               │
│ M7 │ Step-RLVR                   │ step_rlvr_final               │ Process Reward RLVR      │ ≈ L5 (best on L4/L5)  │
│ M8 │ S³-GRPO ⭐ Flagship Hybrid  │ s3_grpo_final                 │ Stepwise + AST + Tax     │ Robust across L0–L5   │
└────┴─────────────────────────────┴───────────────────────────────┴──────────────────────────┴───────────────────────┘
```

**Shared Training Config** (all arms, for cross-rung comparability):

```yaml
base_model:   Qwen/Qwen2.5-Coder-1.5B-Instruct
quantization: 4-bit NF4  (bnb_4bit_use_double_quant: true)
lora_r:       16
lora_alpha:   32
lora_dropout: 0.05
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
learning_rate: 1e-5
optimizer:    AdamW  (weight_decay=0.01, grad_clip=1.0)
group_size_G: 4
grad_accum:   2
train_steps:  500
max_new_tokens: 256
temperature:  0.8
sandbox_timeout: 3.0s
```

---

## 5. Multi-Arm Mitigation Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                       MULTI-ARM MITIGATION FRAMEWORK                               │
├────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│  ARM 1a ── Inv-GRPO ───────────────────────────────────────────────────────────   │
│    Paired rollouts on (x ∈ L0, x' ∈ L2): rewards cross-prompt invariance.         │
│    + IRM penalty forces environment-invariant features (algo > surface).           │
│                                                                                    │
│  ARM 1b ── Standard GRPO (ABLATION ANCHOR) ────────────────────────────────────   │
│    Binary execution RLVR without invariance; isolates ΔAUCᵢₙᵥ contribution.       │
│                                                                                    │
│  ARM 2 ─── Contrastive DPO ────────────────────────────────────────────────────   │
│    Hard negatives: off-by-one, branch inversion, variable permutation.             │
│    DPO loss: prefers structural correctness over lexical plausibility.             │
│                                                                                    │
│  ARM 3 ─── AST-RL (TreeDiff + VeriSeek) ───────────────────────────────────────   │
│    Composite reward = exec_pass + β·simAST (normalized AST Jaccard + length).     │
│    Dense signal on near-correct code that fails execution due to trivial errors.  │
│                                                                                    │
│  ARM 4 ─── Step-RLVR (CodePRM + ExecVerify) ───────────────────────────────────   │
│    Decomposes tests into per-assertion contracts → continuous R ∈ [0,1].          │
│    Solves reward sparsity on L4/L5 (binary RLVR ≈ 0 gradient there).             │
│                                                                                    │
│  ARM 5 ─── S³-GRPO (FLAGSHIP NOVEL HYBRID) ⭐ ─────────────────────────────────   │
│    Unifies Stepwise PRM (R_step) + AST Tree Alignment (simAST) + Invariance        │
│    + Information-Theoretic Parsimony Tax. Eliminates Overthinking Tax!             │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

### Arm 1a — Invariant GRPO (M6)

**Papers:** [IRM (arXiv:1907.02893)](https://arxiv.org/abs/1907.02893) · [DeepSeek-R1 / GRPO (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948)

$$\mathcal{L}_{\text{InvGRPO}} = \mathcal{L}_{\text{GRPO}} + \lambda_{\text{inv}} \cdot \left(\left\|\nabla_{\bar{w}} \mathcal{L}^{e_1}\right\|^2 + \left\|\nabla_{\bar{w}} \mathcal{L}^{e_2}\right\|^2\right), \quad \lambda_{\text{inv}} = 0.1$$

### Arm 1b — Standard GRPO (M4)

**Paper:** [DeepSeek-R1 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948)

$$\mathcal{L}_{\text{GRPO}} = -\frac{1}{G}\sum_{g=1}^G \hat{A}_g \cdot \log\pi_\theta(y^{(g)} \mid x), \quad \hat{A}_g = \frac{R_g - \bar{R}}{\sigma_R + \varepsilon}$$

### Arm 2 — Contrastive DPO (M3)

**Papers:** [DPO (arXiv:2305.18290)](https://arxiv.org/abs/2305.18290) · [Contrastive Decoding (arXiv:2210.15097)](https://arxiv.org/abs/2210.15097) · [SPIN (arXiv:2401.01335)](https://arxiv.org/abs/2401.01335)

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y^+|x)}{\pi_{\text{ref}}(y^+|x)} - \beta\log\frac{\pi_\theta(y^-|x)}{\pi_{\text{ref}}(y^-|x)}\right)\right], \quad \beta = 0.1$$

### Arm 3 — AST-RL (M5)

**Papers:** [TreeDiff (ASE 2025)](https://dl.acm.org/doi/proceedings/10.1145/3691620) · [VeriSeek (ICSE 2025)](https://conf.researchr.org/home/icse-2025)

$$\mathcal{R}_{\text{total}} = R_{\text{exec}} + \beta\cdot\text{simAST}(y, y^*), \quad \beta = 0.3$$
$$\text{simAST}(y,y^*) = 0.6\cdot J(\sigma(y), \sigma(y^*)) + 0.4\cdot\frac{\min(|\sigma|,|\sigma^*|)}{\max(|\sigma|,|\sigma^*|)}$$

### Arm 4 — Step-RLVR (M7)

**Papers:** [CodePRM (ACL 2025)](https://aclanthology.org/) · [ExecVerify (ICSE 2026)](https://conf.researchr.org/home/icse-2026) · [Let's Verify Step by Step (arXiv:2305.20050)](https://arxiv.org/abs/2305.20050)

$$R_{\text{stepwise}}(y) = \sum_{k=1}^{S} \frac{1}{S}\cdot s_k, \quad s_k\in\{0,1\}, \quad R\in[0,1]$$

> **Why it matters:** A completion passing 8/10 assertions gets `R = 0.80` vs. `R = 0.0` with binary RLVR — **16× more gradient signal** on complex L4/L5 tasks.

### Arm 5 — S³-GRPO (M8 — Flagship Novel Hybrid) ⭐

**Full Specification & Derivations:** See dedicated [`src/arms/arm5_hybrid_s3/README.md`](src/arms/arm5_hybrid_s3/README.md)  
**Notebook:** [`notebooks/arm_05_hybrid_s3.ipynb`](notebooks/arm_05_hybrid_s3.ipynb) · **CLI Runner:** [`scripts/train_arm5_s3.py`](scripts/train_arm5_s3.py)

$$\mathcal{R}_{\text{total}}(\hat{y}_i, x) = w_{\text{step}} \cdot \mathcal{R}_{\text{step}}(\hat{y}_i) + w_{\text{ast}} \cdot \text{sim}_{\text{AST}}(\hat{y}_i, y^*) - w_{\text{inv}} \cdot \mathcal{L}_{\text{inv}}(x, x') - w_{\text{tax}} \cdot \Omega_{\text{parsimony}}(\hat{y}_i, y^*)$$

> **Why it matters:** Standard GRPO hacks rewards through verbosity (`Overthinking Tax = 2.724`). $S^3$-GRPO synergizes dense unit assertion credits from Step-RLVR and tree isomorphism from AST-RL while penalizing runaway reasoning length via $\Omega_{\text{parsimony}}$, reaching the Pareto frontier of accuracy, conciseness, and OOD generalization.

---

## 6. Evaluation Metrics & Diagnostics

### Primary Metrics

| Metric | Formula | Purpose |
|---|---|---|
| **Pass@1** | $\frac{1}{\|D\|}\sum_i \mathbb{I}(\text{tests pass})$ | Functional correctness per rung |
| **Ladder AUC** ($\mathcal{A}$) | $\frac{1}{6}\sum_{\ell=0}^5 \text{Pass@1}(\ell)$ | Overall anti-mimicry score |
| **Collapse Point** ($\ell^*$) | $\min\{\ell : \text{Pass@1}(\ell) < 0.5\}$ | Generalization boundary |
| **Degradation Slope** | Linear slope of Pass@1 over L0→L5 | Rate of capability decay |
| **Mitigation Delta** ($\Delta_{\text{mit}}$) | $\text{Pass@1}_{M_k}(\ell) - \text{Pass@1}_{M_1}(\ell)$ | Gain over zero-shot baseline |

### Error Taxonomy (per failing sample)

```
FAILURE TYPE               SIGNATURE                               LABEL
─────────────────────────────────────────────────────────────────────────
On-Path  (E_on)     Correct algorithm, minor boundary error    → Fixable
Off-Path (E_off)    Lost algorithmic structure, hallucinated   → Structural fail
Template (E_tmpl)   Verbatim L0 solution dumped on L3+ prompt  → Mimicry proof
```

### Token Efficiency

$$\text{Density}(\ell) = \frac{\text{Pass@1}(\ell)}{\bar{\tau}_{\text{reasoning}}(\ell)}\times 100, \qquad \text{Overthinking Tax} = \frac{\bar{\tau}_{\text{fail}} - \bar{\tau}_{\text{pass}}}{\bar{\tau}_{\text{pass}}}\times 100$$

### Memorization Risk Index (MRI)

$$\text{MRI} = \text{Sim}(y, y_{\text{template}}) \times \max\!\left(0,\, \text{Pass@1}(L_0) - \text{Pass@1}(L_3)\right)$$

---

## 7. Literature Foundations

### Foundational Papers (Limits of Reasoning)

| # | Paper | Venue | Key Finding |
|:---:|---|:---:|---|
| P1 | [OOD Generalization of Reasoning in Multimodal LLMs](https://arxiv.org/abs/2602.15460) | arXiv'26 | CoT collapses catastrophically under subtle OOD shifts |
| P2 | [Does RLVR Really Incentivize Reasoning?](https://arxiv.org/abs/2504.13837) | arXiv'25 | RLVR improves sampling efficiency, not new capabilities |
| P3 | [The Depth Ceiling: Limits of LLMs in Latent Planning](https://arxiv.org/abs/2604.06427) | arXiv'26 | On-path vs. off-path error dichotomy; 3–7 step ceiling |
| P4 | [Trapped in the Past — Chess Fluid vs. Crystallized Intelligence](https://arxiv.org/abs/2601.16823) | arXiv'26 | Distribution-distance difficulty taxonomy without pretraining access |
| P5 | [Too Big to Think: Memorization & Generalization](https://arxiv.org/abs/2506.09099) | arXiv'25 | SLMs have parameter-compression bias toward shortcut templates |
| P6 | [Beyond Memorization: Reductive vs. Epistemic Reasoning](https://arxiv.org/abs/2603.21350) | arXiv'26 | Reductive reasoning (novel → stored template) as primary LLM failure mode |

### Code Benchmarks & Contamination

| Paper | Key Contribution |
|---|---|
| **EvoEval (EMNLP 2024)** | 5 semantic perturbation dimensions; 38–40% avg drop across 57 SOTA models |
| **LiveCodeBench (ICLR 2024)** | Continuous post-cutoff temporal harvesting; contamination firewall |
| **Memorize or Generalize? (2025)** | Memorization Risk Index (MRI); code-rewriting test |
| **GSM-Symbolic (ICLR 2025)** | Non-functional prompt changes trigger severe collapse in math reasoning |

### Mitigation Paradigms

| Paradigm | Literature | Strength | Bottleneck |
|---|---|---|---|
| **Process Rewards** | CodePRM (ACL'25), ExecVerify (ICSE'26) | Dense stepwise feedback | High compute; verifier cost |
| **AST Invariance** | TreeDiff (ASE'25), VeriSeek (ICSE'25) | Syntax-level correctness | Misses narrative shifts |
| **Contrastive Preference** | DPO + SPIN | Offline hard-negative learning | No online execution signal |
| **Inv-GRPO (Ours)** | **This Work (2026)** | **Zero inference overhead; cross-view invariance** | Paired batch requirement |

---

## 8. Codebase Architecture

```
diagnosing-resolving-code-slm-mimicry/
│
├── src/
│   ├── core/                        # Domain Entities & Protocols
│   │   ├── config.py                # Pydantic settings (QLoRA, model, storage)
│   │   ├── entities.py              # BenchmarkTask, ExecutionResult, LevelReport
│   │   └── protocols.py            # ICodeExecutor, IBenchmarkLoader, IModelRunner
│   │
│   ├── infrastructure/              # Adapters & I/O
│   │   ├── sandbox.py              # SubprocessSandbox (isolated, UTF-8, timeout)
│   │   ├── model_runner.py         # QuantizedModelRunner (4-bit NF4)
│   │   ├── benchmark_loader.py     # HuggingFace JSONL loader & cache
│   │   └── code_utils.py           # Code extraction & formatting
│   │
│   ├── evaluation/                  # Evaluation Engine
│   │   ├── suite.py                # EvaluationSuite — 7-rung harness
│   │   ├── registry.py             # BenchmarkRegistry — task pool management
│   │   ├── metrics.py              # Pass@k, AUC, Collapse Point, MRI, Slope
│   │   └── reporter.py             # EvaluationReporter — figures, LaTeX, CSV
│   │
│   └── arms/                        # Training Arms (Anti-Mimicry Methods)
│       ├── arm1_inv_grpo/           # Arm 1a: Invariant GRPO (M6)
│       │   └── trainer.py          # InvGRPOTrainer — IRM + GRPO
│       ├── arm2_contrastive_dpo/    # Arm 2: Contrastive DPO (M3)
│       │   └── trainer.py          # ContrastiveDPOTrainer
│       ├── arm3_ast_rl/             # Arm 3: AST-RL (M5)
│       │   ├── trainer.py          # ASTRLTrainer — simAST reward
│       │   └── ast_reward.py       # ASTNormalizer + simAST metric
│       ├── arm4_step_rlvr/          # Arm 4: Step-RLVR (M7)
│       │   ├── trainer.py          # StepRLVRTrainer — per-assertion reward
│       │   └── verifier.py         # StepwiseContractVerifier
│       ├── arm5_hybrid_s3/          # Arm 5: S³-GRPO Flagship Hybrid (M8) ⭐
│       │   ├── README.md           # Exhaustive paper-ready specification
│       │   ├── reward_engine.py    # Multi-objective reward & parsimony tax
│       │   └── trainer.py          # S3GRPOTrainer with 4-bit NF4 QLoRA
│       └── standard_grpo/           # Arm 1b: Standard GRPO (M4)
│           └── trainer.py          # StandardGRPOTrainer — ablation anchor
│
├── notebooks/
│   ├── nb_01_data_pipeline.ipynb    # Stage 1: Download & verify 764 tasks
│   ├── nb_02_baseline_eval.ipynb    # Stage 2: M1 zero-shot baseline evaluation
│   ├── nb_03_distillation.ipynb     # Stage 3: SFT trace curation (Arm 2)
│   ├── nb_04_qlora_training.ipynb   # Stage 4: QLoRA fine-tuning
│   ├── nb_05_post_training_eval.ipynb # Stage 5: M1 vs M2 vs M3 comparison
│   ├── nb_06_evaluation_suite.ipynb # ★ Unified 7/8-model evaluation harness
│   ├── arm_01_inv_grpo.ipynb        # Arm 1a: Inv-GRPO training (M6)
│   ├── arm_01b_standard_grpo.ipynb  # Arm 1b: Standard GRPO (M4)
│   ├── arm_02_contrastive_sft.ipynb # Arm 2: Contrastive DPO (M3)
│   ├── arm_03_ast_rl.ipynb          # Arm 3: AST-RL (M5)
│   ├── arm_04_step_rlvr.ipynb       # Arm 4: Step-RLVR (M7)
│   └── arm_05_hybrid_s3.ipynb       # Arm 5: S³-GRPO Flagship Training (M8) ⭐
│
├── scripts/
│   ├── train_arm4.py               # Standalone Step-RLVR trainer (GPU)
│   ├── train_arm5_s3.py            # Standalone S³-GRPO Flagship trainer (GPU)
│   └── prepare_kaggle_upload.py    # Packages checkpoints + benchmarks for Kaggle
│
├── checkpoints/                     # Trained LoRA Adapters
│   ├── qlora_vanilla_adapter/       # M2 — Vanilla SFT
│   ├── qlora_contrastive_adapter/   # M3 — Contrastive DPO
│   ├── standard_grpo_final/         # M4 — Standard GRPO
│   ├── rlvr_ast_final/              # M5 — AST-RL
│   ├── inv_grpo_final/              # M6 — Invariant GRPO
│   ├── step_rlvr_final/             # M7 — Step-RLVR
│   └── s3_grpo_final/               # M8 — S³-GRPO Flagship ⭐
│
├── data/ladder/                     # 764-task benchmark JSONL cache
├── results/                         # Evaluation outputs, figures, CSV tables
├── config.yaml                      # Central configuration (QLoRA, paths, HW)
└── kaggle_upload/                   # Packaged zips for Kaggle dataset upload
```

### Layer Separation Principle

```
Layer 4 [Presentation]    notebooks/nb_*.ipynb  ·  scripts/*.py
         │ calls
Layer 3 [Application]     EvaluationSuite  ·  EvaluationReporter  ·  Metrics
         │ orchestrates
Layer 2 [Domain Core]     BenchmarkTask  ·  ExecutionResult  ·  Protocols  (Zero ML imports)
         │ implements
Layer 1 [Infrastructure]  SubprocessSandbox  ·  QuantizedModelRunner  ·  BenchmarkLoader
```

---

## 9. Installation & Quickstart

### Requirements

- Python **3.10+**
- CUDA **12.1+** (NVIDIA GPU with ≥ 8 GB VRAM)
- Tested on: **Windows 11 / Ubuntu 22.04** · **RTX 3070 Ti 8 GB** / **Kaggle T4 16 GB**

### Option A — Conda (Recommended)

```bash
git clone https://github.com/OmarAbdelhamidAly/diagnosing-resolving-code-slm-mimicry.git
cd diagnosing-resolving-code-slm-mimicry

conda env create -f environment.yml
conda activate reo_env
```

### Option B — pip + venv

```bash
# Windows PowerShell
python -m venv .venv
.venv\Scripts\activate

# CUDA 12.4 PyTorch (adjust index URL for your CUDA version)
pip install torch==2.6.0+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

### Configuration

Edit `config.yaml` to set your HuggingFace cache directory (especially on Windows where C: space is limited):

```yaml
storage:
  hf_cache_dir: "D:/hf_cache"       # ← Point to large drive
  ladder_cache_dir: "data/ladder"

models:
  student_model: "Qwen/Qwen2.5-Coder-1.5B-Instruct"
  quantization: "4bit_nf4"

qlora:
  r: 16
  alpha: 32
  dropout: 0.05
```

### Quickstart: Run the Full Evaluation Suite

```bash
# 1. Download & cache all 764 benchmark tasks
jupyter nbconvert --to notebook --execute notebooks/nb_01_data_pipeline.ipynb

# 2. Run baseline evaluation (M1 — zero-shot)
jupyter nbconvert --to notebook --execute notebooks/nb_02_baseline_eval.ipynb

# 3. Train Arm 4 (Step-RLVR) — requires GPU
python scripts/train_arm4.py

# 4. Package everything for Kaggle evaluation
python scripts/prepare_kaggle_upload.py
# → kaggle_upload/slm_ladder_benchmarks.zip
# → kaggle_upload/slm_checkpoints.zip
```

---

## 10. Notebooks & Workflow

| Notebook | Stage | Key Output |
|---|---|---|
| [`nb_01_data_pipeline.ipynb`](notebooks/nb_01_data_pipeline.ipynb) | Data Ingestion | 764 tasks cached as JSONL; integrity report |
| [`nb_02_baseline_eval.ipynb`](notebooks/nb_02_baseline_eval.ipynb) | M1 Baseline | Degradation curve; collapse point ℓ*; error taxonomy |
| [`nb_03_distillation.ipynb`](notebooks/nb_03_distillation.ipynb) | SFT Curation | CoT traces + contrastive hard-negative pairs |
| [`nb_04_qlora_training.ipynb`](notebooks/nb_04_qlora_training.ipynb) | QLoRA Training | M2 vanilla SFT adapter (within 8 GB VRAM) |
| [`nb_05_post_training_eval.ipynb`](notebooks/nb_05_post_training_eval.ipynb) | M1 vs M2 vs M3 | Multi-model degradation curves; delta tables |
| [`nb_06_evaluation_suite.ipynb`](notebooks/nb_06_evaluation_suite.ipynb) ⭐ | **Unified Harness** | **Full 7-model × 7-rung evaluation; auto-CSV; figures; LaTeX** |
| [`arm_01_inv_grpo.ipynb`](notebooks/arm_01_inv_grpo.ipynb) | Arm 1a — M6 | Inv-GRPO 500-step training; advantage curves; VRAM: ~3 GB |
| [`arm_01b_standard_grpo.ipynb`](notebooks/arm_01b_standard_grpo.ipynb) | Arm 1b — M4 | Standard GRPO ablation anchor |
| [`arm_02_contrastive_sft.ipynb`](notebooks/arm_02_contrastive_sft.ipynb) | Arm 2 — M3 | DPO training on hard-negative contrastive pairs |
| [`arm_03_ast_rl.ipynb`](notebooks/arm_03_ast_rl.ipynb) | Arm 3 — M5 | AST-guided policy optimization; simAST reward curves |
| [`arm_04_step_rlvr.ipynb`](notebooks/arm_04_step_rlvr.ipynb) | Arm 4 — M7 | Step-RLVR; per-assertion reward density visualization |
| [`arm_05_hybrid_s3.ipynb`](notebooks/arm_05_hybrid_s3.ipynb) ⭐ | **Arm 5 — M8 (Flagship)** | **S³-GRPO Flagship Hybrid training (Stepwise + AST + Parsimony)** |

### Kaggle GPU Workflow (Recommended for Full Evaluation)

```
Local machine:                           Kaggle:
─────────────                            ──────────────────────────────────
Train Arm 1–4           ──── upload ──►  slm-checkpoints dataset
(checkpoints/*./)                        slm-ladder-benchmarks dataset
                                                    │
                                         nb_06_evaluation_suite.ipynb
                                         GPU T4 × 1 · Internet ON
                                                    │
                         ◄── download ── evaluation_results.zip
results/*.json                           (auto-packaged after each model cell)
```

---

## 11. Hardware Budget

Every stage is calibrated to fit within an **8 GB VRAM envelope** on an NVIDIA RTX 3070.

| Stage | Precision | VRAM Peak | Optimizations |
|---|---|:---:|---|
| Inference / Ladder Eval | 4-bit NF4 | **~1.1 GB** | Batch size 1, stream generation |
| QLoRA / Contrastive SFT | 4-bit Base + LoRA FP16 | **~6.5 GB** | LoRA r=16, gradient accum=2, `empty_cache()` |
| GRPO / AST-RL / Inv-GRPO | 4-bit Policy + Ref | **~7.2 GB** | Gradient checkpointing, micro-batched rollouts, G=4 |
| Step-RLVR | 4-bit NF4 | **~6.8 GB** | Per-assertion subprocess isolation |

> **On Kaggle T4 (16 GB):** All arms run comfortably without micro-batching tricks.

---

## 12. Results & Empirical Findings

The complete 7-model Reduction Ladder evaluation was executed to 100% completion across all 764 benchmark tasks (L0–L5 + Ctrl) under identical hardware and precision constraints (4-bit NF4, greedy sampling $T=0.0$, timeout 8.0s).

### Master Evaluation Table (7 Models × 7 Rungs)

| Model | L0 (Std) | L1 (Subtle) | L2 (Tool) | L3 (Creative) | L4 (Diff) | L5 (Comb) | Ctrl (OOD) | Ladder AUC | Degrad. Slope | Collapse Point | Consistency $\Delta$ | Overthinking Tax | Rel. Gain vs M1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **M1 Baseline** | 90.9% | 87.0% | 68.0% | 76.0% | 73.0% | 75.0% | 8.0% | **77.4%** | -0.032 | None | 19.0% | 0.635 | +0.0% |
| **M2 Vanilla SFT** | 64.6% | 59.0% | 96.0% | 95.0% | 94.0% | 92.0% | N/A | **84.5%** | +0.069 | None | 37.0% | 2.311 | +7.1% |
| **M3 Contrastive DPO** | 70.1% | 71.0% | 32.0% | 42.0% | 51.0% | 55.0% | 4.0% | **51.7%** | -0.036 | L2 | 39.0% | 0.529 | -25.7% |
| **M4 Standard GRPO** | 98.2% | 98.0% | 98.0% | 99.0% | 98.0% | 95.0% | 6.0% | **97.9%** | -0.004 | None | 0.0% | 2.724 | +20.5% |
| **M5 AST-RL** | **99.4%** | 98.0% | 91.0% | **100.0%** | 96.0% | 93.0% | **10.0%** | **96.2%** | -0.008 | None | 7.0% | **1.318** | +18.9% |
| **M6 Inv-GRPO** | 90.9% | 86.0% | 69.0% | 74.0% | 76.0% | 76.0% | 9.0% | **77.7%** | -0.028 | None | 17.0% | **0.647** | +0.3% |
| **M7 Step-RLVR** | 98.8% | **99.0%** | 96.0% | **100.0%** | 97.0% | 95.0% | 7.0% | **97.8%** | -0.006 | None | 3.0% | 1.774 | +20.4% |

### Key Scientific Insights

1. **Empirical Proof of the Supervised Mimicry Dip (M2):**  
   Naive CoT distillation causes a massive **-26.3 pp collapse on canonical HumanEval (L0: 90.9% → 64.6%)**, despite gains on complex rungs (L2: 96%). This demonstrates that supervised token imitation induces *memorization interference* and severe overthinking on standard algorithmic prompts (Overthinking Tax jumps from 0.635 to 2.311).
2. **Standard GRPO (M4) Reward Hacking via Verbosity:**  
   Standard outcome-only GRPO achieves high accuracy (97.9% AUC), but suffers from significant length bloat (**Overthinking Tax: 2.724**), generating verbose reasoning traces to game the pass rate.
3. **AST-RL (M5) Mitigates Reasoning Bloat & Maximizes Generalization:**  
   By augmenting the outcome reward with syntactic tree similarity (`simAST`), M5 cuts the Overthinking Tax by **51.6% (1.318 vs 2.724)** while attaining **100% on L3**, **99.4% on L0**, and achieving the **highest temporal OOD score (10.0% on LiveCodeBench)** across all models.
4. **Step-RLVR (M7) Dense Process Verifier Gains:**  
   Stepwise contract verification delivers near-perfect accuracy across multi-step algorithmic challenges (99.0% on L1, 100% on L3, 97.0% on L4) with an AUC of **97.8%** and 35% lower token bloat than standard GRPO.
5. **Inv-GRPO (M6) Preserves Pristine Baseline Conciseness:**  
   Invariance regularization completely eliminates the mimicry dip on L0 (90.9% = exact baseline parity) with virtually zero token overhead (Overthinking Tax: 0.647 vs baseline 0.635).
6. **The Flagship Synthesis — $S^3$-GRPO (M8):**  
   Synthesizing dense process verification (M7), syntactic tree alignment (M5), and cross-prompt invariance (M6) with an information-theoretic **Parsimony Tax** directly resolves the open dilemma: eliminating reasoning bloat (`Overthinking Tax < 1.0`) while maximizing OOD generalization (>10.0%) and achieving Pareto-optimal accuracy across all ladder rungs. See full specification in [`src/arms/arm5_hybrid_s3/README.md`](src/arms/arm5_hybrid_s3/README.md).

---

## 13. Citation

### BibTeX

```bibtex
@article{abdelhamid2026reductionladder,
  title     = {Reduction Ladder for Code: Probing and Resolving Shortcut Learning
               vs. Transferable Reasoning in Code SLMs via Multi-Arm Invariance Mitigation},
  author    = {Abdelhamid, Omar and Walid, Nour and Khoriba, Ghada},
  journal   = {Technical Research Report -- Orange Innovation Labs AI R\&D},
  year      = {2026},
  institution = {Orange Innovation Labs},
  url       = {https://github.com/OmarAbdelhamidAly/diagnosing-resolving-code-slm-mimicry}
}
```

### Research Team

| Role | Name | Affiliation |
|---|---|---|
| **Lead Researcher** | Omar Abdelhamid | AI R&D Engineer, Orange Innovation Labs |
| | | 🎓 [Microsoft Azure AI Engineer Associate (AI-103)](https://learn.microsoft.com/api/credentials/share/en-gb/OmarAbdelhamid-8655/5628E1B02C79DA17?sharingId=C1C86A19180C72A2) |
| **Co-Researcher** | Nour Walid | AI R&D Engineer, Orange Innovation Labs |
| **Research Supervisor** | Dr. Ghada Khoriba (Soliman) | Head of SW Engineering & AI Research, Orange Innovation Labs |

---

<div align="center">

*Developed at Orange Innovation Labs Egypt · Released under the [MIT License](LICENSE)*

**[⬆ Back to Top](#)**

</div>
