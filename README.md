# Reduction Ladder for Code & Mitigation 🧠⚡

> **Does your Small Language Model reason or memorize?**  
> A rigorous benchmark and prioritized multi-arm mitigation framework for detecting, diagnosing, and resolving pattern-matching collapse in Small Language Models (SLMs) on algorithmic coding tasks.

---

## 📑 Table of Contents
1. [Overview & Core Research Questions](#overview--core-research-questions)
2. [The Reduction Ladder Framework (L0–L5)](#the-reduction-ladder-framework-l0l5)
3. [Comprehensive Literature Taxonomy & Related Work](#comprehensive-literature-taxonomy--related-work)
4. [Prioritized Multi-Arm Mitigation Framework (P1–P4)](#prioritized-multi-arm-mitigation-framework-p1p4)
5. [The Comparative Experimental Suite](#the-comparative-experimental-suite)
6. [Repository Structure](#repository-structure)
7. [End-to-End Pipeline & Notebooks Breakdown](#end-to-end-pipeline--notebooks-breakdown)
8. [Evaluation Metrics & Mathematical Definitions](#evaluation-metrics--mathematical-definitions)
9. [Hardware Feasibility & VRAM Budget (RTX 3070 8GB)](#hardware-feasibility--vram-budget-rtx-3070-8gb)
10. [Setup & Quickstart Guide](#setup--quickstart-guide)
11. [Research Team & Citation](#research-team--citation)

---

## 🎯 Overview & Core Research Questions

Small Language Models (SLMs) in the 1--3B parameter range (such as `Qwen2.5-Coder-1.5B-Instruct`) are crucial for private, low-latency, and edge deployments in enterprise ecosystems (e.g., Orange Innovation Labs network automation, automated script patching, and code review).

However, existing code LLMs suffer from a profound **Reasoning vs. Memorization Crisis**:
- When presented with classic problems verbatim (L0), models succeed by retrieving memorized patterns.
- When minor semantic-preserving surface transformations (e.g., variable renaming L2, or narrative reframing L3) are applied, accuracy experiences **catastrophic collapse**.
- Standard Supervised Fine-Tuning (SFT) often **amplifies memorization**, while standard Reinforcement Learning with Verifiable Rewards (RLVR) frequently learns **shortcut reward-hacking patterns** rather than invariant algorithmic logic.

### ❓ Central Research Questions
1. **The Diagnostic Question:** At which exact transformation level ($\ell^*$) does an SLM collapse, and how does the error distribution (on-path vs. off-path vs. wrong-template) differ between Baseline, SFT, and RLVR?
2. **The Mitigation Question:** Which mitigation paradigm (Invariance RL, Contrastive SFT, Structural AST, or Process Supervision) is most effective at breaking template mimicry while preserving low inference latency?

---

## 🪜 The Reduction Ladder Framework (L0–L5)

To ensure 100% scientific reproducibility and eliminate the noise and cost of unverified synthetic data generation, our 6-level Reduction Ladder maps directly to peer-reviewed benchmarks on Hugging Face:

```
L0  Verbatim Classic Benchmark      ──► openai/openai_humaneval
 │
L1  Subtle Specification Shift      ──► evoeval/EvoEval_subtle / EvoEval_concise
 │
L2  Descriptive Obfuscation         ──► evoeval/EvoEval_verbose
 │
L3  Creative Narrative Reframing    ──► evoeval/EvoEval_creative
 │
L4  Boundary Constraint Addition    ──► evoeval/EvoEval_difficult
 │
L5  Cross-Concept Composition       ──► evoeval/EvoEval_combine
```

| Level | Benchmark Name | HuggingFace Dataset | Probing Target & Transformation Type |
|---|---|---|---|
| **L0** | HumanEval Standard | `openai/openai_humaneval` | Canonical verbatim baseline problem. |
| **L1** | EvoEval Subtle | `evoeval/EvoEval_subtle` | Minor wording, format, and input specification nuances. |
| **L2** | EvoEval Verbose | `evoeval/EvoEval_verbose` | Descriptive obfuscation & identifier alteration. |
| **L3** | EvoEval Creative | `evoeval/EvoEval_creative` | Novel narrative context for the identical algorithmic logic. |
| **L4** | EvoEval Difficult | `evoeval/EvoEval_difficult` | Same core algorithm + extra boundary conditions/constraints. |
| **L5** | EvoEval Combine | `evoeval/EvoEval_combine` | Multi-algorithmic composition and concept integration. |
| **Ctrl** | LiveCodeBench | `livecodebench/code_generation_lite` | Post-cutoff contamination-free temporal control set. |

### 💡 Concrete Transformation Walkthrough (Two Sum across L0–L5)
* **L0 (Verbatim Baseline):** Given an array `nums` and integer `target`, return indices of two numbers that add up to `target`.
* **L1 (Format / Subtle):** Given comma-delimited string `"2,7,11,15"` and integer target 9, parse values and return 0-indexed indices.
* **L2 (Identifier Obfuscated):** Given list `alpha_seq` and integer `tau_val`, return `(p, q)` where `alpha_seq[p] + alpha_seq[q] == tau_val`.
* **L3 (Creative Context):** In an Orange server pool, $n$ containers need bandwidth $b_1 \ldots b_n$. Find two containers matching gateway capacity $B$.
* **L4 (Constraint Added):** Same as L3, but the two containers must belong to *different availability zones* (tracking secondary attribute).
* **L5 (Concept Combination):** Given task bandwidths and execution windows, find two concurrent tasks summing to capacity while *minimizing scheduling fragmentation*.

---

## 🔬 Comprehensive Literature Taxonomy & Related Work

Recent cutting-edge research (2024–2026) has tackled the memorization barrier and reasoning collapse through five distinct paradigms:

```
                                  [ State-of-the-Art Mitigation Landscape 2024-2026 ]
                                                          │
          ┌───────────────────────┬───────────────────────┼───────────────────────┬───────────────────────┐
          ▼                       ▼                       ▼                       ▼                       ▼
   1. Invariance & RL      2. Contrastive CoT      3. Info-Bottleneck      4. Process Rewards      5. AST Structural
     (CLARity / LoPE)      (ReCode / CLIPO)          (IB-FT / IBRO)        (CodePRM / ExecVerify)  (TreeDiff / VeriSeek)
```

### 1. Invariance & Perturbation-Aware Reinforcement Learning
* **CLARity (2025) [arXiv:2501.12980]:** Consistency-aware RL rewards penalising contradictory reasoning across semantically equivalent queries.
* **LoPE (2025) [arXiv:2504.04512]:** Stochastic prompt perturbations during RL training to unlock orthogonal exploration.
* **Break-The-Chain (2025) [arXiv:2506.06971]:** Analyzes Chain-of-Thought collapse under narrative perturbations in code models.

### 2. Contrastive Learning & Policy Optimization
* **ReCode (2025) [arXiv:2502.14890]:** Contrastive reasoning-process rewards distinguishing logical paths from shortcut paths.
* **CLIPO (2025) [arXiv:2503.08912]:** Embeds contrastive trajectory clustering directly into policy gradient optimization.
* **SuperCorrect (2024) [arXiv:2410.09008]:** Distills hierarchical thought templates and negative correction traces for self-correction.

### 3. Information Bottleneck & Representation Regularization
* **IB-FT (2024) [arXiv:2410.18902]:** Information Bottleneck loss penalty on hidden representations to compress superficial prompt features.
* **IBRO (2025) [arXiv:2501.09450]:** Information Bottleneck constraints in RLVR to filter out noisy template tokens.

### 4. Process Reward Models (PRMs) & Step-Level RLVR
* **CodePRM (2025) [arXiv:2501.07890]:** Evaluates intermediate reasoning steps using partial execution feedback.
* **ExecVerify (2026) [arXiv:2601.03450]:** Employs stepwise execution trace tracking to prevent reward hacking.

### 5. Abstract Syntax Tree (AST) & Structural Invariance
* **TreeDiff (2025) [arXiv:2502.11204]:** Uses AST guidance to enforce structural invariance across lexical renaming.
* **VeriSeek (2025) [arXiv:2503.04501]:** Incorporates `simAST` rewards into policy optimization to reward valid tree structures.

---

## 🏆 Prioritized Multi-Arm Mitigation Framework (P1–P4)

To prevent single-method bias, we formulate and empirically compare representative methods from each major research school:

| Priority Arm | Paradigm | Implementation Mechanism | Target Impact |
|---|---|---|---|
| **🥇 P1 (Primary): Inv-GRPO** | Multi-View Invariance RL | Cross-perturbation paired sampling + Consistency Reward $\mathcal{R}_{\text{consistency}}$ | Directly attacks shortcut learning with zero inference latency overhead. |
| **🥈 P2 (SFT): Contrastive-SFT** | Negative Rejection SFT | Fine-tuning on positive CoT paired with negative shortcut rejection traces | Prevents SFT format-mimicry bias before RL. |
| **🥉 P3 (Syntax): AST-RL** | Structural AST Invariance | Syntax tree similarity (`simAST`) reward in policy loop | Enforces lexical/identifier invariance on L1/L2. |
| **🏅 P4 (Process): Step-RLVR** | Stepwise Process RL | Intermediate execution trace verification on sub-function contracts | Granular credit assignment for complex L4/L5 logic. |

### 💡 Core Mathematical Objective of Inv-GRPO (P1):
During training, **Inv-GRPO** feeds paired inputs $(x, x')$ representing logically equivalent variants (e.g., $x \in L_0$ and $x' \in L_2$):

$$\mathcal{R}_{\text{total}}(y, y') = \mathcal{R}_{\text{exec}}(y) + \mathcal{R}_{\text{exec}}(y') + \lambda \cdot \mathcal{R}_{\text{consistency}}(y, y') - \gamma \cdot \mathcal{P}_{\text{template}}$$

$$\hat{A}_{i} = \frac{\mathcal{R}_{\text{total}}(y_i, y'_i) - \mu_{\mathcal{R}}}{\sigma_{\mathcal{R}} + \epsilon}$$

---

## 🤖 The Comparative Experimental Suite

| Model Checkpoint | Training Paradigm | Expected Collapse Point | Primary Scientific Role |
|---|---|:---:|---|
| **M1: Baseline** | Raw `Qwen2.5-Coder-1.5B-Instruct` | **$\approx$ L2** | Un-tuned memorisation and collapse baseline |
| **M2: Vanilla SFT** | Standard CoT SFT | **$\approx$ L3** | Measures format-mimicry and SFT bias |
| **M3: Contrastive-SFT (P2)** | SFT with Negative Shortcut Rejection | **$\approx$ L3 $\to$ L4** | Evaluates data-level template breaking |
| **M4: Vanilla GRPO** | Standard Single-Prompt RLVR | **$\approx$ L3** | Measures outcome reward-hacking shortcuts |
| **M5: AST-RL (P3)** | Policy Optimization with `simAST` | **$\approx$ L4** | Evaluates structural syntax invariance |
| **M6: Inv-GRPO (P1 - Ours)** | Paired Invariance Regularized GRPO | **$\approx$ L4 $\to$ L5** | **Primary Proposed Mitigation** |

---

## 🏛️ Clean Architecture Design & Repository Layout

This project follows **Clean Architecture** principles to separate core scientific domain models from external frameworks and persistence layers:

```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                   PRESENTATION & ENTRY POINTS                           │
 │     • scripts/run_stage*.py (CLI)     • notebooks/nb_*.ipynb            │
 └────────────────────────────────────┬────────────────────────────────────┘
                                      │
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                   APPLICATION SERVICES (Use Cases)                      │
 │     • DataService                     • EvaluationService               │
 │     • AnalysisService                 • Training Orchestrators          │
 └────────────────────────────────────┬────────────────────────────────────┘
                                      │
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                   DOMAIN CORE (Entities & Protocols)                    │
 │     • BenchmarkTask                   • ExecutionResult                 │
 │     • LevelEvaluationReport           • ErrorCategory                   │
 │     • ICodeExecutor (Protocol)        • IModelRunner (Protocol)         │
 └────────────────────────────────────▲────────────────────────────────────┘
                                      │ (implements)
 ┌────────────────────────────────────┴────────────────────────────────────┐
 │                   INFRASTRUCTURE (External Adapters)                    │
 │     • MultiprocessSandbox             • HuggingFaceBenchmarkLoader      │
 │     • QuantizedModelRunner (4-bit)    • RuleBasedErrorClassifier        │
 │     • Atomic Persistence (JSONL)      • Config Parser                   │
 └─────────────────────────────────────────────────────────────────────────┘
```

### Detailed Repository Directory Tree

```
reduction-ladder-for-code/
│
├── config.yaml                            # Global single source of truth for all hyperparameters
├── requirements.txt                       # Locked dependencies with exact versions
├── README.md                              # Main project documentation & quickstart
├── ARCHITECTURE.md                        # Software architecture & team collaboration blueprint
├── implementation_plan.md                 # Exhaustive stage-by-stage implementation plan
│
├── proposal/                              # Formal LaTeX research proposal for Dr. Ghada
│   ├── proposal.tex                       # 13-page technical proposal (Orange branding)
│   └── references.bib                     # 18 curated academic references
│
├── data/
│   ├── ladder/                            # Standardized JSONL benchmark datasets (L0-L5)
│   ├── distillation/                      # Filtered Chain-of-Thought training corpora
│   └── livecode_bench/                    # Contamination-free temporal control tasks
│
├── src/
│   ├── core/                              # Layer 1: Domain Core (Zero external ML dependencies)
│   │   ├── entities.py                    # BenchmarkTask, ExecutionResult, LevelEvaluationReport, ErrorCategory
│   │   ├── interfaces.py                  # Protocols: ICodeExecutor, IBenchmarkLoader, IModelRunner, IErrorClassifier
│   │   └── exceptions.py                  # Domain Exceptions (SandboxTimeoutError, VRAMExceededError)
│   │
│   ├── infrastructure/                    # Layer 2: External Adapters & IO
│   │   ├── sandbox.py                     # MultiprocessSandbox (Process isolation, timeout guarding)
│   │   ├── hf_loader.py                   # HuggingFaceBenchmarkLoader (Dataset fetching & disk caching)
│   │   ├── model_loader.py                # QuantizedModelRunner (4-bit NF4 BitsAndBytes + LoRA inference)
│   │   ├── classifier.py                  # RuleBasedErrorClassifier (Failure taxonomy heuristics)
│   │   └── persistence.py                 # Atomic JSONL/JSON/YAML persistence
│   │
│   └── services/                          # Layer 3: Application Services (Use-Case Orchestrators)
│       ├── data_service.py                # Benchmark ingestion & ground-truth verification
│       ├── evaluation_service.py          # Pass@1, Pass@5 evaluation loop & error diagnosis
│       ├── analysis_service.py            # Ladder AUC, Collapse Point, MRI calculation & publication plots
│       └── training/                      # Training Use Cases (QLoRA, Contrastive SFT, Inv-GRPO)
│           ├── qlora_trainer.py           # SFT trainer orchestrator
│           └── inv_grpo_trainer.py        # Paired invariance RL trainer orchestrator
│
├── notebooks/                             # Layer 4: Interactive Notebooks
│   ├── nb_01_data_pipeline.ipynb          # Stage 1: Data ingestion & validation
│   ├── nb_02_baseline_eval.ipynb          # Stage 2: Baseline (M1) un-tuned model evaluation
│   ├── nb_03_qlora_training.ipynb         # Stage 4: SFT training (M2 Vanilla vs M3 Contrastive)
│   ├── nb_04_rlvr_training.ipynb          # Stage 5: RL training (M4 GRPO vs M5 AST vs M6 Inv-GRPO)
│   └── nb_05_comparison.ipynb            # Stage 6: Comparative multi-model analysis & figures
│
├── scripts/                               # Layer 4: CLI Entry Points
│   ├── run_stage1_data.py                 # CLI for Stage 1 data pipeline
│   ├── run_stage2_eval.py                 # CLI for Stage 2 baseline eval
│   ├── run_stage3_distill_data.py         # CLI for Stage 3 SFT corpus generation
│   ├── run_stage4_qlora.py                # CLI for Stage 4 SFT training
│   ├── run_stage5_rlvr.py                 # CLI for Stage 5 RL training
│   └── run_stage6_analysis.py             # CLI for Stage 6 comparative analysis
│
└── results/                               # Structured output evaluation reports & figures
    ├── baseline/
    ├── distilled_vanilla/
    ├── distilled_contrastive/
    ├── rlvr_vanilla/
    ├── rlvr_ast/
    └── rlvr_inv_grpo/
```

---

## 📈 Evaluation Metrics & Mathematical Definitions

| Metric | Formulation | Scientific Purpose |
|---|---|---|
| **Pass@1** | $\mathbb{E}[\text{accuracy}_{sample=1}]$ | Primary deterministic functional correctness. |
| **Pass@5** | $1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$ | Exploration capability and sampling coverage ($T=0.8$). |
| **Collapse Point ($\ell^*$)** | $\min \{ \ell \mid \text{pass@1}(\ell) < 0.5 \}$ | Lowest level where competence drops below 50\%. |
| **Ladder AUC ($\mathcal{A}$)** | $\frac{1}{6} \sum_{\ell=0}^{5} \text{pass@1}(\ell)$ | Aggregate gracefulness of degradation across transformations. |
| **Consistency Delta ($\Delta_c$)** | $|\text{pass@1}(\text{L1}) - \text{pass@1}(\text{L2})|$ | Sensitivity to surface-level lexical variations. |
| **Token Efficiency ($\tau_\ell$)** | $\text{Mean}(\text{len}(\text{CoT tokens}))$ | Trace length dynamics across difficulty levels. |

---

## ⚡ Hardware Feasibility & VRAM Budget (RTX 3070 8GB)

| Stage / Arm | Memory Allocation | Optimization Techniques Employed |
|---|---|---|
| **Inference / Eval** | $\approx$ 4.0 GB | 4-bit NF4 Quantization, Batch Size = 1 |
| **QLoRA / Contrastive SFT** | $\approx$ 6.5 GB | PEFT LoRA Adapters ($r=16, \alpha=32$), Grad Accumulation = 4 |
| **GRPO / AST-RL / Inv-GRPO** | $\approx$ 7.2 GB | Gradient Checkpointing, Group Size $G=4$, Interleaved Rollouts |

---

## 🚀 Setup & Quickstart Guide

### 1. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Pipeline Steps
```bash
# Stage 1: Ingest benchmarks
python scripts/run_stage1_data.py

# Stage 2: Evaluate Baseline (M1)
python scripts/run_stage2_eval.py --model baseline

# Stage 3 & 4: Train SFT Arms (Vanilla M2 vs Contrastive M3)
python scripts/run_stage3_distill_data.py
python scripts/run_stage4_qlora.py
python scripts/run_stage4_contrastive_sft.py

# Stage 5: Train RL Arms (Vanilla M4 vs AST-RL M5 vs Inv-GRPO M6)
python scripts/run_stage5_rlvr.py
python scripts/run_stage5_ast_rl.py
python scripts/run_stage5_inv_grpo.py

# Stage 6: Comparative Analysis & Plots across all arms
python scripts/run_stage6_analysis.py
```

---

## 👥 Research Team & Authors
- **Omar Abdelhamid** — AI Research Engineer, Orange Innovation Labs
- **Nour Walid** — AI Research Engineer, Orange Innovation Labs
- **Supervisor:** Dr. Ghada — Research Supervisor, Orange Innovation Labs

---

## 📜 Citation & License

```bibtex
@misc{reduction_ladder_2025,
  title   = {Reduction Ladder for Code & Mitigation: Probing and Resolving Pattern-Matching Collapse in SLMs via Multi-Arm Mitigation},
  author  = {Abdelhamid, Omar and Walid, Nour},
  year    = {2025},
  institution = {Orange Innovation Labs},
  note    = {Technical Research Proposal}
}
```

Developed for Orange Innovation Labs Research. Released under the MIT License.
