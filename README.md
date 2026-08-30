# Reduction Ladder for Code & Mitigation 🧠⚡

> **Probing and Resolving Shortcut Learning vs. Transferable Algorithmic Reasoning in Code-Generating SLMs & LLMs**  
> *A Joint Research Initiative by Orange Innovation Labs (AI R&D Division) & Benha University*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/🤗_Transformers-4.44%2B-yellow.svg?style=for-the-badge)](https://huggingface.co/)
[![Unsloth / BitsAndBytes](https://img.shields.io/badge/⚡_Unsloth-4--bit_NF4-green.svg?style=for-the-badge)](https://github.com/unslothai/unsloth)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg?style=for-the-badge)](LICENSE)

---

## 📑 Comprehensive Table of Contents

1. [Executive Summary & Core Research Questions](#-executive-summary--core-research-questions)
2. [Foundational Rationale: Why the Code Domain?](#-foundational-rationale-why-the-code-domain)
3. [The Reduction Ladder Diagnostic Framework (L0–L5 + Control)](#-the-reduction-ladder-diagnostic-framework-l0l5--control)
   - [Ladder Levels & Hugging Face Grounding](#ladder-levels--hugging-face-grounding)
   - [Concrete Transformation Walkthrough (Two Sum across L0–L5)](#concrete-transformation-walkthrough-two-sum-across-l0l5)
   - [Theoretical Contrast: Reduction Ladder vs. Code-Rewriting (MRI)](#theoretical-contrast-reduction-ladder-vs-code-rewriting-mri)
4. [Comprehensive Literature Taxonomy & Theoretical Foundations](#-comprehensive-literature-taxonomy--theoretical-foundations)
   - [The Foundational Six Papers: Limits of Reasoning](#the-foundational-six-papers-limits-of-reasoning)
   - [Code-Domain Memorization & Contamination Literature](#code-domain-memorization--contamination-literature)
   - [Multi-Dimensional Evaluation Dimensions Beyond Correctness](#multi-dimensional-evaluation-dimensions-beyond-correctness)
   - [State-of-the-Art Mitigation Paradigms (2024–2026)](#state-of-the-art-mitigation-paradigms-20242026)
5. [Model Selection Rationale & Capacity Axis](#-model-selection-rationale--capacity-axis)
   - [Why Qwen2.5-Coder-1.5B-Instruct?](#why-qwen25-coder-15b-instruct)
   - [Survey of Models in Reference Literature](#survey-of-models-in-reference-literature)
   - [Three-Tier Capacity Axis](#three-tier-capacity-axis)
6. [Cross-Paper Comparison Methodology](#-cross-paper-comparison-methodology)
   - [Five-Axis Comparison Framework](#five-axis-comparison-framework)
   - [Direct Reproduction & Extension of EvoEval Baselines](#direct-reproduction--extension-of-evoeval-baselines)
   - [Cross-Paper Benchmark Alignment Matrix](#cross-paper-benchmark-alignment-matrix)
   - [Statistical Rigor & Reporting Standards](#statistical-rigor--reporting-standards)
7. [Prioritized Multi-Arm Mitigation Framework (P1–P4)](#-prioritized-multi-arm-mitigation-framework-p1p4)
   - [Arm 1 (P1 - Primary): Invariance-Regularized GRPO (Inv-GRPO)](#arm-1-p1---primary-invariance-regularized-grpo-inv-grpo)
   - [Arm 2 (P2 - SFT): Contrastive Thought-Template SFT](#arm-2-p2---sft-contrastive-thought-template-sft)
   - [Arm 3 (P3 - Syntax): AST-Guided Policy Optimization (AST-RL)](#arm-3-p3---syntax-ast-guided-policy-optimization-ast-rl)
   - [Arm 4 (P4 - Process): Stepwise Execution-Gated RLVR (Step-RLVR)](#arm-4-p4---process-stepwise-execution-gated-rlvr-step-rlvr)
   - [Intervention Paradigm Analysis: Distillation vs. RLVR vs. RLIR](#intervention-paradigm-analysis-distillation-vs-rlvr-vs-rlir)
8. [The Six-Model Comparative Experimental Suite (M1–M6)](#-the-six-model-comparative-experimental-suite-m1m6)
9. [Mathematical Formulation of Multi-Dimensional Evaluation Metrics](#-mathematical-formulation-of-multi-dimensional-evaluation-metrics)
10. [Clean Architecture Software Engineering Blueprint](#-clean-architecture-software-engineering-blueprint)
11. [Exhaustive 12-Week (3-Month) Execution Roadmap](#-exhaustive-12-week-3-month-execution-roadmap)
12. [Hardware Feasibility & Edge VRAM Budget (RTX 3070 8GB)](#-hardware-feasibility--edge-vram-budget-rtx-3070-8gb)
13. [Installation, Setup & Quickstart Guide](#-installation-setup--quickstart-guide)
14. [Research Authors, Supervision & Citation](#-research-authors-supervision--citation)

---

## 🎯 Executive Summary & Core Research Questions

Small Language Models (SLMs) in the 1–3B parameter bracket (exemplified by `Qwen2.5-Coder-1.5B-Instruct`) are foundational to the future of private, low-latency, on-device, and edge intelligence. Within telecommunications operators like **Orange Innovation Labs**, edge SLMs drive critical workloads: autonomous network script patching, infrastructure configuration verification, self-healing diagnostic routines, and localized developer copilot workflows.

Despite stellar pass rates on static benchmarks, modern code-generating language models suffer from a fundamental **Reasoning vs. Memorization Crisis**:

```
           ┌────────────────────────────────────────────────────────┐
           │              PROMPT PRESENTATION STYLES                │
           └───────────────────────────┬────────────────────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
┌───────────────────────┐                             ┌───────────────────────┐
│ L0: Verbatim Problem  │                             │ L3: Novel Narrative   │
│ (Standard HumanEval)  │                             │ (EvoEval Creative)    │
└───────────┬───────────┘                             └───────────┬───────────┘
            │                                                     │
            ▼                                                     ▼
┌───────────────────────┐                             ┌───────────────────────┐
│ Model Output: 85% Acc │                             │ Model Output: 38% Acc │
│ Pattern Retrieval     │                             │ CATASTROPHIC COLLAPSE │
└───────────────────────┘                             └───────────────────────┘
```

1. **Superficial Pattern Matching over Latent Planning:** When presented with canonical problems verbatim, models recall memorized tokens. However, trivial semantic-preserving surface transformations (e.g., variable obfuscation, helper function abstraction, or narrative reframing) trigger **catastrophic performance collapse**.
2. **SFT Memorization Bias:** Standard Supervised Fine-Tuning (SFT) on reasoning traces often teaches models the *syntactic formatting* of Chain-of-Thought (CoT) without inducing invariant underlying logic.
3. **RLVR Shortcut Learning & Reward Hacking:** Standard Reinforcement Learning with Verifiable Rewards (RLVR) optimizes for unit-test execution passes on single prompts, frequently converging on superficial shortcut heuristics that fail under out-of-distribution variations.

### ❓ The Central Research Questions

* **RQ1 (The Diagnostic Boundary):** At which exact structural transformation level ($\ell^*$) does a code SLM collapse, and how does the error distribution (on-path execution slip vs. off-path logic loss vs. wrong-template shortcut dump) differ systematically across parameter capacities ($\sim$1.5B vs. $\sim$7B vs. Frontier)?
* **RQ2 (The Mitigation Ceiling):** Can a multi-view invariance objective (**Inv-GRPO**) regularize policy rollouts during training to break template mimicry, delay the collapse point ($\Delta\ell^* \ge 2$), and elevate generalization with zero inference-time latency penalty?

---

## 🔬 Foundational Rationale: Why the Code Domain?

A pivotal architectural and scientific design choice in this research is our exclusive focus on **programmatic code synthesis** rather than mathematical word problems (GSM8K/MATH) or natural language QA.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        WHY CODE IS THE OPTIMAL REASONING LAB                          │
├────────────────────────────────┬───────────────────────────────────────────────────────┤
│ 1. Objective Ground-Truth      │ Zero LLM-judge bias; deterministic unit-test pass/fail│
│    Sandbox Verification        │ in isolated subprocesses with sub-millisecond precision│
├────────────────────────────────┼───────────────────────────────────────────────────────┤
│ 2. AST Isomorphism vs.         │ Variable renaming & reordering mutate 100% of surface │
│    Surface Paraphrasing        │ tokens while holding operational semantics invariant. │
├────────────────────────────────┼───────────────────────────────────────────────────────┤
│ 3. Combinatorial State Space   │ Code demands multi-step invariant tracking; shortcut  │
│    & Edge-Case Fragility       │ pattern-matching breaks catastrophically on edges.    │
├────────────────────────────────┼───────────────────────────────────────────────────────┤
│ 4. Direct Industrial Impact    │ Hardening code SLMs empowers private edge-telecom     │
│    for Orange Infrastructure   │ automation without multi-billion-parameter cloud APIs.│
└────────────────────────────────┴───────────────────────────────────────────────────────┘
```

---

## 🪜 The Reduction Ladder Diagnostic Framework (L0–L5 + Control)

To eliminate the noise, cost, and hallucination risks of unverified synthetic datasets, our **Reduction Ladder** grounds each difficulty level in established, peer-reviewed benchmarks hosted on Hugging Face.

```
  L0: Verbatim Classic Baseline    ──►  openai/openai_humaneval (164 tasks)
   │
  L1: Format & Specification Shift ──►  evoeval/EvoEval_subtle  (100 tasks)
   │
  L2: Structural ToolUse Shift     ──►  evoeval/EvoEval_tool_use (100 tasks)
   │
  L3: Creative Narrative Reframe   ──►  evoeval/EvoEval_creative (100 tasks)
   │
  L4: Constraint Augmentation      ──►  evoeval/EvoEval_difficult (100 tasks)
   │
  L5: Cross-Concept Composition    ──►  evoeval/EvoEval_combine (100 tasks)
   │
 [Ctrl]: Temporal Firewall Control ──►  livecodebench/code_generation_lite
```

### Ladder Levels & Hugging Face Grounding

| Level | Benchmark Name | Hugging Face Dataset Path | Exact Transformation Operation | Probing Objective |
|:---:|---|---|---|---|
| **L0** | HumanEval Standard | `openai/openai_humaneval` | Canonical verbatim classic benchmark problems. | Raw memorization / template retrieval baseline. |
| **L1** | EvoEval Subtle | `evoeval/EvoEval_subtle` | Minor formatting, input types, & specification nuances. | Robustness to trivial prompt formatting shifts. |
| **L2** | EvoEval ToolUse | `evoeval/EvoEval_tool_use` | Helper function integration & API abstraction layers. | Structural adaptation & modular contract adherence. |
| **L3** | EvoEval Creative | `evoeval/EvoEval_creative` | Novel narrative context for identical algorithmic logic. | Logic recognition beneath novel storytelling. |
| **L4** | EvoEval Difficult | `evoeval/EvoEval_difficult` | Core algorithm + strict extra boundary constraints. | Algorithmic adaptation vs. rigid template retrieval. |
| **L5** | EvoEval Combine | `evoeval/EvoEval_combine` | Multi-algorithmic composition and concept integration. | Multi-concept composition & genuine generalization. |
| **Ctrl**| LiveCodeBench Lite | `livecodebench/code_generation_lite`| Clean, post-cutoff temporal problem feed. | 0% contamination temporal firewall validation. |

---

### Concrete Transformation Walkthrough (Two Sum across L0–L5)

To understand how semantics remain invariant while surface complexity shifts, consider the canonical *Two Sum* problem:

* **L0 (Verbatim Classic):**
  ```python
  def two_sum(nums: list[int], target: int) -> list[int]:
      """Given an array of integers nums and an integer target, 
      return indices of the two numbers such that they add up to target."""
  ```
* **L1 (Format / Subtle Shift):**
  ```python
  def parse_and_find_indices(data_str: str, target: int) -> tuple[int, int]:
      """Input is a comma-delimited string of numbers '2,7,11,15'. 
      Parse values and return 0-indexed integer tuple of the matching pair."""
  ```
* **L2 (ToolUse / Structural Abstraction):**
  ```python
  def find_pair_with_tool(seq: list[int], target: int, lookup_helper) -> list[int]:
      """Implement pair searching by calling the pre-defined helper function 
      lookup_helper(table, key) to manage complement queries."""
  ```
* **L3 (Creative Narrative Reframing - Telecom Context):**
  ```python
  def match_orange_transceivers(bandwidth_units: list[int], gateway_cap: int) -> list[int]:
      """In an Orange 5G core network pool, n transceivers operate with bandwidths b_1...b_n.
      Identify the IDs of the two transceivers whose combined throughput matches gateway_cap."""
  ```
* **L4 (Boundary Constraint Augmentation):**
  ```python
  def match_transceivers_multi_zone(bandwidths: list[int], zones: list[str], cap: int) -> list[int]:
      """Same as L3, but the two selected transceivers must reside in DIFFERENT 
      availability zones (requiring tracking secondary categorical attributes)."""
  ```
* **L5 (Cross-Concept Composition):**
  ```python
  def schedule_optimal_dual_tasks(tasks: list[dict], total_limit: int) -> list[int]:
      """Given tasks with bandwidth demands and execution windows, identify two concurrent 
      tasks summing to total_limit while MINIMIZING total scheduling fragmentation."""
  ```

---

### Theoretical Contrast: Reduction Ladder vs. Code-Rewriting (MRI)

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               THEORETICAL DICHOTOMY                                    │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   Code-Rewriting (MRI Approach - Yang et al., 2025):                                   │
│   ┌────────────────────────┐         ┌────────────────────────┐                        │
│   │ Surface Syntax: STATIC │   ===>  │  Semantics: MUTATED    │                        │
│   └────────────────────────┘         └────────────────────────┘                        │
│   Probes: Does the model blindly regurgitate old code when requirements changed?       │
│                                                                                        │
│   Reduction Ladder (Our Approach - 2026):                                              │
│   ┌────────────────────────┐         ┌────────────────────────┐                        │
│   │ Semantics: INVARIANT   │   ===>  │  Surface Syntax: MUTATED│                       │
│   └────────────────────────┘         └────────────────────────┘                        │
│   Probes: Does the model fail to apply valid logic simply because framing shifted?     │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📚 Comprehensive Literature Taxonomy & Theoretical Foundations

Our theoretical architecture synthesizes five complementary bodies of peer-reviewed literature across AI reasoning, code generation, and reinforcement learning.

```
                                [ REASONING & MITIGATION TAXONOMY ]
                                                 │
        ┌────────────────────────┬───────────────┴────────────────┬────────────────────────┐
        ▼                        ▼                                ▼                        ▼
 [Foundational Limits]   [Code Benchmarks]               [Diagnostic Dimensions]  [Mitigation Arms]
  • OOD Visual Planning   • EvoEval (EMNLP'24)            • OckBench (Token Dens.) • Inv-GRPO (Ours)
  • RLVR Capacity Limits  • Code-Rewriting / MRI (2025)   • Thinking Longer        • Contrastive-SFT
  • The Depth Ceiling     • LiveCodeBench (ICLR'24)       • Attribution Graphs     • AST-RL (TreeDiff)
  • Trapped in Past       • LeetCodeDataset (2025)        • Flip-Flop Consistency  • Step-RLVR (CodePRM)
  • Too Big to Think      • DynaCode / CRUXEval           • Error Taxonomy         • Info-Bottleneck
  • Beyond Memorization   • GSM-Symbolic (ICLR'25)
```

### The Foundational Six Papers: Limits of Reasoning

1. **P1 — OOD Generalization of Reasoning in Multimodal LLMs for Visual Planning (arXiv:2602.15460):** Proved that Chain-of-Thought provides large boosts in-distribution but suffers catastrophic collapse under subtle out-of-distribution shifts, showing models mimic the *contour* of reasoning rather than executing invariant logic.
2. **P2 — Does Reinforcement Learning Really Incentivize Reasoning Capacity Beyond the Base Model? (arXiv:2504.13837):** Formalized the Pass@1 vs. Pass@$k$ exploration methodology; demonstrated that RLVR primarily elevates sampling efficiency for solutions reachable in the base distribution, whereas distillation is the primary mechanism introducing novel capabilities.
3. **P3 — The Depth Ceiling: Limits of LLMs in Discovering Latent Planning (arXiv:2604.06427):** Identified a structural ceiling in latent planning depth (3–7 steps without CoT), demonstrating that explicit CoT raises this ceiling to $\sim$20 steps. Introduced the foundational **on-path vs. off-path** error dichotomy.
4. **P4 — Trapped in the Past? Fluid vs. Crystallized Intelligence via Chess (arXiv:2601.16823):** Built a difficulty taxonomy based on distance from training distribution without needing access to pre-training corpora.
5. **P5 — Too Big to Think: Capacity, Memorization, and Generalization (arXiv:2506.09099):** Proved mathematically and empirically that lower-capacity models (SLMs) possess an inductive bias toward memorizing shortcut templates due to parameter compression bottlenecks.
6. **P6 — Beyond Memorization: Reductive vs. Epistemic Reasoning via Logic Puzzles (arXiv:2603.21350):** Established *reductive reasoning* (reducing novel problems to stored templates) as the primary failure mode of LLMs, inspiring our Reduction Ladder.

### Code-Domain Memorization & Contamination Literature

* **EvoEval (EMNLP 2024):** Created 5 semantic perturbation dimensions; documented an average 38–40% accuracy degradation across 57 state-of-the-art models.
* **Memorize or Generalize? (2025):** Introduced code-rewriting and the Memorization Risk Index (MRI).
* **LLM Performance for Code Generation on Noisy Tasks (2025):** Discovered "eager pattern matching" where models output classic templates upon seeing familiar keyword tokens.
* **LiveCodeBench (ICLR 2024):** Created continuous post-cutoff temporal problem harvesting.
* **GSM-Symbolic (ICLR 2025):** Proved that non-functional prompt variations trigger severe performance collapse in mathematical reasoning models.

### Multi-Dimensional Evaluation Dimensions Beyond Correctness

* **OckBench — Per-Token Intelligence (2025):** Formalized compute density and reasoning efficiency ratios.
* **Thinking Longer, Not Always Smarter (2025):** Quantified the "Overthinking Tax" (models generate up to 45% more tokens on failed attempts).
* **Flip-Flop Consistency (2025):** Measured significant accuracy swings under neutral variable renames.

### State-of-the-Art Mitigation Paradigms (2024–2026)

| Paradigm | Exemplary Literature | Primary Strength | Critical Bottleneck / Limitation |
|---|---|---|---|
| **Process Rewards (PRMs)** | CodePRM (ACL'25), ExecVerify (ICSE'26) | Dense stepwise feedback prevents reward hacking. | High annotation compute; expensive verifiers. |
| **AST Invariance** | TreeDiff (ASE'25), VeriSeek (ICSE'25) | Guarantees syntax and AST consistency. | Restricted to syntax; misses narrative shifts. |
| **Self-Correction** | SuperCorrect (NeurIPS'24) | Dynamic runtime error recovery. | High inference latency overhead. |
| **Information Bottleneck** | IB-FT (EMNLP'24), IBRO (2025) | Theoretical bounds on representation memorization.| Training instability on constrained SLMs. |
| **Inv-GRPO (Ours)** | **This Work (2026)** | **Zero inference latency overhead; directly regularizes invariant reasoning.** | **Requires paired cross-perturbation training batches.** |

---

## 🤖 Model Selection Rationale & Capacity Axis

### Why Qwen2.5-Coder-1.5B-Instruct?

1. **State-of-the-Art in Sub-2B Code Class:** Holds the highest Pass@1 on HumanEval, MBPP, and MultiPL-E among all open-source models under 2B parameters, ensuring we probe a formidable baseline.
2. **The Canonical Distillation Target:** The DeepSeek-R1 project selected Qwen-1.5B as its primary distillation reference (`DeepSeek-R1-Distill-Qwen-1.5B`), establishing it as the global de-facto benchmark for small-model reasoning research.
3. **Ideal Capacity Bottleneck Laboratory:** Operating under parameter constraints, 1.5B models cannot brute-force memorization of millions of surface variants, making shortcut collapse cleanly observable and measurable.
4. **Edge Deployment Feasibility:** Consumes $\approx$ 1.1 GB VRAM at 4-bit NF4 precision, running efficiently on a single NVIDIA RTX 3070 (8GB) edge node.

### Survey of Models in Reference Literature

| Paper / Initiative | Models Evaluated | Alignment with This Repository |
|---|---|---|
| **DeepSeek-R1 (2025)** | Qwen (1.5B, 7B, 14B, 32B), Llama (8B, 70B) | Qwen-1.5B established as canonical distillation target; our primary model. |
| **EvoEval (EMNLP 2024)** | DeepSeek-Coder (1.3B–33B), CodeLlama (7B–34B), StarCoder2 (3B–15B), GPT-4 | Evaluated 57 models; our L1–L5 ladder directly builds upon their datasets. |
| **Memorize or Generalize? (2025)** | Qwen2.5-Coder (1.5B, 7B), DeepSeek-V2-Lite, Llama-3-8B | Used Qwen2.5-Coder-1.5B for MRI; our MRI metrics directly compare. |
| **SuperCorrect (NeurIPS 2024)** | Qwen2.5-Coder (1.5B, 7B), DeepSeek-Coder (1.3B, 6.7B) | Applied contrastive SFT on Qwen-1.5B; our Arm 2 builds on their loss formulation. |
| **TreeDiff / VeriSeek (2025)** | Qwen2.5-Coder (1.5B, 7B), DeepSeek-Coder-1.3B | AST-guided policy rewards on sub-2B models; reproduced in our Arm 3. |
| **CodePRM / ExecVerify (2025/2026)**| Qwen2.5-Coder-7B, DeepSeek-Coder-6.7B, Llama-3-8B | Process rewards for code; adapted into our Arm 4 stepwise verifier. |
| **OckBench (2025)** | Qwen2.5 (1.5B, 7B, 72B), Llama-3 (8B, 70B), GPT-4o | Formalized per-token intelligence; our token efficiency suite matches their formulation. |
| **LiveCodeBench (2024)** | 50+ models including Qwen2.5-Coder family, DeepSeek, GPT-4o | Temporal contamination control set with published Qwen baselines. |

### Three-Tier Capacity Axis

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               MODEL CAPACITY AXIS                                      │
├───────────────────────┬───────────────────────┬────────────────────────────────────────┤
│ Tier 1: Edge SLM      │ Qwen2.5-Coder-1.5B    │ Primary target. M1–M6 variants tested. │
│                       │ (1.5 Billion Params)  │ Severe capacity bottleneck.            │
├───────────────────────┼───────────────────────┼────────────────────────────────────────┤
│ Tier 2: Mid-Range     │ Qwen2.5-Coder-7B      │ Untreated capacity control.            │
│                       │ (7.0 Billion Params)  │ Measures scaling impact on collapse.   │
├───────────────────────┼───────────────────────┼────────────────────────────────────────┤
│ Tier 3: Frontier LLM  │ Frontier Coding API   │ Empirical reasoning ceiling & upper-   │
│                       │ (>100 Billion Params) │ bound benchmark anchor.                │
└───────────────────────┴───────────────────────┴────────────────────────────────────────┘
```

---

## 📊 Cross-Paper Comparison Methodology

### Five-Axis Comparison Framework

| Axis | Metric Name | What It Measures | Benchmark Comparisons |
|:---:|---|---|---|
| **A1** | **Pass@1 (L0–L5)** | Deterministic accuracy per ladder level | EvoEval reported scores, MRI paper baselines, LiveCodeBench leaderboard |
| **A2** | **Collapse Point ($\ell^*$)** | First level where accuracy drops below 50% | EvoEval degradation profiles (median $\ell^* \approx$ L2 for sub-3B models) |
| **A3** | **Error Taxonomy (%)** | Distribution of on-path / off-path / wrong-template | Depth Ceiling error categories, Noisy Code Tasks eager-matching rates |
| **A4** | **Token Efficiency ($\tau$)** | Tokens per correct solution (Per-Token Intelligence)| OckBench density ratios, Thinking Longer overthinking tax rates |
| **A5** | **MRI Score** | Memorization Risk Index quantifying template mimicry| Memorize-or-Generalize reported MRI values |

### Direct Reproduction & Extension of EvoEval Baselines

1. **Baseline Reproduction:** Reproduce published EvoEval Pass@1 for `Qwen2.5-Coder-1.5B` and `7B` across Subtle, ToolUse, Creative, Difficult, and Combine splits.
2. **Mitigation Delta ($\Delta_{\text{mit}}$):**
   $$\Delta_{\text{mit}}(\ell) = \text{Pass@1}_{\text{M}_k}(\ell) - \text{Pass@1}_{\text{M}_1}(\ell)$$
3. **Collapse Point Shift ($\Delta\ell^*$):**
   $$\Delta\ell^* = \ell^*_{\text{M}_k} - \ell^*_{\text{M}_1} \quad (\text{Target: } \Delta\ell^* \ge 2)$$

### Cross-Paper Benchmark Alignment Matrix

| Reference Paper | Reported Metric | Repository Equivalent Output | Comparison Protocol |
|---|---|---|---|
| **EvoEval (2024)** | Pass@1 per perturbation | Pass@1 per ladder level (L1–L5) | Direct split-by-split absolute accuracy delta ($\Delta_{\text{mit}}$). |
| **MRI Paper (2025)** | Memorization Risk Index | $\text{MRI} = \text{Sim} \times \max(0, \Delta\text{Pass@1})$ | Compute on identical L0$\to$L3 pairs; compare MRI reduction. |
| **Depth Ceiling (2026)** | On-path / Off-path ratio | On-path / Off-path / Wrong-Template % | Extend binary taxonomy with 3rd "Wrong-Template" class. |
| **OckBench (2025)** | Per-Token Intelligence | $\text{Accuracy} / \bar{\tau}_{\text{reasoning}} \times 100$ | Compare density curves across parameter tiers. |
| **Thinking Longer (2025)**| Overthinking Tax (%) | $(\bar{\tau}_{\text{fail}} - \bar{\tau}_{\text{pass}}) / \bar{\tau}_{\text{pass}} \times 100$ | Compare against their reported 45% failure overhead. |
| **LiveCodeBench (2024)** | Post-cutoff Pass@1 | Pass@1 on `code_generation_lite` | Zero-contamination baseline validation. |
| **GSM-Symbolic (2025)** | Symbolic perturbation drop | Consistency Delta ($\Delta_c$) | Code-domain analogue of non-functional prompt variation. |
| **Flip-Flop (2025)** | Variable renaming swing | $\Delta_c$ between L1 and L2 | Evaluate sensitivity magnitude under lexical shifts. |

---

## 🛡️ Prioritized Multi-Arm Mitigation Framework (P1–P4)

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                         MULTI-ARM MITIGATION ARCHITECTURE                              │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│  [P1: Inv-GRPO (Primary)] ──► Paired Rollouts (x, x') + Consistency Reward             │
│                                                                                        │
│  [P2: Contrastive-SFT]    ──► CoT Traces paired with Negative Shortcut Rejection       │
│                                                                                        │
│  [P3: AST-RL (Structural)]──► Policy Optimization guided by Tree-Edit Distance simAST  │
│                                                                                        │
│  [P4: Step-RLVR (Process)]──► Stepwise Contract Verification on Sub-Functions          │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Arm 1 (P1 - Primary): Invariance-Regularized GRPO (Inv-GRPO)

During policy rollouts, Inv-GRPO feeds paired semantically equivalent prompts $(x, x')$ (e.g., $x \in L_0$ and $x' \in L_2$):

$$\mathcal{R}_{\text{total}}(y_i, y'_i) = \mathcal{R}_{\text{exec}}(y_i) + \mathcal{R}_{\text{exec}}(y'_i) + \lambda \cdot \mathcal{R}_{\text{consistency}}(y_i, y'_i) - \gamma \cdot \mathcal{P}_{\text{template}}$$

Where:
* $\mathcal{R}_{\text{exec}}(y) \in \{0, 1\}$ is deterministic sandbox unit-test pass/fail.
* $\mathcal{R}_{\text{consistency}}(y_i, y'_i) = \mathbb{I}(\mathcal{R}_{\text{exec}}(y_i) = 1 \land \mathcal{R}_{\text{exec}}(y'_i) = 1)$ explicitly rewards cross-perturbation invariance.
* $\mathcal{P}_{\text{template}}$ penalizes verbatim classic boilerplate generation on perturbed prompts.
* Group relative advantage is computed over the paired group $G$:
  $$\hat{A}_i = \frac{\mathcal{R}_{\text{total}}(y_i, y'_i) - \text{mean}(\mathcal{R})}{\text{std}(\mathcal{R}) + \epsilon}$$

### Arm 2 (P2 - SFT): Contrastive Thought-Template SFT

Inspired by SuperCorrect, we curate 10K reasoning trajectories from `OpenCodeReasoning`, pairing positive step-by-step traces $y^+$ with negative shortcut failure traces $y^-$:

$$\mathcal{L}_{\text{Contrastive}} = -\sum_{t=1}^{T} \log P_\theta(y_t^+ \mid x, y_{<t}^+) + \alpha \max \left( 0, \log P_\theta(y^- \mid x) - \log P_\theta(y^+ \mid x) + m \right)$$

### Arm 3 (P3 - Syntax): AST-Guided Policy Optimization (AST-RL)

Integrates a deterministic Python AST parser into the RL reward loop to reward structural syntax alignment:

$$\mathcal{R}_{\text{AST}}(y) = \exp\left( -\alpha \cdot \text{TreeDist}(\text{AST}(y), \text{AST}(y^*)) \right)$$

### Arm 4 (P4 - Process): Stepwise Execution-Gated RLVR (Step-RLVR)

Checks pre- and post-conditions of sub-functions during sandbox execution, awarding partial reward credits for correct intermediate algorithmic sub-goals on complex L4/L5 problems.

### Intervention Paradigm Analysis: Distillation vs. RLVR vs. RLIR

| Dimension | 1 — Distillation (SFT) | 2 — Outcome RLVR | 3 — Intrinsic RLIR | 4 — Inv-GRPO (Ours) |
|---|---|---|---|---|
| **Reward Source** | External teacher CoT traces | Binary unit-test sandbox ($0/1$) | Self-rewarding model loop | Paired execution + consistency |
| **Introduces New Capability?**| **Yes** (seeds latent exploration paths) | **Mostly No** (improves sampling efficiency) | **No** (refines consistency) | **Yes** (enforces cross-view invariance) |
| **Primary Failure Mode** | Mimics surface formatting without logic | Reward-hacking shortcut patterns | Collapse into degenerate consensus | Paired batch memory requirements |

---

## 🧪 The Six-Model Comparative Experimental Suite (M1–M6)

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              SIX-MODEL EVALUATION MATRIX                               │
├────┬────────────────────────────┬─────────────────────────────┬───────────────────────┤
│ ID │ Model Checkpoint           │ Training Paradigm           │ Expected Collapse Pt. │
├────┼────────────────────────────┼─────────────────────────────┼───────────────────────┤
│ M1 │ Baseline (Untuned)         │ Qwen2.5-Coder-1.5B-Instruct │ ℓ* ≈ L2 (ToolUse)     │
│ M2 │ Vanilla SFT                │ Standard CoT SFT (10K)      │ ℓ* ≈ L3 (Creative)    │
│ M3 │ Contrastive-SFT (Arm 2)    │ SFT + Negative Rejection    │ ℓ* ≈ L3 → L4          │
│ M4 │ Vanilla GRPO               │ Single-Prompt Outcome RLVR  │ ℓ* ≈ L3 (Creative)    │
│ M5 │ AST-RL (Arm 3)             │ GRPO + simAST Reward        │ ℓ* ≈ L4 (Difficult)   │
│ M6 │ Inv-GRPO (Arm 1 - Proposed)│ Paired Multi-View Invariance│ ℓ* ≈ L4 → L5 (Combine)│
└────┴────────────────────────────┴─────────────────────────────┴───────────────────────┘
```

---

## 📐 Mathematical Formulation of Multi-Dimensional Evaluation Metrics

### 1. Pass@1 (Greedy Functional Correctness)
$$\text{Pass@1} = \frac{1}{|D|} \sum_{i=1}^{|D|} \mathbb{I}(\text{Sample}_1(x_i) \text{ passes all unit tests})$$

### 2. Unbiased Pass@$k$ (Sampling Coverage)
$$\text{Pass@}k = \mathbb{E}_{x \sim D} \left[ 1 - \frac{\binom{n - c}{k}}{\binom{n}{k}} \right] \quad (n=20, k=5, T=0.8)$$

### 3. Collapse Point ($\ell^*$) & Ladder AUC ($\mathcal{A}$)
$$\ell^* = \min \left\{ \ell \in \{0, 1, 2, 3, 4, 5\} \mid \text{Pass@1}(\ell) < 0.50 \right\}, \quad \mathcal{A} = \frac{1}{6} \sum_{\ell=0}^{5} \text{Pass@1}(\ell)$$

### 4. Diagnostic Error Taxonomy Heuristics
When a generated sample fails sandbox execution, it is categorized into:
* **On-Path Failure ($\mathcal{E}_{\text{on}}$):** Algorithmic logic correct; execution failed due to minor boundary condition or off-by-one index.
* **Off-Path Failure ($\mathcal{E}_{\text{off}}$):** Algorithmic structure lost; hallucinated control flow or invalid syntax.
* **Wrong-Template Failure ($\mathcal{E}_{\text{template}}$):** Confidently generated a memorized solution to an unperturbed classic problem. *Direct empirical proof of pattern-matching collapse.*

### 5. Token Efficiency & Compute Density
$$\text{Density}(\ell) = \frac{\text{Pass@1}(\ell)}{\bar{\tau}_{\text{reasoning}}(\ell)} \times 100, \quad \text{Overthinking Tax} = \frac{\bar{\tau}_{\text{fail}} - \bar{\tau}_{\text{pass}}}{\bar{\tau}_{\text{pass}}} \times 100$$

### 6. Consistency Delta ($\Delta_c$)
$$\Delta_c = |\text{Pass@1}(L_1) - \text{Pass@1}(L_2)|$$

### 7. Memorization Risk Index (MRI)
$$\text{MRI} = \text{Similarity}(y, y_{\text{template}}) \times \max(0, \text{Pass@1}(L_0) - \text{Pass@1}(L_3))$$

---

## 🏛️ Clean Architecture Software Engineering Blueprint

The codebase enforces strict **Clean Architecture (Separation of Concerns)** across 4 isolated layers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LAYER 4: PRESENTATION & ENTRY POINTS                     │
│    • CLI Scripts: scripts/run_stage*.py   • Notebooks: notebooks/nb_*.ipynb │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ calls
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LAYER 3: APPLICATION SERVICES (Use Cases)                │
│    • DataService               • EvaluationEngine (src/evaluation/)         │
│    • Metrics Calculator        • Publication Plotting Engine                │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ orchestrates
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LAYER 2: DOMAIN CORE (Pure Python Protocols)             │
│    • Entities: BenchmarkTask, ExecutionResult, LevelEvaluationReport        │
│    • Protocols: ICodeExecutor, IBenchmarkLoader, IModelRunner               │
│    • Exceptions: SandboxTimeoutError, ModelInferenceError (Zero ML Imports) │
└──────────────────────────────────────▲──────────────────────────────────────┘
                                       │ implements
┌──────────────────────────────────────┴──────────────────────────────────────┐
│                    LAYER 1: INFRASTRUCTURE (Adapters & IO)                  │
│    • SubprocessSandbox (-X utf8, stdin streaming, timeout isolation)        │
│    • QuantizedModelRunner (4-bit NF4 BitsAndBytes + HF Cache D:\hf_cache)  │
│    • HuggingFaceBenchmarkLoader (JSONL disk persistence)                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📅 Exhaustive 12-Week (3-Month) Execution Roadmap

```
MONTH 1: DIAGNOSTIC FOUNDATION
├── W1: Ingestion & Environment Setup (JSONL ladder caching, schema validation)
├── W2: Subprocess Sandbox Hardening (100% ground-truth verification on L0-L5)
├── W3: Baseline Inference (M1 Qwen-1.5B & 7B Pass@1/5 across all 664 tasks)
└── W4: Collapse Diagnosis (Calculate baseline ℓ*, Ladder AUC, Error Taxonomy)

MONTH 2: SFT & STRUCTURAL MITIGATION
├── W5: SFT Data Curation (10K OpenCodeReasoning + negative shortcut rejection traces)
├── W6: SFT Model Training (Arm 2A Vanilla M2 & Arm 2B Contrastive M3 via QLoRA)
├── W7: SFT Ladder Probing (Compute MRI reduction & compare vs. SuperCorrect)
└── W8: Standard RLVR Training (Arm 0 Vanilla GRPO M4 & Arm 3 AST-RL M5)

MONTH 3: INVARIANCE OPTIMIZATION & SYNTHESIS
├── W9: Inv-GRPO Training (Paired cross-perturbation sampling & consistency rewards)
├── W10: Process RL & Hyperparameter Tuning (Step-RLVR & λ/γ regularizer tuning)
├── W11: Full Suite Benchmarking (Cross-evaluate M1–M6 on Ladder + LiveCodeBench)
└── W12: Synthesis & Delivery (Compile publication figures & report for Dr. Ghada)
```

---

## ⚡ Hardware Feasibility & Edge VRAM Budget (RTX 3070 8GB)

Every experimental pipeline stage is empirically calibrated to run within an **8GB VRAM envelope**:

| Pipeline Stage | Precision / Mode | VRAM Allocation | Hardware Optimization Techniques |
|---|---|:---:|---|
| **Inference / Ladder Eval** | 4-bit NF4 Quantization | $\approx$ **1.1 GB** | BitsAndBytes NF4, batch size 1, stream generation. |
| **QLoRA / Contrastive SFT** | 4-bit Base + LoRA Float16 | $\approx$ **6.5 GB** | LoRA ($r=16, \alpha=32$), Gradient Accumulation = 4, `torch.cuda.empty_cache()`. |
| **GRPO / AST-RL / Inv-GRPO** | 4-bit Policy + Ref Model | $\approx$ **7.2 GB** | Gradient checkpointing, group size $G=4$, interleaved rollout generation. |

---

## 🚀 Installation, Setup & Quickstart Guide

### 1. Clone & Environment Setup
```bash
# Clone the repository
git clone https://github.com/OmarAbdelhamidAly/reduction-ladder-for-code.git
cd reduction-ladder-for-code

# Create & activate environment (Windows PowerShell)
python -m venv .venv
.venv\Scripts\activate

# Install locked dependencies
pip install -r requirements.txt
```

### 2. Configure Cache & Settings
Ensure `config.yaml` points to your preferred storage drive (especially on Windows where `C:` disk space is constrained):
```yaml
storage:
  hf_cache_dir: "D:/hf_cache"
  ladder_cache_dir: "data/ladder"
```

### 3. Run Stage-by-Stage Pipelines

```bash
# STAGE 1: Download and verify all 6 Ladder benchmarks (664 tasks in < 25s)
python scripts/run_stage1_data.py

# STAGE 2: Run Baseline (M1) 4-bit Evaluation on the Reduction Ladder
python scripts/run_stage2_eval.py --model baseline

# STAGE 3 & 4: Curate Contrastive Data and Train SFT Models
python scripts/run_stage3_distill_data.py
python scripts/run_stage4_qlora.py --arm contrastive

# STAGE 5: Train Invariance-Regularized Policy Optimization (Inv-GRPO)
python scripts/run_stage5_inv_grpo.py

# STAGE 6: Generate Publication Plots & Cross-Paper Metric Tables
python scripts/run_stage6_analysis.py
```

### 4. Interactive Jupyter Notebooks
For interactive inspection, error taxonomy drill-downs, and visualization:
* [`notebooks/nb_01_data_pipeline.ipynb`](notebooks/nb_01_data_pipeline.ipynb) — Ingestion & verification.
* [`notebooks/nb_02_baseline_eval.ipynb`](notebooks/nb_02_baseline_eval.ipynb) — Baseline M1 ladder evaluation.
* [`notebooks/nb_03_qlora_training.ipynb`](notebooks/nb_03_qlora_training.ipynb) — SFT & Contrastive training.
* [`notebooks/nb_04_rlvr_training.ipynb`](notebooks/nb_04_rlvr_training.ipynb) — GRPO & Inv-GRPO training.
* [`notebooks/nb_05_comparison.ipynb`](notebooks/nb_05_comparison.ipynb) — Publication-grade comparative plots.

---

## 👥 Research Authors, Supervision & Citation

### Research Authors
* **Omar Abdelhamid** — AI R&D Engineer, Orange Innovation Labs | M.Sc. AI Researcher, Benha University
* **Nour Walid** — AI R&D Engineer, Orange Innovation Labs

### Research Supervision
* **Dr. Ghada Soliman** — Head of Software Engineering & AI Research, Orange Innovation Labs

### BibTeX Citation

```bibtex
@article{abdelhamid2026reductionladder,
  title     = {Reduction Ladder for Code: Probing and Resolving Shortcut Learning vs. Transferable Reasoning in Code SLMs via Multi-Arm Invariance Mitigation},
  author    = {Abdelhamid, Omar and Walid, Nour and Soliman, Ghada},
  journal   = {Technical Research Report -- Orange Innovation Labs AI R\&D},
  year      = {2026},
  institution = {Orange Innovation Labs \& Benha University},
  url       = {https://github.com/OmarAbdelhamidAly/reduction-ladder-for-code}
}
```

---
*Developed with ❤️ at Orange Innovation Labs Egypt. Released under the [MIT License](LICENSE).*
