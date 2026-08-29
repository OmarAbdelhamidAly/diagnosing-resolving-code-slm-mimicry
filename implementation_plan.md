# 🧠 Implementation Plan — Reduction Ladder for Code & Multi-Arm Mitigation
**Orange Innovation Labs — Research & Advanced AI Division**  
**Document Version:** 2.2 — Exhaustive Technical Specification  
**Authors:** Omar Abdelhamid, Nour Walid  
**Research Supervisor:** Dr. Ghada — Orange Innovation Labs  
**Target Hardware:** NVIDIA RTX 3070 (8 GB VRAM), 16+ GB System RAM  

---

## 📑 Table of Contents
1. [Executive Summary & Research Objectives](#1-executive-summary--research-objectives)
2. [Verified Reduction Ladder Benchmark Mapping](#2-verified-reduction-ladder-benchmark-mapping)
3. [Full Repository Layout & File Responsibilities](#3-full-repository-layout--file-responsibilities)
4. [Stage 1: Benchmark Ingestion & Sandbox Verification (`nb_01_data_pipeline.ipynb`)](#4-stage-1-benchmark-ingestion--sandbox-verification)
5. [Stage 2: Baseline Evaluation & Diagnostic Error Taxonomy (`nb_02_baseline_eval.ipynb`)](#5-stage-2-baseline-evaluation--diagnostic-error-taxonomy)
6. [Stage 3 & 4: Supervised Fine-Tuning Arms (`nb_03_qlora_training.ipynb`)](#6-stage-3--4-supervised-fine-tuning-arms)
7. [Stage 5: Reinforcement Learning & Multi-Arm Mitigation (`nb_04_rlvr_training.ipynb`)](#7-stage-5-reinforcement-learning--multi-arm-mitigation)
8. [Stage 6: Comparative Multi-Model Analysis & Plots (`nb_05_comparison.ipynb`)](#8-stage-6-comparative-multi-model-analysis--plots)
9. [Hardware Feasibility, Memory Budgets & Execution Commands](#9-hardware-feasibility-memory-budgets--execution-commands)
10. [Risk Register & Contingency Protocols](#10-risk-register--contingency-protocols)

---

## 1. Executive Summary & Research Objectives

### 🎯 The Core Problem
Small Language Models (SLMs) in the 1--3B parameter regime (such as `Qwen2.5-Coder-1.5B-Instruct`) exhibit brittle pattern-matching behavior. While they score high on verbatim benchmarks (L0), their competence collapses when tasks are perturbed (L1--L5). Supervised Fine-Tuning (SFT) often increases format mimicry, while outcome-based Reinforcement Learning (RLVR) exploits shortcut syntax hacks.

### 🔬 Project Goals
1. **Diagnosis:** Quantify the exact transformation level $\ell^*$ where model competence drops below 50\% across six progressive ladder levels.
2. **Mitigation:** Implement and empirically compare a **Prioritized Multi-Arm Mitigation Framework** (P1: Inv-GRPO, P2: Contrastive SFT, P3: AST-RL, P4: Step-RLVR) to raise the collapse point to L4/L5 with zero inference latency overhead.

---

## 2. Verified Reduction Ladder Benchmark Mapping

To eliminate synthetic data generation risks and guarantee 100\% reproducible execution, the 6 ladder levels map directly to peer-reviewed benchmarks on Hugging Face:

| Level | Benchmark Identifier | Hugging Face Dataset Path | Transformation & Probing Objective |
|---|---|---|---|
| **L0** | HumanEval Standard | `openai/openai_humaneval` | Canonical verbatim baseline problem statement. |
| **L1** | EvoEval Subtle | `evoeval/EvoEval_subtle` | Minor wording, format, and input specification shifts. |
| **L2** | EvoEval Verbose | `evoeval/EvoEval_verbose` | Descriptive obfuscation & identifier/variable alteration. |
| **L3** | EvoEval Creative | `evoeval/EvoEval_creative` | Novel narrative context for the identical algorithmic logic. |
| **L4** | EvoEval Difficult | `evoeval/EvoEval_difficult` | Same core algorithm + extra boundary constraints. |
| **L5** | EvoEval Combine | `evoeval/EvoEval_combine` | Multi-algorithmic composition and concept integration. |
| **Control** | LiveCodeBench | `livecodebench/code_generation_lite` | Post-cutoff temporal generalization control set. |

---

## 3. Full Repository Layout & File Responsibilities

```
reduction-ladder-for-code/
│
├── config.yaml                            # Global single source of truth for all hyperparameters
├── requirements.txt                       # Locked dependencies with exact versions
├── README.md                              # Comprehensive project overview & documentation
│
├── proposal/                              # LaTeX proposal for Dr. Ghada (Orange Innovation)
│   ├── proposal.tex                       # Complete 13-page technical proposal
│   └── references.bib                     # 18 curated academic references
│
├── data/
│   ├── ladder/                            # Standardized JSONL benchmark datasets
│   │   ├── L0_humaneval.jsonl             # 164 canonical tasks
│   │   ├── L1_subtle.jsonl               # 164 subtle perturbation tasks
│   │   ├── L2_verbose.jsonl              # 164 verbose/identifier obfuscated tasks
│   │   ├── L3_creative.jsonl             # 164 creative narrative tasks
│   │   ├── L4_difficult.jsonl            # 164 augmented constraint tasks
│   │   └── L5_combine.jsonl              # 164 multi-concept combined tasks
│   ├── distillation/                      # Filtered Chain-of-Thought training corpora
│   │   ├── sft_positive_cot.jsonl         # Positive reasoning trajectories
│   │   └── sft_contrastive_pairs.jsonl    # Paired positive + negative shortcut traces
│   └── livecode_bench/                    # Contamination-free temporal control tasks
│
├── src/
│   ├── data_pipeline/
│   │   ├── loader.py                      # Ingestion & schema normalization from HuggingFace
│   │   └── verifier.py                    # Multi-process sandbox validation for ground-truth
│   │
│   ├── evaluation/
│   │   ├── harness.py                     # Deterministic Pass@1 and Pass@5 execution engine
│   │   ├── sandbox.py                     # Memory-limited, timeout-guarded subprocess runner
│   │   ├── classifier.py                  # Error taxonomy: on-path, off-path, wrong-template
│   │   └── consistency.py                 # Cross-perturbation delta calculation engine
│   │
│   ├── distillation/
│   │   ├── dataset_builder.py             # SFT CoT formatter from OpenCodeReasoning
│   │   └── contrastive_builder.py         # Negative shortcut synthesis for Contrastive-SFT
│   │
│   ├── training/
│   │   ├── qlora_finetune.py              # Arm 2A: Standard 4-bit QLoRA SFT training
│   │   ├── contrastive_sft.py             # Arm 2B (P2): Contrastive Thought-Template SFT
│   │   ├── grpo_rlvr.py                   # Arm 0: Vanilla single-prompt GRPO RLVR
│   │   ├── ast_rl.py                      # Arm 3 (P3): AST-guided structural reward policy
│   │   ├── step_rlvr.py                   # Arm 4 (P4): Stepwise execution-gated reward policy
│   │   └── inv_grpo.py                    # Arm 1 (P1): Invariance-Regularized paired GRPO
│   │
│   └── analysis/
│       ├── metrics.py                     # Ladder AUC, Collapse Point, MRI, Consistency Delta
│       └── plots.py                       # High-resolution publication figure generators
│
├── notebooks/
│   ├── nb_01_data_pipeline.ipynb          # Stage 1: Data ingestion, normalization & validation
│   ├── nb_02_baseline_eval.ipynb          # Stage 2: Baseline (M1) un-tuned model evaluation
│   ├── nb_03_qlora_training.ipynb         # Stage 4: SFT training (M2 Vanilla vs M3 Contrastive)
│   ├── nb_04_rlvr_training.ipynb          # Stage 5: RL training (M4 GRPO vs M5 AST vs M6 Inv-GRPO)
│   └── nb_05_comparison.ipynb            # Stage 6: Comparative multi-model analysis & figures
│
├── scripts/
│   ├── run_stage1_data.py                 # CLI wrapper for Stage 1
│   ├── run_stage2_eval.py                 # CLI wrapper for Stage 2
│   ├── run_stage3_distill_data.py         # CLI wrapper for Stage 3
│   ├── run_stage4_qlora.py                # CLI wrapper for Stage 4
│   ├── run_stage5_rlvr.py                 # CLI wrapper for Stage 5
│   └── run_stage6_analysis.py             # CLI wrapper for Stage 6
│
├── results/                               # Structured output metrics & raw generation logs
│   ├── baseline/                          # M1 raw samples & JSON evaluation logs
│   ├── distilled_vanilla/                 # M2 raw samples & JSON evaluation logs
│   ├── distilled_contrastive/             # M3 raw samples & JSON evaluation logs
│   ├── rlvr_vanilla/                      # M4 raw samples & JSON evaluation logs
│   ├── rlvr_ast/                          # M5 raw samples & JSON evaluation logs
│   └── rlvr_inv_grpo/                     # M6 raw samples & JSON evaluation logs
│
└── checkpoints/                           # Saved LoRA adapter weights & final models
    ├── qlora_vanilla_adapter/
    ├── qlora_contrastive_adapter/
    ├── rlvr_vanilla_final/
    ├── rlvr_ast_final/
    └── rlvr_inv_grpo_final/
```

---

## 4. Stage 1: Benchmark Ingestion & Sandbox Verification

**Notebook:** `notebooks/nb_01_data_pipeline.ipynb`  
**Core Module:** `src/data_pipeline/loader.py`, `src/data_pipeline/verifier.py`

### 📋 Detailed Implementation Steps
1. **Automated Loading:** Use Hugging Face `datasets` library to pull:
   - `openai/openai_humaneval` $\to$ Level 0
   - `evoeval/EvoEval_subtle` $\to$ Level 1
   - `evoeval/EvoEval_verbose` $\to$ Level 2
   - `evoeval/EvoEval_creative` $\to$ Level 3
   - `evoeval/EvoEval_difficult` $\to$ Level 4
   - `evoeval/EvoEval_combine` $\to$ Level 5
   - `livecodebench/code_generation_lite` $\to$ Control set
2. **Schema Normalization Function:**
   ```python
   def normalize_task(raw_item: dict, level: str) -> dict:
       return {
           "task_id": raw_item["task_id"],
           "ladder_level": level,
           "prompt": raw_item["prompt"],
           "canonical_solution": raw_item["canonical_solution"],
           "test": raw_item["test"],
           "entry_point": raw_item.get("entry_point", "")
       }
   ```
3. **Ground-Truth Sandbox Verification:**
   - Execute each `canonical_solution` against its paired `test` block inside a sandboxed subprocess with a 5-second timeout.
   - Assert 100\% pass rate across all 164 tasks per level before serialization.
4. **Persistence:** Save verified tasks as structured JSON Lines in `data/ladder/L{0-5}.jsonl`.

---

## 5. Stage 2: Baseline Evaluation & Diagnostic Error Taxonomy

**Notebook:** `notebooks/nb_02_baseline_eval.ipynb`  
**Core Module:** `src/evaluation/harness.py`, `src/evaluation/sandbox.py`, `src/evaluation/classifier.py`

### 📋 Detailed Implementation Steps
1. **Model Loading (4-Bit NF4):**
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
   import torch

   bnb_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_compute_dtype=torch.bfloat16,
       bnb_4bit_use_double_quant=True
   )
   tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
   model = AutoModelForCausalLM.from_pretrained(
       "Qwen/Qwen2.5-Coder-1.5B-Instruct",
       quantization_config=bnb_config,
       device_map="auto"
   )
   ```
2. **Sampling Protocol:**
   - **Pass@1:** Greedy decoding ($T = 0.0$, `do_sample=False`, `max_new_tokens=1024`).
   - **Pass@5:** Nucleus sampling ($T = 0.8$, $\text{top\_p} = 0.95$, $N = 5$ samples per task).
3. **Execution Sandbox & Isolation:**
   - Run in separate worker subprocesses using `concurrent.futures.ProcessPoolExecutor`.
   - Resource limits enforced via `signal.alarm(10)` or Windows subprocess timeout flags.
4. **Automated Error Taxonomy Classification:**
   ```python
   def classify_failure(generated_code: str, test_output: str, task: dict) -> str:
       if "SyntaxError" in test_output or "IndentationError" in test_output:
           return "syntax_error"
       elif "AssertionError" in test_output:
           # Check for wrong template / memorized boilerplate signatures
           if detect_unrelated_problem_signature(generated_code, task):
               return "wrong_template"
           elif is_near_correct_algorithm(generated_code, task):
               return "on_path"  # Algorithmic logic correct, off-by-one or edge case
           else:
               return "off_path" # Completely erroneous algorithmic structure
       return "runtime_error"
   ```
5. **Output Metrics Computed:** Pass@1, Pass@5, Error Type Ratios, and baseline Collapse Point ($\ell^*$).

---

## 6. Stage 3 & 4: Supervised Fine-Tuning Arms

**Notebook:** `notebooks/nb_03_qlora_training.ipynb`  
**Core Module:** `src/training/qlora_finetune.py`, `src/training/contrastive_sft.py`

### 📋 Arm 2A: Standard QLoRA SFT (`M2: Distilled Vanilla`)
- **Corpus:** 10,000 high-quality Python algorithmic reasoning traces from `OpenCodeReasoning`.
- **QLoRA Hyperparameters:**
  ```yaml
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  learning_rate: 2.0e-4
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.03
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4  # Effective batch = 16
  num_train_epochs: 3
  max_seq_length: 2048
  fp16: false
  bf16: true
  ```

### 📋 Arm 2B (Priority 2): Contrastive Thought-Template SFT (`M3: Contrastive SFT`)
- **Objective:** Fix the SFT format-mimicry bias by teaching the model to reject memorized shortcut templates.
- **Data Construction:**
  - Input: Problem $x$ with potential shortcut decoy prompt.
  - Target CoT:
    ```
    <thought>
    Initial intuition might be to apply the classic two-pointer template from Problem X.
    However, constraint Y in this specification renders that template invalid.
    Instead, we must maintain an invariant dynamic programming state.
    </thought>
    ```
- **Training Loss:** Standard cross-entropy with loss active only on the target reasoning and solution tokens.

---

## 7. Stage 5: Reinforcement Learning & Multi-Arm Mitigation

**Notebook:** `notebooks/nb_04_rlvr_training.ipynb`  
**Core Module:** `src/training/grpo_rlvr.py`, `src/training/inv_grpo.py`, `src/training/ast_rl.py`

### 📋 Arm 1 (Priority 1 — Primary Mitigation): Invariance-Regularized GRPO (`M6: Inv-GRPO`)
- **Batch Construction:** Paired prompt loading $(x, x')$ where $x \in L_0$ and $x' \in L_2$ (or $L_3$).
- **Rollout Generation:** Generate $G=4$ completions for $x$ and $G=4$ completions for $x'$.
- **Dual Invariance Reward Implementation:**
  ```python
  def compute_inv_grpo_reward(y_0, y_pert, test_0, test_pert) -> float:
      pass_0 = float(sandbox_run(y_0, test_0))
      pass_pert = float(sandbox_run(y_pert, test_pert))
      
      # Consistency bonus
      consistency = 1.0 if (pass_0 == 1.0 and pass_pert == 1.0) else 0.0
      
      # Template penalty
      penalty = 0.5 if detect_template_collapse(y_pert) else 0.0
      
      # Total reward
      return (pass_0 + pass_pert) + 0.5 * consistency - penalty
  ```
- **Joint Advantage Estimation:**
  $$\hat{A}_i = \frac{\mathcal{R}_{\text{total}}(y_i, y'_i) - \mu_{\mathcal{R}}}{\sigma_{\mathcal{R}} + 10^{-8}}$$
- **KL Coefficient $\beta$:** $0.1$ against the SFT reference model.

### 📋 Arm 3 (Priority 3): AST-Guided Policy Optimization (`M5: AST-RL`)
- **Reward Formulation:**
  $$\mathcal{R}(y) = \mathcal{R}_{\text{exec}}(y) + 0.3 \cdot \text{simAST}(\text{AST}(y), \text{AST}(y^*))$$
- **AST Tree Distance:** Calculated using `zss` (Zhang-Shasha tree edit distance) on Python `ast.parse` nodes to reward correct algorithmic tree structures under variable renaming.

### 📋 Arm 0 (Baseline RLVR): Vanilla GRPO (`M4: Vanilla GRPO`)
- Standard outcome-based binary reward on single isolated prompts ($R \in \{0, 1\}$).

---

## 8. Stage 6: Comparative Multi-Model Analysis & Plots

**Notebook:** `notebooks/nb_05_comparison.ipynb`  
**Core Module:** `src/analysis/metrics.py`, `src/analysis/plots.py`

### 📊 Mathematical Metric Formulations

1. **Pass@1 \& Pass@5 ($p_1, p_5$):**
   $$p_1 = \frac{1}{|D|} \sum_{i=1}^{|D|} \mathbb{I}(\text{sample}_1 == \text{PASS}), \quad p_5 = \frac{1}{|D|} \sum_{i=1}^{|D|} \left( 1 - \frac{\binom{n-c}{5}}{\binom{n}{5}} \right)$$
2. **Collapse Point ($\ell^*$):**
   $$\ell^* = \min \left\{ \ell \in \{0, 1, 2, 3, 4, 5\} \mid \text{pass@1}(\ell) < 0.50 \right\}$$
3. **Ladder Area Under Curve ($\mathcal{A}$):**
   $$\mathcal{A} = \frac{1}{6} \sum_{\ell=0}^{5} \text{pass@1}(\ell)$$
4. **Consistency Delta ($\Delta_c$):**
   $$\Delta_c = |\text{pass@1}(L_1) - \text{pass@1}(L_2)|$$
5. **Memorisation Risk Index (MRI):**
   $$\text{MRI} = \text{Similarity}(y, y_{\text{canonical}}) \times \max(0, \text{pass@1}(L_0) - \text{pass@1}(L_3))$$

### 📈 Generated Publication Figures
1. `fig1_ladder_accuracy_curves.pdf`: Pass@1 vs Level (L0--L5) for M1 through M6.
2. `fig2_collapse_point_histogram.pdf`: Distribution of $\ell^*$ across problem paradigms.
3. `fig3_error_taxonomy_breakdown.pdf`: Stacked bar chart (On-path vs Off-path vs Wrong-template).
4. `fig4_consistency_delta_comparison.pdf`: Sensitivity to surface perturbations across arms.
5. `fig5_livecodebench_temporal_control.pdf`: Unseen post-cutoff generalization verification.

---

## 9. Hardware Feasibility, Memory Budgets & Execution Commands

### ⚡ VRAM Budget on NVIDIA RTX 3070 (8 GB GDDR6)

| Phase / Arm | Base Model | Precision | Batch / Group | VRAM Peak | Feasibility Strategy |
|---|---|---|---|---|---|
| **Inference (Stage 2)** | Qwen2.5-Coder-1.5B | 4-bit NF4 | Batch = 1 | $\approx$ 3.8 GB | Native `bitsandbytes` 4-bit |
| **SFT (Stage 4)** | Qwen2.5-Coder-1.5B | 4-bit NF4 | Batch = 4, GA = 4 | $\approx$ 6.4 GB | LoRA rank $r=16$, Target modules only |
| **Vanilla GRPO (Stage 5)** | Qwen2.5-Coder-1.5B | 4-bit NF4 | Group $G=4$, Batch=1 | $\approx$ 7.1 GB | Gradient Checkpointing enabled |
| **Inv-GRPO (Stage 5)** | Qwen2.5-Coder-1.5B | 4-bit NF4 | Group $G=4$, Batch=1 | $\approx$ 7.3 GB | Sequential paired rollout generation |

### 🚀 CLI Execution Workflow
```bash
# 1. Activate Environment
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Execute Data Pipeline
python scripts/run_stage1_data.py

# 3. Evaluate Baseline (M1)
python scripts/run_stage2_eval.py --model baseline

# 4. Train SFT Arms (M2 & M3)
python scripts/run_stage3_distill_data.py
python scripts/run_stage4_qlora.py --variant vanilla
python scripts/run_stage4_qlora.py --variant contrastive

# 5. Train RL Arms (M4, M5, M6)
python scripts/run_stage5_rlvr.py --algorithm vanilla_grpo
python scripts/run_stage5_rlvr.py --algorithm ast_rl
python scripts/run_stage5_rlvr.py --algorithm inv_grpo

# 6. Run Comparative Diagnostics & Generate Report
python scripts/run_stage6_analysis.py
```

---

## 10. Risk Register & Contingency Protocols

| Risk Identifier | Severity | Likelihood | Mitigation & Contingency Protocol |
|---|---|---|---|
| **VRAM Out-Of-Memory (OOM)** | High | Medium | Reduce group size $G$ from 4 to 2; enable CPU offloading for optimizer states. |
| **Reward Hacking in RL** | High | Low | Penalize empty/boilerplate outputs via $\mathcal{P}_{\text{template}}$ and enforce timeout assertions. |
| **AST Parsing Failures on Malformed Code** | Medium | Low | Return $\text{simAST} = 0.0$ on `SyntaxError` without crashing the training loop. |
| **Evaluation Sandbox Hangs** | Medium | Low | Isolated subprocess worker with hard kill `signal.SIGKILL` after 10.0s elapsed time. |
