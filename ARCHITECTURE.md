# Software Architecture & Design Blueprint
**Project:** Diagnosing & Resolving Code SLM Mimicry  
**Organization:** Orange Innovation Labs — AI Research Division  
**Authors:** Omar Abdelhamid, Nour Walid  
**Supervisor:** Dr. Ghada  

---

## Architectural Philosophy

This codebase is designed following **Robert C. Martin's Clean Architecture** principles adapted for scientific machine learning research. The primary objectives are:

1. **Separation of Concerns:** Business logic (reasoning diagnosis, reward computation, evaluation metrics) is completely decoupled from external frameworks (Hugging Face, PyTorch, BitsAndBytes, OS subprocessing).
2. **Dependency Inversion Principle (DIP):** High-level application services depend upon abstract protocols (`Protocol` / `ABC`), not on concrete infrastructure classes.
3. **Reproducibility & Testability:** Every component (sandbox execution, error classification, prompt formatting) can be tested in isolation with mocked inputs.
4. **Team Scalability:** Clear layer boundaries enable multiple researchers to build new mitigation arms (e.g. DPO, PPO, Tree-Search) without modifying existing evaluation pipelines.

---

## Layered Architecture Overview

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
 │     • QLoRATrainerService             • InvGRPOTrainerService           │
 │     • AnalysisService                 • MetricsAggregator               │
 └────────────────────────────────────┬────────────────────────────────────┘
                                      │
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                   DOMAIN CORE (Entities & Protocols)                    │
 │     • BenchmarkTask                   • ExecutionResult                 │
 │     • LevelEvaluationReport           • ErrorCategory                   │
 │     • ICodeExecutor (Protocol)        • IModelRunner (Protocol)         │
 │     • IBenchmarkLoader (Protocol)     • IErrorClassifier (Protocol)     │
 └────────────────────────────────────▲────────────────────────────────────┘
                                      │ (implements)
 ┌────────────────────────────────────┴────────────────────────────────────┐
 │                   INFRASTRUCTURE (External Adapters)                    │
 │     • MultiprocessSandbox (Subprocess Code Isolation)                   │
 │     • HuggingFaceBenchmarkLoader (HF Datasets Loader)                   │
 │     • QuantizedModelRunner (4-bit NF4 BitsAndBytes + PEFT)              │
 │     • RuleBasedErrorClassifier (Failure Mode Taxonomy)                  │
 │     • Atomic File Persistence (JSONL / YAML / JSON)                     │
 └─────────────────────────────────────────────────────────────────────────┘
```

---

## Repository Directory Responsibilities

```
diagnosing-resolving-code-slm-mimicry/
│
├── config.yaml                            # Global single source of truth for all hyperparameters
├── requirements.txt                       # Locked dependencies with exact versions
├── README.md                              # Main project documentation & quickstart
├── ARCHITECTURE.md                        # This software design blueprint
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
│   │   ├── __init__.py
│   │   ├── entities.py                    # BenchmarkTask, ExecutionResult, LevelEvaluationReport, ErrorCategory
│   │   ├── interfaces.py                  # Protocols: ICodeExecutor, IBenchmarkLoader, IModelRunner, IErrorClassifier
│   │   └── exceptions.py                  # Domain Exceptions (SandboxTimeoutError, VRAMExceededError)
│   │
│   ├── infrastructure/                    # Layer 2: External Adapters & IO
│   │   ├── __init__.py
│   │   ├── sandbox.py                     # MultiprocessSandbox (Process isolation, timeout guarding)
│   │   ├── hf_loader.py                   # HuggingFaceBenchmarkLoader (Dataset fetching & disk caching)
│   │   ├── model_loader.py                # QuantizedModelRunner (4-bit NF4 BitsAndBytes + LoRA inference)
│   │   ├── classifier.py                  # RuleBasedErrorClassifier (Failure taxonomy heuristics)
│   │   └── persistence.py                 # Atomic JSONL/JSON/YAML persistence
│   │
│   ├── shared/                            # Shared Cross-Stage Components
│   │   ├── __init__.py
│   │   ├── data_service.py                # Benchmark ingestion & ground-truth verification
│   │   ├── engine.py                      # Unified EvaluationEngine (inference, sandbox, reports)
│   │   ├── metrics.py                     # Ladder AUC, Collapse Point, MRI calculations
│   │   └── plots.py                       # Publication-grade degradation & taxonomy plots
│   │
│   ├── stage2_baseline/                   # Stage 2 Baseline Probing
│   │   ├── __init__.py
│   │   └── evaluation_service.py          # Level & suite evaluation loop
│   │
│   ├── stage3_distillation/               # Stage 3 SFT Distillation
│   │   ├── __init__.py
│   │   ├── dataset_builder.py             # Vanilla CoT distillation builder
│   │   └── contrastive_builder.py         # Shortcut-rejection contrastive builder
│   │
│   ├── stage4_training/                   # Stage 4 QLoRA SFT Training
│   │   ├── __init__.py
│   │   └── qlora_finetune.py              # 4-bit NF4 QLoRA fine-tuner
│   │
│   └── stage5_comparison/                 # Stage 5 Post-Training Comparison
│       ├── __init__.py
│       └── analysis_service.py            # Cross-model summary tables & degradation comparison
│
├── notebooks/                             # Layer 4: Interactive Notebooks
│   ├── nb_01_data_pipeline.ipynb          # Stage 1: Benchmark Data Ingestion & Ground-Truth Verification
│   ├── nb_02_baseline_eval.ipynb          # Stage 2: Baseline (M1) Evaluation & Collapse Probing
│   ├── nb_03_distillation.ipynb           # Stage 3: Vanilla CoT & Contrastive SFT Dataset Construction
│   ├── nb_04_qlora_training.ipynb         # Stage 4: QLoRA Fine-Tuning (M2 Vanilla vs M3 Contrastive)
│   └── nb_05_post_training_eval.ipynb     # Stage 5: Comparative Evaluation & Publication Figures
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

## Team Collaboration Guidelines (For Omar & Nour)

### 1. Adding a New Benchmark Level
To add a new benchmark (e.g. MBPP or a new EvoEval split):
1. Open `src/infrastructure/hf_loader.py` and register the dataset path in `LADDER_DATASET_CONFIGS`.
2. Do **not** modify `DataService` or `EvaluationService` — they operate directly on `BenchmarkTask` entities.

### 2. Adding a New Model or Custom Sampling Strategy
1. Implement the `IModelRunner` protocol in `src/infrastructure/model_loader.py` (e.g. `vLLMRunner` or `APIRunner`).
2. Pass the new runner instance into `EvaluationService(model_runner=...)`.

### 3. Adding a New Mitigation Arm
1. Create your trainer orchestrator in `src/services/training/my_new_arm.py`.
2. Ensure your reward calculation implements `IRewardComputer` from `src/core/interfaces.py`.
3. Create a CLI runner in `scripts/run_stage5_my_arm.py`.

---

## Coding Standards
- **Strict Type Hinting:** All functions must include complete Python type hints (`typing.List`, `Dict`, `Optional`, `Tuple`).
- **No Heavy Frameworks in Domain Core:** `src/core/` must never import `torch`, `transformers`, `datasets`, or `bitsandbytes`.
- **Atomic File Writing:** Always use `src.infrastructure.persistence.save_jsonl` / `save_json` to prevent partial corrupted files during crashes.
- **Process Isolation:** Never run user-generated code with raw `exec()` in the main thread; always use `MultiprocessSandbox` with explicit timeouts.
