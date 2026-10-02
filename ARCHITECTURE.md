# 🏛️ Software Architecture & Design Blueprint
**Project:** Diagnosing & Resolving Code SLM Mimicry via Invariance-Regularized Policy Optimization  
**Organization:** Orange Innovation Labs — AI Research & Advanced Innovation Division  
**Authors:** Omar Abdelhamid, Nour Walid  
**Academic Supervisor:** Dr. Ghada Soliman  

---

## 1. Architectural Philosophy

This codebase is designed following **Robert C. Martin's Clean Architecture** principles adapted for modern scientific machine learning research. The primary objectives are:

1. **Separation of Concerns:** Core domain logic (evaluation metrics, error classification, reward computation) is decoupled from external execution engines (PyTorch, BitsAndBytes, HuggingFace Transformers, OS Subprocesses).
2. **Dependency Inversion Principle (DIP):** High-level orchestration services depend on abstract protocols (`Protocol` / `ABC`), not concrete infrastructure implementations.
3. **Reproducibility & Determinism:** Every component (sandbox execution, AST normalization, metric calculation) is fully deterministic and testable in isolation without GPU access.
4. **Independent Research Arms:** Each mitigation technique is isolated in its own self-contained module within `src/arms/`, allowing parallel development and ablation without regression risks.

---

## 2. Layered Architecture Diagram

```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                   PRESENTATION & ENTRY POINTS                           │
 │     • notebooks/nb_06_evaluation_suite.ipynb (Master Ladder Benchmark)  │
 │     • notebooks/arm_01_*.ipynb ... arm_04_*.ipynb (Interactive Training)│
 │     • scripts/prepare_kaggle_upload.py & scripts/train_arm4.py          │
 └────────────────────────────────────┬────────────────────────────────────┘
                                      │
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                   APPLICATION & EVALUATION SERVICES                     │
 │     • EvaluationSuite (Batched Inference + Parallel Sandbox Testing)    │
 │     • BenchmarkRegistry (764-Task Pool across L0-L5 + Ctrl)             │
 │     • SuiteReporter (Master Table, Figures, LaTeX Exporter)             │
 │     • Mitigation Trainers (InvGRPO, ContrastiveDPO, AST-RL, StepRLVR)   │
 └────────────────────────────────────┬────────────────────────────────────┘
                                      │
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                   DOMAIN CORE (Zero External ML Dependencies)           │
 │     • BenchmarkTask                   • ExecutionResult                 │
 │     • LevelEvaluationReport           • ErrorCategory                   │
 │     • ICodeExecutor (Protocol)        • IModelRunner (Protocol)         │
 │     • IBenchmarkLoader (Protocol)     • IErrorClassifier (Protocol)     │
 │     • Metrics Engine (9 Pure Literature Functions)                      │
 └────────────────────────────────────▲────────────────────────────────────┘
                                      │ (implements)
 ┌────────────────────────────────────┴────────────────────────────────────┐
 │                   INFRASTRUCTURE & EXTERNAL ADAPTERS                    │
 │     • SubprocessSandbox / MultiprocessSandbox (Process-isolated exec)   │
 │     • QuantizedModelRunner (4-bit NF4 BitsAndBytes + LoRA PEFT)         │
 │     • RuleBasedErrorClassifier (6-Way Failure Mode Heuristics)          │
 │     • extract_code (Torch-Free Markdown & Code Syntax Normalizer)       │
 │     • Centralized Settings (Pydantic v2 + config.yaml + .env)           │
 └─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Repository Directory Responsibilities

```
diagnosing-resolving-code-slm-mimicry/
│
├── README.md                              # Main publication documentation & quickstart
├── ARCHITECTURE.md                        # This software design blueprint
├── WORKFLOW.md                            # Comprehensive stage-by-stage execution workflow
├── config.yaml                            # Global single source of truth for all hyperparameters
├── requirements.txt                       # Locked Python dependencies with CUDA 12.4 compatibility
├── environment.yml                        # Conda environment definition
├── .env.example                           # Template for local environment variables & cache paths
│
├── proposal/                              # Formal LaTeX research proposal for Orange Labs
│   ├── proposal.tex                       # 13-page technical proposal (Orange branding)
│   └── references.bib                     # Academic bibliography
│
├── data/
│   ├── ladder/                            # 764 standardized JSONL benchmark tasks (L0–L5 + Ctrl)
│   │   ├── L0_humaneval_standard.jsonl    # 164 tasks (Canonical HumanEval)
│   │   ├── L1_evoeval_subtle.jsonl        # 100 tasks (Subtle prompt perturbations)
│   │   ├── L2_evoeval_tooluse.jsonl       # 100 tasks (Tool-use & verbose perturbations)
│   │   ├── L3_evoeval_creative.jsonl      # 100 tasks (Creative reasoning tasks)
│   │   ├── L4_evoeval_difficult.jsonl     # 100 tasks (Algorithmic & edge-case challenges)
│   │   ├── L5_evoeval_combine.jsonl       # 100 tasks (Combinatorial & compositional tasks)
│   │   └── Ctrl_livecode_lite.jsonl       # 100 tasks (Temporal OOD control: LiveCodeBench Lite)
│   └── distillation/                      # Filtered Chain-of-Thought corpora for SFT distillation
│
├── src/
│   ├── core/                              # Layer 1: Domain Core (Zero external ML dependencies)
│   │   ├── entities.py                    # BenchmarkTask, ExecutionResult, LevelEvaluationReport
│   │   ├── interfaces.py                  # Protocols: ICodeExecutor, IModelRunner, IErrorClassifier
│   │   ├── exceptions.py                  # Domain Exceptions (SandboxTimeoutError, VRAMExceededError)
│   │   └── config.py                      # Centralized Pydantic v2 Settings Loader
│   │
│   ├── infrastructure/                    # Layer 2: External Adapters & IO
│   │   ├── sandbox.py                     # SubprocessSandbox (Process isolation, UTF-8, auto-indent)
│   │   ├── model_loader.py                # QuantizedModelRunner (4-bit NF4, batched inference, multi-EOS)
│   │   ├── classifier.py                  # RuleBasedErrorClassifier (Failure taxonomy heuristics)
│   │   ├── code_utils.py                  # extract_code (Torch-free markdown & code extraction)
│   │   └── persistence.py                 # Atomic JSONL/JSON streaming utilities
│   │
│   ├── evaluation/                        # Unified Evaluation Framework
│   │   ├── registry.py                    # BenchmarkRegistry (Pre-cached task pool loader)
│   │   ├── metrics.py                     # 9 Pure academic metric functions (Ladder AUC, MRI, etc.)
│   │   ├── suite.py                       # EvaluationSuite (Batched generation + parallel sandbox)
│   │   └── reporter.py                    # SuiteReporter (Publication figures, LaTeX & master table)
│   │
│   └── arms/                              # 5 Modular Mitigation Arms
│       ├── standard_grpo/                 # Baseline: Standard GRPO (Outcome reward only)
│       ├── arm1_inv_grpo/                 # Primary Contribution: Invariance-Regularized Policy Optimization
│       ├── arm2_contrastive_dpo/          # Arm 2: Contrastive Shortcut-Rejection DPO
│       ├── arm3_ast_rl/                   # Arm 3: AST-Guided Structural Policy Optimization
│       └── arm4_step_rlvr/                # Arm 4: Stepwise Contract-Verified Process RLVR
│
├── notebooks/                             # Interactive Research & Execution Notebooks
│   ├── nb_01_data_pipeline.ipynb          # Benchmark Ingestion & Ground-Truth Verification
│   ├── nb_02_baseline_eval.ipynb          # Baseline (M1) Initial Probing
│   ├── nb_03_distillation.ipynb           # CoT Distillation Corpus Creation
│   ├── nb_04_qlora_training.ipynb         # QLoRA Training for M2
│   ├── nb_05_post_training_eval.ipynb     # Comparative Probing
│   ├── nb_06_evaluation_suite.ipynb       # 🏆 Master Evaluation Suite for all 7 Models
│   ├── arm_01_inv_grpo.ipynb              # Training Notebook for Arm 1 (M6)
│   ├── arm_02_contrastive_sft.ipynb       # Training Notebook for Arm 2 (M3)
│   ├── arm_03_ast_rl.ipynb                # Training Notebook for Arm 3 (M5)
│   └── arm_04_step_rlvr.ipynb             # Training Notebook for Arm 4 (M7)
│
├── checkpoints/                           # LoRA Adapter Weights (Saved per arm)
│   ├── qlora_vanilla_adapter/             # M2: Vanilla SFT
│   ├── qlora_contrastive_adapter/         # M3: Contrastive DPO
│   ├── standard_grpo_final/               # M4: Standard GRPO
│   ├── rlvr_ast_final/                    # M5: AST-RL
│   ├── inv_grpo_final/                    # M6: Inv-GRPO (Primary Contribution)
│   └── step_rlvr_final/                   # M7: Step-RLVR
│
├── scripts/                               # Automation & Packaging Utilities
│   ├── prepare_kaggle_upload.py           # Clean checkpoint packaging & benchmark zipping
│   └── train_arm4.py                      # Standalone CLI training runner for Arm 4
│
└── results/                               # Generated Evaluation Artifacts
    ├── master_comparison_table.csv        # Cumulative multi-model standings
    └── <MODEL_ID>/eval_report.json        # Per-model detailed rung reports & task records
```

---

## 4. Design Patterns & Best Practices

### A. Subprocess Sandboxing & Execution Safety
User-generated code from language models can trigger infinite loops, memory leaks, or execution errors.  
- All test runs are executed in an isolated OS process via `SubprocessSandbox`.
- Communication happens over `sys.stdin` to prevent command-line character limit overflow on Windows.
- Per-task timeout enforcement (default: 8.0s) prevents hanging threads.

### B. Batched Generation on Edge & Cloud Hardware
- `QuantizedModelRunner.generate_batch()` uses left-padding (`padding_side="left"`) to evaluate batches of 16 tasks simultaneously.
- Reduces inference latency from ~40s/task to ~2.5s/task on NVIDIA T4 GPUs.
- Multi-token EOS termination (`<|im_end|>`, `<|endoftext|>`) eliminates runaway token generation.

### C. Pure Functional Metrics Engine
- All metrics in `src/evaluation/metrics.py` are stateless, pure mathematical functions.
- Independent of PyTorch, Transformers, or GPU hardware.
- Tested and verifiable directly against the literature (Chen et al. 2021, Jiang et al. 2024, Aly et al. 2026).
