# 🔬 Arm 2: Contrastive SFT & Direct Preference Optimization (Contrastive-DPO)

**Project:** Diagnosing and Resolving Code Small Language Model Mimicry via a Reduction Ladder  
**Institution:** Orange Innovation Labs — AI Research & Development Division  
**Authors:** Omar Abdelhamid, Nour Walid (Equal Contribution)  
**Supervisor:** Dr. Ghada Khoriba  
**Model Identifier:** `M3_contrastive_dpo` | **Base Model:** Qwen2.5-Coder-1.5B-Instruct  
**Target Milestone:** Week 09 Evaluation & Mitigation Suite  

---

## 📑 Table of Contents
1. [Executive Summary](#-executive-summary)
2. [Problem Statement: Why Vanilla SFT Induces Shortcut Mimicry](#-problem-statement-why-vanilla-sft-induces-shortcut-mimicry)
3. [Theoretical Foundations & Literature Gap](#-theoretical-foundations--literature-gap)
4. [Mathematical Formulation: Contrastive Loss & DPO](#-mathematical-formulation-contrastive-loss--dpo)
5. [Contrastive Pair Dataset Architecture](#-contrastive-pair-dataset-architecture)
6. [Engineering Implementation Walkthrough](#-engineering-implementation-walkthrough)
7. [Empirical Hypotheses & Target Metrics](#-empirical-hypotheses--target-metrics)
8. [Hardware & Hyperparameter Specifications](#-hardware--hyperparameter-specifications)
9. [Formal Scientific Bibliography](#-formal-scientific-bibliography)

---

## 📌 Executive Summary

**Arm 2 (Contrastive SFT / DPO)** addresses the primary vulnerability identified in Small Language Models (~1.5B parameters): **shortcut memorization**. When an SLM is fine-tuned exclusively on positive demonstration pairs $(x, y^+)$, it frequently learns superficial token associations rather than genuine algorithmic invariant rules. When presented with a perturbed task $x'$ (such as an API restriction or renamed variables), the model blindly emits memorized canonical code from pre-training benchmarks (e.g., HumanEval $L_0$).

Arm 2 reframes mitigation as a **preference-guided unlearning problem**. By pairing each perturbed problem statement $x'$ with:
1. A **chosen completion ($y^+$)**: An invariant reasoning path that explicitly acknowledges the constraint shift and derives the correct algorithm.
2. A **rejected decoy shortcut ($y^-$)**: The verbatim memorized canonical solution that solves the unperturbed problem but fails under the altered specification.

We optimize the policy $\pi_\theta$ using Direct Preference Optimization (DPO) and Contrastive SFT, applying a direct negative gradient penalty to the memorized decoy shortcut while increasing the log-likelihood of the invariant program.

---

## 🎯 Problem Statement: Why Vanilla SFT Induces Shortcut Mimicry

In Week 09, our foundational benchmarks revealed the **Mimicry Dilemma**:
* **Baseline M1 ($L_0$ HumanEval):** $92.7\%$
* **Vanilla SFT M2 ($L_0$ HumanEval):** $64.6\%$ ($\Delta = -28.1\text{ pp}$ collapse)
* **Vanilla SFT M2 ($L_2$–$L_5$ Complex Rungs):** $+13$ to $+18\text{ pp}$ gain over M1.

### Root-Cause Diagnosis
Vanilla SFT maximizes likelihood over positive reasoning traces:
$$\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(x, y)} \left[ \sum_{t=1}^T \log \pi_\theta(y_t \mid x, y_{<t}) \right]$$
Because the model has finite parameter capacity ($1.5\text{B}$), optimizing likelihood over transformed rungs without explicitly penalizing canonical shortcuts creates catastrophic interference. The model unlearns standard canonical structures while failing to generalize to subtle shifts.

---

## 📚 Theoretical Foundations & Literature Gap

Arm 2 bridges three seminal works in preference learning and code unlearning:

```
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│       DPO (Rafailov, 2023)      │       │     SuperCorrect (ICLR 2025)    │
│  Closed-Form Preference Loss    │       │   Unlearning Erroneous Shortcut │
│    [Elimination of Reward Model]│       │   [Targeted Negative Gradients] │
└────────────────┬────────────────┘       └────────────────┬────────────────┘
                 │                                         │
                 └────────────────────┬────────────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │    Arm 2: Contrastive-DPO     │
                      │   x  : Perturbed Problem      │
                      │   y+ : Invariant Algorithm    │
                      │   y- : Memorized L0 Decoy     │
                      └───────────────────────────────┘
                                      ▲
                                      │
                 ┌────────────────────┴────────────────────┐
                 │                                         │
┌────────────────┴────────────────┐       ┌────────────────┴────────────────┐
│   Contrastive CoT (Chia, 2023)  │       │       ReCode (ASE 2025)         │
│  Contrastive Reasoning Traces   │       │   Code Robustness & Perturbation│
│  [Invalid vs Valid Logic]       │       │   [Syntactic Decoy Generation]  │
└─────────────────────────────────┘       └─────────────────────────────────┘
```

1. **Direct Preference Optimization (Rafailov et al., NeurIPS 2023) [1]:**
   Eliminates the unstable Reinforcement Learning from Human Feedback (RLHF) reward model by deriving the optimal policy directly from the Bradley-Terry preference model in closed form.
2. **SuperCorrect: Error-Driven Alignment for Code (ICLR 2025) [2]:**
   Demonstrates that fine-tuning models on paired error corrections where the negative sample represents common model hallucinations drastically reduces repeat errors compared to positive-only SFT.
3. **Contrastive Chain-of-Thought (Chia et al., 2023) [3]:**
   Proves that conditioning language models on contrastive pairs (demonstrating why a plausible but incorrect reasoning path fails) significantly sharpens reasoning boundaries on logical benchmarks.
4. **ReCode: Robustness Benchmark for Code Models (Wang et al., 2023) [4]:**
   Formalizes code transformations (syntax, docstring, format) and demonstrates that top models degrade by up to $30\%$ under semantics-preserving perturbations.

---

## 📐 Mathematical Formulation: Contrastive Loss & DPO

Given a dataset of triplets $\mathcal{D}_{\text{contrastive}} = \{(x_i, y_i^+, y_i^-)\}_{i=1}^N$, where:
* $x_i$: The problem prompt under a Diagnostic Reduction Ladder transformation (e.g., $L_2$ ToolUse or $L_4$ Difficult).
* $y_i^+$: Ground-truth invariant implementation that satisfies the altered constraints.
* $y_i^-$: The memorized decoy shortcut from canonical $L_0$ that fails the test suite.

### 1. Direct Preference Optimization Loss
We optimize the policy $\pi_\theta$ against frozen reference policy $\pi_{\text{ref}}$ (Model M1):
$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y^+, y^-) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y^+ \mid x)}{\pi_{\text{ref}}(y^+ \mid x)} - \beta \log \frac{\pi_\theta(y^- \mid x)}{\pi_{\text{ref}}(y^- \mid x)} \right) \right]$$
Where:
* $\beta \in [0.05, 0.2]$ is the temperature controlling divergence from the reference policy.
* $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid activation.

### 2. Gradient Dynamics on Shortcuts
The gradient with respect to parameter $\theta$ decomposes into:
$$\nabla_\theta \mathcal{L}_{\text{DPO}}(\theta) = -\beta \sigma(\hat{r}_\theta(x, y^-) - \hat{r}_\theta(x, y^+)) \left[ \nabla_\theta \log \pi_\theta(y^+ \mid x) - \nabla_\theta \log \pi_\theta(y^- \mid x) \right]$$
Where $\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$ is the implicit reward.
* **Positive Gradient:** Pushes probability mass toward the invariant synthesis $y^+$.
* **Negative Gradient:** Actively suppresses the token logits corresponding to the memorized canonical shortcut $y^-$.

---

## 📦 Contrastive Pair Dataset Architecture

The contrastive dataset is constructed via [`src/arms/arm2_contrastive_dpo/dataset.py`](file:///c:/Users/Lenovo/Downloads/Reasoning/reo/src/arms/arm2_contrastive_dpo/dataset.py) from the distillation corpus:

```json
{
  "task_id": "EvoEval_ToolUse_042",
  "ladder_level": "L2",
  "prompt": "def find_max_subarray(nums: List[int], logger: DiagnosticLogger) -> int:\n    \"\"\"Find max subarray sum and log each step using logger.record().\"\"\"",
  "chosen": "    logger.record('init', nums)\n    max_so_far = nums[0]\n    curr_max = nums[0]\n    for x in nums[1:]:\n        curr_max = max(x, curr_max + x)\n        max_so_far = max(max_so_far, curr_max)\n        logger.record('step', curr_max)\n    return max_so_far",
  "rejected": "    max_so_far = nums[0]\n    curr_max = nums[0]\n    for x in nums[1:]:\n        curr_max = max(x, curr_max + x)\n        max_so_far = max(max_so_far, curr_max)\n    return max_so_far"
}
```

* **Notice:** The `rejected` sample represents the exact HumanEval $L_0$ canonical implementation. It computes the correct mathematical value but completely ignores the `logger` interface constraint, representing the classic shortcut mimicry failure mode.

---

## 🛠️ Engineering Implementation Walkthrough

The module is structured under `src/arms/arm2_contrastive_dpo/`:
* [`dataset.py`](file:///c:/Users/Lenovo/Downloads/Reasoning/reo/src/arms/arm2_contrastive_dpo/dataset.py): Dataset parser converting JSONL records into HuggingFace/TRL DPO preference triplets (`prompt`, `chosen`, `rejected`).
* [`trainer.py`](file:///c:/Users/Lenovo/Downloads/Reasoning/reo/src/arms/arm2_contrastive_dpo/trainer.py): Production 500-step QLoRA trainer implementing contrastive fine-tuning.
* [`notebooks/arm_02_contrastive_sft.ipynb`](file:///c:/Users/Lenovo/Downloads/Reasoning/reo/notebooks/arm_02_contrastive_sft.ipynb): End-to-end training notebook with token length distribution plots and loss tracking.

---

## 📊 Empirical Hypotheses & Target Metrics

| Metric | M1 Baseline | M2 Vanilla SFT | M3 Contrastive Target | Scientific Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **$L_0$ HumanEval** | **92.7%** | 64.6% | **$\ge 88.0\%$** | Negative decoy penalty prevents canonical distribution collapse. |
| **$L_2$ ToolUse** | 84.0% | **96.0%** | **$\ge 92.0\%$** | Preserves high complex-rung execution accuracy. |
| **Ladder AUC** | 81.57% | 83.33% | **$\ge 85.0\%$** | Substantial overall improvement across the full spectrum. |
| **Degradation Slope**| -0.035 | -0.041 | **$\ge -0.025$** | Flatter curve indicating genuine algorithmic invariance. |

---

## ⚙️ Hardware & Hyperparameter Specifications

* **Base Model:** `Qwen/Qwen2.5-Coder-1.5B-Instruct`
* **LoRA Rank ($r$):** 16 | **LoRA Alpha ($\alpha$):** 32 | **Dropout:** 0.05
* **Target Modules:** `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
* **DPO Beta ($\beta$):** 0.1
* **Learning Rate:** $2 \times 10^{-4}$ (Cosine schedule with 10% warmup)
* **Optimization Steps:** 500 steps | **Batch Size:** 2 (gradient accumulation = 8)
* **Quantization:** 4-bit NF4 with double quantization (bitsandbytes)
* **Hardware Ceiling:** Peak VRAM $\le 5.2\text{ GB}$ on NVIDIA RTX 3070 Ti 8GB.

---

## 📖 Formal Scientific Bibliography

1. **Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023).** *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* Advances in Neural Information Processing Systems (NeurIPS 2023). [arXiv:2305.18290](https://arxiv.org/abs/2305.18290).
2. **Li, Z., et al. (2025).** *SuperCorrect: Supervising Self-Correction in Language Models via Contrastive Feedback.* International Conference on Learning Representations (ICLR 2025).
3. **Chia, Y. K., et al. (2023).** *Contrastive Chain-of-Thought Prompting.* Association for Computational Linguistics (ACL 2023).
4. **Wang, S., et al. (2023).** *ReCode: Robustness Evaluation of Code Generation Models.* Annual Meeting of the Association for Computational Linguistics (ACL 2023). [arXiv:2212.10264](https://arxiv.org/abs/2212.10264).
