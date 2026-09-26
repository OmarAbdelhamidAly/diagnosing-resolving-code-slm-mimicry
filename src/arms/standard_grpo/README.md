# 🔬 Baseline Ablation: Standard GRPO (Vanilla RLVR)

**Project:** Diagnosing and Resolving Code Small Language Model Mimicry via a Reduction Ladder  
**Institution:** Orange Innovation Labs — AI Research & Development Division  
**Authors:** Omar Abdelhamid, Nour Walid (Equal Contribution)  
**Supervisor:** Dr. Ghada Khoriba  
**Model Identifier:** `M4_standard_grpo` | **Base Model:** Qwen2.5-Coder-1.5B-Instruct  
**Target Milestone:** Week 09 Evaluation & Mitigation Suite (Primary Scientific Ablation Anchor)  

---

## 📑 Table of Contents
1. [Executive Summary](#-executive-summary)
2. [Scientific Role: The Essential Ablation Anchor](#-scientific-role-the-essential-ablation-anchor)
3. [Theoretical Foundations & Literature Basis](#-theoretical-foundations--literature-basis)
4. [Mathematical Formulation: Group Relative Policy Optimization](#-mathematical-formulation-group-relative-policy-optimization)
5. [Comparative Analysis: Standard GRPO (M4) vs. Inv-GRPO (M6)](#-comparative-analysis-standard-grpo-m4-vs-inv-grpo-m6)
6. [Engineering Architecture (`trainer.py`)](#-engineering-architecture-trainerpy)
7. [Empirical Hypotheses & Target Metrics](#-empirical-hypotheses--target-metrics)
8. [Hardware & Hyperparameter Specifications](#-hardware--hyperparameter-specifications)
9. [Formal Scientific Bibliography](#-formal-scientific-bibliography)

---

## 📌 Executive Summary

**Model M4 (Standard GRPO)** serves as the critical **ablation baseline** for our research suite. It implements standard Group Relative Policy Optimization (GRPO) as formulated in DeepSeekMath (Shao et al., 2024), evaluating coding prompts in isolation and computing rewards strictly through binary unit-test execution:
$$R_{\text{exec}} \in \{0, 1\}$$

By training Model M4 under identical hardware, LoRA ranks, learning rates, and base model checkpoints as our primary contribution **M6 (Inv-GRPO)**, we isolate and rigorously quantify the exact empirical advantage provided by our **Invariance Regularization term ($\lambda \mathcal{D}_{\text{Inv}}$)**.

---

## 🎯 Scientific Role: The Essential Ablation Anchor

In scientific literature on reinforcement learning for reasoning, claiming an improvement requires establishing an uncompromised ablation anchor:
1. **Did RLVR itself fix the mimicry, or was it our Invariance Regularizer?**
   * If M4 (Standard GRPO) achieves the same $L_0$ preservation and $L_5$ robustness as M6, then shortcut mitigation is a trivial byproduct of test-driven RL.
   * If M4 suffers from representation collapse on perturbed rungs while M6 succeeds, we definitively prove that **Invariance Regularization is mathematically required** to overcome the capacity bottleneck in Small Language Models.
2. **Outcome-Only vs. Cross-Perturbation Feedback:**
   * M4 evaluates prompt $x$ and receives reward $r(y)$. It has no awareness of transformed prompt $x'$.
   * M6 evaluates paired rollouts $(x, x')$ and penalizes divergence between equivalent algorithms.

---

## 📚 Theoretical Foundations & Literature Basis

```
┌────────────────────────────────────────────────────────┐
│           DeepSeekMath (Shao et al., 2024)             │
│        Group Relative Policy Optimization (GRPO)       │
│                                                        │
│  • Eliminates PPO Value Critic Model (saves 50% VRAM)  │
│  • Computes baseline from group of G sampled rollouts   │
│  • Single-prompt isolated evaluation: x -> {y1...yG}   │
└───────────────────────────┬────────────────────────────┘
                            │ Direct Adoption as Baseline
                            ▼
┌────────────────────────────────────────────────────────┐
│             Model M4: Standard GRPO (Ours)             │
│  • Objective: Pure RLVR without Invariance Penalty     │
│  • Evaluates single coding prompts                     │
│  • Reward: R in {0, 1} from SubprocessSandbox          │
│  • Scientific Purpose: Primary Ablation for M6         │
└────────────────────────────────────────────────────────┘
```

1. **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (Shao et al., 2024) [1]:**
   Introduced GRPO to train mathematical reasoning models without requiring a separate critic network, estimating advantage by normalizing rewards across a group of sampled responses.
2. **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (2025) [2]:**
   Demonstrates that large-scale RLVR induces emergent reasoning and self-verification behaviors purely through rule-based outcome verification.
3. **Qwen2.5-Coder Technical Report (Hui et al., Alibaba 2024) [3]:**
   Establishes the foundational instruction-following and code synthesis capabilities of the Qwen2.5-Coder architecture.

---

## 📐 Mathematical Formulation: Group Relative Policy Optimization

Given a coding prompt $x \sim \mathcal{D}$, the policy $\pi_\theta$ generates a group of $G$ candidate solutions $\{y_1, y_2, \dots, y_G\}$.

### 1. Group Relative Advantage Estimation
Each solution $y_i$ is executed in our isolated native `SubprocessSandbox` against unit test suite $T_x$, yielding a binary reward:
$$r_i = \mathbb{I}(\text{Execute}(y_i, T_x) == \text{PASS}) \in \{0, 1\}$$
The relative advantage $\hat{A}_i$ is computed by standardizing rewards across the group:
$$\bar{r} = \frac{1}{G}\sum_{j=1}^G r_j, \quad \sigma = \sqrt{\frac{1}{G}\sum_{j=1}^G (r_j - \bar{r})^2 + \epsilon}$$
$$\hat{A}_i = \frac{r_i - \bar{r}}{\sigma}$$

### 2. Clipped Surrogate Objective
The policy parameters $\theta$ are updated by maximizing the clipped surrogate objective with KL divergence regularization against reference policy $\pi_{\text{ref}}$:
$$\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, \{y_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{G}\sum_{i=1}^G \min\left( \rho_i(\theta)\hat{A}_i, \text{clip}(\rho_i(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_i \right) - \beta \mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) \right]$$
Where:
* $\rho_i(\theta) = \frac{\pi_\theta(y_i \mid x)}{\pi_{\theta_{\text{old}}}(y_i \mid x)}$ is the importance sampling ratio.
* $\epsilon = 0.2$ is the PPO clipping parameter.
* $\beta = 0.04$ is the KL penalty coefficient preventing excessive policy drift.

---

## ⚖️ Comparative Analysis: Standard GRPO (M4) vs. Inv-GRPO (M6)

| Architectural Dimension | Standard GRPO (Model M4) | Inv-GRPO (Model M6 — Ours) |
| :--- | :--- | :--- |
| **Input Structure** | Single isolated prompt $x$ | Paired prompts $(x, x')$ across Ladder levels |
| **Rollout Sampling** | $G$ completions for $x$ | $G/2$ for $x$ and $G/2$ for $x'$ |
| **Reward Function** | Binary execution $\mathcal{R}_{\text{exec}} \in \{0, 1\}$ | $\mathcal{R}_{\text{exec}} + \text{Consistency Bonus} - \text{Decoy Penalty}$ |
| **Cross-Task Regularizer**| None ($\lambda = 0$) | Semantic Invariance Regularizer ($\lambda \mathcal{D}_{\text{Inv}}$) |
| **Shortcut Mitigation** | Indirect (fails tests on $x'$) | Explicit negative gradient on canonical template |
| **Theoretical Goal** | Maximize single-task Pass@1 | Achieve Pareto-optimal Ladder AUC & invariance |

---

## 🛠️ Engineering Architecture (`trainer.py`)

Implemented in [`src/arms/standard_grpo/trainer.py`](file:///c:/Users/Lenovo/Downloads/Reasoning/reo/src/arms/standard_grpo/trainer.py):
* **Class:** `StandardGRPOTrainer`
* **Features:**
  * 4-bit NF4 quantized base model initialization with double quantization.
  * LoRA adapter integration on all attention and MLP projection weights ($r=16, \alpha=32$).
  * Group rollout generation with temperature sampling ($T=0.8$).
  * Isolated subprocess execution via `SubprocessSandbox` with 3.0s timeout and UTF-8 encoding.
  * Dynamic advantage standardization and token-level log-ratio backpropagation.

---

## 📊 Empirical Hypotheses & Target Metrics

| Metric | M1 Baseline | M4 Standard GRPO Target | M6 Inv-GRPO Target | Diagnostic Insight |
| :--- | :---: | :---: | :---: | :--- |
| **$L_0$ HumanEval** | **92.7%** | 88.0% | **92.1%** | Standard GRPO suffers minor canonical regression. |
| **$L_2$ ToolUse** | 84.0% | 88.0% | **89.0%** | Both RL methods improve over zero-shot base. |
| **$L_5$ Combine** | 77.0% | 80.0% | **84.0%** | Inv-GRPO shows distinct advantage on composite rungs. |
| **Ladder AUC** | 81.57% | 83.50% | **86.02%** | $\Delta \text{AUC}_{\text{Inv}} = +2.52\text{ pp}$ proves invariance value. |
| **Degradation Slope**| -0.035 | -0.030 | **-0.019** | Inv-GRPO maintains the flattest degradation curve. |

---

## ⚙️ Hardware & Hyperparameter Specifications

* **Base Model:** `Qwen/Qwen2.5-Coder-1.5B-Instruct`
* **LoRA Rank ($r$):** 16 | **LoRA Alpha ($\alpha$):** 32 | **Dropout:** 0.05
* **Target Modules:** `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
* **Group Size ($G$):** 4 completions per prompt
* **Learning Rate:** $1 \times 10^{-5}$ (AdamW optimizer)
* **PPO Clip Ratio ($\epsilon$):** 0.2 | **KL Penalty ($\beta$):** 0.04
* **Max Generation Length:** 256 tokens | **Temperature:** 0.8
* **Hardware Ceiling:** Peak VRAM $\le 4.9\text{ GB}$ on NVIDIA RTX 3070 Ti 8GB.

---

## 📖 Formal Scientific Bibliography

1. **Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Zhang, M., ... & Guo, D. (2024).** *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* arXiv preprint [arXiv:2402.03300](https://arxiv.org/abs/2402.03300).
2. **DeepSeek-AI. (2025).** *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* arXiv preprint [arXiv:2501.12948](https://arxiv.org/abs/2501.12948).
3. **Hui, B., Yang, J., Cui, Z., Yang, X., Liu, D., Zhang, L., ... & Lin, J. (2024).** *Qwen2.5-Coder Technical Report.* arXiv preprint [arXiv:2409.12186](https://arxiv.org/abs/2409.12186).
