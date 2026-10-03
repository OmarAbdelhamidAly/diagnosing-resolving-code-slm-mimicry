# Arm 5: $S^3$-GRPO — Structural, Stepwise & Invariant Policy Optimization

**Model Identifier:** `M8_hybrid_s3_grpo`  
**Checkpoint Output:** `checkpoints/s3_grpo_final/`  
**Base Architecture:** `Qwen/Qwen2.5-Coder-1.5B-Instruct` (4-bit NF4 QLoRA, $r=16, \alpha=32$)  
**Primary Research Goal:** Synthesize process rewards, syntactic tree alignment, and invariance regularizers to completely eradicate **Reasoning Bloat (Overthinking Tax)** and **Supervised Mimicry Collapse**.

---

## 1. Empirical Motivation & Research Gap

In our systematic 7-model Reduction Ladder evaluation across **5,348 real execution trials**, four critical phenomena were quantitatively discovered:

1. **The Supervised Mimicry Dip (M2 Vanilla SFT):**  
   Naive Chain-of-Thought distillation caused a catastrophic **-26.3 pp collapse on canonical HumanEval (L0: 90.9% → 64.6%)**, while inflating token length by 264% (`Overthinking Tax = 2.311`). Naive imitation causes severe memorization interference on straightforward algorithmic tasks.
2. **Reward Hacking via Verbosity in Standard GRPO (M4):**  
   Standard outcome-only GRPO ($R \in \{0, 1\}$) achieved 97.9% in-distribution AUC, but at the cost of **massive verbosity hacking (`Overthinking Tax = 2.724`)** and poor out-of-distribution transfer (**LiveCodeBench OOD: 6.0%**). The policy learned to generate lengthy boilerplate to statistically maximize assertion coverage.
3. **AST Tree Alignment Eliminates Reasoning Bloat (M5 AST-RL):**  
   Enforcing Abstract Syntax Tree similarity ($sim_{AST}$) cut the Overthinking Tax by **51.6% (1.318 vs 2.724)** while achieving **100% on L3** and attaining the **highest OOD generalization across all models (10.0%)**.
4. **Stepwise Process Rewards Solve Credit Assignment (M7 Step-RLVR):**  
   Evaluating independent sub-assertions awarded dense partial credits, unlocking **99.0% on L1** and **100.0% on L3**.

### The Research Question:
> *Can we formulate a unified, multi-objective policy gradient framework that inherits the dense credit of Step-RLVR and the structural parsimony of AST-RL, while penalizing runaway reasoning tokens to achieve state-of-the-art accuracy with minimal inference cost?*

---

## 2. Mathematical Formulation

$S^3$-GRPO optimizes the policy $\pi_\theta$ using Group Relative Policy Optimization (GRPO) over a group of $G$ sampled rollouts $\{\hat{y}_1, \dots, \hat{y}_G\}$ for prompt $x$:

$$\mathcal{L}_{S^3\text{-GRPO}}(\theta) = -\frac{1}{G} \sum_{i=1}^G \left[ \min\left( \frac{\pi_\theta(\hat{y}_i \mid x)}{\pi_{\text{old}}(\hat{y}_i \mid x)} \hat{A}_i, \, \text{clip}\left(\frac{\pi_\theta(\hat{y}_i \mid x)}{\pi_{\text{old}}(\hat{y}_i \mid x)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_i \right) - \beta_{\text{KL}} D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) \right]$$

### Composite Reward Objective:
For each rollout $\hat{y}_i$, the reward combines four distinct signals:

$$\mathcal{R}_{\text{total}}(\hat{y}_i, x) = w_{\text{step}} \cdot \mathcal{R}_{\text{step}}(\hat{y}_i) + w_{\text{ast}} \cdot \text{sim}_{\text{AST}}(\hat{y}_i, y^*) - w_{\text{inv}} \cdot \mathcal{L}_{\text{inv}}(x, x') - w_{\text{tax}} \cdot \Omega_{\text{parsimony}}(\hat{y}_i, y^*)$$

Where:

### 1. Dense Stepwise Process Reward ($\mathcal{R}_{\text{step}}$)
Evaluates $K$ unit test assertions independently via `StepwiseContractVerifier`:
$$\mathcal{R}_{\text{step}}(\hat{y}_i) = \frac{1}{K} \sum_{k=1}^K \mathbb{I}\left[\text{exec}(\hat{y}_i, \text{assert}_k) == \text{PASS}\right] \in [0, 1]$$

### 2. Normalized AST Structural Similarity ($\text{sim}_{\text{AST}}$)
Measures syntactic alignment against canonical reference $y^*$ after variable anonymization:
$$\text{sim}_{\text{AST}}(\hat{y}_i, y^*) = 0.6 \cdot \text{Jaccard}(\text{AST}(\hat{y}_i), \text{AST}(y^*)) + 0.4 \cdot \frac{\min(|\hat{y}_i|_{\text{nodes}}, |y^*|_{\text{nodes}})}{\max(|\hat{y}_i|_{\text{nodes}}, |y^*|_{\text{nodes}})} \in [0, 1]$$

### 3. Cross-Prompt Invariance Regularizer ($\mathcal{L}_{\text{inv}}$)
Penalizes divergence in functional execution under surface perturbations $(x, x')$:
$$\mathcal{L}_{\text{inv}} = \left| \mathcal{R}_{\text{step}}(\hat{y}_i \mid x) - \mathcal{R}_{\text{step}}(\hat{y}_i' \mid x') \right|$$

### 4. Information-Theoretic Parsimony Penalty ($\Omega_{\text{parsimony}}$)
Directly suppresses boilerplate hallucination and reasoning bloat:
$$\Omega_{\text{parsimony}}(\hat{y}_i, y^*) = \min\left(1.0, \, \max\left(0, \frac{|\text{Tokens}(\hat{y}_i)| - |\text{Tokens}(y^*)|}{|\text{Tokens}(y^*)|}\right)\right)$$

### 5. Group Relative Advantage ($\hat{A}_i$)
$$\hat{A}_i = \frac{\mathcal{R}_{\text{total}}(\hat{y}_i) - \text{mean}(\{\mathcal{R}_{\text{total}}\})}{\text{std}(\{\mathcal{R}_{\text{total}}\}) + \epsilon}$$

---

## 3. Implementation & Architecture Mapping

All components follow Robert C. Martin's Clean Architecture and are isolated in `src/arms/arm5_hybrid_s3/`:

| Component | Source File | Exact Class / Function | Responsibility |
|---|---|---|---|
| **Reward Engine** | `reward_engine.py` | `S3RewardEngine` | Computes 4-component composite reward & group advantages |
| **Step Verifier** | `arm4_step_rlvr/verifier.py` | `StepwiseContractVerifier` | Subprocess assertion instrumentation & partial credit |
| **AST Normalizer** | `arm3_ast_rl/ast_engine.py` | `simAST`, `ASTNormalizer` | AST node anonymization & Jaccard tree distance |
| **Trainer** | `trainer.py` | `S3GRPOTrainer` | 500-step QLoRA GRPO with micro-batched backward passes |
| **CLI Runner** | `scripts/train_arm5_s3.py` | Standalone script | Production training runner with device auto-detection |
| **Notebook** | `notebooks/arm_05_hybrid_s3.ipynb` | Jupyter Notebook | Kaggle T4/RTX 3070 interactive training & curves |

---

## 4. Hyperparameter Calibration

Calibrated to fit within an **8 GB VRAM envelope** on consumer edge hardware (RTX 3070) or cloud T4 (16 GB):

| Hyperparameter | Value | Scientific Rationale |
|---|:---:|---|
| Base Model | `Qwen2.5-Coder-1.5B-Instruct` | Target SLM under reasoning study |
| Quantization | 4-bit NF4 double quant | Keeps base footprint at ~1.1 GB VRAM |
| LoRA Rank $r$ | 16 | Sufficient rank for RL policy adaptation |
| LoRA Alpha $\alpha$ | 32 | Scaling factor = 2.0 |
| Target Modules | `all-linear` (7 projection layers) | Captures full attention & MLP policy dynamics |
| Optimizer | AdamW ($\beta_1=0.9, \beta_2=0.999$) | Stable policy gradients |
| Learning Rate | $1.0 \times 10^{-5}$ | Conservative rate preventing catastrophic forgetting |
| Group Size $G$ | 4 rollouts | Optimal balance of advantage variance vs VRAM |
| Max Generation Tokens | 384 tokens | Enforces conciseness (sufficient for all benchmarks) |
| $w_{\text{step}}$ (Stepwise weight) | 0.50 | Primary functional correctness signal |
| $w_{\text{ast}}$ (AST weight) | 0.30 | Inductive syntactic structure bias |
| $w_{\text{inv}}$ (Invariance weight) | 0.20 | Cross-view surface robustness |
| $w_{\text{parsimony}}$ (Tax weight) | 0.15 | Anti-overthinking length regularization |
| Training Steps | 500 steps | Matches M4, M5, and M7 budget for fair comparison |

---

## 5. Expected Empirical Impact (The Research Hypothesis)

```
                     Ladder AUC % (Higher is Better)
M1 Baseline      [77.4%]
M4 Std GRPO      [97.9%] ──► But Overthinking Tax = 2.724 (Heavy Bloat!)
M5 AST-RL        [96.2%] ──► Overthinking Tax = 1.318 (Clean, OOD = 10%)
M7 Step-RLVR     [97.8%] ──► Dense Credit, 99% on L1
M8 S³-GRPO (Hyp) [98.5%+] ──► Low Tax (<1.0), High OOD (>10%), No Mimicry Dip!
```

By constraining generation with both process verification and tree structural parsimony, $S^3$-GRPO is hypothesized to achieve the **Pareto frontier** of high reasoning accuracy, minimal token inference cost, and robust cross-distribution generalization.
