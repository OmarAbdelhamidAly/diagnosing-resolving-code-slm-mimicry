# 🔬 Arm 4: Stepwise Contract Verifier & Process Reward Optimization (Step-RLVR)

**Project:** Diagnosing and Resolving Code Small Language Model Mimicry via a Reduction Ladder  
**Institution:** Orange Innovation Labs — AI Research & Development Division  
**Authors:** Omar Abdelhamid, Nour Walid (Equal Contribution)  
**Supervisor:** Dr. Ghada Khoriba  
**Model Identifier:** `M7_step_rlvr` | **Base Model:** Qwen2.5-Coder-1.5B-Instruct  
**Target Milestone:** Week 09 Evaluation & Mitigation Suite  

---

## 📑 Table of Contents
1. [Executive Summary](#-executive-summary)
2. [Problem Statement: The Terminal Credit Assignment Trap](#-problem-statement-the-terminal-credit-assignment-trap)
3. [Theoretical Foundations & Literature Gap](#-theoretical-foundations--literature-gap)
4. [Mathematical Formulation: Stepwise Contracts & Dense Partial Credits](#-mathematical-formulation-stepwise-contracts--dense-partial-credits)
5. [The Stepwise Contract Verification Architecture (`verifier.py`)](#-the-stepwise-contract-verification-architecture-verifierpy)
6. [Process Reward Engine & Advantage Calculation](#-process-reward-engine--advantage-calculation)
7. [Engineering Implementation & Training Dynamics](#-engineering-implementation--training-dynamics)
8. [Empirical Hypotheses & Target Metrics](#-empirical-hypotheses--target-metrics)
9. [Hardware & Hyperparameter Specifications](#-hardware--hyperparameter-specifications)
10. [Formal Scientific Bibliography](#-formal-scientific-bibliography)

---

## 📌 Executive Summary

**Arm 4 (Step-RLVR)** introduces execution-gated **process supervision** for reinforcement learning on complex multi-step coding tasks. On difficult and compound benchmarks (e.g., $L_4$ Difficult and $L_5$ Combine), algorithms require chained sub-routines (e.g., parsing, transformation, dynamic programming, output formatting). 

Standard outcome-only RLVR evaluates only the final return value ($R_{\text{terminal}} \in \{0, 1\}$). If a model correctly derives 3 complex sub-routines but commits an off-by-one error in the 4th step, terminal verification assigns $R=0$. The policy gradient treats the entire sequence as equally erroneous, actively destroying correct intermediate reasoning steps.

Step-RLVR decomposes multi-step tasks into independent, executable **Sub-Function Contracts**. By executing contract assertions at intermediate boundaries within our isolated subprocess sandbox, Step-RLVR awards dense, granular partial credits:
$$R_{\text{stepwise}}(y) = \sum_{s=1}^S w_s \cdot \mathbb{I}(\text{Contract}_s(y) == \text{Valid})$$
This eliminates gradient starvation on long-horizon problems, transforming complex reasoning tasks from all-or-nothing lotteries into learnable curriculum surfaces.

---

## 🎯 Problem Statement: The Terminal Credit Assignment Trap

In multi-goal and composition tasks:
1. **Extreme Reward Sparsity:** On $L_5$ (Combine), models must satisfy multiple orthogonal constraints simultaneously. Exploratory pass rates for a 1.5B model drop below $5\%$, starving the policy of positive reinforcement.
2. **Pathological Credit Assignment:** Terminal rewards cannot identify *where* an algorithm failed. Did the model fail at data ingestion, mathematical core logic, or return type casting? A zero reward flattens all token gradients indiscriminately.
3. **Superficial Mimicry as a Coping Strategy:** Unable to learn multi-step chains from sparse rewards, SLMs default to memorized single-step templates that produce high surface token likelihood but fail functional execution.

---

## 📚 Theoretical Foundations & Literature Gap

Step-RLVR bridges literature in Process Reward Models (PRMs) and software testing contracts:

```
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│       CodePRM (ACL 2025)        │       │      ExecVerify (ICSE 2026)     │
│  Process Reward Models for Code │       │   Execution-Gated Verification  │
│   [Token-Level Value Estimates] │       │    [Sub-Routine Unit Contracts] │
└────────────────┬────────────────┘       └────────────────┬────────────────┘
                 │                                         │
                 └────────────────────┬────────────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │       Arm 4: Step-RLVR        │
                      │   Executable Step Contracts   │
                      │    + Granular Partial Credits │
                      │    + Dense Policy Advantages  │
                      └───────────────────────────────┘
                                      ▲
                                      │
                 ┌────────────────────┴────────────────────┐
                 │                                         │
┌────────────────┴────────────────┐       ┌────────────────┴────────────────┐
│    Math-Shepherd (ACL 2024)     │       │    PRM800K (Lightman et al.)    │
│  Automated Process Supervision  │       │   Let's Verify Step by Step     │
│  [Elimination of Human Labels]  │       │   [Step-Level Advantage Scaling]│
└─────────────────────────────────┘       └─────────────────────────────────┘
```

1. **CodePRM: Process Reward Models for Code Generation (ACL 2025) [1]:**
   Demonstrates that evaluating intermediate execution states during code synthesis outperforms outcome-only verifiers by $18.4\%$ on complex algorithms.
2. **ExecVerify: Stepwise Execution-Gated Verification (ICSE 2026) [2]:**
   Introduces programmatic contracts for unit testing sub-functions in generated code, showing that verifying intermediate invariants drastically reduces runtime exceptions.
3. **Math-Shepherd: Automated Process Supervision (Wang et al., ACL 2024) [3]:**
   Proves that process rewards can be automatically synthesized via programmatic rollout verification without requiring costly human step-by-step annotation.
4. **Let's Verify Step by Step (Lightman et al., OpenAI 2023) [4]:**
   Foundational proof that active step-level supervision solves reasoning distribution drift in multi-step inference chains.

---

## 📐 Mathematical Formulation: Stepwise Contracts & Dense Partial Credits

### 1. Sub-Function Step Contracts
A complex problem is partitioned into $S$ sequential programmatic contracts:
$$\mathcal{C} = \{\text{Contract}_1, \text{Contract}_2, \dots, \text{Contract}_S\}$$
Where each contract $\text{Contract}_s = (n_s, e_s, w_s, T_s)$ specifies:
* $n_s$: Semantic step identifier (e.g., `parse_graph_input`, `compute_shortest_path`).
* $e_s$: Specific sub-routine entry point.
* $w_s \in (0, 1]$: Normalized step importance weight ($\sum_{s=1}^S w_s = 1.0$).
* $T_s$: Unit test assertion suite dedicated strictly to step $s$.

### 2. Stepwise Reward Formulation
Let $y$ denote the complete generated program. The execution sandbox evaluates each contract independently:
$$v_s(y) = \mathbb{I}(\text{Execute}(y, T_s) == \text{PASS}) \in \{0, 1\}$$
The composite dense stepwise reward is:
$$\mathcal{R}_{\text{stepwise}}(y) = \sum_{s=1}^S w_s \cdot v_s(y) \in [0, 1]$$

### 3. Stepwise Group Advantage in Policy Gradient
In Group Relative Policy Optimization (GRPO), we sample $G$ rollouts $\{y_i\}_{i=1}^G$. The advantage $\hat{A}_i$ is computed from the normalized stepwise rewards:
$$\bar{\mathcal{R}} = \frac{1}{G}\sum_{j=1}^G \mathcal{R}_{\text{stepwise}}(y_j), \quad \sigma = \sqrt{\frac{1}{G}\sum_{j=1}^G (\mathcal{R}_{\text{stepwise}}(y_j) - \bar{\mathcal{R}})^2 + \epsilon}$$
$$\hat{A}_i = \frac{\mathcal{R}_{\text{stepwise}}(y_i) - \bar{\mathcal{R}}}{\sigma}$$
* **Key Advantage:** Rollouts that solve 3 out of 4 steps achieve positive relative advantage over rollouts that solve 1 out of 4 steps, preserving and reinforcing the correct sub-routine representations.

---

## 🔍 The Stepwise Contract Verification Architecture (`verifier.py`)

Implemented in [`src/arms/arm4_step_rlvr/verifier.py`](file:///c:/Users/Lenovo/Downloads/Reasoning/reo/src/arms/arm4_step_rlvr/verifier.py):

```python
@dataclass
class StepContract:
    name: str
    entry_point: str
    weight: float
    test: str

class StepwiseContractVerifier:
    def __init__(self, sandbox: ICodeExecutor):
        self.sandbox = sandbox

    def evaluate_steps(self, full_code: str, step_specs: List[StepContract], prompt: str = "") -> Dict[str, Any]:
        total_reward = 0.0
        step_results = []
        for step in step_specs:
            res = self.sandbox.execute(
                prompt=prompt,
                solution=full_code,
                test=step.test,
                entry_point=step.entry_point,
            )
            credits = step.weight if res.passed else 0.0
            total_reward += credits
            step_results.append({
                "name": step.name,
                "passed": res.passed,
                "credits": round(credits, 3),
            })
        return {
            "total_stepwise_reward": round(total_reward, 3),
            "steps": step_results,
        }
```

---

## 🏗️ Process Reward Engine & Advantage Calculation

* [`verifier.py`](file:///c:/Users/Lenovo/Downloads/Reasoning/reo/src/arms/arm4_step_rlvr/verifier.py): Implements `StepwiseRewardEngine.compare_rewards()`, calculating:
  * $\Delta_{\text{reward}} = \mathcal{R}_{\text{stepwise}} - \mathcal{R}_{\text{terminal}}$: Quantifying the amount of valid reasoning credit recovered from failed terminal runs.
  * Partial credit attribution for intermediate algorithmic stages.

---

## 🛠️ Engineering Implementation & Training Dynamics

The module is structured under `src/arms/arm4_step_rlvr/`:
* [`verifier.py`](file:///c:/Users/Lenovo/Downloads/Reasoning/reo/src/arms/arm4_step_rlvr/verifier.py): Contract dataclasses, contract execution engine, partial credit accumulator.
* [`trainer.py`](file:///c:/Users/Lenovo/Downloads/Reasoning/reo/src/arms/arm4_step_rlvr/trainer.py): Step-RLVR policy trainer with per-step advantage normalization.
* [`notebooks/arm_04_step_rlvr.ipynb`](file:///c:/Users/Lenovo/Downloads/Reasoning/reo/notebooks/arm_04_step_rlvr.ipynb): Interactive notebook demonstrating step verification across complex rungs.

---

## 📊 Empirical Hypotheses & Target Metrics

| Metric | M1 Baseline | M4 Standard GRPO | M7 Step-RLVR Target | Scientific Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **$L_5$ Combine** | 77.0% | 81.0% | **$\ge 89.0\%$** | Dense sub-goal contracts solve multi-constraint composition. |
| **$L_4$ Difficult** | 76.0% | 80.0% | **$\ge 87.0\%$** | Recovers partial gradient from complex multi-branch algorithms. |
| **Ladder AUC** | 81.57% | 83.50% | **$\ge 86.5\%$** | Eliminates the steep drop on higher-order reduction rungs. |
| **Degradation Slope**| -0.035 | -0.031 | **$\ge -0.015$** | Flattest degradation curve across complex task depths. |

---

## ⚙️ Hardware & Hyperparameter Specifications

* **Base Model:** `Qwen/Qwen2.5-Coder-1.5B-Instruct`
* **LoRA Rank ($r$):** 16 | **LoRA Alpha ($\alpha$):** 32 | **Dropout:** 0.05
* **Group Size ($G$):** 4 rollouts per prompt
* **Step Contracts per Task:** $S \in [2, 4]$ sub-routines
* **Learning Rate:** $1 \times 10^{-5}$ (AdamW)
* **Sampling Temperature:** 0.8 | **Max New Tokens:** 384
* **Sandbox Timeout:** 4.0s cumulative per step suite
* **Peak VRAM:** $\sim 5.1\text{ GB}$ on NVIDIA RTX 3070 Ti 8GB.

---

## 📖 Formal Scientific Bibliography

1. **Luo, Z., et al. (2025).** *CodePRM: Process Reward Models for Code Reasoning via Sub-Goal Verification.* Annual Meeting of the Association for Computational Linguistics (ACL 2025).
2. **Banerjee, S., et al. (2026).** *ExecVerify: Execution-Gated Verification for Multi-Step Code Synthesis.* International Conference on Software Engineering (ICSE 2026).
3. **Wang, P., et al. (2024).** *Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations.* Association for Computational Linguistics (ACL 2024). [arXiv:2312.08935](https://arxiv.org/abs/2312.08935).
4. **Lightman, H., et al. (2023).** *Let's Verify Step by Step.* OpenAI Technical Report. [arXiv:2305.20050](https://arxiv.org/abs/2305.20050).
