"""Multi-Arm Mitigation Package for Small Language Model Mimicry.

Contains modular implementations of the 4 prioritized mitigation arms:
  - arm1_inv_grpo:       Invariance-Regularized Policy Optimization (Primary Innovation)
  - arm2_contrastive_sft: Contrastive Thought-Template SFT / DPO
  - arm3_ast_rl:          AST-Guided Policy Optimization (TreeDiff/VeriSeek)
  - arm4_step_rlvr:       Stepwise Execution-Gated Process RLVR (CodePRM)
"""
