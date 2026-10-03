from src.arms.arm4_step_rlvr.verifier import (
    StepContract,
    StepwiseContractVerifier,
    StepwiseRewardEngine,
)


def __getattr__(name: str):
    if name == "StepRLVRTrainer":
        from src.arms.arm4_step_rlvr.trainer import StepRLVRTrainer
        return StepRLVRTrainer
    raise AttributeError(f"module 'src.arms.arm4_step_rlvr' has no attribute '{name}'")


__all__ = [
    "StepContract",
    "StepwiseContractVerifier",
    "StepwiseRewardEngine",
    "StepRLVRTrainer",
]

