"""Core domain package containing pure entities, interfaces, and exceptions."""

from src.core.entities import (
    BenchmarkTask,
    ExecutionResult,
    ModelSample,
    LevelEvaluationReport,
    ModelEvaluationSuiteReport,
    ErrorCategory,
    LadderLevel,
    InvGRPORewardSignal,
)
from src.core.interfaces import (
    ICodeExecutor,
    IBenchmarkLoader,
    IModelRunner,
    IErrorClassifier,
    IRewardComputer,
)
from src.core.exceptions import (
    ReductionLadderException,
    DatasetIngestionError,
    SandboxTimeoutError,
    SandboxCrashError,
    ModelInferenceError,
    VRAMExceededError,
    ConfigurationError,
)

__all__ = [
    "BenchmarkTask",
    "ExecutionResult",
    "ModelSample",
    "LevelEvaluationReport",
    "ModelEvaluationSuiteReport",
    "ErrorCategory",
    "LadderLevel",
    "InvGRPORewardSignal",
    "ICodeExecutor",
    "IBenchmarkLoader",
    "IModelRunner",
    "IErrorClassifier",
    "IRewardComputer",
    "ReductionLadderException",
    "DatasetIngestionError",
    "SandboxTimeoutError",
    "SandboxCrashError",
    "ModelInferenceError",
    "VRAMExceededError",
    "ConfigurationError",
]
