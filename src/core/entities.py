"""Domain entities and data structures for Reduction Ladder research.

Pure domain models with zero dependency on heavy external ML frameworks.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional


class ErrorCategory(str, Enum):
    """Categorization of model generation failures."""
    PASS = "pass"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    WRONG_TEMPLATE = "wrong_template"  # Recalled unrelated textbook template
    ON_PATH = "on_path"                # Correct logic, minor off-by-one or edge case
    OFF_PATH = "off_path"              # Completely incorrect algorithmic reasoning
    TIMEOUT = "timeout"
    CRASH = "crash"


class LadderLevel(str, Enum):
    """Reduction Ladder difficulty and transformation levels."""
    L0 = "L0"  # Verbatim HumanEval Classic
    L1 = "L1"  # Subtle Nuance / Format Shift
    L2 = "L2"  # Structural API / Tool-Use Shift
    L3 = "L3"  # Creative Narrative Context
    L4 = "L4"  # Augmented Constraints
    L5 = "L5"  # Multi-Concept Combination
    CTRL = "Ctrl"  # Temporal Control (LiveCodeBench)


@dataclass(frozen=True)
class BenchmarkTask:
    """Represents a single coding problem in the Reduction Ladder."""
    task_id: str
    ladder_level: str
    benchmark: str
    prompt: str
    canonical_solution: str
    test: str
    entry_point: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "ladder_level": self.ladder_level,
            "benchmark": self.benchmark,
            "prompt": self.prompt,
            "canonical_solution": self.canonical_solution,
            "test": self.test,
            "entry_point": self.entry_point,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkTask":
        return cls(
            task_id=str(data["task_id"]),
            ladder_level=str(data.get("ladder_level", "")),
            benchmark=str(data.get("benchmark", "")),
            prompt=str(data.get("prompt", "")),
            canonical_solution=str(data.get("canonical_solution", "")),
            test=str(data.get("test", "")),
            entry_point=str(data.get("entry_point", "")),
        )


@dataclass
class ExecutionResult:
    """Output of executing code inside the isolated sandbox."""
    passed: bool
    status: str
    error_message: str
    execution_time_seconds: float = 0.0


@dataclass
class ModelSample:
    """A single code generation sample from a model."""
    task_id: str
    sample_index: int
    generated_code: str
    raw_response: str
    execution_result: Optional[ExecutionResult] = None
    error_category: Optional[ErrorCategory] = None


@dataclass
class LevelEvaluationReport:
    """Aggregated evaluation metrics for a single ladder level."""
    ladder_level: str
    benchmark_name: str
    total_tasks: int
    pass_at_1: float
    pass_at_5: Optional[float] = None
    error_breakdown: Dict[str, int] = field(default_factory=dict)
    average_token_length: float = 0.0
    task_results: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ModelEvaluationSuiteReport:
    """Comprehensive multi-level evaluation report for a model checkpoint."""
    model_name: str
    checkpoint_path: Optional[str]
    timestamp: str
    level_reports: Dict[str, LevelEvaluationReport] = field(default_factory=dict)
    collapse_point: Optional[str] = None
    ladder_auc: float = 0.0
    consistency_delta: float = 0.0
    memorization_risk_index: float = 0.0


@dataclass
class InvGRPORewardSignal:
    """Fine-grained breakdown of Invariance-Regularized GRPO reward components."""
    exec_reward_original: float
    exec_reward_perturbed: float
    consistency_bonus: float
    template_penalty: float
    total_reward: float
