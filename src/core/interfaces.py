"""Abstract interfaces and protocols for Clean Architecture.

Enforces Dependency Inversion Principle (DIP): High-level business logic
depends on these abstractions, not on concrete infrastructure classes.
"""

from typing import Protocol, List, Dict, Any, Optional
from src.core.entities import BenchmarkTask, ExecutionResult, ModelSample, LevelEvaluationReport


class ICodeExecutor(Protocol):
    """Interface for isolated code sandbox execution."""

    def execute(
        self,
        prompt: str,
        solution: str,
        test: str,
        entry_point: str,
        timeout_seconds: float = 5.0
    ) -> ExecutionResult:
        """Run code in isolated environment and return execution result."""
        ...


class IBenchmarkLoader(Protocol):
    """Interface for loading and normalizing benchmark datasets."""

    def load_level(self, level_key: str) -> List[BenchmarkTask]:
        """Load and normalize a specific ladder benchmark level."""
        ...

    def load_all_levels(self) -> Dict[str, List[BenchmarkTask]]:
        """Load and normalize all available ladder benchmark levels."""
        ...


class IModelRunner(Protocol):
    """Interface for generating code from an LLM/SLM."""

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        num_samples: int = 1,
        max_new_tokens: int = 1024
    ) -> List[str]:
        """Generate code completions for a given prompt."""
        ...


class IErrorClassifier(Protocol):
    """Interface for diagnosing and categorizing execution failures."""

    def classify(
        self,
        task: BenchmarkTask,
        generated_code: str,
        execution_result: ExecutionResult
    ) -> str:
        """Classify a failure into a domain error category."""
        ...


class IRewardComputer(Protocol):
    """Interface for computing reinforcement learning rewards."""

    def compute_reward(
        self,
        task: BenchmarkTask,
        completion: str,
        **kwargs
    ) -> float:
        """Compute scalar reward for a completion."""
        ...
