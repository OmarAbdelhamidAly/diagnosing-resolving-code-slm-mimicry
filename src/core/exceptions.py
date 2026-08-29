"""Domain exceptions for Reduction Ladder research framework."""


class ReductionLadderException(Exception):
    """Base exception for all domain errors."""
    pass


class DatasetIngestionError(ReductionLadderException):
    """Raised when benchmark data cannot be fetched or normalized."""
    pass


class SandboxTimeoutError(ReductionLadderException):
    """Raised when code execution in sandbox exceeds allowed time."""
    pass


class SandboxCrashError(ReductionLadderException):
    """Raised when worker subprocess crashes."""
    pass


class ModelInferenceError(ReductionLadderException):
    """Raised when model loading or forward pass fails."""
    pass


class VRAMExceededError(ReductionLadderException):
    """Raised when GPU VRAM allocation fails (OOM)."""
    pass


class ConfigurationError(ReductionLadderException):
    """Raised when config.yaml is invalid or missing required keys."""
    pass
