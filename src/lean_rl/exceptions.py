"""
Custom exceptions for LeanReinforcement.

This module defines a hierarchy of custom exceptions for better error handling
and debugging throughout the codebase.
"""

from typing import Optional, Any


class LeanRLError(Exception):
    """Base exception for all LeanRL errors."""

    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self):
        if self.details:
            return f"{self.message}. Details: {self.details}"
        return self.message


class EnvironmentError(LeanRLError):
    """Base class for environment-related errors."""

    pass


class DojoEnvironmentError(EnvironmentError):
    """Errors related to Dojo environment operations."""

    pass


class EnvironmentInitializationError(EnvironmentError):
    """Error during environment initialization."""

    pass


class EnvironmentResetError(EnvironmentError):
    """Error during environment reset."""

    pass


class AgentError(LeanRLError):
    """Base class for agent-related errors."""

    pass


class ModelError(AgentError):
    """Errors related to model operations."""

    pass


class TrainingError(LeanRLError):
    """Base class for training-related errors."""

    pass


class ConfigurationError(LeanRLError):
    """Errors related to configuration."""

    pass


class DataError(LeanRLError):
    """Errors related to data processing."""

    pass


class RepositoryError(DataError):
    """Errors related to repository operations."""

    pass


class TheoremError(LeanRLError):
    """Errors related to theorem operations."""

    pass


# Convenience functions for creating specific errors
def create_dojo_error(
    message: str, original_exception: Optional[Exception] = None
) -> DojoEnvironmentError:
    """Create a DojoEnvironmentError with optional original exception details."""
    details = {}
    if original_exception:
        details.update(
            {
                "original_exception": str(original_exception),
                "exception_type": type(original_exception).__name__,
            }
        )
    return DojoEnvironmentError(message, details)


def create_model_error(message: str, model_name: Optional[str] = None) -> ModelError:
    """Create a ModelError with optional model information."""
    details = {}
    if model_name:
        details["model"] = model_name
    return ModelError(message, details)


def create_training_error(
    message: str, epoch: Optional[int] = None, step: Optional[int] = None
) -> TrainingError:
    """Create a TrainingError with optional training state information."""
    details = {}
    if epoch is not None:
        details["epoch"] = epoch
    if step is not None:
        details["step"] = step
    return TrainingError(message, details)
