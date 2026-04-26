"""Custom exceptions for the SLIDERS project."""


class SLIDERSError(Exception):
    """Base exception for SLIDERS project."""


class EmbeddingDimensionMismatch(SLIDERSError):
    """Raised when embedding dimensions don't match expected values."""


class InvalidSliderConfig(SLIDERSError):
    """Raised when slider configuration is invalid."""
