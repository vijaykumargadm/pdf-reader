"""Typed exceptions so callers can react to specific failure modes."""

from __future__ import annotations


class PdfReaderError(Exception):
    """Base class for every error this project raises deliberately."""


class ConfigError(PdfReaderError):
    """Settings or environment are unusable."""


class PdfExtractionError(PdfReaderError):
    """The file could not be read as a PDF or contained no usable text."""


class OcrError(PdfReaderError):
    """Scanned pages could not be transcribed."""


class SummarizationError(PdfReaderError):
    """A model call failed for a non-retryable reason."""


class ModelNotFoundError(SummarizationError):
    """The configured model is not available to this API key."""


class RateLimitError(SummarizationError):
    """Retries were exhausted while the provider kept rate limiting us."""
