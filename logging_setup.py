"""Logging setup shared by the CLI and the Streamlit app."""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"

_NOISY = ("httpx", "httpx2", "httpcore", "httpcore2", "urllib3", "google_genai", "google.auth")


def configure_logging(level: str = "INFO", *, force: bool = False) -> None:
    """Install a single stderr handler and quiet down chatty third parties."""
    root = logging.getLogger()
    if root.handlers and not force:
        root.setLevel(level.upper())
        return

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)


def mask_secret(value: str, *, keep: int = 4) -> str:
    """Render a key safely for logs and UI: ``AQ.A...KA (53 chars)``."""
    if not value:
        return "<empty>"
    if len(value) <= keep * 2:
        return f"{'*' * len(value)} ({len(value)} chars)"
    return f"{value[:keep]}...{value[-2:]} ({len(value)} chars)"
