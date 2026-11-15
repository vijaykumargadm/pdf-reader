"""Opt-in tests that hit the real Gemini API and therefore cost tokens.

They are deselected by default (see ``pyproject.toml``). Run them with::

    pytest -m live

They need a working ``.env`` with ``GEMINI_API_KEY``.
"""

from __future__ import annotations

import pytest

from config import get_settings
from pipeline import summarize_pdf
from summarizer import GeminiSummarizer

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def settings():
    try:
        loaded = get_settings()
    except Exception as exc:  # missing or invalid .env
        pytest.skip(f"live tests need a configured .env: {exc}")
    return loaded


def test_healthcheck_returns_text_and_counts_tokens(settings):
    engine = GeminiSummarizer(settings)
    reply = engine.healthcheck()

    assert reply.strip()
    assert engine.ledger.calls == 1
    assert engine.ledger.input_tokens > 0
    assert engine.ledger.output_tokens > 0


def test_full_pipeline_on_a_small_pdf(sample_pdf, settings):
    settings = settings.model_copy(update={"chunk_size": 100, "chunk_overlap": 20})
    result = summarize_pdf(sample_pdf, style="brief", settings=settings)

    assert result.summary.strip()
    assert result.page_count == 3
    assert len(result.chunks) > 1
    # One call per chunk plus the combine step.
    assert result.ledger.calls == len(result.chunks) + 1
    assert result.ledger.total_tokens > 0
    assert result.used_ocr is False
