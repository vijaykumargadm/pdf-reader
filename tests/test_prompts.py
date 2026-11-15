"""Prompts render with the variables the code passes, and unknown styles fail loudly."""

from __future__ import annotations

import pytest

from prompts import (
    ANALYSIS,
    CHUNK_SUMMARY,
    COMBINE,
    OCR_PAGE,
    QUOTES,
    REDUCE,
    STYLES,
    style_instruction,
)


@pytest.mark.parametrize("style", sorted(STYLES))
def test_chunk_summary_renders_every_style(style):
    messages = CHUNK_SUMMARY.format_messages(
        style=style_instruction(style),
        document_name="report.pdf",
        page_label="pp.2-3",
        text="Some excerpt.",
    )
    assert len(messages) == 2
    assert messages[0].type == "system"
    assert style_instruction(style) in messages[1].content
    assert "pp.2-3" in messages[1].content


def test_combine_renders():
    messages = COMBINE.format_messages(
        style="brief", summaries="[1] first\n\n---\n\n[2] second", document_name="report.pdf"
    )
    assert "report.pdf" in messages[1].content
    assert "[2] second" in messages[1].content


def test_reduce_analysis_quotes_render():
    assert "tight" in REDUCE.format_messages(style="brief", summaries="x")[1].content
    assert "Document type" in ANALYSIS.format_messages(text="x")[1].content
    assert "verbatim" in QUOTES.format_messages(text="x")[1].content


def test_ocr_prompt_includes_page_number():
    messages = OCR_PAGE.format_messages(page_number=7)
    assert "7" in messages[1].content
    assert "Transcribe" in messages[1].content


def test_unknown_style_raises():
    with pytest.raises(KeyError, match="unknown summary style"):
        style_instruction("poetic")


def test_prompts_do_not_leak_curly_braces():
    """A stray brace would make LangChain treat text as a template variable."""
    for style in STYLES.values():
        CHUNK_SUMMARY.format_messages(
            style=style, document_name="d.pdf", page_label="p.1", text="t"
        )
