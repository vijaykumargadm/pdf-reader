"""Chunking: page awareness, token bounds and overlap."""

from __future__ import annotations

import pytest

from errors import PdfExtractionError
from pdf_processor import PageText, chunk_pages, count_tokens, extract_pages


def test_chunks_are_numbered_and_page_aware(sample_pdf):
    pages = extract_pages(sample_pdf)
    chunks = chunk_pages(pages, chunk_size=120, chunk_overlap=20)

    assert len(chunks) > 1
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert chunks[0].start_page == 1
    for chunk in chunks:
        assert 1 <= chunk.start_page <= chunk.end_page <= 3
        assert chunk.text.strip()
        assert chunk.tokens > 0


def test_chunk_page_ranges_move_forward(sample_pdf):
    chunks = chunk_pages(extract_pages(sample_pdf), chunk_size=100, chunk_overlap=10)
    starts = [chunk.start_page for chunk in chunks]
    assert starts == sorted(starts)


def test_chunks_respect_the_token_budget(long_pdf):
    limit = 100
    chunks = chunk_pages(extract_pages(long_pdf), chunk_size=limit, chunk_overlap=15)
    assert len(chunks) > 1
    assert all(chunk.tokens <= limit for chunk in chunks)


def test_overlap_duplicates_content(long_pdf):
    pages = extract_pages(long_pdf)
    document_tokens = count_tokens(" ".join(page.text for page in pages))
    chunks = chunk_pages(pages, chunk_size=120, chunk_overlap=40)
    assert sum(chunk.tokens for chunk in chunks) > document_tokens


def test_no_overlap_means_no_duplication(long_pdf):
    pages = extract_pages(long_pdf)
    document_tokens = count_tokens(" ".join(page.text for page in pages))
    chunks = chunk_pages(pages, chunk_size=120, chunk_overlap=0)
    # Boundary handling can duplicate a character or two, never whole chunks.
    assert sum(chunk.tokens for chunk in chunks) <= document_tokens + len(chunks) * 2


def test_empty_document_is_rejected():
    with pytest.raises(PdfExtractionError):
        chunk_pages([PageText(1, "   ")], chunk_size=100, chunk_overlap=10)


def test_page_label_reads_naturally():
    chunk = chunk_pages([PageText(1, "word " * 200)], chunk_size=50, chunk_overlap=5)[0]
    assert chunk.page_label == "p.1"
