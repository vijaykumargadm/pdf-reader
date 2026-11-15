"""Text cleaning and page handling."""

from __future__ import annotations

from pdf_processor import (
    PageText,
    clean_text,
    count_tokens,
    extract_pages,
    is_text_sparse,
    strip_running_headers,
)


def test_clean_text_repairs_hyphenated_line_breaks():
    assert clean_text("summa-\nrisation works") == "summarisation works"


def test_clean_text_collapses_whitespace_and_blank_lines():
    messy = "Title\n\n\n\nBody   with    gaps\tand trailing   \n"
    assert clean_text(messy) == "Title\n\nBody with gaps and trailing"


def test_clean_text_normalises_unicode_and_soft_hyphens():
    assert clean_text("caf\u00e9\u00ad") == "café"


def test_count_tokens_is_monotonic():
    assert count_tokens("") == 0
    assert count_tokens("hello world") < count_tokens("hello world " * 50)


def test_strip_running_headers_drops_repeated_edge_lines():
    pages = [
        PageText(1, "REPORT\nReal content one\nPage 1"),
        PageText(2, "REPORT\nReal content two\nPage 2"),
        PageText(3, "REPORT\nReal content three\nPage 3"),
    ]
    cleaned = strip_running_headers(pages)
    assert all("REPORT" not in page.text for page in cleaned)
    assert all("Page " not in page.text for page in cleaned)
    assert "Real content two" in cleaned[1].text


def test_strip_running_headers_keeps_repeated_body_text():
    pages = [PageText(1, "Introduction\nbody a"), PageText(2, "Introduction\nbody b")]
    assert strip_running_headers(pages) == pages  # fewer than three pages: untouched


def test_extract_pages_reads_generated_pdf(sample_pdf):
    pages = extract_pages(sample_pdf)
    assert len(pages) == 3
    assert [page.page_number for page in pages] == [1, 2, 3]
    assert "LangChain" in pages[0].text
    assert "page 3" in pages[2].text
    assert all("ACME RESEARCH" not in page.text for page in pages)


def test_is_text_sparse_flags_scans():
    scanned = [PageText(1, ""), PageText(2, " ")]
    assert is_text_sparse(scanned) is True
    assert is_text_sparse([PageText(1, "real words " * 60)]) is False
