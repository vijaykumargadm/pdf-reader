"""Shared fixtures. PDFs are generated at test time so no binaries live in the repo."""

from __future__ import annotations

from pathlib import Path

import pytest
from fpdf import FPDF

BODY = (
    "LangChain is a framework for building applications on top of large language "
    "models. It provides prompt templates, document loaders, text splitters and "
    "retrieval helpers that keep application code small and testable."
)

SECOND = (
    "Map-reduce summarisation splits a long document into chunks, summarises each "
    "chunk independently, and then merges the partial summaries into one answer. "
    "This keeps every model call inside the context window."
)


def _paragraph(pdf: FPDF, text: str) -> None:
    pdf.multi_cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)


def _build(path: Path, *, pages: int = 3) -> Path:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for number in range(1, pages + 1):
        pdf.add_page()
        pdf.set_font("Helvetica", size=9)
        pdf.cell(0, 6, "ACME RESEARCH - INTERNAL DRAFT", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=12)
        _paragraph(pdf, f"Page {number}. {BODY}")
        _paragraph(pdf, f"Page {number}. {SECOND}")
        _paragraph(pdf, f"This paragraph belongs to page {number} of the sample document.")

        # Footer: written with page breaks disabled so it cannot spill onto a new page.
        pdf.set_auto_page_break(auto=False)
        pdf.set_y(-18)
        pdf.set_font("Helvetica", size=9)
        pdf.cell(0, 6, f"Page {number}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_auto_page_break(auto=True, margin=15)
    pdf.output(str(path))
    return path


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A three-page text PDF with a repeated header and footer on every page."""
    return _build(tmp_path_factory.mktemp("pdfs") / "sample.pdf")


@pytest.fixture(scope="session")
def long_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A PDF with enough prose to require several chunks."""
    path = tmp_path_factory.mktemp("pdfs") / "long.pdf"
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    for number in range(1, 13):
        _paragraph(pdf, f"Section {number}. {BODY}")
        _paragraph(pdf, f"Section {number}. {SECOND}")
    pdf.output(str(path))
    return path
