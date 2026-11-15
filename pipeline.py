"""Orchestration: file on disk -> clean pages -> chunks -> summary -> result."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from config import Settings, get_settings
from errors import OcrError
from pdf_processor import (
    Chunk,
    PageText,
    chunk_pages,
    extract_pages,
    is_text_sparse,
    render_pages,
)
from summarizer import GeminiSummarizer
from utils import CostLedger, format_cost

log = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]


def _scaled(progress: ProgressFn | None, low: float, high: float) -> ProgressFn | None:
    """Map a sub-step's 0..1 progress onto the parent's [low, high] window."""
    if progress is None:
        return None

    def inner(fraction: float, message: str) -> None:
        clamped = min(max(fraction, 0.0), 1.0)
        progress(low + (high - low) * clamped, message)

    return inner


@dataclass
class SummaryResult:
    """Everything one run produced, including the cost trail."""

    document: str
    style: str
    summary: str
    chunk_summaries: list[str]
    chunks: list[Chunk]
    ledger: CostLedger
    page_count: int
    used_ocr: bool = False
    ocr_pages: int = 0
    analysis: str | None = None
    quotes: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def cost_label(self) -> str:
        return format_cost(self.ledger)

    def to_markdown(self) -> str:
        lines = [
            f"# Summary — {self.document}",
            "",
            f"- **Style:** {self.style}",
            f"- **Pages:** {self.page_count}",
            f"- **Chunks:** {len(self.chunks)}",
            f"- **Model:** {self.ledger.model}",
            f"- **OCR fallback:** {'yes' if self.used_ocr else 'no'}",
            f"- **Tokens:** {self.ledger.input_tokens:,} in / {self.ledger.output_tokens:,} out",
            f"- **Estimated cost:** {self.cost_label}",
            "",
            "## Summary",
            "",
            self.summary.strip(),
            "",
        ]
        if self.analysis:
            lines += ["## Structure analysis", "", self.analysis.strip(), ""]
        if self.quotes:
            lines += ["## Key quotes", "", self.quotes.strip(), ""]
        if self.warnings:
            lines += ["## Warnings", "", *[f"- {warning}" for warning in self.warnings], ""]
        lines += [
            "## Chunk statistics",
            "",
            "| # | Pages | Tokens | Characters |",
            "|---|---|---|---|",
        ]
        for chunk in self.chunks:
            lines.append(
                f"| {chunk.index + 1} | {chunk.page_label} | {chunk.tokens} | {len(chunk.text)} |"
            )
        lines += [
            "",
            "## Usage by step",
            "",
            "| Step | Calls | Input | Output | Total |",
            "|---|---|---|---|---|",
        ]
        for row in self.ledger.rows():
            lines.append(
                f"| {row['step']} | {row['calls']} | {row['input_tokens']:,} | "
                f"{row['output_tokens']:,} | {row['total_tokens']:,} |"
            )
        return "\n".join(lines) + "\n"


def load_pages(
    path: str | Path,
    settings: Settings,
    *,
    summarizer: GeminiSummarizer | None = None,
    progress: ProgressFn | None = None,
) -> tuple[list[PageText], bool, int, int]:
    """Extract text, falling back to vision OCR when the PDF is scanned.

    Returns ``(pages, used_ocr, ocr_page_count, total_page_count)``.
    """
    pdf_path = Path(path)
    pages = extract_pages(pdf_path)
    total_pages = len(pages)
    if progress:
        progress(0.1, f"Extracted {total_pages} pages")

    if not is_text_sparse(pages) or not settings.ocr_enabled:
        return pages, False, 0, total_pages

    log.info("%s looks scanned; running OCR fallback", pdf_path.name)
    if progress:
        progress(0.12, "Scanned PDF detected — rendering pages for OCR")
    engine = summarizer or GeminiSummarizer(settings)
    images = render_pages(pdf_path, dpi=settings.ocr_dpi, max_pages=settings.ocr_max_pages)
    ocr_pages = engine.transcribe_pages(images, progress=_scaled(progress, 0.12, 0.22))
    if not any(page.text.strip() for page in ocr_pages):
        raise OcrError(
            f"OCR produced no text for {pdf_path.name}. The scan may be too low quality."
        )
    return ocr_pages, True, len(ocr_pages), total_pages


def summarize_pdf(
    path: str | Path,
    *,
    style: str = "detailed",
    settings: Settings | None = None,
    include_analysis: bool = False,
    include_quotes: bool = False,
    progress: ProgressFn | None = None,
    engine: GeminiSummarizer | None = None,
) -> SummaryResult:
    """Run the whole map-reduce pipeline for one PDF.

    Pass ``engine`` to share one client (and ledger) with follow-up steps such as RAG.
    """
    settings = settings or get_settings()
    pdf_path = Path(path)
    engine = engine or GeminiSummarizer(settings)
    warnings: list[str] = []

    pages, used_ocr, ocr_count, total_pages = load_pages(
        pdf_path, settings, summarizer=engine, progress=progress
    )

    chunks = chunk_pages(
        pages, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    if progress:
        progress(0.2, f"Split into {len(chunks)} chunks")

    summaries = engine.summarize_chunks(
        chunks, style, pdf_path.name, progress=_scaled(progress, 0.2, 0.7)
    )
    summary = engine.combine(summaries, style, pdf_path.name, progress=_scaled(progress, 0.7, 0.8))

    analysis = None
    quotes = None
    if include_analysis or include_quotes:
        # Auxiliary steps read the source (page-labelled), not the merged summary,
        # so quotes stay verbatim and structure analysis sees the real document.
        material = [f"[{chunk.page_label}]\n{chunk.text}" for chunk in chunks]
        if include_analysis:
            analysis = engine.analyze_structure(material, progress=_scaled(progress, 0.8, 0.9))
        if include_quotes:
            quotes = engine.extract_quotes(material, progress=_scaled(progress, 0.9, 1.0))

    if used_ocr and ocr_count < total_pages:
        warnings.append(
            f"Only the first {ocr_count} of {total_pages} pages were OCR'd "
            f"(ocr_max_pages={settings.ocr_max_pages})."
        )

    if progress:
        progress(1.0, "Done")

    return SummaryResult(
        document=pdf_path.name,
        style=style,
        summary=summary,
        chunk_summaries=summaries,
        chunks=chunks,
        ledger=engine.ledger,
        page_count=total_pages,
        used_ocr=used_ocr,
        ocr_pages=ocr_count,
        analysis=analysis,
        quotes=quotes,
        warnings=warnings,
    )
