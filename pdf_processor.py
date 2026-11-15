"""PDF text extraction, cleaning, token counting and chunking.

This module never talks to an LLM: it turns bytes on disk into clean, page-aware
text and then into token-bounded chunks. OCR (which needs a model) lives in
``summarizer.transcribe_images`` and is orchestrated by ``pipeline``.
"""

from __future__ import annotations

import io
import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from errors import PdfExtractionError

log = logging.getLogger(__name__)

#: Below this many alphanumeric characters per page a PDF is treated as scanned.
SPARSE_CHARS_PER_PAGE = 100

_HYPHEN_BREAK = re.compile(r"(?<=[a-z])-[ \t]*\n[ \t]*(?=[a-z])")
_TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)
_INLINE_SPACE = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_PAGE_NUMBER = re.compile(r"^(?:page\s*)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?$", re.IGNORECASE)


@dataclass(frozen=True)
class PageText:
    """Text extracted from a single PDF page."""

    page_number: int  # 1-based, as a human counts pages
    text: str


@dataclass(frozen=True)
class Chunk:
    """A token-bounded slice of the document, aware of the pages it came from."""

    index: int
    text: str
    start_page: int
    end_page: int
    tokens: int

    @property
    def page_label(self) -> str:
        if self.start_page == self.end_page:
            return f"p.{self.start_page}"
        return f"pp.{self.start_page}-{self.end_page}"


@lru_cache(maxsize=1)
def _encoder() -> tiktoken.Encoding | None:
    """Load the tokeniser once. Returns None when the BPE file is unavailable."""
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # offline fallback is intentional
        log.warning("tiktoken encoding unavailable (%s); falling back to char estimate", exc)
        return None


def count_tokens(text: str) -> int:
    """Approximate Gemini token count.

    Gemini's own tokeniser is not public, so cl100k is used as a close proxy for
    budgeting. If the BPE file cannot be downloaded the function degrades to the
    usual four-characters-per-token rule instead of failing.
    """
    if not text:
        return 0
    encoder = _encoder()
    if encoder is None:
        return max(1, len(text) // 4)
    return len(encoder.encode(text, disallowed_special=()))


def clean_text(text: str) -> str:
    """Normalise whitespace and repair words split across line breaks."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00ad", "")  # soft hyphen
    text = _HYPHEN_BREAK.sub("", text)
    text = _INLINE_SPACE.sub(" ", text)
    text = _TRAILING_SPACE.sub("", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def strip_running_headers(pages: list[PageText], *, ratio: float = 0.6) -> list[PageText]:
    """Drop repeated top/bottom lines (letterheads, footers, page numbers).

    Only the first and last three non-empty lines of each page are candidates, so
    a repeated *body* heading is never removed.
    """
    if len(pages) < 3:
        return pages

    seen: Counter[str] = Counter()
    for page in pages:
        lines = [line.strip() for line in page.text.splitlines() if line.strip()]
        for line in set(lines[:3] + lines[-3:]):
            seen[line] += 1

    threshold = max(2, round(len(pages) * ratio))
    noise = {line for line, hits in seen.items() if hits >= threshold}
    if not noise:
        return pages

    cleaned: list[PageText] = []
    for page in pages:
        kept: list[str] = []
        lines = page.text.splitlines()
        non_empty = [i for i, line in enumerate(lines) if line.strip()]
        edge = set(non_empty[:3] + non_empty[-3:])
        for index, line in enumerate(lines):
            stripped = line.strip()
            if index in edge and (stripped in noise or _PAGE_NUMBER.match(stripped)):
                continue
            kept.append(line)
        cleaned.append(replace(page, text=clean_text("\n".join(kept))))

    log.debug("removed %d repeated header/footer lines", len(noise))
    return cleaned


def _read_pdf(path: Path) -> PdfReader:
    if not path.exists():
        raise PdfExtractionError(f"file not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise PdfExtractionError(f"not a PDF file: {path.name}")
    try:
        reader = PdfReader(str(path))
    except (PdfReadError, OSError) as exc:
        raise PdfExtractionError(f"cannot read {path.name}: {exc}") from exc

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:  # pypdf raises several types
            raise PdfExtractionError(f"{path.name} is password protected") from exc
    return reader


def extract_pages(path: str | Path) -> list[PageText]:
    """Extract and clean the text of every page, keeping page numbers."""
    pdf_path = Path(path)
    reader = _read_pdf(pdf_path)

    pages: list[PageText] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:  # one bad page should not kill the run
            log.warning("page %d of %s could not be parsed: %s", number, pdf_path.name, exc)
            raw = ""
        pages.append(PageText(page_number=number, text=clean_text(raw)))

    if not any(page.text.strip() for page in pages):
        raise PdfExtractionError(f"{pdf_path.name} has no extractable text (scanned PDF?)")
    return strip_running_headers(pages)


def is_text_sparse(pages: list[PageText], *, threshold: int = SPARSE_CHARS_PER_PAGE) -> bool:
    """True when the document looks like scans rather than real text."""
    if not pages:
        return True
    total = sum(sum(ch.isalnum() for ch in page.text) for page in pages)
    return total / len(pages) < threshold


def render_pages(path: str | Path, *, dpi: int = 200, max_pages: int | None = None) -> list[bytes]:
    """Rasterise pages to PNG bytes for the vision OCR path."""
    import pypdfium2 as pdfium

    pdf_path = Path(path)
    document = pdfium.PdfDocument(str(pdf_path))
    try:
        limit = len(document) if max_pages is None else min(len(document), max_pages)
        images: list[bytes] = []
        for index in range(limit):
            page = document[index]
            bitmap = page.render(scale=dpi / 72)
            image = bitmap.to_pil()
            if image.mode not in {"RGB", "L"}:
                image = image.convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", optimize=True)
            images.append(buffer.getvalue())
            page.close()
        return images
    finally:
        document.close()


def _flatten(pages: list[PageText]) -> tuple[str, list[tuple[int, int, int]]]:
    """Join pages into one string and remember which character range is which page."""
    parts: list[str] = []
    spans: list[tuple[int, int, int]] = []
    cursor = 0
    for page in pages:
        text = page.text.strip()
        if not text:
            continue
        start = cursor
        parts.append(text)
        cursor += len(text)
        spans.append((start, cursor, page.page_number))
        parts.append("\n\n")
        cursor += 2
    return "".join(parts), spans


def _pages_for(spans: list[tuple[int, int, int]], start: int, end: int) -> tuple[int, int]:
    covered = [
        number for span_start, span_end, number in spans if span_start < end and span_end > start
    ]
    if not covered:
        return 1, 1
    return min(covered), max(covered)


def chunk_pages(
    pages: list[PageText],
    *,
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    """Split a document into overlapping, token-bounded chunks with page ranges."""
    document, spans = _flatten(pages)
    if not document.strip():
        raise PdfExtractionError("document contains no text to chunk")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""],
    )

    chunks: list[Chunk] = []
    search_from = 0
    for index, piece in enumerate(splitter.split_text(document)):
        if not piece.strip():
            continue
        start = document.find(piece, search_from)
        if start < 0:  # overlapping chunks can normalise slightly; rescan from 0
            start = document.find(piece)
        if start < 0:
            log.debug("chunk %d could not be located in the source text", index)
            start = search_from
        end = start + len(piece)
        search_from = start + 1
        first, last = _pages_for(spans, start, end)
        chunks.append(
            Chunk(
                index=len(chunks),
                text=piece.strip(),
                start_page=first,
                end_page=last,
                tokens=count_tokens(piece),
            )
        )

    if not chunks:
        raise PdfExtractionError("chunking produced no output")
    log.debug("produced %d chunks from %d pages", len(chunks), len(pages))
    return chunks
