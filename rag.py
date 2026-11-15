"""Chat with a PDF: Gemini embeddings, a local Chroma store and grounded answers.

The index is rebuilt per document into a content-addressed collection, so the same
file reuses its vectors while a different file never pollutes the results.
"""

from __future__ import annotations

import os

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import hashlib
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import Settings
from errors import ConfigError
from pdf_processor import Chunk
from prompts import NOT_IN_DOCUMENT

if TYPE_CHECKING:
    from summarizer import GeminiSummarizer

log = logging.getLogger(__name__)

#: Cosine relevance below this is treated as "not in the document".
DEFAULT_MIN_SCORE = 0.30
DEFAULT_TOP_K = 5

REFUSAL = (
    "I could not find that in this document. Try rephrasing, or ask about something "
    "that appears in the text."
)


@dataclass(frozen=True)
class Passage:
    """One retrieved chunk, ready to show as a citation."""

    text: str
    page_label: str
    page_number: int
    score: float


@dataclass
class RagAnswer:
    question: str
    answer: str
    sources: list[Passage] = field(default_factory=list)
    grounded: bool = True


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return cleaned[:40] or "document"


def collection_name_for(document_name: str, chunks: Sequence[Chunk]) -> str:
    """Deterministic, Chroma-safe collection name tied to the document content."""
    digest = hashlib.sha1(document_name.encode("utf-8"))
    for chunk in chunks:
        digest.update(chunk.text.encode("utf-8"))
    return f"{_slug(document_name)}-{digest.hexdigest()[:10]}"


class PdfIndex:
    """A persisted Chroma collection over one document's chunks."""

    def __init__(
        self,
        settings: Settings,
        *,
        embeddings: Embeddings | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.settings = settings
        self.collection_name = collection_name or "pdf-chunks"
        self._embeddings = embeddings or self._build_embeddings()
        self._store: Chroma | None = None

    def _build_embeddings(self) -> Embeddings:
        try:
            return GoogleGenerativeAIEmbeddings(
                model=self.settings.embedding_model,
                google_api_key=self.settings.api_key,
            )
        except Exception as exc:  # constructor validation
            raise ConfigError(f"cannot initialise embeddings: {exc}") from exc

    @property
    def store(self) -> Chroma:
        if self._store is None:
            self._store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self._embeddings,
                persist_directory=str(self.settings.chroma_dir),
                collection_metadata={"hnsw:space": "cosine"},
            )
        return self._store

    def build(self, chunks: Sequence[Chunk], document_name: str) -> int:
        """(Re)build the collection for one document. Returns the chunk count."""
        self.collection_name = collection_name_for(document_name, chunks)
        self._store = None  # force re-open with the new collection name
        store = self.store

        existing = store.get(limit=1)
        if existing.get("ids"):
            log.info("resetting existing collection %s", self.collection_name)
            store.reset_collection()

        store.add_texts(
            texts=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "page_number": chunk.start_page,
                    "page_label": chunk.page_label,
                    "chunk_index": chunk.index,
                    "document": document_name,
                }
                for chunk in chunks
            ],
            ids=[f"{self.collection_name}-{chunk.index}" for chunk in chunks],
        )
        log.info("indexed %d chunks into %s", len(chunks), self.collection_name)
        return len(chunks)

    def count(self) -> int:
        return len(self.store.get(limit=10_000).get("ids", []))

    def search(
        self,
        question: str,
        *,
        k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> list[Passage]:
        """Return passages above ``min_score``, best first. Empty list means no match."""
        if not question.strip():
            return []
        if self.count() == 0:
            return []
        results = self.store.similarity_search_with_relevance_scores(question, k=k)
        passages: list[Passage] = []
        for document, score in results:
            if score < min_score:
                continue
            metadata = document.metadata or {}
            passages.append(
                Passage(
                    text=document.page_content,
                    page_label=str(metadata.get("page_label", "?")),
                    page_number=int(metadata.get("page_number", 0) or 0),
                    score=round(float(score), 4),
                )
            )
        return passages


class RagChat:
    """Ties retrieval to the chat model, refusing when nothing relevant is found."""

    def __init__(self, index: PdfIndex, summarizer: GeminiSummarizer) -> None:
        self.index = index
        self.summarizer = summarizer

    def ask(
        self,
        question: str,
        *,
        history: Sequence[tuple[str, str]] | None = None,
        k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> RagAnswer:
        passages = self.index.search(question, k=k, min_score=min_score)
        if not passages:
            log.info("no passages above threshold; refusing to answer")
            return RagAnswer(question=question, answer=REFUSAL, sources=[], grounded=False)

        reply = self.summarizer.answer_question(question, passages, history=history)
        if reply.strip().upper().startswith(NOT_IN_DOCUMENT):
            return RagAnswer(question=question, answer=REFUSAL, sources=passages, grounded=False)
        return RagAnswer(question=question, answer=reply, sources=passages, grounded=True)


def build_index(
    chunks: Sequence[Chunk],
    document_name: str,
    settings: Settings,
    *,
    embeddings: Embeddings | None = None,
) -> PdfIndex:
    """Convenience helper used by the CLI and the UI."""
    index = PdfIndex(settings, embeddings=embeddings)
    index.build(chunks, document_name)
    return index
