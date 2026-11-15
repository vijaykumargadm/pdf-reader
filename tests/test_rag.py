"""RAG: index building, retrieval quality and grounded-answer behaviour.

Uses a deterministic local embedding stub so the suite never calls Gemini.
"""

from __future__ import annotations

import math
import re
import zlib
from dataclasses import dataclass, field

import pytest
from langchain_core.embeddings import Embeddings

from config import Settings
from pdf_processor import Chunk
from rag import REFUSAL, PdfIndex, RagChat, build_index, collection_name_for

DIM = 512


class FakeEmbeddings(Embeddings):
    """Bag-of-words hashing embedding: stable, local, good enough for retrieval tests."""

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * DIM
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vector[zlib.crc32(word.encode()) % DIM] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@dataclass
class StubSummarizer:
    reply: str = "Revenue grew 18 percent [1]."
    calls: list[tuple[str, int]] = field(default_factory=list)

    def answer_question(self, question, passages, *, history=None):
        self.calls.append((question, len(passages)))
        return self.reply


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        gemini_api_key="test-key",
        chroma_dir=tmp_path / "chroma",
        data_dir=tmp_path / "uploads",
    )


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        Chunk(
            index=0,
            text="Northwind revenue grew 18 percent to 4.8 million dollars in the third quarter.",
            start_page=1,
            end_page=1,
            tokens=20,
        ),
        Chunk(
            index=1,
            text="The largest risk is customer concentration and open engineering roles.",
            start_page=5,
            end_page=5,
            tokens=15,
        ),
        Chunk(
            index=2,
            text="Management recommends adding a second database region before the traffic peak.",
            start_page=6,
            end_page=6,
            tokens=15,
        ),
    ]


@pytest.fixture
def index(settings, chunks) -> PdfIndex:
    return build_index(chunks, "demo.pdf", settings, embeddings=FakeEmbeddings())


def test_build_indexes_every_chunk(index, chunks):
    assert index.count() == len(chunks)


def test_search_finds_the_right_page(index):
    passages = index.search("revenue grew 18 percent", k=3, min_score=0.0)
    assert passages
    assert passages[0].page_label == "p.1"
    assert "18 percent" in passages[0].text
    assert 0.0 <= passages[0].score <= 1.0


def test_search_respects_min_score(index):
    assert index.search("zebra quantum bicycle", k=3, min_score=0.9) == []


def test_empty_question_returns_nothing(index):
    assert index.search("   ") == []


def test_rebuild_resets_instead_of_duplicating(settings, chunks):
    first = build_index(chunks, "demo.pdf", settings, embeddings=FakeEmbeddings())
    count = first.count()
    second = build_index(chunks, "demo.pdf", settings, embeddings=FakeEmbeddings())
    assert second.count() == count == len(chunks)


def test_collection_name_is_content_addressed(chunks):
    name = collection_name_for("demo.pdf", chunks)
    assert name.startswith("demo-pdf-")
    assert name == collection_name_for("demo.pdf", chunks)
    other = [Chunk(index=0, text="different", start_page=1, end_page=1, tokens=1)]
    assert collection_name_for("demo.pdf", other) != name


def test_rag_answers_with_citations(index):
    stub = StubSummarizer()
    answer = RagChat(index, stub).ask("how much did revenue grow?", k=2, min_score=0.0)
    assert answer.grounded is True
    assert answer.answer == "Revenue grew 18 percent [1]."
    assert answer.sources
    assert stub.calls and stub.calls[0][1] == 2


def test_rag_refuses_without_retrieval(index):
    stub = StubSummarizer()
    answer = RagChat(index, stub).ask("what is the ceo's shoe size?", min_score=0.99)
    assert answer.grounded is False
    assert answer.answer == REFUSAL
    assert stub.calls == []  # no model call, no tokens spent


def test_rag_treats_model_sentinel_as_refusal(index):
    stub = StubSummarizer(reply="NOT_IN_DOCUMENT")
    answer = RagChat(index, stub).ask("how much did revenue grow?", min_score=0.0)
    assert answer.grounded is False
    assert answer.answer == REFUSAL


def test_rag_passes_history(index):
    stub = StubSummarizer()
    RagChat(index, stub).ask(
        "and the risks?", min_score=0.0, history=[("how much revenue?", "4.8m")]
    )
    assert stub.calls[0][0] == "and the risks?"
