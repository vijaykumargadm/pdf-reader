"""Pipeline orchestration, exercised with a stubbed model layer."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, field

import pytest

import pipeline
from config import Price, Settings
from pdf_processor import Chunk, PageText
from utils import CostLedger


@dataclass
class FakeEngine:
    settings: Settings
    calls: list[str] = field(default_factory=list)
    ledger: CostLedger = field(init=False)

    def __post_init__(self) -> None:
        self.ledger = CostLedger(model=self.settings.gemini_model, price=Price(1.0, 1.0))

    def summarize_chunks(self, chunks, style, document_name, progress=None):
        self.calls.append(f"map:{len(chunks)}:{style}")
        return [f"summary of {chunk.page_label}" for chunk in chunks]

    def combine(self, summaries, style, document_name, progress=None):
        self.calls.append(f"combine:{len(summaries)}")
        return "FINAL SUMMARY"

    def analyze_structure(self, summaries, progress=None):
        self.calls.append("analysis")
        return "## Structure"

    def extract_quotes(self, summaries, progress=None):
        self.calls.append("quotes")
        return "- a quote"

    def transcribe_pages(self, images, start_page=1, progress=None):
        self.calls.append("ocr")
        return [PageText(1, "ocr text")]


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        gemini_api_key="test-key",
        gemini_model="gemini-2.5-flash",
        chunk_size=120,
        chunk_overlap=20,
        data_dir=tmp_path / "uploads",
        chroma_dir=tmp_path / "chroma",
    )


@pytest.fixture
def fake_engine(monkeypatch, settings):
    created: list[FakeEngine] = []

    def factory(_settings):
        engine = FakeEngine(settings=settings)
        created.append(engine)
        return engine

    monkeypatch.setattr(pipeline, "GeminiSummarizer", factory)
    return created


def test_summarize_pdf_wires_map_and_combine(sample_pdf, settings, fake_engine):
    result = pipeline.summarize_pdf(sample_pdf, style="brief", settings=settings)

    assert result.summary == "FINAL SUMMARY"
    assert result.style == "brief"
    assert result.page_count == 3
    assert result.used_ocr is False
    assert result.analysis is None
    assert len(result.chunk_summaries) == len(result.chunks) > 1
    assert fake_engine[0].calls[0].startswith("map:")
    assert fake_engine[0].calls[1].startswith("combine:")


def test_optional_steps_only_run_when_asked(sample_pdf, settings, fake_engine):
    result = pipeline.summarize_pdf(
        sample_pdf, style="detailed", settings=settings, include_analysis=True, include_quotes=True
    )
    assert result.analysis == "## Structure"
    assert result.quotes == "- a quote"
    assert fake_engine[0].calls[-2:] == ["analysis", "quotes"]


def test_progress_reaches_one(sample_pdf, settings, fake_engine):
    seen: list[float] = []
    pipeline.summarize_pdf(sample_pdf, settings=settings, progress=lambda f, _m: seen.append(f))
    assert seen[0] > 0
    assert seen[-1] == 1.0
    assert seen == sorted(seen)


def test_scanned_pdf_falls_back_to_ocr(sample_pdf, settings, fake_engine, monkeypatch):
    monkeypatch.setattr(pipeline, "is_text_sparse", lambda _pages: True)
    monkeypatch.setattr(pipeline, "render_pages", lambda *a, **k: [b"png"])
    result = pipeline.summarize_pdf(sample_pdf, settings=settings)
    assert result.used_ocr is True
    assert "ocr" in fake_engine[0].calls
    assert result.page_count == 3


def test_ocr_disabled_keeps_extracted_text(sample_pdf, settings, fake_engine, monkeypatch):
    monkeypatch.setattr(pipeline, "is_text_sparse", lambda _pages: True)
    settings = settings.model_copy(update={"ocr_enabled": False})
    result = pipeline.summarize_pdf(sample_pdf, settings=settings)
    assert result.used_ocr is False
    assert "ocr" not in fake_engine[0].calls


def test_markdown_report_contains_every_section(sample_pdf, settings, fake_engine):
    result = pipeline.summarize_pdf(
        sample_pdf, settings=settings, include_analysis=True, include_quotes=True
    )
    markdown = result.to_markdown()
    for heading in (
        "## Summary",
        "## Structure analysis",
        "## Key quotes",
        "## Chunk statistics",
        "## Usage by step",
    ):
        assert heading in markdown
    assert "FINAL SUMMARY" in markdown


def test_chunk_dataclass_is_immutable():
    chunk = Chunk(index=0, text="x", start_page=1, end_page=1, tokens=1)
    with pytest.raises(FrozenInstanceError):
        chunk.text = "y"
