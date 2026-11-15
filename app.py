"""Streamlit UI for pdf-reader.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from config import DEFAULT_EMBEDDING_MODEL, KNOWN_MODELS, SUMMARY_STYLES, Settings, get_settings
from errors import PdfReaderError
from logging_setup import configure_logging, mask_secret
from pipeline import SummaryResult, summarize_pdf
from rag import PdfIndex, RagChat, build_index
from summarizer import GeminiSummarizer

log = logging.getLogger(__name__)

st.set_page_config(page_title="PDF Reader", page_icon="📄", layout="wide")

STYLE_LABELS = {
    "brief": "Brief — 2-3 sentences",
    "detailed": "Detailed — key points and conclusions",
    "bullet_points": "Bullet points — scannable list",
    "executive": "Executive — decisions and risks",
}


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "document.pdf"
    return cleaned if cleaned.lower().endswith(".pdf") else f"{cleaned}.pdf"


def _digest(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _settings_or_stop() -> Settings:
    try:
        return get_settings()
    except Exception as exc:  # pydantic ValidationError when .env is missing
        st.error(
            "Configuration problem: "
            f"{exc}\n\nCreate a `.env` file next to `app.py` with `GEMINI_API_KEY=...` "
            "and restart Streamlit."
        )
        st.stop()
        raise


def _sidebar(base: Settings) -> tuple[Settings, str, bool, bool]:
    with st.sidebar:
        st.header("Settings")

        model = st.selectbox(
            "Model",
            options=list(KNOWN_MODELS),
            index=list(KNOWN_MODELS).index(base.gemini_model)
            if base.gemini_model in KNOWN_MODELS
            else 0,
            help="Any model your API key can access. Override the list in config.py.",
        )
        style = st.selectbox(
            "Summary style",
            options=list(SUMMARY_STYLES),
            format_func=lambda value: STYLE_LABELS.get(value, value),
            index=list(SUMMARY_STYLES).index("detailed"),
        )

        with st.expander("Chunking", expanded=False):
            chunk_size = st.slider("Chunk size (tokens)", 200, 8000, base.chunk_size, step=100)
            chunk_overlap = st.slider("Overlap (tokens)", 0, 1000, base.chunk_overlap, step=25)

        ocr_enabled = st.toggle(
            "OCR fallback for scans",
            value=base.ocr_enabled,
            help="Renders pages and transcribes them with the vision model. Costs extra tokens.",
        )
        analysis = st.checkbox("Structure analysis", value=False)
        quotes = st.checkbox("Key quotes", value=False)

        st.divider()
        if st.button("Test API key", use_container_width=True):
            with st.spinner("Calling Gemini…"):
                try:
                    engine = GeminiSummarizer(base)
                    reply = engine.healthcheck()
                    st.success(f"Key works. Reply: {reply[:60]!r}")
                except PdfReaderError as exc:
                    st.error(str(exc))
        st.caption(f"Key: `{mask_secret(base.api_key)}`")
        st.caption(f"Embeddings: `{base.embedding_model or DEFAULT_EMBEDDING_MODEL}`")

        st.divider()
        if st.button("Clear results", use_container_width=True):
            for key in ("results", "indexes", "chats", "errors"):
                st.session_state.pop(key, None)
            st.rerun()

    settings = base.model_copy(
        update={
            "gemini_model": model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "ocr_enabled": ocr_enabled,
        }
    )
    return settings, style, analysis, quotes


def _run_key(digest: str, settings: Settings, style: str, analysis: bool, quotes: bool) -> str:
    return "|".join(
        [
            digest,
            settings.gemini_model,
            style,
            str(settings.chunk_size),
            str(settings.chunk_overlap),
            str(settings.ocr_enabled),
            str(analysis),
            str(quotes),
        ]
    )


def _process_uploads(uploads, settings: Settings, style: str, analysis: bool, quotes: bool) -> None:
    results: dict[str, SummaryResult] = st.session_state.setdefault("results", {})
    errors: dict[str, str] = st.session_state.setdefault("errors", {})
    settings.ensure_dirs()

    for upload in uploads:
        data = upload.getvalue()
        digest = _digest(data)
        key = _run_key(digest, settings, style, analysis, quotes)
        if key in results:
            continue

        path = settings.data_dir / f"{digest[:8]}-{_safe_name(upload.name)}"
        path.write_bytes(data)

        bar = st.progress(0.0, text=f"Preparing {upload.name}…")
        engine = GeminiSummarizer(settings)

        def progress(fraction: float, message: str, _bar=bar, _name: str = upload.name) -> None:
            _bar.progress(min(max(fraction, 0.0), 1.0), text=f"{_name}: {message}")

        try:
            result = summarize_pdf(
                path,
                style=style,
                settings=settings,
                include_analysis=analysis,
                include_quotes=quotes,
                progress=progress,
                engine=engine,
            )
        except PdfReaderError as exc:
            errors[upload.name] = str(exc)
            bar.empty()
            st.error(f"{upload.name}: {exc}")
            continue

        results[key] = result
        errors.pop(upload.name, None)
        bar.empty()
        st.success(
            f"{upload.name}: {result.page_count} pages, {len(result.chunks)} chunks, "
            f"{result.ledger.calls} calls, est. {result.cost_label}"
        )


def _documents() -> dict[str, SummaryResult]:
    results: dict[str, SummaryResult] = st.session_state.get("results", {})
    documents: dict[str, SummaryResult] = {}
    for result in results.values():
        documents[f"{result.document} · {result.style}"] = result
    return documents


def _downloads(result: SummaryResult) -> None:
    left, middle, right = st.columns(3)
    stem = Path(result.document).stem
    left.download_button(
        "Download Markdown",
        data=result.to_markdown(),
        file_name=f"{stem}-summary.md",
        mime="text/markdown",
        use_container_width=True,
    )
    middle.download_button(
        "Download text",
        data=result.summary,
        file_name=f"{stem}-summary.txt",
        mime="text/plain",
        use_container_width=True,
    )
    right.download_button(
        "Download stats CSV",
        data=pd.DataFrame(result.ledger.rows()).to_csv(index=False).encode("utf-8"),
        file_name=f"{stem}-stats.csv",
        mime="text/csv",
        use_container_width=True,
    )


def _summary_tab(result: SummaryResult) -> None:
    st.markdown(result.summary)
    _downloads(result)
    if result.warnings:
        for warning in result.warnings:
            st.warning(warning)


def _index_for(result: SummaryResult, settings: Settings) -> PdfIndex:
    indexes: dict[str, PdfIndex] = st.session_state.setdefault("indexes", {})
    key = f"{result.document}|{result.ledger.model}|{len(result.chunks)}"
    if key not in indexes:
        with st.spinner("Indexing the document for chat…"):
            indexes[key] = build_index(result.chunks, result.document, settings)
    return indexes[key]


def _chat_tab(result: SummaryResult, settings: Settings) -> None:
    chats: dict[str, list[dict]] = st.session_state.setdefault("chats", {})
    history = chats.setdefault(result.document, [])

    st.caption(
        "Answers come only from this document, with page citations. "
        "Embedding calls are not counted in the token ledger."
    )

    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                st.caption(message["sources"])

    question = st.chat_input("Ask a question about this document…")
    if not question:
        return

    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            index = _index_for(result, settings)
            engine = GeminiSummarizer(settings)
            pairs = [
                (turn["content"], history[i + 1]["content"])
                for i, turn in enumerate(history[:-1])
                if turn["role"] == "user" and i + 1 < len(history)
            ]
            with st.spinner("Searching the document…"):
                answer = RagChat(index, engine).ask(question, history=pairs)
        except PdfReaderError as exc:
            st.error(str(exc))
            history.pop()
            return

        st.markdown(answer.answer)
        source_line = ""
        if answer.sources:
            source_line = "Sources: " + ", ".join(
                f"[{i}] {passage.page_label} ({passage.score:.2f})"
                for i, passage in enumerate(answer.sources, start=1)
            )
            st.caption(source_line)
            with st.expander("Retrieved passages"):
                for i, passage in enumerate(answer.sources, start=1):
                    st.markdown(f"**[{i}] {passage.page_label}** · score {passage.score:.2f}")
                    st.text(passage.text[:1200])

    history.append({"role": "assistant", "content": answer.answer, "sources": source_line})


def _stats_tab(result: SummaryResult) -> None:
    first, second, third, fourth = st.columns(4)
    first.metric("Model calls", result.ledger.calls)
    second.metric("Input tokens", f"{result.ledger.input_tokens:,}")
    third.metric("Output tokens", f"{result.ledger.output_tokens:,}")
    fourth.metric("Est. cost", result.cost_label)

    st.subheader("Usage by step")
    st.dataframe(pd.DataFrame(result.ledger.rows()), use_container_width=True, hide_index=True)

    st.subheader("Chunks")
    chunks = pd.DataFrame(
        [
            {
                "chunk": chunk.index + 1,
                "pages": chunk.page_label,
                "tokens": chunk.tokens,
                "characters": len(chunk.text),
            }
            for chunk in result.chunks
        ]
    )
    st.dataframe(chunks, use_container_width=True, hide_index=True)

    if result.cost_usd is None:
        st.info(
            f"No published price for `{result.ledger.model}` in `config.PRICING`, "
            "so the dollar estimate is unavailable. Token counts above are exact."
        )


def main() -> None:
    configure_logging("INFO")
    base = _settings_or_stop()
    settings, style, analysis, quotes = _sidebar(base)

    st.title("📄 PDF Reader")
    st.caption(
        "LangChain + Gemini · map-reduce summaries · structure analysis · verbatim quotes · "
        "chat with citations"
    )

    uploads = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
    if uploads and st.button("Summarise", type="primary"):
        _process_uploads(uploads, settings, style, analysis, quotes)

    documents = _documents()
    if not documents:
        st.info("Upload a PDF and press **Summarise**. Your first run costs a few thousand tokens.")
        return

    label = st.selectbox("Showing results for", list(documents))
    result = documents[label]

    tabs = st.tabs(["Summary", "Chat", "Structure", "Quotes", "Stats & Cost"])
    with tabs[0]:
        _summary_tab(result)
    with tabs[1]:
        _chat_tab(result, settings)
    with tabs[2]:
        if result.analysis:
            st.markdown(result.analysis)
        else:
            st.info(
                "Tick **Structure analysis** in the sidebar and summarise again to see this tab."
            )
    with tabs[3]:
        if result.quotes:
            st.markdown(result.quotes)
        else:
            st.info("Tick **Key quotes** in the sidebar and summarise again to see this tab.")
    with tabs[4]:
        _stats_tab(result)


if __name__ == "__main__":
    main()
