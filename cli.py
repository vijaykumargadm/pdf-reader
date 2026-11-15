"""Command line entry point.

python cli.py report.pdf --style detailed --analysis --quotes --out summary.md
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

from config import DEFAULT_MODEL, SUMMARY_STYLES, Settings, __version__, get_settings
from errors import PdfReaderError
from logging_setup import configure_logging, mask_secret
from pipeline import summarize_pdf
from summarizer import GeminiSummarizer

log = logging.getLogger("cli")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf-reader",
        description="Summarise a PDF with LangChain + Gemini.",
    )
    parser.add_argument("pdf", nargs="?", type=Path, help="path to the PDF to summarise")
    parser.add_argument("--version", action="version", version=f"pdf-reader {__version__}")
    parser.add_argument(
        "--style",
        choices=SUMMARY_STYLES,
        default="detailed",
        help="summary style (default: detailed)",
    )
    parser.add_argument("--analysis", action="store_true", help="also analyse document structure")
    parser.add_argument("--quotes", action="store_true", help="also extract key quotes")
    parser.add_argument("--out", type=Path, help="write the Markdown report to this path")
    parser.add_argument("--stats", type=Path, help="write per-step token usage as CSV")
    parser.add_argument(
        "--model", default=None, help=f"override the model (default: {DEFAULT_MODEL})"
    )
    parser.add_argument("--chunk-size", type=int, default=None, help="tokens per chunk")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="overlap between chunks")
    parser.add_argument(
        "--ask", default=None, help="ask a question about the PDF after summarising"
    )
    parser.add_argument("--top-k", type=int, default=5, help="passages to retrieve (default: 5)")
    parser.add_argument("--no-ocr", action="store_true", help="skip the OCR fallback for scans")
    parser.add_argument("--quiet", action="store_true", help="no progress output")
    parser.add_argument("--check", action="store_true", help="verify the key and model, then exit")
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    return parser


def _progress(fraction: float, message: str) -> None:
    print(f"\r{fraction * 100:5.1f}%  {message:<58}", end="", file=sys.stderr, flush=True)


def _load_settings(args: argparse.Namespace) -> Settings:
    settings = get_settings()
    updates: dict[str, object] = {}
    if args.model:
        updates["gemini_model"] = args.model
    if args.no_ocr:
        updates["ocr_enabled"] = False
    if args.chunk_size:
        updates["chunk_size"] = args.chunk_size
    if args.chunk_overlap is not None:
        updates["chunk_overlap"] = args.chunk_overlap
    if updates:
        settings = settings.model_copy(update=updates)
    settings.ensure_dirs()
    return settings


def _write_stats(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configure_logging(args.log_level)

    try:
        settings = _load_settings(args)
    except Exception as exc:  # pydantic ValidationError and friends
        print(f"configuration error: {exc}", file=sys.stderr)
        print("check that .env exists and defines GEMINI_API_KEY", file=sys.stderr)
        return 2

    if args.check:
        engine = GeminiSummarizer(settings)
        try:
            answer = engine.healthcheck()
        except PdfReaderError as exc:
            print(f"FAIL  {exc}", file=sys.stderr)
            return 1
        print(f"OK    key {mask_secret(settings.api_key)} | model {settings.gemini_model}")
        print(f"      reply {answer!r}")
        print(f"      {engine.ledger.summary()}")
        return 0

    if args.pdf is None:
        _parser().error("a PDF path is required unless --check is used")
        return 2

    engine = GeminiSummarizer(settings)
    try:
        result = summarize_pdf(
            args.pdf,
            style=args.style,
            settings=settings,
            include_analysis=args.analysis,
            include_quotes=args.quotes,
            progress=None if args.quiet else _progress,
            engine=engine,
        )
    except PdfReaderError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130

    if not args.quiet:
        print(file=sys.stderr)

    print(result.summary.strip())
    print(file=sys.stderr)

    if args.ask:
        from rag import RagChat, build_index

        try:
            index = build_index(result.chunks, result.document, settings)
            reply = RagChat(index, engine).ask(args.ask, k=args.top_k)
        except PdfReaderError as exc:
            print(f"\nrag error: {exc}", file=sys.stderr)
            return 1

        print(f"Q: {args.ask}")
        print()
        print(reply.answer)
        if reply.sources:
            print("\nSources:", file=sys.stderr)
            for position, passage in enumerate(reply.sources, start=1):
                print(
                    f"  [{position}] {passage.page_label} (score {passage.score:.2f})",
                    file=sys.stderr,
                )
        print(file=sys.stderr)

    print(result.ledger.summary(), file=sys.stderr)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result.to_markdown(), encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    if args.stats:
        _write_stats(args.stats, result.ledger.rows())
        print(f"wrote {args.stats}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
