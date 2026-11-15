"""All Gemini interaction, wrapped in retries, typed errors and token accounting."""

from __future__ import annotations

import base64
import logging
import random
import time
from collections.abc import Callable, Sequence

from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config import Settings
from errors import ConfigError, ModelNotFoundError, RateLimitError, SummarizationError
from pdf_processor import Chunk, PageText, count_tokens
from prompts import ANALYSIS, CHUNK_SUMMARY, COMBINE, OCR_PAGE, QUOTES, REDUCE, style_instruction
from utils import CostLedger, truncate_to_tokens

log = logging.getLogger(__name__)

#: Above this many tokens of partial summaries we condense before the final merge.
COMBINE_BUDGET_TOKENS = 8_000

#: Material handed to the analysis / quote prompts is capped to keep costs sane.
AUX_INPUT_TOKENS = 8_000

ProgressFn = Callable[[float, str], None]

_RETRY_MARKERS = (
    "429",
    "resource_exhausted",
    "rate limit",
    "500",
    "502",
    "503",
    "504",
    "deadline",
    "unavailable",
    "overloaded",
    "internal error",
    "temporarily",
)
_MODEL_MARKERS = ("404", "not_found", "not found", "does not exist", "unsupported model")
_KEY_MARKERS = ("api_key_invalid", "api key not valid", "permission_denied", "401", "403")


def _message_text(message: AIMessage) -> str:
    """Flatten a LangChain message whose content may be a string or block list."""
    content = message.content
    if isinstance(content, str):
        return content.strip()
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(parts).strip()


def _usage_of(message: AIMessage) -> tuple[int, int]:
    usage = getattr(message, "usage_metadata", None) or {}
    return int(usage.get("input_tokens", 0) or 0), int(usage.get("output_tokens", 0) or 0)


def _classify(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(marker in text for marker in _KEY_MARKERS) and "quota" not in text:
        return "auth"
    if any(marker in text for marker in _MODEL_MARKERS):
        return "model"
    if any(marker in text for marker in _RETRY_MARKERS):
        return "retry"
    return "fatal"


def _group_by_budget(items: Sequence[str], budget: int) -> list[list[str]]:
    """Greedily pack summaries so each group stays under the token budget."""
    groups: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for item in items:
        tokens = count_tokens(item)
        if current and current_tokens + tokens > budget:
            groups.append(current)
            current, current_tokens = [], 0
        current.append(item)
        current_tokens += tokens
    if current:
        groups.append(current)
    return groups


class GeminiSummarizer:
    """One object per document run, so the ledger aggregates every call."""

    def __init__(self, settings: Settings, ledger: CostLedger | None = None) -> None:
        self.settings = settings
        self.ledger = ledger or CostLedger(model=settings.gemini_model, price=settings.price_for())
        self._llm = self._build_llm()

    def _build_llm(self) -> ChatGoogleGenerativeAI:
        try:
            return ChatGoogleGenerativeAI(
                model=self.settings.gemini_model,
                google_api_key=self.settings.api_key,
                temperature=self.settings.temperature,
                max_output_tokens=self.settings.max_output_tokens,
                max_retries=0,  # retries are handled here so backoff is visible
            )
        except Exception as exc:  # constructor validation errors
            raise ConfigError(f"cannot initialise Gemini client: {exc}") from exc

    # ------------------------------------------------------------------ plumbing

    def _invoke(self, messages, label: str) -> str:
        """Call the model with exponential backoff and record token usage."""
        attempts = self.settings.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                response = self._llm.invoke(messages)
            except Exception as exc:  # provider raises many types
                kind = _classify(exc)
                if kind == "auth":
                    raise ConfigError(
                        f"Gemini rejected the API key while running '{label}': {exc}"
                    ) from exc
                if kind == "model":
                    raise ModelNotFoundError(
                        f"model {self.settings.gemini_model!r} is not available to this key "
                        f"(step '{label}'). Set GEMINI_MODEL to one from the model list."
                    ) from exc
                if kind == "retry" and attempt < attempts:
                    delay = self.settings.retry_base_delay * 2 ** (attempt - 1)
                    delay += random.uniform(0, delay * 0.25)
                    log.warning(
                        "step '%s' hit a transient error (attempt %d/%d), retrying in %.1fs: %s",
                        label,
                        attempt,
                        attempts,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
                    continue
                if kind == "retry":
                    raise RateLimitError(
                        f"step '{label}' still failing after {attempts} attempts: {exc}"
                    ) from exc
                raise SummarizationError(f"step '{label}' failed: {exc}") from exc

            input_tokens, output_tokens = _usage_of(response)
            self.ledger.record(label, input_tokens, output_tokens)
            text = _message_text(response)
            if not text:
                raise SummarizationError(f"step '{label}' returned an empty response")
            log.debug("step '%s' ok (%d in / %d out)", label, input_tokens, output_tokens)
            return text

        raise RateLimitError(f"step '{label}' exhausted retries")

    def _render(self, prompt, label: str, **variables) -> str:
        return self._invoke(prompt.format_messages(**variables), label)

    # ------------------------------------------------------------------- steps

    def summarize_chunks(
        self,
        chunks: Sequence[Chunk],
        style: str,
        document_name: str,
        progress: ProgressFn | None = None,
    ) -> list[str]:
        """Map step: one summary per chunk, in document order."""
        instruction = style_instruction(style)
        total = len(chunks)
        summaries: list[str] = []
        for position, chunk in enumerate(chunks, start=1):
            summary = self._render(
                CHUNK_SUMMARY,
                "map",
                style=instruction,
                document_name=document_name,
                page_label=chunk.page_label,
                text=chunk.text,
            )
            summaries.append(summary)
            if progress:
                progress(
                    0.75 * position / total, f"Summarised {chunk.page_label} ({position}/{total})"
                )
        return summaries

    def _condense(self, group: Sequence[str], instruction: str) -> str:
        joined = "\n\n---\n\n".join(group)
        return self._render(REDUCE, "reduce", style=instruction, summaries=joined)

    def combine(
        self,
        summaries: Sequence[str],
        style: str,
        document_name: str,
        progress: ProgressFn | None = None,
    ) -> str:
        """Reduce step: merge partial summaries, condensing first if they are huge."""
        if not summaries:
            raise SummarizationError("nothing to combine: the map step produced no summaries")
        instruction = style_instruction(style)
        pending = list(summaries)

        rounds = 0
        while len(pending) > 1 and count_tokens("\n\n".join(pending)) > COMBINE_BUDGET_TOKENS:
            pending = [
                self._condense(group, instruction)
                for group in _group_by_budget(pending, COMBINE_BUDGET_TOKENS)
            ]
            rounds += 1
            log.debug("condense round %d -> %d partial summaries", rounds, len(pending))
            if rounds >= 6:
                break

        if progress:
            progress(0.85, "Merging partial summaries")
        joined = "\n\n---\n\n".join(f"[{index}] {text}" for index, text in enumerate(pending, 1))
        return self._render(
            COMBINE,
            "combine",
            style=instruction,
            summaries=joined,
            document_name=document_name,
        )

    def transcribe_pages(
        self,
        images: Sequence[bytes],
        *,
        start_page: int = 1,
        progress: ProgressFn | None = None,
    ) -> list[PageText]:
        """OCR fallback: ask the vision model to transcribe rendered page images."""
        pages: list[PageText] = []
        total = len(images)
        for offset, png in enumerate(images):
            page_number = start_page + offset
            messages = OCR_PAGE.format_messages(page_number=page_number)
            encoded = base64.b64encode(png).decode("ascii")
            messages[-1] = HumanMessage(
                content=[
                    {"type": "text", "text": str(messages[-1].content)},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    },
                ]
            )
            text = self._invoke(messages, "ocr")
            if text.strip() == "EMPTY_PAGE":
                text = ""
            pages.append(PageText(page_number=page_number, text=text))
            if progress:
                progress(0.2 * (offset + 1) / total, f"OCR page {page_number}/{total}")
        return pages

    def _aux_material(self, material: Sequence[str]) -> str:
        joined = "\n\n---\n\n".join(material)
        return truncate_to_tokens(joined, AUX_INPUT_TOKENS, count_tokens)

    def analyze_structure(self, material: Sequence[str], progress: ProgressFn | None = None) -> str:
        if progress:
            progress(0.9, "Analysing document structure")
        return self._render(ANALYSIS, "analysis", text=self._aux_material(material))

    def extract_quotes(self, material: Sequence[str], progress: ProgressFn | None = None) -> str:
        if progress:
            progress(0.95, "Extracting key quotes")
        return self._render(QUOTES, "quotes", text=self._aux_material(material))

    def healthcheck(self) -> str:
        """Tiny call used by the UI to prove the key and model still work."""
        return self._render(
            CHUNK_SUMMARY,
            "healthcheck",
            style="Reply with the single word OK.",
            document_name="healthcheck",
            page_label="p.1",
            text="This is a healthcheck.",
        )


__all__ = ["COMBINE_BUDGET_TOKENS", "GeminiSummarizer"]
