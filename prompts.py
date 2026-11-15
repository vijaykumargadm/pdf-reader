"""Every prompt used by the app, as LangChain ``ChatPromptTemplate`` objects.

Keeping them here means the UI, the CLI and the tests all send identical
instructions. Add a new summary style by adding an entry to ``STYLES``.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

SYSTEM = (
    "You are a meticulous document analyst. You work only from the text supplied "
    "to you. Never invent facts, names, numbers or citations. If something is not "
    "present in the text, say so plainly instead of guessing. Reply in clean "
    "Markdown with no preamble and no meta commentary."
)

STYLES: dict[str, str] = {
    "brief": (
        "Write a 2-3 sentence overview that captures the document's purpose and "
        "its single most important conclusion."
    ),
    "detailed": (
        "Write a thorough summary covering the key points, the supporting "
        "arguments, any data or figures that matter, and the conclusions. Use "
        "short paragraphs with bold lead-ins."
    ),
    "bullet_points": (
        "Produce a scannable bullet list of the key points. One idea per bullet, "
        "at most 20 words each, ordered by importance. No introduction."
    ),
    "executive": (
        "Write an executive briefing for a decision maker: what this document is, "
        "why it matters, the risks or open questions, and concrete "
        "recommendations. Use a short paragraph plus a bulleted action list."
    ),
}


def _template(human: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", human)])


def style_instruction(style: str) -> str:
    """Return the instruction text for a style, with a helpful error if unknown."""
    try:
        return STYLES[style]
    except KeyError:
        raise KeyError(f"unknown summary style {style!r}; choose from {sorted(STYLES)}") from None


CHUNK_SUMMARY = _template(
    "Summarise the excerpt below.\n\n"
    "Style: {style}\n\n"
    "Document: {document_name}, {page_label}\n\n"
    'Excerpt:\n"""\n{text}\n"""\n\n'
    "Return only the summary of this excerpt."
)

COMBINE = _template(
    "You are merging partial summaries of {document_name} into one final "
    "summary.\n\n"
    "Style: {style}\n\n"
    "Partial summaries, in document order:\n\n{summaries}\n\n"
    "Write the final summary. Remove repetition, resolve ordering, and keep every "
    "distinct fact that matters. Do not mention that the input was partial and do "
    "not describe your process."
)

REDUCE = _template(
    "Condense the following partial summaries into a single tighter summary, "
    "keeping all distinct facts and their document order.\n\n"
    "Style: {style}\n\n"
    "{summaries}\n\n"
    "Return only the condensed summary."
)

ANALYSIS = _template(
    "Analyse the structure of the document below. Cover, in Markdown sections:\n"
    "1. **Document type** - what kind of document this is.\n"
    "2. **Logical sections** - the main parts in order, with a one-line purpose "
    "for each.\n"
    "3. **Argument or narrative flow** - how the document builds its case.\n"
    "4. **Notable elements** - tables, figures, definitions, appendices, "
    "references, if present.\n\n"
    'Material:\n"""\n{text}\n"""'
)

QUOTES = _template(
    "Extract the 5 to 10 most important verbatim quotes from the material below.\n\n"
    "Rules:\n"
    "- Copy the wording exactly, do not paraphrase.\n"
    "- One bullet per quote, in document order.\n"
    "- After each quote, add the source page in parentheses when it is known.\n"
    "- If a quote would be longer than 40 words, trim it with an ellipsis.\n\n"
    'Material:\n"""\n{text}\n"""'
)

#: Sentinel the model must return when the retrieved passages cannot answer.
NOT_IN_DOCUMENT = "NOT_IN_DOCUMENT"

RAG_ANSWER = _template(
    "Answer the question using only the numbered passages below.\n\n"
    "Rules:\n"
    "- Cite the passages you rely on as [1], [2] inline, right after the claim.\n"
    "- Copy figures and names exactly as they appear.\n"
    "- If the passages do not contain the answer, reply with exactly: NOT_IN_DOCUMENT\n"
    "- Keep it under 200 words unless the question asks for detail.\n\n"
    "Passages:\n{passages}\n\n"
    "Earlier conversation:\n{history}\n\n"
    "Question: {question}"
)

OCR_PAGE = _template(
    "Transcribe every word on this page image exactly as it appears.\n\n"
    "Rules:\n"
    "- Preserve reading order, paragraph breaks, headings and list markers.\n"
    "- Do not summarise, translate, correct or comment.\n"
    "- Ignore decorative borders and watermarks.\n"
    "- If the page contains no text, reply with exactly: EMPTY_PAGE\n\n"
    "Page number: {page_number}"
)
