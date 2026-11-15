# Roadmap

Enhanced, end-to-end rebuild inspired by
[pdfsummarizer-gemini](https://github.com/niti007/pdfsummarizer-gemini), extended with
multi-PDF input, OCR fallback, chat-with-PDF (RAG) and token/cost accounting.

Target: local-only Streamlit app, private repo `vijaykumargadm/pdf-reader`,
work on `feat/pdf-summarizer`, merge by PR, tag `v1.0.0`.

## Phase 0 — Gate ✅

- [x] Install Git 2.56, GitHub CLI 2.104, Python 3.12.10 (winget)
- [x] `git config --global` user.name / user.email / init.defaultBranch / autocrlf
- [x] `gh auth login` → account `vijaykumargadm`, scopes `gist`, `read:org`, `repo`
- [x] Rebuild `.venv` on Python 3.12.10 and install `requirements.txt`
- [x] Validate `GEMINI_API_KEY` live: 26 text models visible, `gemini-3.1-flash-lite` returns text
- [x] `.gitignore` protecting `.env`, `.env.example` template
- [x] Private repo created, scaffold pushed

**Exit criteria:** key returns text · `gh auth status` green · `.env` provably untracked · repo reachable.

## Phase 1 — Skeleton & config

- [ ] `config.py` — pydantic-settings, model allow-list, price table
- [ ] Logging setup, typed errors
- [ ] Pin `requirements.txt`, add dev deps (`pytest`, `ruff`)
- [ ] `pyproject.toml` (ruff + pytest config)
- [ ] Empty modules so imports resolve

**Exit:** `python -c "import config, prompts, pdf_processor, summarizer, utils"` clean; `ruff check .` clean.

## Phase 2 — Ingestion

- [ ] pypdf extraction with per-page text + metadata
- [ ] Cleaning: hyphen joins, whitespace, repeated header/footer removal
- [ ] tiktoken token counting
- [ ] Recursive chunking with overlap
- [ ] OCR fallback: render pages with `pypdfium2`, transcribe via Gemini vision
- [ ] Tests: chunk boundaries, empty doc, scanned PDF

**Exit:** fixture PDF → chunks with page numbers; scanned fixture triggers OCR path.

## Phase 3 — Summarisation core

- [ ] `ChatPromptTemplate`s, 4 styles (brief, detailed, bullets, executive)
- [ ] Map→reduce summarisation
- [ ] Retry/backoff for 429 / 404 / transient 5xx
- [ ] Token + cost ledger from `usage_metadata`
- [ ] Structure analysis and key-quote extraction
- [ ] CLI smoke test on a fixture

**Exit:** one command summarises a PDF in all 4 styles with a cost report.

## Phase 4 — Chat with PDF (RAG)

- [ ] Gemini embeddings, local Chroma persistence
- [ ] Chunk metadata carries page numbers
- [ ] Grounded answer chain with citations
- [ ] Refusal when retrieval returns nothing
- [ ] Tests with a stubbed retriever

**Exit:** question → answer with page citations; empty retrieval → explicit "not in the document".

## Phase 5 — Streamlit UI

- [ ] Multi-PDF upload, sidebar (model, style, chunk size/overlap, OCR toggle)
- [ ] Progress reporting per chunk
- [ ] Tabs: Summary · Chat · Structure · Quotes · Stats & Cost
- [ ] Downloads: markdown, txt, CSV
- [ ] Session-state caching so re-runs don't re-bill

**Exit:** full flow works from a cold `streamlit run app.py`.

## Phase 6 — Tests & CI

- [ ] Unit tests (chunking, cleaning, cost math, prompts)
- [ ] Mocked-LLM tests, one opt-in live test (`-m live`)
- [ ] `ruff check` + `pytest` green locally
- [ ] GitHub Actions workflow (needs `workflow` scope on the gh token)

**Exit:** CI green on the PR.

## Phase 7 — Docs & release

- [ ] README: architecture diagram, screenshots, cost notes, troubleshooting
- [ ] PR from `feat/pdf-summarizer` → `main`
- [ ] Tag `v1.0.0`, verify remote state

**Exit:** `main` matches local, tag visible on GitHub, `.env` never in history.
