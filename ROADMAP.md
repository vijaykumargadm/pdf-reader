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

## Phase 1 — Skeleton & config ✅

- [x] `config.py` — pydantic-settings, model allow-list, price table
- [x] Logging setup, typed errors (`errors.py`, `logging_setup.py`)
- [x] Pin `requirements.txt`, dev deps in `requirements-dev.txt`
- [x] `pyproject.toml` (ruff + pytest config)
- [x] Modules import cleanly, `ruff check .` passes

**Exit:** ✅ `ruff check .` clean · `pytest` 36 passed.

## Phase 2 — Ingestion ✅

- [x] pypdf extraction with per-page text + metadata
- [x] Cleaning: hyphen joins, whitespace, repeated header/footer removal
- [x] tiktoken token counting (with offline char fallback)
- [x] Recursive chunking with overlap and page ranges
- [x] OCR fallback: render pages with `pypdfium2`, transcribe via Gemini vision
- [x] Tests: chunk boundaries, repeated headers, empty doc, page ranges

**Exit:** ✅ 6-page demo → page-labelled chunks; scanned path wired and unit-tested.

## Phase 3 — Summarisation core ✅

- [x] `ChatPromptTemplate`s, 4 styles (brief, detailed, bullets, executive)
- [x] Map→reduce summarisation with automatic condense rounds
- [x] Retry/backoff for 429 / 404 / transient 5xx, typed errors
- [x] Token + cost ledger from `usage_metadata`
- [x] Structure analysis and verbatim key quotes with page refs
- [x] CLI (`cli.py`) with `--check`, styles, exports, stats CSV

**Exit:** ✅ live run: 5 calls, 2,424 in / 1,424 out, quotes cited by page.

## Phase 4 — Chat with PDF (RAG) ✅

- [x] Gemini embeddings (`gemini-embedding-001`), local Chroma persistence
- [x] Chunk metadata carries page numbers and labels
- [x] Grounded answer chain with inline `[n]` citations
- [x] Refusal when retrieval returns nothing (no model call, no tokens)
- [x] Content-addressed collections so re-indexing never duplicates
- [x] Tests with a local embedding stub

**Exit:** ✅ live: 2 chunks indexed, answer cited `[1] pp.5-6` (score 0.71).

## Phase 5 — Streamlit UI ✅

- [x] Multi-PDF upload, sidebar (model, style, chunk size/overlap, OCR toggle)
- [x] Progress reporting per chunk, per file
- [x] Tabs: Summary · Chat · Structure · Quotes · Stats & Cost
- [x] Downloads: markdown, txt, CSV
- [x] Session-state caching keyed on file digest + settings so re-runs don't re-bill

**Exit:** ✅ `streamlit run app.py` renders; `AppTest` smoke test in the suite.

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
