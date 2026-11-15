# pdf-reader

An end-to-end PDF summariser built on **LangChain** and **Google Gemini**, with a
Streamlit UI. Upload one or more PDFs, get a summary in the style you choose, read
structure analysis and key quotes, chat with the document (RAG), and see exactly
what each run cost in tokens.

Built in phases — see [ROADMAP.md](ROADMAP.md) for the plan and current status.

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Toolchain, credentials, repo | ✅ complete |
| 1 | Skeleton & config | ✅ complete |
| 2 | PDF ingestion (+ OCR fallback) | ✅ complete |
| 3 | Summarisation core | ✅ complete |
| 4 | Chat with PDF (RAG) | ✅ complete |
| 5 | Streamlit UI | ✅ complete |
| 6 | Tests & CI | ⬜ next |
| 7 | Docs & release | ⬜ |

## Requirements

- Python 3.12
- A Google AI Studio API key ([aistudio.google.com/apikey](https://aistudio.google.com/apikey))

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env    # then put your key in .env
```

## Run

### Streamlit UI

```powershell
streamlit run app.py
```

Upload one or more PDFs, pick a style, and press **Summarise**. Tabs give you the
summary, chat-with-document (RAG, with page citations), structure analysis, key
quotes, and a token/cost breakdown. Results are cached in session state, so
changing a tab never re-bills you.

### CLI

```powershell
# verify the key and model
python cli.py --check

# summarise one PDF
python cli.py data\uploads\demo.pdf --style detailed --analysis --quotes --out summary.md

# force several chunks to exercise map-reduce
python cli.py data\uploads\demo.pdf --chunk-size 300 --chunk-overlap 40 --stats stats.csv

# ask a question about the document (builds a Chroma index)
python cli.py data\uploads\demo.pdf --ask "What is the biggest risk?"
```

Styles: `brief`, `detailed`, `bullet_points`, `executive`.

## Tests

```powershell
pip install -r requirements-dev.txt
ruff check .
pytest
```

## Layout

```
app.py            Streamlit UI (tabs, uploads, downloads)
cli.py            Command-line entry point
config.py         Settings, model + pricing tables
errors.py         Typed exceptions
logging_setup.py  Logging + key masking
pdf_processor.py  Extraction, cleaning, chunking, OCR rendering
pipeline.py       Orchestration: file -> pages -> chunks -> summary
prompts.py        All ChatPromptTemplate definitions
rag.py            Gemini embeddings, Chroma store, grounded chat
summarizer.py     Gemini calls: chunk summaries, combine, analysis, quotes, RAG
utils.py          Token accounting, cost ledger, formatting helpers
tests/            pytest suite
```

## Security

`.env` is git-ignored and must never be committed. Only `.env.example` is tracked.
