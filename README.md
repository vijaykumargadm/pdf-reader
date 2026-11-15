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
| 4 | Chat with PDF (RAG) | ⬜ next |
| 5 | Streamlit UI | ⬜ |
| 6 | Tests & CI | ⬜ |
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

The pipeline is fully usable from the CLI today:

```powershell
# verify the key and model
python cli.py --check

# summarise one PDF
python cli.py data\uploads\demo.pdf --style detailed --analysis --quotes --out summary.md

# force several chunks to exercise map-reduce
python cli.py data\uploads\demo.pdf --chunk-size 300 --chunk-overlap 40 --stats stats.csv
```

Styles: `brief`, `detailed`, `bullet_points`, `executive`. The Streamlit UI lands in
Phase 5 (`streamlit run app.py`).

## Tests

```powershell
pip install -r requirements-dev.txt
ruff check .
pytest
```

## Layout

```
app.py            Streamlit UI
config.py         Settings, model + pricing tables
pdf_processor.py  Extraction, cleaning, chunking, OCR fallback
summarizer.py     Gemini calls: chunk summaries, combine, analysis, quotes
prompts.py        All ChatPromptTemplate definitions
rag.py            Embeddings, Chroma store, retrieval chain
utils.py          Display, exports, statistics, token accounting
tests/            pytest suite
```

## Security

`.env` is git-ignored and must never be committed. Only `.env.example` is tracked.
