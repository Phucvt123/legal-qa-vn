---
title: Legal QA VN
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.49.1"
python_version: "3.10"
app_file: app.py
pinned: false
short_description: RAG question answering on Vietnam Labor Code 2019
---

# Vietnamese Labor Law QA Assistant (RAG)

A Retrieval-Augmented Generation (RAG) assistant that answers questions about the Vietnamese Labor Code 2019 (Law No. 45/2019/QH14) with article-level citations, and refuses questions that are out of scope or not supported by the source text.

- Live demo: https://huggingface.co/spaces/Phucvt123/RAG
- Source data: Bo luat Lao dong 2019, 220 articles

> Disclaimer: This project is for learning and demonstration only. Answers are for reference and do not replace professional legal advice.

## Features

- **Grounded answers with citations.** Every answer must cite the article it relies on, in the form `(Điều X, 45/2019/QH14)`. Citations are checked after generation, and a warning is shown if the model cites an article that was not in the retrieved context.
- **Refusal when there is no legal basis.** If the retrieved context does not support an answer, or the question is outside labor law, the assistant replies with a fixed refusal message instead of guessing.
- **OCR-aware ingestion.** The source document is an OCR scan. The cleaning step fixes common OCR mistakes (for example "Điền 194." becoming "Điều 194.") but **never changes digits**, because a single wrong number (06 vs 08 months) changes the legal meaning.
- **Structure validation.** The parser checks that article numbers increase monotonically and reports missing or duplicated articles. The build fails in `--strict` mode if the article count does not match the expected value.
- **Structure-aware chunking.** One article becomes one chunk. Long articles are split along clause boundaries, and each chunk keeps its article header.
- **Streaming and fallback.** Responses are streamed token by token. If the primary LLM hits its rate limit, the app retries and then falls back to a secondary model.

## Architecture

```
            Offline (build_index.py)                          Online (app.py)
 ┌──────────────────────────────────────────┐   ┌───────────────────────────────────────┐
 │ DOCX/PDF -> load -> clean (no digit edit)│   │ Question                              │
 │        -> parse Chapter/Article/Clause   │   │   -> embed (bge-m3, CPU)              │
 │        -> validate (220 articles)        │   │   -> dense top-k (NumPy dot product,  │
 │        -> chunk per article              │   │      active documents only)           │
 │        -> embed (BAAI/bge-m3)            │   │   -> prompt with context              │
 │        -> artifacts/                     │   │   -> LLM (Groq, streaming, fallback)  │
 │           chunks.json, embeddings.npy,   │   │   -> citation check + sources         │
 │           manifest.json                  │   │                                       │
 └──────────────────────────────────────────┘   └───────────────────────────────────────┘
```

At about 230 chunks a vector database is unnecessary, so the index is a NumPy matrix kept in memory. Moving to Qdrant is on the roadmap.

## Design decisions

The pipeline was chosen through experiments in a Colab proof of concept before writing the project code.

| Decision | Options compared | Choice | Reason |
|---|---|---|---|
| Chunking | per article vs fixed 1000 chars (overlap 200) | per article | Higher recall, and chunks map directly to citable units |
| Embedding | bge-m3, vietnamese-bi-encoder, multilingual-e5-base | bge-m3 | Best recall, especially on informal questions |
| Retrieval | BM25, dense, hybrid (RRF), dense + cross-encoder rerank | dense | Highest recall; reranking added ~2.6 s per query without improving recall |
| Out-of-scope guard | reranker score threshold vs LLM refusal | LLM refusal | Score threshold let too many unrelated legal questions through |

Proof-of-concept results (small evaluation set, mostly LLM-generated questions, numbers are indicative only):

| Metric | Value | Sample size |
|---|---|---|
| Recall@5 (dense, test split) | 0.991 | ~106 questions |
| Recall@5 (BM25 / hybrid / rerank) | 0.764 / 0.877 / 0.953 | ~106 questions |
| Answer correctness (LLM-as-judge) | 0.95 | 40 questions |
| Faithfulness (LLM-as-judge) | 1.00 | 40 questions |
| Correct article cited | 0.95 | 40 questions |
| Out-of-scope refusal | 100% | 8 questions |

Retrieval latency on a Colab T4 GPU was about 26 ms for dense retrieval and about 2.6 s with reranking.

## Project structure

```
legal-qa-vn/
├── app.py                   # Gradio app (entry point for Hugging Face Spaces)
├── requirements.txt         # Runtime dependencies (CPU torch)
├── requirements-dev.txt     # Build, OCR and test dependencies
├── pytest.ini
├── .env.example
├── legal_qa/
│   ├── config.py            # Settings from environment variables
│   ├── ingest/
│   │   ├── loaders.py       # DOCX (paragraphs + tables), PDF, TXT
│   │   ├── cleaning.py      # OCR cleanup that never edits digits
│   │   ├── parser.py        # Chapter / Article / Clause parsing and validation
│   │   └── chunker.py       # One article per chunk, split long articles by clause
│   ├── retriever.py         # NumPy dense retrieval, active documents only
│   ├── llm.py               # OpenAI-compatible client, retry, model fallback
│   ├── prompts.py           # System prompt, refusal and disclaimer text
│   └── rag.py               # Retrieval + generation + citation check
├── scripts/
│   ├── build_index.py       # Builds artifacts/ from data/raw/
│   ├── eval_retrieval.py    # Recall@k and MRR@k on data/eval/qa.jsonl
│   └── ocr_pdf.py           # Fallback OCR for scanned PDFs (Tesseract)
├── data/
│   ├── raw/                 # Source documents
│   ├── metadata.csv         # Document number, status, effective date, source
│   ├── ocr_fixes.csv        # Extra phrase fixes (must not contain digits)
│   └── eval/qa.jsonl        # Evaluation questions
├── artifacts/               # Built index (committed for deployment)
│   ├── chunks.json
│   ├── embeddings.npy
│   └── manifest.json
└── tests/
    ├── test_cleaning.py
    ├── test_parser.py
    └── test_chunker.py
```

## Getting started

Requirements: Python 3.10 or 3.11 and a free Groq API key from https://console.groq.com/keys.

```bash
git clone https://github.com/<username>/legal-qa-vn.git
cd legal-qa-vn
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env                 # then fill in GROQ_API_KEY
```

### 1. Run the tests

```bash
pytest
```

### 2. Build the index

Check parsing first without computing embeddings:

```bash
python scripts/build_index.py --skip-embed
```

The expected output is `220 articles / expected 220`. The cleaned text is written to `artifacts/debug/` for manual inspection. If an article is reported missing, search the cleaned text around the reported gap. Fix letter-level OCR errors with a rule in `cleaning.py`, and fix digit errors by hand in the source document.

Then build the full index:

```bash
python scripts/build_index.py --strict
```

Encoding about 230 chunks takes a few minutes on CPU.

### 3. Evaluate retrieval

```bash
python scripts/eval_retrieval.py
```

### 4. Run the app

```bash
python app.py
```

Open http://127.0.0.1:7860.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | (required) | API key for the LLM provider |
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` | Any OpenAI-compatible endpoint |
| `LLM_MODELS` | `openai/gpt-oss-120b,openai/gpt-oss-20b` | Primary model, then fallback models |
| `EMBED_MODEL` | `BAAI/bge-m3` | Must match `artifacts/manifest.json` |
| `TOP_K` | `5` | Number of retrieved chunks |

## OCR fallback

The repository ships with an OCR'd DOCX, so OCR is not needed to run the project. To process a new scanned PDF:

1. Install Tesseract and the Vietnamese language pack (`tesseract-ocr-vie`).
2. Run:

```bash
python scripts/ocr_pdf.py input.pdf data/raw/output.docx --dpi 300 --min-conf 80
```

Low-confidence words are reported so that numbers can be checked against the official text before indexing.

## Deployment

The demo runs on Hugging Face Spaces (Gradio SDK). Embeddings run on CPU, so no GPU quota is used.

1. Create a Gradio Space and add `GROQ_API_KEY` under Settings > Secrets.
2. Track the binary index with Git LFS, then push:

```bash
git lfs install
git lfs track "*.npy"
git add .gitattributes artifacts/
git commit -m "Update index"
git push
```

3. The first start downloads bge-m3 (about 2.3 GB) and takes a few minutes. The Space sleeps when idle, so the first request after a long pause is slow.

The YAML block at the top of this file is the Hugging Face Spaces configuration. GitHub shows it as a small table.

## Limitations

- Only the Labor Code 2019 is indexed. Decrees and circulars that guide it are not included yet, so some detailed questions cannot be answered.
- Each question is answered on its own. Follow-up questions that depend on chat history are not supported.
- The evaluation set is small and mostly LLM-generated, so the reported numbers have wide uncertainty.
- The free Groq tier allows a limited number of requests per day. When the limit is reached, the app reports that it is overloaded.

## Roadmap

- [ ] Index the Labor Code 2012 (expired) and warn when an answer relies on an expired document
- [ ] Add a larger hand-written evaluation set (informal and multi-article questions)
- [ ] Add generation evaluation with a judge model from a different model family
- [ ] Move to FastAPI + Qdrant + Docker Compose
- [ ] Run evaluation in CI on every change to retrieval or prompts

## License

Code: MIT. Vietnamese legal documents are not protected by copyright (Article 15, Law on Intellectual Property).
