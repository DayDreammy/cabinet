# Cabinet MVP (Golden Pipeline)

A minimal, transparent research assistant that surfaces raw quotes from a private corpus.
It prioritizes recall via brute-force search and accuracy via LLM review, then returns
quote-centric evidence with source links.

## Quick start
1) Start the backend (UI expects port 8002):

```
python -m uvicorn main:app --host 127.0.0.1 --port 8002
```

2) Open the UI:

```
http://127.0.0.1:8002/
```

## Data
- Input dataset: `data/ps_2026-01-07.json`
- Fields: `id`, `title`, `question`, `content`, `url`, `publishedAt`, `updatedAt`, `proofread`

## Core endpoints
- `GET /stream_research?query=...`
  - SSE stream: logs, candidates, per-term search traces, merge summary, and final results
  - Final payload includes `text_report` for copy/paste usage
- `GET /extract_keywords?query=...`
  - LLM-based concept expansion for long questions
- `GET /doc/{id}`
  - Returns full document content for preview/highlight
- `GET /debug_review?doc_id=...&query=...`
  - LLM input/output for review debugging

## LLM integration
- Uses local OpenAI-compatible endpoint: `http://127.0.0.1:8000/v1/chat/completions`
- Review model: `GLM-4-Flash`
- Keyword extraction model: `glm-4.7` with `thinking=disabled`

## Tests
```
python scripts/test_quote_match.py
```
