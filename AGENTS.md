# Agent Guide

This file is for automated agents or maintainers who need fast context.

## System goals
- Quote-centric: only return exact sentences from source content.
- Trust/traceability: every quote must map back to `content` offsets.
- Simple pipeline: brute-force search -> LLM review -> assembly.

## Key files
- `main.py`: FastAPI endpoints, SSE stream, thresholds, text report format.
- `search.py`: weighted keyword search.
- `review.py`: LLM review + quote matching + keyword extraction.
- `public/index.html`: test UI, debug tables, copyable text report.
- `scripts/test_quote_match.py`: regression tests for quote matching.

## Constants (main.py)
- `SCORE_MIN = 5.0`
- `SCORE_RECOMMEND = 8.0`
- `SCORE_MUST = 10.0`
- `EXTENDED_LIMIT = 10`

These drive tiers and the text report output.

## SSE event contract
- `log`, `log_skip`
- `candidates`
- `term_search`
- `merge_summary`
- `card_found`
- `done` (includes `results` + `text_report`)

## Text report format
Generated in `_build_text_report`:
- Sections: 推荐先阅读 / 推荐阅读 / 扩展阅读
- Each item:
  - `# <title>`
  - quote (blank lines removed)
  - source URL
  - blank line x2

## Keyword extraction
- Model: `glm-4.7` with `thinking=disabled`.
- Prompt includes a large candidate keyword list; prefer items from the list.
- Fallback: if no content, retry with `GLM-4-Flash`.

## Quote matching
- Must return a substring of `content`.
- Normalization handles curly quotes and whitespace differences.

## UI notes
- UI hardcodes SSE endpoint to `http://127.0.0.1:8002/stream_research`.
- Result cards support “View” to open the full doc with highlighted quote.
