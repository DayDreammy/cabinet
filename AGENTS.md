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
- `scripts/ps_tool.py`: local-only retrieval CLI (safe snippets + quote offsets).

## Constants (main.py)
- `SCORE_MIN = 1.0`
- `SCORE_RECOMMEND = 8.0`
- `SCORE_MUST = 10.0`
- `EXTENDED_LIMIT = 10`

These drive tiers and the text report output.

## SSE event contract
Quick search (`/stream_research`):
- `log`, `log_skip`
- `candidates`
- `term_search`
- `merge_summary`
- `card_found`
- `done` (includes `results` + `text_report`)

Deep research (`/stream_codex_research`):
- `phase_start`
- `log`
- `stream_log` (structured JSON logs: `event` / `id` / `status` / `ts`)
- `codex_event` (raw Codex JSON event)
- `codex_reasoning`
- `codex_command`
- `codex_message`
- `done` (includes `final_message`, `usage`, trace metadata)

Deep structured log phases (`stream_log.event`):
- `start`
- `progress`
- `complete`
- `error`

Deep streaming status examples (`stream_log.status`):
- `running`, `heartbeat`, `event`, `thought`, `call`, `response`, `turn_completed`, `failed`, `success`, `timeout`

Deep research local-only enforcement:
- Agent is instructed to only retrieve evidence from `data/ps_2026-01-07.json`.
- If Codex attempts forbidden commands (e.g. `curl/wget/pip` or any `http(s)://`), the runner aborts with `failure_reason=forbidden_command`.

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
- UI default service address is `http://127.0.0.1:8002/`.
- UI supports two modes:
  - `Quick Search (cabinet)` -> `/stream_research`
  - `Deep Research (codex ...)` -> `/stream_codex_research`
- Result cards support “View” to open the full doc with highlighted quote.

## Local runbook
Start backend (recommended command):
```bash
cd /home/yy/project/ai_arch_lesson/cabinet_repo && mkdir -p logs && nohup python3 -m uvicorn main:app --host 0.0.0.0 --port 8002 > logs/uvicorn_8002.log 2>&1 & echo $!
```

Check startup log:
```bash
tail -f /home/yy/project/ai_arch_lesson/cabinet_repo/logs/uvicorn_8002.log
```

Open UI:
```text
http://127.0.0.1:8002/
```

CLI-first deep research (no frontend):
```bash
cd /home/yy/project/ai_arch_lesson/cabinet_repo
./scripts/codexr "你的问题" --context "补充上下文"
```

Optional alias for "codex xxxx"-style workflow:
```bash
alias codexx='/home/yy/project/ai_arch_lesson/cabinet_repo/scripts/codexr'
codexx "你的问题"
```

Deep endpoint useful params:
- `timeout_sec`: per-request timeout (default 1200s).
- `sandbox_mode`: `read-only|workspace-write|danger-full-access` (used when `privilege_mode=default`).
- `privilege_mode`: `default|full-auto|danger`.
  - recommended: `danger` (default) to avoid Codex `LandlockRestrict` blocking local command execution.

CLI script params (`scripts/codex_stream_cli.py` / `scripts/codexr`):
- `--timeout-sec`
- `--proxy` / `--unset-proxy`
- `--sandbox-mode`
- `--privilege-mode`
- `--retries` (retry timeout-like failures)
- `--print-final`

NUL-byte safety:
- Deep endpoint sanitizes `query/context` with `text.replace("\\0", "")` before composing the Codex prompt.

## API For Other Callers
Synchronous API (returns final `text_report` + `results`):
- `POST /api/deep_research`
- Request JSON fields: `query`, `context`, `token`
- Auth: compares `token` with env `CABINET_API_TOKEN` (if env is unset, token check is skipped with a warning)

## Local Retrieval CLI
The deep-research agent should prefer these instead of ad-hoc `python3 - <<'PY' ... json.load(...)`:

```bash
cd /home/yy/project/ai_arch_lesson/cabinet_repo
scripts/ps stats
scripts/ps question-grep --contains "如何坦然地面对慢慢变老" --topk 20
scripts/ps search --query "亲密关系 边界" --topk 10
scripts/ps substring-scan --phrase "爱的本质" --phrase "边界" --topk 30
scripts/ps sentence-grep --id <id> --contains "边界" --max-results 20
scripts/ps slice --id <id> --start <quote_start> --end <quote_end>
scripts/ps locate --id <id> --quote "原句摘录" --normalize-quotes
```

Output safety rule:
- Never print/cat full `data/ps_2026-01-07.json` or full `content`; use snippet/preview outputs to avoid context blow-ups and timeouts.
