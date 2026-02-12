---
name: cabinet-codex-research
description: Drive Cabinet deep research with transparent Codex streaming. Use when the user wants `codex ...` style deep retrieval (iterative planning, multi-round search, and evidence-first output) instead of quick keyword search, and when process transparency via SSE event streaming is required.
---

# Cabinet Codex Research

Use this skill to execute deep retrieval tasks through a separate Codex process while keeping each operation visible in a live stream.

## Workflow

1. Validate user input:
- Require a clear `query`.
- Accept optional `context` to constrain scope, audience, or depth.

2. Select mode:
- Use quick mode (`/stream_research`) for fast quote extraction from local corpus.
- Use deep mode (`/stream_codex_research`) for iterative investigation (`codex exec --json`).

3. Execute deep mode:
- Call `/stream_codex_research` with `query`, `context`, and sensible `timeout_sec`.
- Consume SSE events and surface them in real time:
  - `phase_start`
  - `log`
  - `codex_reasoning`
  - `codex_command`
  - `codex_message`
  - `done`

4. Deliver final output:
- Use `done.final_message` as the primary deep-research artifact.
- Also expose trace metadata (`exit_code`, `timed_out`, `elapsed_sec`, `commands`).

## Output Rules

- Keep evidence-first style: prefer source quotes, links, and explicit uncertainty.
- Separate facts from assumptions.
- If evidence coverage is incomplete, include unresolved questions and next search steps.

## Prompt Templates

Read `references/prompts.md` for reusable system/user prompt templates.

## Event Contract

Read `references/event-contract.md` for the deep mode SSE contract and UI mapping.
