# Deep Mode SSE Event Contract

Endpoint: `/stream_codex_research`

## Events

- `phase_start`
  - Payload: `{ "phase": "codex_boot", "query": "<query>" }`
  - Meaning: deep mode has started and Codex process is being initialized.

- `log`
  - Payload: plain text
  - Meaning: lifecycle logs, raw non-JSON Codex output, and turn summaries.

- `codex_event`
  - Payload: raw Codex JSON event object (`thread.started`, `item.completed`, etc.)
  - Meaning: full-fidelity event stream for advanced debugging.

- `codex_reasoning`
  - Payload: `{ "text": "<reasoning summary>" }`
  - Meaning: model reasoning item captured from Codex event stream.

- `codex_command`
  - Payload:
    ```json
    {
      "id": "item_x",
      "command": "...",
      "status": "in_progress|completed",
      "exit_code": 0,
      "output": "..."
    }
    ```
  - Meaning: command execution trace from deep run.

- `codex_message`
  - Payload: `{ "text": "<assistant message>" }`
  - Meaning: assistant intermediate/final text chunks.

- `done`
  - Payload:
    ```json
    {
      "query": "...",
      "context": "...",
      "elapsed_sec": 12.34,
      "exit_code": 0,
      "timed_out": false,
      "final_message": "...",
      "messages": ["..."],
      "reasoning": ["..."],
      "commands": [{ "...": "..." }]
    }
    ```
  - Meaning: terminal event containing final artifact and execution trace summary.
