#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List


def load_cases(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("cases file must be a JSON array")
    return data


def stream_codex_research(
    base_url: str,
    query: str,
    context: str,
    timeout_sec: int,
    socket_timeout: int,
    max_seconds: int,
) -> Dict[str, Any]:
    params = {
        "query": query,
        "context": context,
        "timeout_sec": timeout_sec,
    }
    url = (
        f"{base_url.rstrip('/')}/stream_codex_research?"
        f"{urllib.parse.urlencode(params)}"
    )

    event_counts: Dict[str, int] = {}
    done_payload: Dict[str, Any] = {}
    logs: List[str] = []
    stream_logs: List[Dict[str, Any]] = []
    start_time = time.time()

    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=socket_timeout) as resp:
        current_event = ""
        data_lines: List[str] = []

        while True:
            if max_seconds and (time.time() - start_time) > max_seconds:
                raise TimeoutError(f"stream exceeded {max_seconds}s")

            raw_line = resp.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8").rstrip("\n")

            if line.startswith("event:"):
                current_event = line[len("event:") :].strip()
                continue
            if line.startswith("data:"):
                data_lines.append(line[len("data:") :].lstrip())
                continue
            if line == "":
                if not current_event:
                    data_lines = []
                    continue
                payload_text = "\n".join(data_lines)
                event_counts[current_event] = event_counts.get(current_event, 0) + 1
                if current_event in ("log", "log_skip"):
                    logs.append(payload_text)
                elif current_event == "stream_log":
                    try:
                        stream_logs.append(json.loads(payload_text))
                    except Exception:
                        logs.append(f"invalid stream_log payload: {payload_text}")
                elif current_event == "done":
                    done_payload = json.loads(payload_text)
                    break
                data_lines = []
                current_event = ""

    return {
        "done": done_payload,
        "events": event_counts,
        "logs": logs,
        "stream_logs": stream_logs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run codex deep research E2E cases.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8002",
        help="Cabinet backend base URL.",
    )
    parser.add_argument(
        "--cases",
        default=str(Path(__file__).with_name("codex_deep_research_cases.json")),
        help="Path to JSON cases file.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=180,
        help="timeout_sec passed to /stream_codex_research.",
    )
    parser.add_argument(
        "--socket-timeout",
        type=int,
        default=180,
        help="Socket timeout seconds.",
    )
    parser.add_argument(
        "--max-seconds",
        type=int,
        default=300,
        help="Maximum seconds to wait per stream.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional output JSON file to store results.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Limit number of cases to run (<=0 means all).",
    )
    args = parser.parse_args()

    cases_path = Path(args.cases)
    if not cases_path.exists():
        raise FileNotFoundError(f"cases file not found: {cases_path}")

    cases = load_cases(cases_path)
    if args.limit > 0:
        cases = cases[: args.limit]

    results: List[Dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        case_id = str(case.get("id", f"case-{idx}"))
        query = str(case.get("query", "")).strip()
        context = str(case.get("context", "")).strip()

        if not query:
            results.append({"id": case_id, "query": query, "error": "missing query"})
            continue

        print(f"{idx}. {case_id} -> {query}")
        try:
            response = stream_codex_research(
                args.base_url,
                query,
                context,
                args.timeout_sec,
                args.socket_timeout,
                args.max_seconds,
            )
        except Exception as exc:
            print(f"   error: {exc}")
            results.append({"id": case_id, "query": query, "error": str(exc)})
            continue

        done = response.get("done", {})
        result = {
            "id": case_id,
            "query": query,
            "context": context,
            "events": response.get("events", {}),
            "stream_logs": response.get("stream_logs", []),
            "done": done,
            "error": "",
        }
        results.append(result)
        print(
            "   done: "
            f"exit={done.get('exit_code')} "
            f"elapsed={done.get('elapsed_sec')} "
            f"timed_out={done.get('timed_out')}"
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Wrote results to {out_path}")

    failed = [item for item in results if item.get("error")]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
