#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_cases(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("cases file must be a JSON array")
    return data


def stream_research(
    base_url: str,
    query: str,
    top_k: int,
    max_workers: int,
    score_threshold: float,
    chat_url: Optional[str],
    timeout: int,
    max_seconds: int,
) -> Dict[str, Any]:
    params = {
        "query": query,
        "top_k": top_k,
        "max_workers": max_workers,
        "score_threshold": score_threshold,
    }
    if chat_url:
        params["chat_url"] = chat_url
    url = f"{base_url.rstrip('/')}/stream_research?{urllib.parse.urlencode(params)}"

    event_counts: Dict[str, int] = {}
    done_payload: Dict[str, Any] = {}
    logs: List[str] = []
    start_time = time.time()

    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
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
                elif current_event == "done":
                    done_payload = json.loads(payload_text)
                    return {
                        "done": done_payload,
                        "events": event_counts,
                        "logs": logs,
                    }
                data_lines = []
                current_event = ""

    return {
        "done": done_payload,
        "events": event_counts,
        "logs": logs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stream_research E2E cases.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8001",
        help="Cabinet backend base URL.",
    )
    parser.add_argument(
        "--cases",
        default=str(Path(__file__).with_name("stream_research_cases.json")),
        help="Path to JSON cases file.",
    )
    parser.add_argument(
        "--chat-url",
        default="",
        help="Override chat_url passed to /stream_research.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Socket timeout seconds.",
    )
    parser.add_argument(
        "--max-seconds",
        type=int,
        default=180,
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
        top_k = int(case.get("top_k", 5))
        max_workers = int(case.get("max_workers", 5))
        score_threshold = float(case.get("score_threshold", 5.0))

        if not query:
            results.append(
                {
                    "id": case_id,
                    "query": query,
                    "error": "missing query",
                }
            )
            continue

        print(f"{idx}. {case_id} -> {query}")
        try:
            response = stream_research(
                args.base_url,
                query,
                top_k,
                max_workers,
                score_threshold,
                args.chat_url or None,
                args.timeout,
                args.max_seconds,
            )
        except Exception as exc:
            results.append({"id": case_id, "query": query, "error": str(exc)})
            print(f"   error: {exc}")
            continue

        done = response.get("done", {})
        result = {
            "id": case_id,
            "query": query,
            "params": {
                "top_k": top_k,
                "max_workers": max_workers,
                "score_threshold": score_threshold,
            },
            "events": response.get("events", {}),
            "done": done,
            "error": "",
        }
        results.append(result)
        results_count = len(done.get("results", [])) if isinstance(done, dict) else 0
        print(f"   done: results={results_count} elapsed={done.get('elapsed_sec')}")

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
