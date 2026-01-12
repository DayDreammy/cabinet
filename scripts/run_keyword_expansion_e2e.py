#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
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


def fetch_keywords(
    base_url: str,
    query: str,
    max_keywords: int,
    chat_url: Optional[str],
    timeout: int,
) -> Dict[str, Any]:
    params = {"query": query, "max_keywords": max_keywords}
    if chat_url:
        params["chat_url"] = chat_url
    url = f"{base_url.rstrip('/')}/extract_keywords?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def render_result(item: Dict[str, Any]) -> str:
    keywords = item.get("keywords") or []
    if isinstance(keywords, list):
        keywords_text = ", ".join(str(k) for k in keywords)
    else:
        keywords_text = str(keywords)
    error = item.get("error") or ""
    if error:
        return f"error={error}"
    return f"keywords=[{keywords_text}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run keyword expansion E2E cases.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8002",
        help="Cabinet backend base URL.",
    )
    parser.add_argument(
        "--cases",
        default=str(Path(__file__).with_name("keyword_expansion_cases.json")),
        help="Path to JSON cases file.",
    )
    parser.add_argument(
        "--chat-url",
        default="",
        help="Override chat_url passed to /extract_keywords.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=90,
        help="Request timeout seconds.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional output JSON file to store results.",
    )
    args = parser.parse_args()

    cases_path = Path(args.cases)
    if not cases_path.exists():
        raise FileNotFoundError(f"cases file not found: {cases_path}")

    cases = load_cases(cases_path)
    results = []
    for idx, case in enumerate(cases, start=1):
        case_id = str(case.get("id", f"case-{idx}"))
        query = str(case.get("query", "")).strip()
        max_keywords = int(case.get("max_keywords", 10))
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
            response = fetch_keywords(
                args.base_url, query, max_keywords, args.chat_url or None, args.timeout
            )
        except urllib.error.URLError as exc:
            error = getattr(exc, "reason", str(exc))
            result = {"id": case_id, "query": query, "error": str(error)}
            results.append(result)
            print(f"   error: {error}")
            continue
        except Exception as exc:
            result = {"id": case_id, "query": query, "error": str(exc)}
            results.append(result)
            print(f"   error: {exc}")
            continue

        result = {
            "id": case_id,
            "query": query,
            "keywords": response.get("keywords", []),
            "raw_text": response.get("raw_text", ""),
            "parsed": response.get("parsed", {}),
            "error": response.get("error", ""),
        }
        results.append(result)
        print(f"   {render_result(result)}")
        time.sleep(0.2)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Wrote results to {out_path}")

    failed = [item for item in results if item.get("error")]
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
