#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from review import parse_review_response


def run_case(name: str, content: str, quote: str, expected_score: float) -> None:
    doc = {"content": content}
    response = {
        "choices": [
            {
                "message": {
                    "content": f"```json\n{{\"quote\": \"{quote}\", \"score\": {expected_score}}}\n```"
                }
            }
        ]
    }
    parsed = parse_review_response(doc, response)
    ok = bool(parsed.get("quote")) and parsed.get("score") == expected_score
    strategy = parsed.get("match_strategy")
    if ok:
        print(f"PASS: {name} (strategy={strategy})")
        return
    print(f"FAIL: {name}")
    print(f"  quote_raw: {parsed.get('quote_raw')}")
    print(f"  quote: {parsed.get('quote')}")
    print(f"  score: {parsed.get('score')}")
    print(f"  strategy: {strategy}")
    sys.exit(1)


def main() -> None:
    case1_content = "爱不是'love'，爱是caritas，爱是慈善。"
    case1_quote = "爱不是‘love’，爱是caritas，爱是慈善。"
    run_case("curly-quote-normalize", case1_content, case1_quote, 10)

    case2_content = "真爱最好办。\n\n一句“我不希望这样”，如果是真爱，那就解决了。"
    case2_quote = "真爱最好办。一句“我不希望这样”，如果是真爱，那就解决了。"
    run_case("whitespace-normalize", case2_content, case2_quote, 8)

    print("All tests passed.")


if __name__ == "__main__":
    main()
