from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any, Dict, Iterable, List

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "ps_2026-01-07.json")

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")
WEIGHTS = {"title": 2.0, "question": 1.5, "content": 1.0}

try:
    import jieba  # type: ignore

    def tokenize(text: str) -> List[str]:
        return [t.strip() for t in jieba.cut(text) if t.strip()]

except Exception:  # pragma: no cover - optional dependency

    def tokenize(text: str) -> List[str]:
        return _TOKEN_RE.findall(text)


def tokenize_query(query: str) -> List[str]:
    return sorted(set(tokenize(query)))


def load_db(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def _score_text(text: str, query_tokens: Iterable[str]) -> int:
    if not text:
        return 0
    counts = Counter(tokenize(text))
    return sum(counts.get(token, 0) for token in query_tokens)


def _score_doc(doc: Dict[str, Any], query_tokens: List[str]) -> float:
    title = doc.get("title", "")
    question = doc.get("question", "")
    content = doc.get("content", "")

    return (
        WEIGHTS["title"] * _score_text(title, query_tokens)
        + WEIGHTS["question"] * _score_text(question, query_tokens)
        + WEIGHTS["content"] * _score_text(content, query_tokens)
    )


def search_db(docs: List[Dict[str, Any]], query: str, top_k: int = 20) -> List[Dict[str, Any]]:
    query_tokens = tokenize_query(query)
    scored: List[Dict[str, Any]] = []
    for doc in docs:
        score = _score_doc(doc, query_tokens)
        item = dict(doc)
        item["search_score"] = score
        scored.append(item)

    scored.sort(key=lambda item: item.get("search_score", 0), reverse=True)
    return scored[:top_k]
