from __future__ import annotations

import heapq
import json
import os
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

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


def _ensure_doc_counts(doc: Dict[str, Any]) -> Dict[str, Counter]:
    counts = doc.get("_token_counts")
    if counts is not None:
        return counts
    counts = {
        "title": Counter(tokenize(doc.get("title", ""))),
        "question": Counter(tokenize(doc.get("question", ""))),
        "content": Counter(tokenize(doc.get("content", ""))),
    }
    doc["_token_counts"] = counts
    return counts


def _score_counts(counts: Dict[str, Counter], query_tokens: Iterable[str]) -> float:
    if not query_tokens:
        return 0.0
    return (
        WEIGHTS["title"] * sum(counts["title"].get(token, 0) for token in query_tokens)
        + WEIGHTS["question"]
        * sum(counts["question"].get(token, 0) for token in query_tokens)
        + WEIGHTS["content"]
        * sum(counts["content"].get(token, 0) for token in query_tokens)
    )


def _score_doc(doc: Dict[str, Any], query_tokens: Iterable[str]) -> float:
    counts = _ensure_doc_counts(doc)
    return _score_counts(counts, query_tokens)


def _select_top_k(items: List[Tuple[float, int, Dict[str, Any]]], top_k: int) -> List[Dict[str, Any]]:
    if not items:
        return []
    items.sort(key=lambda row: (-row[0], row[1]))
    return [row[2] for row in items[:top_k]]


def _update_heap(
    heap: List[Tuple[float, int, Dict[str, Any]]],
    score: float,
    doc_idx: int,
    doc: Dict[str, Any],
    top_k: int,
) -> None:
    if len(heap) < top_k:
        item = dict(doc)
        item["search_score"] = score
        heapq.heappush(heap, (score, doc_idx, item))
        return
    if score > heap[0][0]:
        item = dict(doc)
        item["search_score"] = score
        heapq.heapreplace(heap, (score, doc_idx, item))


def search_db_multi(
    docs: List[Dict[str, Any]], queries: List[str], top_k: int = 20
) -> Dict[str, List[Dict[str, Any]]]:
    unique_queries = list(dict.fromkeys(queries))
    token_map = {query: tokenize_query(query) for query in unique_queries}
    heaps: Dict[str, List[Tuple[float, int, Dict[str, Any]]]] = {
        query: [] for query in unique_queries
    }

    for doc_idx, doc in enumerate(docs):
        counts = _ensure_doc_counts(doc)
        for query, tokens in token_map.items():
            score = _score_counts(counts, tokens)
            _update_heap(heaps[query], score, doc_idx, doc, top_k)

    return {query: _select_top_k(heap, top_k) for query, heap in heaps.items()}


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
