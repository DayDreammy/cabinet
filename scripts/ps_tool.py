#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple


REPO_DIR = os.path.dirname(os.path.dirname(__file__))
DEFAULT_DB_PATH = os.path.join(REPO_DIR, "data", "ps_2026-01-07.json")

# Ensure repo root is importable (for `import search`).
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)


QUOTE_NORMALIZE_BASE = str.maketrans(
    {
        ord("“"): '"',
        ord("”"): '"',
        ord("„"): '"',
        ord("‟"): '"',
        ord("«"): '"',
        ord("»"): '"',
        ord("‹"): '"',
        ord("›"): '"',
        ord("‘"): "'",
        ord("’"): "'",
        ord("‚"): "'",
        ord("‛"): "'",
        ord("「"): '"',
        ord("」"): '"',
        ord("『"): '"',
        ord("』"): '"',
        ord("《"): '"',
        ord("》"): '"',
        ord("〈"): '"',
        ord("〉"): '"',
        ord("＂"): '"',
        ord("＇"): "'",
    }
)


def _read_db(path: str) -> List[Dict[str, Any]]:
    # Local-only source: JSON array.
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _doc_by_id(docs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for doc in docs:
        doc_id = doc.get("id")
        if doc_id:
            by_id[str(doc_id)] = doc
    return by_id


def _percentile(values: List[int], p: float) -> int:
    if not values:
        return 0
    values_sorted = sorted(values)
    idx = int((len(values_sorted) - 1) * p)
    return values_sorted[idx]


def _snippet_around(text: str, at: int, window: int) -> str:
    if not text:
        return ""
    if at < 0:
        at = 0
    start = max(0, at - window)
    end = min(len(text), at + window)
    snippet = text[start:end]
    # Make it single-line for log safety.
    return snippet.replace("\n", "\\n")


def _split_sentences_with_offsets(text: str) -> List[Tuple[int, int]]:
    # Very simple sentence splitting that preserves exact substrings + offsets.
    # We treat newline as a boundary too, because many docs are stanza-like.
    seps = set("。！？!?;\n")
    spans: List[Tuple[int, int]] = []
    start = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in seps:
            end = i + 1
            if end > start:
                spans.append((start, end))
            start = end
        i += 1
    if start < len(text):
        spans.append((start, len(text)))
    # Drop empty/whitespace-only spans, but preserve offsets by not trimming text itself.
    cleaned: List[Tuple[int, int]] = []
    for s, e in spans:
        if text[s:e].strip():
            cleaned.append((s, e))
    return cleaned


def cmd_stats(args: argparse.Namespace) -> None:
    path = args.db
    size = os.path.getsize(path) if os.path.exists(path) else 0
    docs = _read_db(path)
    keys = Counter()
    for d in docs[: min(len(docs), 200)]:
        keys.update(d.keys())
    lens = [len((d.get("content") or "")) for d in docs]
    payload = {
        "db_path": path,
        "db_size_bytes": size,
        "docs": len(docs),
        "sample_keys": sorted(keys.keys()),
        "content_len": {
            "min": min(lens) if lens else 0,
            "p50": _percentile(lens, 0.50),
            "p90": _percentile(lens, 0.90),
            "max": max(lens) if lens else 0,
            "mean": int(statistics.mean(lens)) if lens else 0,
        },
    }
    print(json.dumps(payload, ensure_ascii=False))


def cmd_search(args: argparse.Namespace) -> None:
    from search import search_db, tokenize_query

    docs = _read_db(args.db)
    results = search_db(docs, args.query, top_k=args.topk)
    tokens = tokenize_query(args.query)
    rows: List[Dict[str, Any]] = []
    for item in results:
        content = item.get("content") or ""
        pos = -1
        for t in tokens:
            p = content.find(t)
            if p != -1 and (pos == -1 or p < pos):
                pos = p
        if pos == -1:
            pos = 0
        rows.append(
            {
                "id": item.get("id"),
                "title": item.get("title"),
                "question": item.get("question"),
                "url": item.get("url"),
                "search_score": item.get("search_score", 0.0),
                "content_len": len(content),
                "snippet": _snippet_around(content, pos, args.snippet_window),
            }
        )
    print(json.dumps({"query": args.query, "topk": args.topk, "results": rows}, ensure_ascii=False))


def cmd_multi_search(args: argparse.Namespace) -> None:
    from search import search_db_multi, tokenize_query

    docs = _read_db(args.db)
    query_list = [q for q in args.query if q.strip()]
    out: Dict[str, Any] = {"topk": args.topk, "results": {}}
    results = search_db_multi(docs, query_list, top_k=args.topk)
    for q, items in results.items():
        tokens = tokenize_query(q)
        rows: List[Dict[str, Any]] = []
        for item in items:
            content = item.get("content") or ""
            pos = -1
            for t in tokens:
                p = content.find(t)
                if p != -1 and (pos == -1 or p < pos):
                    pos = p
            if pos == -1:
                pos = 0
            rows.append(
                {
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "question": item.get("question"),
                    "url": item.get("url"),
                    "search_score": item.get("search_score", 0.0),
                    "content_len": len(content),
                    "snippet": _snippet_around(content, pos, args.snippet_window),
                }
            )
        out["results"][q] = rows
    print(json.dumps(out, ensure_ascii=False))


def cmd_get(args: argparse.Namespace) -> None:
    docs = _read_db(args.db)
    by_id = _doc_by_id(docs)
    doc = by_id.get(args.id)
    if not doc:
        print(json.dumps({"error": "not_found", "id": args.id}, ensure_ascii=False))
        return
    content = doc.get("content") or ""
    payload = {
        "id": doc.get("id"),
        "title": doc.get("title"),
        "question": doc.get("question"),
        "url": doc.get("url"),
        "publishedAt": doc.get("publishedAt"),
        "updatedAt": doc.get("updatedAt"),
        "content_len": len(content),
        "content_preview": (content[: args.max_chars]).replace("\n", "\\n"),
        "content_truncated": len(content) > args.max_chars,
    }
    print(json.dumps(payload, ensure_ascii=False))


def _find_all(haystack: str, needle: str) -> List[int]:
    if not needle:
        return []
    out: List[int] = []
    start = 0
    while True:
        i = haystack.find(needle, start)
        if i == -1:
            return out
        out.append(i)
        start = i + 1


def cmd_locate(args: argparse.Namespace) -> None:
    docs = _read_db(args.db)
    by_id = _doc_by_id(docs)
    doc = by_id.get(args.id)
    if not doc:
        print(json.dumps({"error": "not_found", "id": args.id}, ensure_ascii=False))
        return
    content = doc.get("content") or ""
    quote = args.quote or ""

    matches = _find_all(content, quote)
    exact = True
    suggested_quote = ""

    if not matches and args.normalize_quotes and quote:
        norm_content = content.translate(QUOTE_NORMALIZE_BASE)
        norm_quote = quote.translate(QUOTE_NORMALIZE_BASE)
        matches = _find_all(norm_content, norm_quote)
        exact = False if matches else True
        if matches:
            # Provide a quote that IS an exact substring of original content.
            s = matches[0]
            suggested_quote = content[s : s + len(quote)]

    payload = {
        "id": doc.get("id"),
        "title": doc.get("title"),
        "url": doc.get("url"),
        "quote": quote,
        "exact_match": exact if matches else False,
        "match_count": len(matches),
        "matches": [{"quote_start": i, "quote_end": i + len(quote)} for i in matches],
    }
    if suggested_quote:
        payload["suggested_quote"] = suggested_quote
    print(json.dumps(payload, ensure_ascii=False))


def cmd_sentence_grep(args: argparse.Namespace) -> None:
    docs = _read_db(args.db)
    by_id = _doc_by_id(docs)
    doc = by_id.get(args.id)
    if not doc:
        print(json.dumps({"error": "not_found", "id": args.id}, ensure_ascii=False))
        return
    content = doc.get("content") or ""
    needle = args.contains or ""
    spans = _split_sentences_with_offsets(content)
    rows: List[Dict[str, Any]] = []
    for s, e in spans:
        seg = content[s:e]
        if needle and needle not in seg:
            continue
        # Provide a small display version; full quote can be fetched via `slice`.
        quote_disp = seg.strip().replace("\n", "\\n")
        if len(quote_disp) > args.max_chars:
            quote_disp = quote_disp[: args.max_chars] + "..."
        row: Dict[str, Any] = {"quote_display": quote_disp, "quote_start": s, "quote_end": e}
        if args.include_quote:
            row["quote"] = seg
        rows.append(row)
        if len(rows) >= args.max_results:
            break
    payload = {
        "id": doc.get("id"),
        "title": doc.get("title"),
        "url": doc.get("url"),
        "contains": needle,
        "results": rows,
    }
    print(json.dumps(payload, ensure_ascii=False))


def cmd_slice(args: argparse.Namespace) -> None:
    docs = _read_db(args.db)
    by_id = _doc_by_id(docs)
    doc = by_id.get(args.id)
    if not doc:
        print(json.dumps({"error": "not_found", "id": args.id}, ensure_ascii=False))
        return
    content = doc.get("content") or ""
    start = max(0, int(args.start))
    end = min(len(content), int(args.end))
    if end < start:
        start, end = end, start
    payload = {
        "id": doc.get("id"),
        "title": doc.get("title"),
        "url": doc.get("url"),
        "quote_start": start,
        "quote_end": end,
        "quote": content[start:end],
    }
    print(json.dumps(payload, ensure_ascii=False))


def cmd_substring_scan(args: argparse.Namespace) -> None:
    docs = _read_db(args.db)
    needles = [n for n in args.phrase if n]
    rows: List[Dict[str, Any]] = []
    for d in docs:
        content = d.get("content") or ""
        score = 0.0
        hits: List[Dict[str, Any]] = []
        for ph in needles:
            n = content.count(ph)
            if not n:
                continue
            w = 0.2 if ph in args.downweight else 1.0
            score += w * n
            hits.append({"phrase": ph, "count": n})
        if score <= 0:
            continue
        rows.append(
            {
                "score": score,
                "id": d.get("id"),
                "title": d.get("title"),
                "question": d.get("question"),
                "url": d.get("url"),
                "content_len": len(content),
                "hits": hits,
            }
        )
    rows.sort(key=lambda r: (-float(r.get("score", 0.0)), -int(r.get("content_len", 0))))
    out = {"phrases": needles, "topk": args.topk, "results": rows[: args.topk]}
    print(json.dumps(out, ensure_ascii=False))


def cmd_question_grep(args: argparse.Namespace) -> None:
    docs = _read_db(args.db)
    needle = args.contains or ""
    rows: List[Dict[str, Any]] = []
    for d in docs:
        q = d.get("question") or ""
        t = d.get("title") or ""
        if needle and (needle not in q) and (needle not in t):
            continue
        rows.append(
            {
                "id": d.get("id"),
                "title": t,
                "question": q,
                "url": d.get("url"),
                "content_len": len(d.get("content") or ""),
            }
        )
        if len(rows) >= args.topk:
            break
    print(json.dumps({"contains": needle, "topk": args.topk, "results": rows}, ensure_ascii=False))


def cmd_content_grep(args: argparse.Namespace) -> None:
    docs = _read_db(args.db)
    needle = args.contains or ""
    rows: List[Dict[str, Any]] = []
    for d in docs:
        content = d.get("content") or ""
        if needle and needle not in content:
            continue
        pos = content.find(needle) if needle else 0
        rows.append(
            {
                "id": d.get("id"),
                "title": d.get("title"),
                "question": d.get("question"),
                "url": d.get("url"),
                "content_len": len(content),
                "first_pos": pos,
                "snippet": _snippet_around(content, pos, args.snippet_window),
            }
        )
        if len(rows) >= args.topk:
            break
    print(
        json.dumps(
            {"contains": needle, "topk": args.topk, "results": rows},
            ensure_ascii=False,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local-only retrieval tools for Cabinet data/*.")
    p.add_argument("--db", default=DEFAULT_DB_PATH, help="Path to local JSON array db.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stats", help="Show db stats (size, doc count, content length).")
    s.set_defaults(func=cmd_stats)

    s = sub.add_parser("search", help="Weighted keyword search (token-based).")
    s.add_argument("--query", required=True)
    s.add_argument("--topk", type=int, default=10)
    s.add_argument("--snippet-window", type=int, default=180)
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("multi-search", help="Search multiple queries in one pass.")
    s.add_argument("--query", action="append", required=True)
    s.add_argument("--topk", type=int, default=8)
    s.add_argument("--snippet-window", type=int, default=160)
    s.set_defaults(func=cmd_multi_search)

    s = sub.add_parser("get", help="Get a doc preview by id (truncated).")
    s.add_argument("--id", required=True)
    s.add_argument("--max-chars", type=int, default=1200)
    s.set_defaults(func=cmd_get)

    s = sub.add_parser("locate", help="Locate a quote (substring) offsets in content.")
    s.add_argument("--id", required=True)
    s.add_argument("--quote", required=True)
    s.add_argument(
        "--normalize-quotes",
        action="store_true",
        help="If exact quote not found, try quote normalization and provide suggested_quote.",
    )
    s.set_defaults(func=cmd_locate)

    s = sub.add_parser("sentence-grep", help="Extract sentence candidates (exact substrings) that contain phrase.")
    s.add_argument("--id", required=True)
    s.add_argument("--contains", required=True)
    s.add_argument("--max-results", type=int, default=20)
    s.add_argument("--max-chars", type=int, default=220)
    s.add_argument(
        "--include-quote",
        action="store_true",
        help="Include full exact quote in output (may be large). Otherwise use `slice` to fetch by offsets.",
    )
    s.set_defaults(func=cmd_sentence_grep)

    s = sub.add_parser("slice", help="Extract exact substring by offsets (quote_start/quote_end).")
    s.add_argument("--id", required=True)
    s.add_argument("--start", required=True, type=int)
    s.add_argument("--end", required=True, type=int)
    s.set_defaults(func=cmd_slice)

    s = sub.add_parser("substring-scan", help="Count literal phrase hits across docs (fast triage).")
    s.add_argument("--phrase", action="append", required=True)
    s.add_argument("--topk", type=int, default=30)
    s.add_argument("--downweight", action="append", default=["爱是"])
    s.set_defaults(func=cmd_substring_scan)

    s = sub.add_parser("question-grep", help="Find docs whose title/question contains literal text.")
    s.add_argument("--contains", required=True)
    s.add_argument("--topk", type=int, default=20)
    s.set_defaults(func=cmd_question_grep)

    s = sub.add_parser("content-grep", help="Find docs whose content contains literal text (with snippet).")
    s.add_argument("--contains", required=True)
    s.add_argument("--topk", type=int, default=20)
    s.add_argument("--snippet-window", type=int, default=180)
    s.set_defaults(func=cmd_content_grep)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
