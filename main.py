from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse

from review import (
    DEFAULT_CHAT_URL,
    MODEL_NAME,
    build_review_payload,
    extract_keywords,
    parse_review_response,
    post_json,
    review_doc,
)
from search import DEFAULT_DB_PATH, WEIGHTS, load_db, search_db, tokenize_query

app = FastAPI()

DOCS: List[Dict[str, Any]] = []
DOCS_BY_ID: Dict[str, Dict[str, Any]] = {}
PUBLIC_DIR = os.path.join(os.path.dirname(__file__), "public")
LOGGER = logging.getLogger("uvicorn.error")
LOGGER.setLevel(logging.INFO)

if os.path.isdir(PUBLIC_DIR):
    app.mount("/public", StaticFiles(directory=PUBLIC_DIR), name="public")


@app.on_event("startup")
def _load_docs() -> None:
    global DOCS, DOCS_BY_ID
    DOCS = load_db(DEFAULT_DB_PATH)
    DOCS_BY_ID = {}
    for doc in DOCS:
        doc_id = doc.get("id")
        if doc_id:
            DOCS_BY_ID[str(doc_id)] = doc


def _format_sse(event: str, data: Any) -> str:
    if isinstance(data, str):
        payload = data
    else:
        payload = json.dumps(data, ensure_ascii=False)
    lines = payload.splitlines() or [""]
    formatted = [f"event: {event}\n"]
    formatted.extend(f"data: {line}\n" for line in lines)
    formatted.append("\n")
    return "".join(formatted)


def _log(message: str) -> None:
    LOGGER.info(message)


def _merge_candidates(candidate_lists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for candidates in candidate_lists:
        for doc in candidates:
            doc_id = str(doc.get("id", ""))
            if not doc_id:
                continue
            score = float(doc.get("search_score", 0))
            if doc_id not in merged:
                item = dict(doc)
                item["search_score"] = score
                item["search_hits"] = 1
                merged[doc_id] = item
            else:
                merged_item = merged[doc_id]
                merged_item["search_score"] += score
                merged_item["search_hits"] = merged_item.get("search_hits", 1) + 1

    merged_list = list(merged.values())
    merged_list.sort(key=lambda item: item.get("search_score", 0), reverse=True)
    return merged_list


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(PUBLIC_DIR, "index.html"))


@app.get("/doc/{doc_id}")
def get_doc(doc_id: str) -> Dict[str, Any]:
    doc = DOCS_BY_ID.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="doc not found")
    return {
        "id": doc.get("id", ""),
        "title": doc.get("title", ""),
        "question": doc.get("question", ""),
        "content": doc.get("content", ""),
        "url": doc.get("url", ""),
        "publishedAt": doc.get("publishedAt", ""),
        "updatedAt": doc.get("updatedAt", ""),
    }


@app.get("/debug_review")
def debug_review(
    doc_id: str = Query(..., min_length=1),
    query: str = Query(..., min_length=1),
    chat_url: str = Query(DEFAULT_CHAT_URL),
) -> Dict[str, Any]:
    doc = DOCS_BY_ID.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="doc not found")

    payload = build_review_payload(doc, query, model=MODEL_NAME)
    error = ""
    try:
        response = post_json(chat_url, payload)
    except Exception as exc:
        error = str(exc)
        response = {}

    parsed = parse_review_response(doc, response)
    return {
        "doc_id": doc_id,
        "query": query,
        "payload": payload,
        "response": response,
        "parsed": parsed,
        "error": error,
    }


@app.get("/extract_keywords")
def extract_keywords_api(
    query: str = Query(..., min_length=1),
    max_keywords: int = Query(10, ge=1, le=10),
    chat_url: str = Query(DEFAULT_CHAT_URL),
) -> Dict[str, Any]:
    result = extract_keywords(query, chat_url, max_keywords=max_keywords)
    _log(f"keywords: {result.get('keywords', [])}")
    _log(f"keywords payload: {json.dumps(result.get('payload', {}), ensure_ascii=False)}")
    _log(f"keywords raw_text: {result.get('raw_text', '')}")
    _log(f"keywords parsed: {json.dumps(result.get('parsed', {}), ensure_ascii=False)}")
    _log(f"keywords model: {result.get('keyword_model', '')}")
    _log(
        "keywords response primary: "
        f"{json.dumps(result.get('response_primary', {}), ensure_ascii=False)}"
    )
    if result.get("response_fallback"):
        _log(
            "keywords response fallback: "
            f"{json.dumps(result.get('response_fallback', {}), ensure_ascii=False)}"
        )
    if result.get("response"):
        _log(f"keywords response: {json.dumps(result.get('response', {}), ensure_ascii=False)}")
    if result.get("error"):
        _log(f"keywords error: {result.get('error')}")
    return result


@app.get("/stream_research")
def stream_research(
    query: str = Query(..., min_length=1),
    top_k: int = Query(20, ge=1, le=50),
    max_workers: int = Query(100, ge=1, le=100),
    score_threshold: float = Query(8.0, ge=0.0, le=10.0),
    chat_url: str = Query(DEFAULT_CHAT_URL),
) -> StreamingResponse:
    def event_generator() -> Iterable[str]:
        started = time.time()
        msg = f"search start: {query}"
        _log(msg)
        yield _format_sse("log", msg)

        tokens = tokenize_query(query)
        msg = f"query tokens ({len(tokens)}): {tokens}"
        _log(msg)
        yield _format_sse("log", msg)

        msg = (
            "weights: "
            f"title={WEIGHTS['title']} "
            f"question={WEIGHTS['question']} "
            f"content={WEIGHTS['content']}"
        )
        _log(msg)
        yield _format_sse("log", msg)

        search_terms = [query]
        if len(query) >= 20 or len(tokens) >= 6:
            msg = "long query detected, extracting keywords"
            _log(msg)
            yield _format_sse("log", msg)
            keyword_result = extract_keywords(query, chat_url, max_keywords=10)
            keywords = keyword_result.get("keywords", [])
            if keywords:
                search_terms = keywords
            msg = f"keywords: {keywords}"
            _log(msg)
            yield _format_sse("log", msg)
            _log(
                f"keywords payload: {json.dumps(keyword_result.get('payload', {}), ensure_ascii=False)}"
            )
            _log(f"keywords raw_text: {keyword_result.get('raw_text', '')}")
            _log(
                f"keywords parsed: {json.dumps(keyword_result.get('parsed', {}), ensure_ascii=False)}"
            )
            _log(f"keywords model: {keyword_result.get('keyword_model', '')}")
            _log(
                "keywords response primary: "
                f"{json.dumps(keyword_result.get('response_primary', {}), ensure_ascii=False)}"
            )
            if keyword_result.get("response_fallback"):
                _log(
                    "keywords response fallback: "
                    f"{json.dumps(keyword_result.get('response_fallback', {}), ensure_ascii=False)}"
                )
            if keyword_result.get("response"):
                _log(
                    "keywords response: "
                    f"{json.dumps(keyword_result.get('response', {}), ensure_ascii=False)}"
                )
            if keyword_result.get("error"):
                _log(f"keywords error: {keyword_result.get('error')}")

        search_terms = list(dict.fromkeys(search_terms))
        msg = f"search terms ({len(search_terms)}): {search_terms}"
        _log(msg)
        yield _format_sse("log", msg)

        candidate_lists = []
        for term in search_terms:
            term_candidates = search_db(DOCS, term, top_k=top_k)
            candidate_lists.append(term_candidates)
            term_preview = [
                {
                    "id": str(item.get("id", "")),
                    "title": item.get("title", ""),
                    "search_score": round(float(item.get("search_score", 0)), 2),
                }
                for item in term_candidates[:3]
            ]
            preview = ", ".join(
                f"{idx + 1}.{item.get('title', '(untitled)')}[{item.get('search_score', 0):.1f}]"
                for idx, item in enumerate(term_candidates[:3])
            )
            _log(
                f"term search: {term} -> {len(term_candidates)} candidates | top: {preview}"
            )
            yield _format_sse(
                "term_search",
                {
                    "term": term,
                    "count": len(term_candidates),
                    "top": term_preview,
                },
            )

        candidates = (
            candidate_lists[0]
            if len(candidate_lists) == 1
            else _merge_candidates(candidate_lists)
        )
        if len(candidate_lists) > 1:
            total = sum(len(items) for items in candidate_lists)
            _log(
                "merge summary: "
                f"terms={len(search_terms)} total={total} unique={len(candidates)}"
            )
            yield _format_sse(
                "merge_summary",
                {
                    "terms": len(search_terms),
                    "total": total,
                    "unique": len(candidates),
                },
            )
        msg = f"search done: {len(candidates)} candidates, start review"
        _log(msg)
        yield _format_sse("log", msg)

        if candidates:
            candidates_payload = [
                {
                    "id": str(item.get("id", "")),
                    "title": item.get("title", ""),
                    "search_score": round(float(item.get("search_score", 0)), 2),
                    "search_hits": item.get("search_hits", 1),
                }
                for item in candidates
            ]
            yield _format_sse("candidates", candidates_payload)

            preview = ", ".join(
                f"{idx + 1}.{item.get('title', '(untitled)')}[{item.get('search_score', 0):.1f}]"
                for idx, item in enumerate(candidates[:5])
            )
            msg = f"top candidates: {preview}"
            _log(msg)
            yield _format_sse("log", msg)

        hits: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            msg = f"review threads: {max_workers}"
            _log(msg)
            yield _format_sse("log", msg)

            futures = [executor.submit(review_doc, doc, query, chat_url) for doc in candidates]

            for future in as_completed(futures):
                result = future.result()
                score = result.get("score", 0)
                quote = result.get("quote", "")
                error = result.get("error", "")
                if quote and score >= score_threshold:
                    result["must_read"] = score >= 10
                    hits.append(result)
                    quote_len = len(quote)
                    msg = (
                        "hit: "
                        f"{result.get('title', '(unknown)')} "
                        f"score={score:.1f} "
                        f"quote_len={quote_len}"
                    )
                    _log(msg)
                    yield _format_sse("log", msg)
                    _log("hit: card_found")
                    yield _format_sse("card_found", result)
                else:
                    title = result.get("title", "(unknown)")
                    if error:
                        msg = f"skip: {title} (error: {error})"
                    elif not quote:
                        msg = f"skip: {title} (no quote, score={score:.1f})"
                    else:
                        msg = f"skip: {title} (score={score:.1f} < {score_threshold})"
                    _log(msg)
                    yield _format_sse("log_skip", msg)

        hits.sort(key=lambda item: item.get("score", 0), reverse=True)
        final_hits = hits
        msg = f"hits: {len(hits)}, return: {len(final_hits)}"
        _log(msg)
        yield _format_sse("log", msg)

        elapsed = round(time.time() - started, 2)
        _log("done")
        yield _format_sse(
            "done",
            {
                "query": query,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "elapsed_sec": elapsed,
                "results": final_hits,
            },
        )

    return StreamingResponse(event_generator(), media_type="text/event-stream")
