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


@app.get("/stream_research")
def stream_research(
    query: str = Query(..., min_length=1),
    top_k: int = Query(20, ge=1, le=50),
    max_workers: int = Query(5, ge=1, le=20),
    score_threshold: float = Query(7.0, ge=0.0, le=10.0),
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

        candidates = search_db(DOCS, query, top_k=top_k)
        msg = f"search done: {len(candidates)} candidates, start review"
        _log(msg)
        yield _format_sse("log", msg)

        if candidates:
            candidates_payload = [
                {
                    "id": str(item.get("id", "")),
                    "title": item.get("title", ""),
                    "search_score": round(float(item.get("search_score", 0)), 2),
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
        final_hits = hits[:8]
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
