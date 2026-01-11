from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse

from search import DEFAULT_DB_PATH, WEIGHTS, load_db, search_db, tokenize_query
from review import DEFAULT_CHAT_URL, review_doc

app = FastAPI()

DOCS: List[Dict[str, Any]] = []
PUBLIC_DIR = os.path.join(os.path.dirname(__file__), "public")

if os.path.isdir(PUBLIC_DIR):
    app.mount("/public", StaticFiles(directory=PUBLIC_DIR), name="public")


@app.on_event("startup")
def _load_docs() -> None:
    global DOCS
    DOCS = load_db(DEFAULT_DB_PATH)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(PUBLIC_DIR, "index.html"))


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
        yield _format_sse("log", f"search start: {query}")
        tokens = tokenize_query(query)
        yield _format_sse("log", f"query tokens ({len(tokens)}): {tokens}")
        yield _format_sse(
            "log",
            (
                "weights: "
                f"title={WEIGHTS['title']} "
                f"question={WEIGHTS['question']} "
                f"content={WEIGHTS['content']}"
            ),
        )

        candidates = search_db(DOCS, query, top_k=top_k)
        yield _format_sse(
            "log",
            f"search done: {len(candidates)} candidates, start review",
        )
        if candidates:
            preview = ", ".join(
                f"{idx + 1}.{item.get('title', '(untitled)')}[{item.get('search_score', 0):.1f}]"
                for idx, item in enumerate(candidates[:5])
            )
            yield _format_sse("log", f"top candidates: {preview}")

        hits: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            yield _format_sse("log", f"review threads: {max_workers}")
            futures = [executor.submit(review_doc, doc, query, chat_url) for doc in candidates]

            for future in as_completed(futures):
                result = future.result()
                score = result.get("score", 0)
                quote = result.get("quote", "")
                error = result.get("error", "")
                if quote and score >= score_threshold:
                    hits.append(result)
                    quote_len = len(quote)
                    yield _format_sse(
                        "log",
                        (
                            "hit: "
                            f"{result.get('title', '(unknown)')} "
                            f"score={score:.1f} "
                            f"quote_len={quote_len}"
                        ),
                    )
                    yield _format_sse("card_found", result)
                else:
                    title = result.get("title", "(unknown)")
                    if error:
                        yield _format_sse("log_skip", f"skip: {title} (error: {error})")
                    elif not quote:
                        yield _format_sse(
                            "log_skip",
                            f"skip: {title} (no quote, score={score:.1f})",
                        )
                    else:
                        yield _format_sse(
                            "log_skip",
                            f"skip: {title} (score={score:.1f} < {score_threshold})",
                        )

        hits.sort(key=lambda item: item.get("score", 0), reverse=True)
        final_hits = hits[:8]
        yield _format_sse("log", f"hits: {len(hits)}, return: {len(final_hits)}")

        elapsed = round(time.time() - started, 2)
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
