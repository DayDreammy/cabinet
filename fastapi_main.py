import asyncio
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from pathlib import Path
from typing import Any
from datetime import datetime

import httpx
try:
    import markdown
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    markdown = None
from bs4 import BeautifulSoup
from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    Query,
    Response,
)
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from review import DEFAULT_CHAT_URL, extract_keywords, review_doc
from search import DEFAULT_DB_PATH, load_db, search_db, search_db_multi, tokenize_query

SCORE_MIN = 1.0
SCORE_RECOMMEND = 8.0
SCORE_MUST = 10.0
EXTENDED_LIMIT = 10
DOCS: list[dict[str, Any]] = []
DOCS_BY_ID: dict[str, dict[str, Any]] = {}


class HelloResponse(BaseModel):
    message: str = Field(..., example="hello, yy")


class ChatMessage(BaseModel):
    role: str = Field(..., example="user")
    content: str = Field(..., example="hello")


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., example="glm-4.7")
    messages: list[ChatMessage]
    stream: bool | None = Field(False, example=False)
    max_tokens: int | None = Field(None, example=256)
    temperature: float | None = Field(None, example=0.7)
    thinking: dict[str, Any] | None = Field(
        default_factory=lambda: {"type": "disabled"},
        description=(
            "Reasoning control payload for GLM. Defaults to "
            "{'type': 'disabled'} to suppress reasoning_content."
        ),
        example={"type": "disabled"},
    )
    agentic: bool | None = Field(True, example=True)

    class Config:
        extra = "allow"


class WebSearchRequest(BaseModel):
    search_query: str = Field(..., max_length=70, example="FastAPI 教程")
    search_engine: str = Field(..., example="search_std")
    search_intent: bool = Field(..., example=False)
    count: int | None = Field(10, ge=1, le=50, example=10)
    search_domain_filter: str | None = None
    search_recency_filter: str | None = Field("noLimit", example="noLimit")
    content_size: str | None = Field(None, example="medium")
    request_id: str | None = None
    user_id: str | None = Field(None, min_length=6, max_length=128)

    class Config:
        extra = "allow"


class ReadPageRequest(BaseModel):
    url: str = Field(..., description="HTTP or HTTPS URL to fetch.")

    class Config:
        extra = "allow"


class CabinetSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, example="恋人在争吵后的冷静期，适合做些什么？")
    top_k: int | None = Field(30, ge=1, le=50, example=30)
    max_workers: int | None = Field(150, ge=1, le=150, example=150)
    score_threshold: float | None = Field(SCORE_MIN, ge=0.0, le=10.0, example=1.0)
    chat_url: str | None = Field(DEFAULT_CHAT_URL, example=DEFAULT_CHAT_URL)
    max_keywords: int | None = Field(10, ge=1, le=10, example=10)

    class Config:
        extra = "allow"


class CabinetFullTextRequest(BaseModel):
    doc_id: str = Field(..., alias="id", min_length=1, example="123")

    class Config:
        allow_population_by_field_name = True
        extra = "allow"


app = FastAPI(
    title="Demo FastAPI",
    description=(
        "Small demo service for greeting users and proxying OpenAI-compatible "
        "chat requests to GLM."
    ),
    version="1.0.0",
)

GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
GLM_SEARCH_URL = "https://open.bigmodel.cn/api/paas/v4/web_search"
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agentic")
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def load_cabinet_docs() -> None:
    global DOCS, DOCS_BY_ID
    DOCS = load_db(DEFAULT_DB_PATH)
    DOCS_BY_ID = {}
    for doc in DOCS:
        doc_id = doc.get("id")
        if doc_id is not None:
            DOCS_BY_ID[str(doc_id)] = doc


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get(
    "/hello",
    response_model=HelloResponse,
    summary="Greet a user by name",
    description=(
        "Returns a greeting message with the provided user name. The name must "
        "be at least 1 character long."
    ),
    response_description="Greeting message payload",
    tags=["demo"],
)
def hello(
    name: str = Query(
        ...,
        min_length=1,
        description="User name to greet.",
        example="yy",
    )
) -> HelloResponse:
    return HelloResponse(message=f"hello, {name}")


@app.post(
    "/v1/chat/completions",
    summary="OpenAI-compatible chat endpoint",
    description=(
        "Forwards OpenAI-style chat completion requests to GLM. When "
        "`agentic=true` (default), the model may call the built-in web search "
        "and read-page tools; supports parallel tool calls and up to 3 tool "
        "rounds. Streaming is not supported in agentic mode."
    ),
    response_model=None,
    tags=["chat"],
)
async def chat_completions(
    background_tasks: BackgroundTasks,
    payload: ChatCompletionRequest,
) -> Response:
    api_key = os.getenv("GLM_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GLM_API_KEY is not set.")

    payload_dict: dict[str, Any] = payload.dict(exclude_none=True)
    ensure_system_prompt(payload_dict)
    messages_for_log = payload_dict.get("messages", [])
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    stream = bool(payload_dict.get("stream"))
    agentic = bool(payload_dict.pop("agentic", True))
    timeout = httpx.Timeout(60.0, connect=10.0)

    if stream and agentic:
        raise HTTPException(
            status_code=400, detail="stream is not supported in agentic mode."
        )

    if stream:
        client = httpx.AsyncClient(timeout=timeout)
        resp = await client.stream(
            "POST",
            GLM_BASE_URL,
            headers=headers,
            json=payload_dict,
        )
        if resp.status_code != 200:
            body_text = (await resp.aread()).decode("utf-8", errors="ignore")
            await resp.aclose()
            await client.aclose()
            return JSONResponse(
                status_code=resp.status_code,
                content={"detail": body_text},
            )
        background_tasks.add_task(resp.aclose)
        background_tasks.add_task(client.aclose)
        return StreamingResponse(resp.aiter_bytes(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=timeout) as client:
        if not agentic:
            logger.info("agentic: disabled, forwarding chat request.")
            resp = await client.post(
                GLM_BASE_URL,
                headers=headers,
                json=payload_dict,
            )
            try:
                data = resp.json()
            except ValueError:
                data = {"detail": resp.text}
            add_html_content(data)
            return JSONResponse(status_code=resp.status_code, content=data)

        logger.info("agentic: enabled, requesting tool decision.")
        payload_dict.setdefault(
            "tools",
            [
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Search the web for up-to-date information.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "search_query": {
                                    "type": "string",
                                    "description": "Query to search for.",
                                },
                                "search_engine": {
                                    "type": "string",
                                    "enum": [
                                        "search_std",
                                        "search_pro",
                                        "search_pro_sogou",
                                        "search_pro_quark",
                                    ],
                                },
                                "search_intent": {
                                    "type": "boolean",
                                    "description": "Whether to use intent detection.",
                                },
                                "count": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 50,
                                },
                                "search_domain_filter": {"type": "string"},
                                "search_recency_filter": {
                                    "type": "string",
                                    "enum": [
                                        "oneDay",
                                        "oneWeek",
                                        "oneMonth",
                                        "oneYear",
                                        "noLimit",
                                    ],
                                },
                                "content_size": {
                                    "type": "string",
                                    "enum": ["medium", "high"],
                                },
                                "request_id": {"type": "string"},
                                "user_id": {"type": "string"},
                            },
                            "required": [
                                "search_query",
                                "search_engine",
                                "search_intent",
                            ],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "cabinet_search",
                        "description": "Search the local cabinet corpus and return quoted answers.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "User query to search for.",
                                },
                                "top_k": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 50,
                                },
                                "max_workers": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 150,
                                },
                                "score_threshold": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 10,
                                },
                                "chat_url": {"type": "string"},
                                "max_keywords": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 10,
                                },
                            },
                            "required": ["query"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "cabinet_full_text",
                        "description": "Fetch the full cabinet JSON item by id.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": "Document id to fetch.",
                                }
                            },
                            "required": ["id"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "read_page",
                        "description": "Fetch a URL and return the main text content.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "url": {
                                    "type": "string",
                                    "description": "HTTP or HTTPS URL to read.",
                                }
                            },
                            "required": ["url"],
                        },
                    },
                },
            ],
        )
        payload_dict.setdefault("tool_choice", "auto")

        messages = list(payload_dict.get("messages", []))
        for round_index in range(1, 7):
            logger.info("--【第%s轮】--", round_index)
            if round_index == 1:
                log_user_message(messages_for_log)
            resp = await client.post(
                GLM_BASE_URL,
                headers=headers,
                json=payload_dict,
            )
            try:
                data = resp.json()
            except ValueError:
                data = {"detail": resp.text}

            if resp.status_code != 200:
                return JSONResponse(status_code=resp.status_code, content=data)

            log_model_answer(data)
            add_html_content(data)
            tool_calls = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("tool_calls", [])
                or []
            )
            if not tool_calls:
                content = (
                    data.get("choices", [{}])[0].get("message", {}).get("content", "")
                )
                parsed_calls = parse_tool_calls_from_content(content)
                if parsed_calls:
                    logger.info("agentic: parsed tool call markup from content.")
                    tool_calls = parsed_calls
                else:
                    logger.info("agentic: no tool call requested by model.")
                    add_html_content(data)
                    return JSONResponse(status_code=resp.status_code, content=data)

            tool_call_payloads = await build_tool_calls(
                tool_calls,
                headers,
                timeout,
                round_index,
            )
            if not tool_call_payloads:
                logger.info("agentic: no supported tool calls executed.")
                add_html_content(data)
                return JSONResponse(status_code=resp.status_code, content=data)

            assistant_tool_calls, tool_messages, tool_outputs = tool_call_payloads
            messages.append({"role": "assistant", "tool_calls": assistant_tool_calls})
            messages.extend(tool_messages)

            payload_dict["messages"] = messages
            payload_dict["stream"] = False

        logger.info("agentic: max tool rounds reached.")
        add_html_content(data)
        return JSONResponse(status_code=resp.status_code, content=data)


@app.post(
    "/web_search",
    summary="Web search proxy",
    description=(
        "Forwards web search requests to GLM Web Search API. Set the "
        "`GLM_API_KEY` environment variable before calling."
    ),
    response_model=None,
    tags=["search"],
)
async def web_search(payload: WebSearchRequest) -> Response:
    api_key = os.getenv("GLM_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GLM_API_KEY is not set.")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload_dict: dict[str, Any] = payload.dict(exclude_none=True)
    timeout = httpx.Timeout(60.0, connect=10.0)
    result = await web_search_internal(payload_dict, headers, timeout)
    return JSONResponse(status_code=result["status_code"], content=result["data"])


async def web_search_internal(
    payload_dict: dict[str, Any],
    headers: dict[str, str],
    timeout: httpx.Timeout,
) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            GLM_SEARCH_URL,
            headers=headers,
            json=payload_dict,
        )
    try:
        data = resp.json()
    except ValueError:
        data = {"detail": resp.text}
    return {"status_code": resp.status_code, "data": data}


@app.post(
    "/cabinet_search",
    summary="Cabinet search tool",
    description=(
        "Searches the local cabinet corpus, reviews candidates, and returns "
        "quoted answers with scores."
    ),
    response_model=None,
    tags=["tools"],
)
async def cabinet_search(payload: CabinetSearchRequest) -> Response:
    result = await cabinet_search_internal(payload.dict(exclude_none=True))
    return JSONResponse(status_code=result["status_code"], content=result["data"])


async def cabinet_search_internal(payload_dict: dict[str, Any]) -> dict[str, Any]:
    def run_search() -> dict[str, Any]:
        query = str(payload_dict.get("query", "")).strip()
        if not query:
            return {"status_code": 400, "data": {"detail": "query is required"}}
        top_k = int(payload_dict.get("top_k", 30))
        max_workers = int(payload_dict.get("max_workers", 150))
        score_threshold = float(payload_dict.get("score_threshold", SCORE_MIN))
        chat_url = str(payload_dict.get("chat_url") or DEFAULT_CHAT_URL)
        max_keywords = int(payload_dict.get("max_keywords", 10))

        started = time.time()
        tokens = tokenize_query(query)
        search_terms = [query]
        if len(query) >= 20 or len(tokens) >= 6:
            keyword_result = extract_keywords(query, chat_url, max_keywords=max_keywords)
            keywords = keyword_result.get("keywords", [])
            if keywords:
                search_terms = keywords

        search_terms = list(dict.fromkeys(search_terms))
        candidate_lists = []
        term_candidates_map = (
            search_db_multi(DOCS, search_terms, top_k=top_k)
            if len(search_terms) > 1
            else {}
        )
        for term in search_terms:
            if term_candidates_map:
                term_candidates = term_candidates_map.get(term, [])
            else:
                term_candidates = search_db(DOCS, term, top_k=top_k)
            candidate_lists.append(term_candidates)

        if not candidate_lists:
            candidates = []
        elif len(candidate_lists) == 1:
            candidates = candidate_lists[0]
        else:
            candidates = _merge_candidates(candidate_lists)

        hits: list[dict[str, Any]] = []
        if candidates:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(review_doc, doc, query, chat_url)
                    for doc in candidates
                ]
                for future in as_completed(futures):
                    result = future.result()
                    score = result.get("score", 0)
                    quote = result.get("quote", "")
                    if quote and score > 0 and score >= score_threshold:
                        result["must_read"] = score >= SCORE_MUST
                        result["tier"] = (
                            "core" if score >= SCORE_RECOMMEND else "extended"
                        )
                        hits.append(result)

        hits.sort(key=lambda item: item.get("score", 0), reverse=True)
        text_report = _build_text_report(hits)
        elapsed = round(time.time() - started, 2)
        return {
            "status_code": 200,
            "data": {
                "query": query,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "elapsed_sec": elapsed,
                "results": hits,
                "text_report": text_report,
            },
        }

    return await asyncio.to_thread(run_search)


@app.post(
    "/cabinet_full_text",
    summary="Cabinet full text tool",
    description="Fetches a cabinet JSON item by id.",
    response_model=None,
    tags=["tools"],
)
async def cabinet_full_text(payload: CabinetFullTextRequest) -> Response:
    result = await cabinet_full_text_internal(payload.doc_id)
    return JSONResponse(status_code=result["status_code"], content=result["data"])


async def cabinet_full_text_internal(doc_id: str) -> dict[str, Any]:
    doc = DOCS_BY_ID.get(str(doc_id))
    if not doc:
        return {"status_code": 404, "data": {"detail": "doc not found"}}
    data = dict(doc)
    data.pop("_token_counts", None)
    return {"status_code": 200, "data": data}


@app.post(
    "/read_page",
    summary="Read page proxy",
    description="Fetches a URL and extracts the main text content.",
    response_model=None,
    tags=["tools"],
)
async def read_page(payload: ReadPageRequest) -> Response:
    timeout = httpx.Timeout(60.0, connect=10.0)
    result = await read_page_internal(payload.url, timeout)
    return JSONResponse(status_code=result["status_code"], content=result["data"])


async def read_page_internal(
    url: str,
    timeout: httpx.Timeout,
) -> dict[str, Any]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return {
            "status_code": 400,
            "data": {"detail": "Invalid URL scheme. Use http or https."},
        }

    headers = {
        "User-Agent": "demo2-fastapi/1.0 (+https://example.local)",
        "Accept": "text/html,application/xhtml+xml",
    }
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        resp = await client.get(url, headers=headers)

    html = resp.text or ""
    title, content, truncated = extract_main_text(html)
    return {
        "status_code": resp.status_code,
        "data": {
            "url": str(resp.url),
            "title": title,
            "content": content,
            "content_truncated": truncated,
        },
    }


async def build_tool_calls(
    tool_calls: list[dict[str, Any]],
    headers: dict[str, str],
    timeout: httpx.Timeout,
    round_index: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, str]],
] | None:
    tasks: list[asyncio.Task] = []
    call_specs: list[dict[str, Any]] = []
    assistant_tool_calls: list[dict[str, Any]] = []

    for tool_call in tool_calls:
        function_info = tool_call.get("function", {})
        function_name = function_info.get("name")
        if function_name in {"web_search", "search"}:
            display_name = "web_search"
        elif function_name in {"cabinet_search"}:
            display_name = "cabinet_search"
        elif function_name in {"cabinet_full_text"}:
            display_name = "cabinet_full_text"
        elif function_name in {"read_page"}:
            display_name = "read_page"
        else:
            logger.info(
                "agentic: tool call requested: %s (unsupported).",
                function_name,
            )
            continue

        raw_arguments = function_info.get("arguments", "{}")
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                logger.info("agentic: tool arguments invalid JSON: %s", raw_arguments)
                continue
        elif isinstance(raw_arguments, dict):
            arguments = raw_arguments
        else:
            logger.info("agentic: tool arguments invalid format: %s", type(raw_arguments))
            continue

        if display_name == "web_search":
            if not arguments.get("search_query"):
                logger.info("agentic: tool arguments missing search_query.")
                continue
            if not arguments.get("search_engine"):
                arguments["search_engine"] = "search_std"
            if "search_intent" not in arguments or arguments["search_intent"] is None:
                arguments["search_intent"] = False
            normalize_search_arguments(arguments)
        elif display_name == "cabinet_search":
            if not arguments.get("query"):
                logger.info("agentic: tool arguments missing query.")
                continue
            if "top_k" not in arguments or arguments["top_k"] is None:
                arguments["top_k"] = 30
            if "max_workers" not in arguments or arguments["max_workers"] is None:
                arguments["max_workers"] = 150
            if "score_threshold" not in arguments or arguments["score_threshold"] is None:
                arguments["score_threshold"] = SCORE_MIN
            if "max_keywords" not in arguments or arguments["max_keywords"] is None:
                arguments["max_keywords"] = 10
        elif display_name == "cabinet_full_text":
            if not arguments.get("id"):
                if arguments.get("doc_id"):
                    arguments["id"] = arguments.get("doc_id")
                else:
                    logger.info("agentic: tool arguments missing id.")
                    continue
        elif display_name == "read_page":
            if not arguments.get("url"):
                logger.info("agentic: tool arguments missing url.")
                continue

        tool_call_id = tool_call.get("id") or f"call_tool_{len(call_specs) + 1}"
        assistant_tool_calls.append(
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": display_name,
                    "arguments": json.dumps(arguments, ensure_ascii=True),
                },
            }
        )
        call_specs.append(
            {
                "id": tool_call_id,
                "name": display_name,
                "arguments": arguments,
            }
        )

        if display_name == "web_search":
            logger.info(
                "[Agent] Decided to call tool: '%s' query='%s'",
                display_name,
                truncate_query(arguments.get("search_query")),
            )
            logger.info(
                "agentic: calling tool=%s via /web_search args=%s",
                display_name,
                json.dumps(arguments, ensure_ascii=False),
            )
            tasks.append(
                asyncio.create_task(
                    web_search_internal(arguments, headers, timeout)
                )
            )
        elif display_name == "cabinet_search":
            logger.info(
                "[Agent] Decided to call tool: '%s' query='%s'",
                display_name,
                truncate_query(arguments.get("query")),
            )
            logger.info(
                "agentic: calling tool=%s via /cabinet_search args=%s",
                display_name,
                json.dumps(arguments, ensure_ascii=False),
            )
            tasks.append(
                asyncio.create_task(
                    cabinet_search_internal(arguments)
                )
            )
        elif display_name == "cabinet_full_text":
            logger.info(
                "[Agent] Decided to call tool: '%s' id='%s'",
                display_name,
                truncate_query(arguments.get("id")),
            )
            logger.info(
                "agentic: calling tool=%s via /cabinet_full_text args=%s",
                display_name,
                json.dumps(arguments, ensure_ascii=False),
            )
            tasks.append(
                asyncio.create_task(
                    cabinet_full_text_internal(arguments.get("id", ""))
                )
            )
        elif display_name == "read_page":
            logger.info(
                "[Agent] Decided to call tool: '%s' url='%s'",
                display_name,
                truncate_url(arguments.get("url")),
            )
            logger.info(
                "agentic: calling tool=%s via /read_page args=%s",
                display_name,
                json.dumps(arguments, ensure_ascii=False),
            )
            tasks.append(
                asyncio.create_task(
                    read_page_internal(arguments.get("url"), timeout)
                )
            )

    if not tasks:
        return None

    results = await asyncio.gather(*tasks)
    tool_messages: list[dict[str, Any]] = []
    tool_outputs: list[dict[str, str]] = []
    for spec, result in zip(call_specs, results):
        logger.info(
            "agentic: tool result tool=%s status=%s",
            spec["name"],
            result["status_code"],
        )
        if result["status_code"] != 200:
            continue
        output_payload = json.dumps(result["data"], ensure_ascii=False)
        tool_outputs.append({"name": spec["name"], "output": output_payload})
        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": spec["id"],
                "name": spec["name"],
                "content": json.dumps(result["data"], ensure_ascii=True),
            }
        )

    if not tool_messages:
        return None

    for tool in tool_outputs:
        tool_name = tool["name"]
        output = tool["output"]
        logger.info(
            "[System] Tool Output (%s): '%s'",
            tool_name,
            truncate_tool_output(tool_name, output),
        )

    return assistant_tool_calls, tool_messages, tool_outputs


def parse_tool_calls_from_content(content: str) -> list[dict[str, Any]]:
    if not content:
        return []

    if "&lt;tool_call&gt;" in content:
        content = (
            content.replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&amp;", "&")
        )

    tool_pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    pair_pattern = re.compile(
        r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL
    )
    tool_calls: list[dict[str, Any]] = []

    for match_index, match in enumerate(tool_pattern.finditer(content), start=1):
        inner = match.group(1).strip()
        name_match = re.match(r"^([^\n<]+)", inner)
        if not name_match:
            continue
        name = name_match.group(1).strip()
        arguments: dict[str, Any] = {}
        for key, value in pair_pattern.findall(inner):
            arguments[key.strip()] = value.strip()
        if not arguments:
            continue
        tool_calls.append(
            {
                "id": f"call_markup_{match_index}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=True),
                },
            }
        )

    return tool_calls


def normalize_search_arguments(arguments: dict[str, Any]) -> None:
    intent_value = arguments.get("search_intent")
    if isinstance(intent_value, str):
        arguments["search_intent"] = intent_value.strip().lower() in {
            "true",
            "1",
            "yes",
        }

    count_value = arguments.get("count")
    if isinstance(count_value, str) and count_value.isdigit():
        arguments["count"] = int(count_value)


def extract_main_text(html: str, max_chars: int = 4000) -> tuple[str, str, bool]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)

    candidates = [node for node in (soup.find("article"), soup.find("main"), soup.body) if node]
    if not candidates:
        candidates = [soup]

    def clean_text(text: str) -> str:
        lines = [line.strip() for line in text.splitlines()]
        return "\n".join([line for line in lines if line])

    best_text = ""
    for node in candidates:
        text = clean_text(node.get_text(separator="\n"))
        if len(text) > len(best_text):
            best_text = text

    truncated = False
    if len(best_text) > max_chars:
        best_text = f"{best_text[:max_chars].rstrip()}…"
        truncated = True

    return title, best_text, truncated


def truncate_query(query: Any, limit: int = 10) -> str:
    if not query:
        return ""
    text = str(query)
    return text if len(text) <= limit else f"{text[:limit]}…"


def truncate_url(url: Any, limit: int = 60) -> str:
    if not url:
        return ""
    text = str(url)
    return text if len(text) <= limit else f"{text[:limit]}…"


def truncate_search_content(output: str, limit: int = 20) -> str:
    if not output:
        return ""
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        text = output.strip().replace("\n", " ")
        return text if len(text) <= limit else f"{text[:limit]}…"

    results = data.get("search_result") or []
    if not results:
        text = json.dumps(data, ensure_ascii=False)
        return text if len(text) <= limit else f"{text[:limit]}…"

    snippet_parts: list[str] = []
    for item in results:
        content = (item or {}).get("content") or ""
        if content:
            snippet = content.strip().replace("\n", " ")
            snippet_parts.append(snippet[:limit])
        if len(snippet_parts) >= 3:
            break
    preview = " | ".join(snippet_parts)
    return preview if len(preview) <= limit else f"{preview[:limit]}…"


def truncate_read_page_content(output: str, limit: int = 200) -> str:
    if not output:
        return ""
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        text = output.strip().replace("\n", " ")
        return text if len(text) <= limit else f"{text[:limit]}…"

    content = data.get("content") or ""
    text = content.strip().replace("\n", " ")
    return text if len(text) <= limit else f"{text[:limit]}…"


def truncate_tool_output(tool_name: str, output: str) -> str:
    if tool_name == "web_search":
        return truncate_search_content(output)
    if tool_name == "read_page":
        return truncate_read_page_content(output)
    if tool_name == "cabinet_search":
        return truncate_cabinet_search_output(output)
    if tool_name == "cabinet_full_text":
        return truncate_cabinet_full_text_output(output)
    text = output.strip().replace("\n", " ")
    return text if len(text) <= 200 else f"{text[:200]}…"


def truncate_cabinet_search_output(output: str, limit: int = 200) -> str:
    if not output:
        return ""
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        text = output.strip().replace("\n", " ")
        return text if len(text) <= limit else f"{text[:limit]}…"

    results = data.get("results") or []
    if not results:
        return "results=0"
    first = results[0]
    title = (first or {}).get("title") or ""
    score = (first or {}).get("score")
    summary = f"results={len(results)}"
    if title:
        summary = f"{summary} top={title}"
    if score is not None:
        summary = f"{summary} score={score}"
    return summary if len(summary) <= limit else f"{summary[:limit]}…"


def truncate_cabinet_full_text_output(output: str, limit: int = 200) -> str:
    if not output:
        return ""
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        text = output.strip().replace("\n", " ")
        return text if len(text) <= limit else f"{text[:limit]}…"

    doc_id = data.get("id")
    title = data.get("title") or ""
    summary = f"id={doc_id}" if doc_id is not None else "id="
    if title:
        summary = f"{summary} title={title}"
    return summary if len(summary) <= limit else f"{summary[:limit]}…"


def _build_text_report(items: list[dict[str, Any]]) -> str:
    def sort_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(rows, key=lambda item: item.get("score", 0), reverse=True)

    sections = [
        (
            "推荐先阅读",
            sort_items([i for i in items if i.get("score", 0) >= SCORE_MUST]),
        ),
        (
            "推荐阅读",
            sort_items(
                [i for i in items if SCORE_RECOMMEND <= i.get("score", 0) < SCORE_MUST]
            ),
        ),
        (
            "扩展阅读",
            sort_items(
                [i for i in items if SCORE_MIN <= i.get("score", 0) < SCORE_RECOMMEND]
            )[:EXTENDED_LIMIT],
        ),
    ]

    lines: list[str] = []
    for name, rows in sections:
        lines.append(name)
        if rows:
            for row in rows:
                title = row.get("title", "")
                quote = row.get("quote", "")
                if quote:
                    quote = "\n".join(
                        line.strip() for line in quote.splitlines() if line.strip()
                    )
                url = row.get("url", "")
                lines.append(f"# {title}")
                lines.append(quote)
                lines.append(url)
                lines.append("")
                lines.append("")
        else:
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def _merge_candidates(candidate_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
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


def ensure_system_prompt(payload_dict: dict[str, Any]) -> None:
    year = datetime.now().year
    system_prompt = f"Current year is {year}."
    messages = payload_dict.get("messages")
    if not isinstance(messages, list):
        payload_dict["messages"] = [{"role": "system", "content": system_prompt}]
        return
    if messages and messages[0].get("role") == "system":
        if system_prompt not in messages[0].get("content", ""):
            messages[0]["content"] = (
                f"{messages[0].get('content', '').rstrip()}\n{system_prompt}".strip()
            )
        return
    messages.insert(0, {"role": "system", "content": system_prompt})


def log_user_message(messages: list[dict[str, Any]]) -> None:
    if not messages:
        return
    last_user = next(
        (msg for msg in reversed(messages) if msg.get("role") == "user"),
        None,
    )
    if last_user and last_user.get("content"):
        logger.info("[User] %s", last_user["content"])


def log_model_answer(data: dict[str, Any]) -> None:
    message = data.get("choices", [{}])[0].get("message", {})
    content = message.get("content") or message.get("reasoning_content") or ""
    if content:
        logger.info("[LLM] %s", content)


def add_html_content(data: dict[str, Any]) -> None:
    message = data.get("choices", [{}])[0].get("message", {})
    content = message.get("content")
    if not content:
        return
    if markdown is None:
        message["content_html"] = content
        return
    message["content_html"] = markdown.markdown(
        content,
        extensions=["extra", "sane_lists"],
    )
