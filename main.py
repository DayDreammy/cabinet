from __future__ import annotations

import json
import logging
import os
import re
import selectors
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, Iterator, List, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from review import (
    DEFAULT_CHAT_URL,
    MODEL_NAME,
    build_review_payload,
    extract_keywords,
    locate_quote,
    parse_review_response,
    post_json,
    review_doc,
)
from search import DEFAULT_DB_PATH, WEIGHTS, load_db, search_db, search_db_multi, tokenize_query

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DOCS: List[Dict[str, Any]] = []
DOCS_BY_ID: Dict[str, Dict[str, Any]] = {}
PUBLIC_DIR = os.path.join(os.path.dirname(__file__), "public")
REPO_DIR = os.path.dirname(__file__)
LOGGER = logging.getLogger("uvicorn.error")
LOGGER.setLevel(logging.INFO)
API_TOKEN_ENV = "CABINET_API_TOKEN"
SCORE_MIN = 1.0
SCORE_RECOMMEND = 8.0
SCORE_MUST = 10.0
EXTENDED_LIMIT = 10
CODEX_TIMEOUT_SEC = 1200
CODEX_HEARTBEAT_SEC = 5.0
CODEX_RAW_FATAL_PATTERNS = [
    re.compile(r"codex_api::endpoint::responses:\s*error=(?P<reason>[\w\-]+)"),
]
CODEX_RAW_ROLLOUT_MISSING_PATTERN = re.compile(
    r"codex_core::rollout::list:\s*state db missing rollout path for thread\s+(?P<thread>[0-9a-f\-]+)"
)
CODEX_FORBIDDEN_COMMAND_PATTERNS = [
    re.compile(r"\b(curl|wget)\b", re.IGNORECASE),
    re.compile(r"\b(web_search|open_page|find_in_page)\b", re.IGNORECASE),
    re.compile(r"\b(playwright|chromium)\b", re.IGNORECASE),
    re.compile(r"\b(pip3?|conda|npm|yarn|apt(-get)?|brew)\b", re.IGNORECASE),
    re.compile(r"https?://", re.IGNORECASE),
]
CODEX_PROXY_ENV_KEYS = [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
]

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


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _log(message: str) -> None:
    LOGGER.info(message)


def _truncate_text(text: str, limit: int = 400) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _remove_nul_bytes(text: str) -> tuple[str, int]:
    if "\x00" not in text:
        return text, 0
    count = text.count("\x00")
    return text.replace("\x00", ""), count


def _build_stream_log_payload(
    task_id: str,
    event: str,
    status: str,
    message: str = "",
    **extra: Any,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "event": event,
        "id": task_id,
        "status": status,
        "ts": _utc_now_iso(),
    }
    if message:
        payload["message"] = message
    for key, value in extra.items():
        payload[key] = value
    return payload


def _build_codex_prompt(query: str, context: str) -> str:
    clean_query = query.strip()
    clean_context = context.strip() if context.strip() else "（无）"
    return (
        "你是 Cabinet 的深度研究代理。请围绕用户问题进行主动、深度、全面、迭代检索，"
        "目标是找到可核验的原文章证据，像博物馆导览员一样给出高光。\n\n"
        "执行要求：\n"
        f"0) 本任务只允许使用本地信源：{DEFAULT_DB_PATH}。\n"
        "   - 数据格式：JSON 数组，共约 3355 条；文件约 16MB。\n"
        "   - 字段：id/title/question/content/url/publishedAt/updatedAt/proofread。\n"
        "   - 单条示例结构：{\"id\":\"...\",\"title\":\"...\",\"question\":\"...\",\"content\":\"...\",\"url\":\"...\"}\n"
        "   - content 中有大量换行；长度中位数约 867 字符，最大约 19502 字符。\n"
        "   禁止任何网络访问/网页工具/HTTP 请求/下载依赖。\n"
        "   你必须通过命令行工具在本地文件中完成检索。\n"
        "   强烈建议只使用本仓库提供的本地检索 CLI（避免你自己反复 json.load / 打印大段文本导致上下文膨胀/超时）：\n"
        "   - `scripts/ps stats`\n"
        "   - `scripts/ps question-grep --contains '...' --topk 20`\n"
        "   - `scripts/ps search --query '...' --topk 10`\n"
        "   - `scripts/ps substring-scan --phrase '...' --phrase '...' --topk 30`\n"
        "   - `scripts/ps sentence-grep --id <id> --contains '...' --max-results 20`（默认只返回 offsets + display）\n"
        "   - `scripts/ps slice --id <id> --start <quote_start> --end <quote_end>`（取回可直接引用的原句）\n"
        "   - `scripts/ps locate --id <id> --quote '原句' --normalize-quotes`\n"
        "   输出控制：任何时候都不要 `cat` 整个 data 文件，也不要打印整篇 content；只看上述工具返回的 snippet/preview。\n"
        "   你最终给出的所有引用证据，必须来自该数据源的 content 原文子串。\n"
        "\n"
        "最终输出格式（很重要）：\n"
        "6) 你可以在正文中自由说明，但在最后必须额外附上一段可机器解析的 JSON，且必须放在标记之间：\n"
        "   <FINAL_JSON>\n"
        "   {\"results\": [{\"id\": \"...\", \"quote\": \"...\", \"quote_start\": 0, \"quote_end\": 0, \"score\": 0-10}], \"notes\": \"...\"}\n"
        "   </FINAL_JSON>\n"
        "   约束：\n"
        "   - `results` 最多 16 条。\n"
        "   - `quote` 必须是 content 的原文子串（可包含换行），且 (quote_start, quote_end) 必须精确对应。\n"
        "   - `score` 用于排序（10 最强）。\n"
        "1) 先做研究计划：拆出子问题、检索关键词和潜在盲区。\n"
        "2) 至少进行两轮迭代：每轮说明新发现和剩余缺口。\n"
        "3) 证据优先给“原句摘录 + 出处（id/title/url）”；不能确认原句时明确标注。\n"
        "   每条引用必须是 content 的原文子串，并尽量给出 quote_start/quote_end（content 中的字符偏移）。\n"
        "4) 最后输出三个部分：\n"
        "   A. 研究结论（简洁）\n"
        "   B. 证据清单（按重要性排序）\n"
        "   C. 未覆盖问题与下一步建议\n"
        "5) 全程尽量调用工具完成检索，不要只给泛化建议。\n\n"
        f"用户问题：{clean_query}\n"
        f"用户上下文：{clean_context}\n"
    )


def _parse_codex_json_line(line: str) -> Dict[str, Any] | None:
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


_FINAL_JSON_RE = re.compile(r"<FINAL_JSON>\s*(?P<json>\{.*\})\s*</FINAL_JSON>", re.DOTALL)


def _extract_final_json_block(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    match = _FINAL_JSON_RE.search(text)
    if not match:
        return {}
    raw = match.group("json").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _build_hits_from_codex_final_json(final_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = final_json.get("results")
    if not isinstance(results, list):
        return []

    hits: List[Dict[str, Any]] = []
    for row in results[:16]:
        if not isinstance(row, dict):
            continue
        doc_id = str(row.get("id", "")).strip()
        if not doc_id:
            continue
        doc = DOCS_BY_ID.get(doc_id, {})
        content = str(doc.get("content", "") or "")
        title = str(doc.get("title", "") or "")
        url = str(doc.get("url", "") or "")

        quote = str(row.get("quote", "") or "")
        start = row.get("quote_start", 0)
        end = row.get("quote_end", 0)
        try:
            start_i = int(start)
            end_i = int(end)
        except (TypeError, ValueError):
            start_i = 0
            end_i = 0

        error = ""
        if not doc:
            error = "doc_not_found"

        # Fill/validate quote and offsets against local content.
        if content:
            if start_i >= 0 and end_i >= 0 and start_i < end_i and end_i <= len(content):
                slice_quote = content[start_i:end_i]
                if quote and quote != slice_quote:
                    # Try to reconcile by locating the provided quote.
                    s2, e2, q2, _ = locate_quote(content, quote)
                    if s2 != -1:
                        start_i, end_i, quote = s2, e2, q2
                    else:
                        # Keep offsets; override quote to exact slice so downstream is consistent.
                        quote = slice_quote
                        if not error:
                            error = "quote_mismatch_fixed_to_slice"
                elif not quote:
                    quote = slice_quote
            elif quote:
                s2, e2, q2, _ = locate_quote(content, quote)
                if s2 != -1:
                    start_i, end_i, quote = s2, e2, q2
                else:
                    quote = ""
                    start_i = 0
                    end_i = 0
                    if not error:
                        error = "quote_not_found"

        try:
            score = float(row.get("score", 8.0))
        except (TypeError, ValueError):
            score = 8.0

        item = {
            "id": doc_id,
            "title": title,
            "url": url,
            "quote": quote,
            "quote_start": start_i,
            "quote_end": end_i,
            "score": score,
            "error": error,
        }
        item["must_read"] = score >= SCORE_MUST
        item["tier"] = "core" if score >= SCORE_RECOMMEND else "extended"
        hits.append(item)

    hits.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return hits


class DeepResearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    context: str = Field("", min_length=0)
    token: str = Field(..., min_length=1)


def _require_api_token(token: str) -> None:
    expected = os.environ.get(API_TOKEN_ENV, "").strip()
    if not expected:
        # Soft mode: allow if not configured.
        LOGGER.warning("%s is not set; token check skipped", API_TOKEN_ENV)
        return
    if token != expected:
        raise HTTPException(status_code=401, detail="invalid token")


def _iter_codex_research_events(
    query: str,
    context: str,
    timeout_sec: int,
    sandbox_mode: str,
    privilege_mode: str,
    proxy: str,
    unset_proxy: bool,
) -> Iterator[Tuple[str, Any]]:
    started = time.time()
    task_id = f"codex-{int(started * 1000)}"

    clean_query, query_nul_removed = _remove_nul_bytes(query)
    clean_context, context_nul_removed = _remove_nul_bytes(context)
    prompt = _build_codex_prompt(clean_query, clean_context)

    cmd = ["codex", "-C", REPO_DIR]
    if privilege_mode == "danger":
        cmd.append("--dangerously-bypass-approvals-and-sandbox")
    elif privilege_mode == "full-auto":
        cmd.append("--full-auto")
    else:
        cmd.extend(
            [
                "--sandbox",
                sandbox_mode,
                "--ask-for-approval",
                "never",
            ]
        )
    cmd.extend(["exec", "--json", "--skip-git-repo-check", prompt])

    def stream_log(event: str, status: str, message: str = "", **extra: Any) -> None:
        payload = _build_stream_log_payload(
            task_id=task_id,
            event=event,
            status=status,
            message=message,
            **extra,
        )
        _log(f"stream_log: {json.dumps(payload, ensure_ascii=False)}")
        yield_events.append(("stream_log", payload))

    yield_events: List[Tuple[str, Any]] = []

    msg = f"codex start: mode={privilege_mode} sandbox={sandbox_mode} timeout={timeout_sec}s"
    _log(msg)
    yield ("log", msg)
    stream_log(
        "start",
        "running",
        "codex process preparing",
        query=clean_query,
        context_present=bool(clean_context.strip()),
        privilege_mode=privilege_mode,
        sandbox_mode=sandbox_mode,
        timeout_sec=timeout_sec,
        proxy_enabled=bool(proxy) and not unset_proxy,
    )
    yield from yield_events
    yield_events.clear()

    env = _build_codex_subprocess_env(proxy=proxy, unset_proxy=unset_proxy)
    yield ("phase_start", {"phase": "codex_boot", "query": clean_query})
    if query_nul_removed or context_nul_removed:
        nul_msg = (
            "prompt sanitized: "
            f"query_nul_removed={query_nul_removed} "
            f"context_nul_removed={context_nul_removed}"
        )
        _log(nul_msg)
        yield ("log", nul_msg)
        stream_log(
            "progress",
            "sanitized",
            nul_msg,
            query_nul_removed=query_nul_removed,
            context_nul_removed=context_nul_removed,
        )
        yield from yield_events
        yield_events.clear()

    try:
        process = subprocess.Popen(
            cmd,
            cwd=REPO_DIR,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
    except Exception as exc:
        msg = f"codex spawn failed: {exc}"
        _log(msg)
        yield ("log", msg)
        stream_log("error", "failed", msg, error_type="spawn_failed")
        yield from yield_events
        yield_events.clear()
        yield (
            "done",
            {
                "query": clean_query,
                "context": clean_context,
                "elapsed_sec": round(time.time() - started, 2),
                "exit_code": -1,
                "timed_out": False,
                "failure_reason": "spawn_failed",
                "final_message": "codex 子进程启动失败。",
                "final_json": {},
                "results": [],
                "text_report": "",
                "messages": [],
                "reasoning": [],
                "commands": [],
                "turn_completed": False,
                "usage": {},
                "stream_stats": {"raw_lines": 0, "json_lines": 0, "events": 0},
                "noise_stats": {"rollout_missing_count": 0, "rollout_missing_threads_sample": []},
            },
        )
        return

    if process.stdout is None:
        msg = "failed to capture codex output"
        _log(msg)
        yield ("log", msg)
        stream_log("error", "failed", msg, error_type="stdout_unavailable")
        yield from yield_events
        yield_events.clear()
        yield (
            "done",
            {
                "query": clean_query,
                "context": clean_context,
                "elapsed_sec": round(time.time() - started, 2),
                "exit_code": -1,
                "timed_out": False,
                "failure_reason": "stdout_unavailable",
                "final_message": "未能捕获 codex 输出。",
                "final_json": {},
                "results": [],
                "text_report": "",
                "messages": [],
                "reasoning": [],
                "commands": [],
                "turn_completed": False,
                "usage": {},
                "stream_stats": {"raw_lines": 0, "json_lines": 0, "events": 0},
                "noise_stats": {"rollout_missing_count": 0, "rollout_missing_threads_sample": []},
            },
        )
        return

    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    stream_log("progress", "running", "codex process started")
    yield from yield_events
    yield_events.clear()

    final_messages: List[str] = []
    reasoning_messages: List[str] = []
    command_steps: List[Dict[str, Any]] = []
    command_map: Dict[str, Dict[str, Any]] = {}
    timed_out = False
    turn_completed = False
    failure_reason = ""
    usage: Dict[str, Any] = {}
    should_stop = False
    raw_line_count = 0
    json_line_count = 0
    event_count = 0
    rollout_missing_count = 0
    rollout_missing_threads_sample: List[str] = []
    last_heartbeat_ts = started

    try:
        while True:
            elapsed = time.time() - started
            if elapsed > timeout_sec:
                timed_out = True
                process.kill()
                msg = f"codex timeout after {timeout_sec}s"
                _log(msg)
                yield ("log", msg)
                stream_log(
                    "error",
                    "timeout",
                    msg,
                    timeout_sec=timeout_sec,
                    elapsed_sec=round(elapsed, 2),
                )
                yield from yield_events
                yield_events.clear()
                failure_reason = "service_timeout"
                break

            ready = selector.select(timeout=0.5)
            if not ready:
                if process.poll() is not None:
                    break
                now = time.time()
                if now - last_heartbeat_ts >= CODEX_HEARTBEAT_SEC:
                    heartbeat = (
                        "codex heartbeat: "
                        f"elapsed={round(now - started, 1)}s "
                        f"events={event_count} raw={raw_line_count} "
                        f"rollout_missing={rollout_missing_count}"
                    )
                    _log(heartbeat)
                    yield ("log", heartbeat)
                    stream_log(
                        "progress",
                        "heartbeat",
                        heartbeat,
                        elapsed_sec=round(now - started, 1),
                        events=event_count,
                        raw_lines=raw_line_count,
                        rollout_missing=rollout_missing_count,
                    )
                    yield from yield_events
                    yield_events.clear()
                    last_heartbeat_ts = now
                continue

            for key, _ in ready:
                line = key.fileobj.readline()
                if line == "":
                    continue
                text_line = line.rstrip("\r\n")
                if not text_line:
                    continue

                raw_line_count += 1
                parsed = _parse_codex_json_line(text_line)
                if parsed is None:
                    rollout_thread = _extract_codex_rollout_missing_thread(text_line)
                    if rollout_thread:
                        rollout_missing_count += 1
                        if (
                            rollout_thread not in rollout_missing_threads_sample
                            and len(rollout_missing_threads_sample) < 12
                        ):
                            rollout_missing_threads_sample.append(rollout_thread)
                        if rollout_missing_count <= 2:
                            msg = (
                                f"codex raw[{raw_line_count}]: "
                                f"{_truncate_text(text_line, limit=1500)}"
                            )
                            _log(msg)
                            yield ("log", msg)
                        elif rollout_missing_count % 20 == 0:
                            msg = (
                                "codex rollout-missing (suppressed): "
                                f"count={rollout_missing_count} "
                                f"latest_thread={rollout_thread}"
                            )
                            _log(msg)
                            yield ("log", msg)
                        continue

                    msg = f"codex raw[{raw_line_count}]: {_truncate_text(text_line, limit=1500)}"
                    _log(msg)
                    yield ("log", msg)
                    raw_fatal_reason = _extract_codex_raw_fatal_reason(text_line)
                    if raw_fatal_reason:
                        failure_reason = f"codex_api_{raw_fatal_reason}"
                        _log(f"codex fatal raw detected: {failure_reason}")
                        yield ("log", f"codex fatal raw detected: {failure_reason}")
                        stream_log(
                            "error",
                            "failed",
                            f"codex fatal raw detected: {failure_reason}",
                            raw_reason=raw_fatal_reason,
                        )
                        yield from yield_events
                        yield_events.clear()
                        process.kill()
                        should_stop = True
                    continue

                json_line_count += 1
                event_count += 1
                yield ("codex_event", parsed)
                stream_log(
                    "progress",
                    "event",
                    "codex event",
                    event_type=str(parsed.get("type", "")),
                    event_index=event_count,
                )
                yield from yield_events
                yield_events.clear()

                event_type = str(parsed.get("type", ""))
                if event_type in {"thread.started", "turn.started"}:
                    yield ("log", event_type)
                    continue

                if event_type == "turn.completed":
                    usage = parsed.get("usage", {}) if isinstance(parsed, dict) else {}
                    yield (
                        "log",
                        (
                            "turn completed: "
                            f"input={usage.get('input_tokens', 0)} "
                            f"output={usage.get('output_tokens', 0)}"
                        ),
                    )
                    stream_log("progress", "turn_completed", "codex turn completed", usage=usage)
                    yield from yield_events
                    yield_events.clear()
                    turn_completed = True
                    failure_reason = ""
                    should_stop = True
                    continue

                if event_type not in {"item.started", "item.completed"}:
                    continue

                item = parsed.get("item", {})
                if not isinstance(item, dict):
                    continue
                item_type = str(item.get("type", ""))
                item_id = str(item.get("id", ""))

                if item_type == "agent_message" and event_type == "item.completed":
                    text = str(item.get("text", "")).strip()
                    if text:
                        final_messages.append(text)
                        yield ("codex_message", {"text": text})
                        stream_log("progress", "response", _truncate_text(text, limit=300), item_id=item_id)
                        yield from yield_events
                        yield_events.clear()
                    continue

                if item_type == "reasoning" and event_type == "item.completed":
                    text = str(item.get("text", "")).strip()
                    if text:
                        reasoning_messages.append(text)
                        yield ("codex_reasoning", {"text": text})
                        stream_log("progress", "thought", _truncate_text(text, limit=300), item_id=item_id)
                        yield from yield_events
                        yield_events.clear()
                    continue

                if item_type != "command_execution":
                    continue

                record = command_map.get(item_id)
                if not record:
                    record = {
                        "id": item_id,
                        "type": "command_execution",
                        "command": "",
                        "status": "",
                        "exit_code": None,
                        "output": "",
                    }
                    command_map[item_id] = record
                    command_steps.append(record)

                record["command"] = item.get("command", "") or record["command"]
                record["status"] = item.get("status", "") or record["status"]
                if item.get("exit_code") is not None:
                    record["exit_code"] = item.get("exit_code")
                output = str(item.get("aggregated_output", "") or "")
                if output:
                    record["output"] = _truncate_text(output, limit=1200)

                if _is_forbidden_codex_command(record.get("command", "")):
                    failure_reason = "forbidden_command"
                    msg = f"forbidden command detected; aborting: {record.get('command', '')}"
                    _log(msg)
                    yield ("log", msg)
                    stream_log("error", "failed", "forbidden command detected; aborting", item_id=record.get("id", ""), command=record.get("command", ""))
                    yield from yield_events
                    yield_events.clear()
                    process.kill()
                    should_stop = True
                    continue

                yield (
                    "codex_command",
                    {
                        "id": record.get("id", ""),
                        "command": record.get("command", ""),
                        "status": record.get("status", ""),
                        "exit_code": record.get("exit_code"),
                        "output": record.get("output", ""),
                    },
                )
                stream_log(
                    "progress",
                    "call",
                    _truncate_text(record.get("command", ""), limit=300),
                    item_id=record.get("id", ""),
                    command_status=record.get("status", ""),
                    exit_code=record.get("exit_code"),
                )
                yield from yield_events
                yield_events.clear()

            if should_stop:
                break
    finally:
        selector.close()
        if process.poll() is None:
            if turn_completed:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
            else:
                process.kill()

    return_code = process.wait()
    elapsed_sec = round(time.time() - started, 2)
    final_message = "\n\n".join(final_messages).strip()
    if not final_message:
        if failure_reason:
            final_message = f"codex 执行失败，原因：{failure_reason}"
        elif timed_out:
            final_message = "codex 执行超时。"
        else:
            final_message = "codex 未返回最终文本。"

    final_json = _extract_final_json_block(final_message)
    deep_hits = _build_hits_from_codex_final_json(final_json) if final_json else []
    deep_text_report = _build_text_report(deep_hits) if deep_hits else ""

    done_payload = {
        "query": clean_query,
        "context": clean_context,
        "elapsed_sec": elapsed_sec,
        "exit_code": return_code,
        "timed_out": timed_out,
        "failure_reason": failure_reason,
        "final_message": final_message,
        "final_json": final_json,
        "results": deep_hits,
        "text_report": deep_text_report,
        "messages": final_messages,
        "reasoning": reasoning_messages,
        "commands": command_steps,
        "turn_completed": turn_completed,
        "usage": usage,
        "stream_stats": {
            "raw_lines": raw_line_count,
            "json_lines": json_line_count,
            "events": event_count,
        },
        "noise_stats": {
            "rollout_missing_count": rollout_missing_count,
            "rollout_missing_threads_sample": rollout_missing_threads_sample,
        },
    }
    complete_status = "success" if return_code == 0 and not timed_out else "failed"
    stream_log(
        "complete",
        complete_status,
        "codex run finished",
        exit_code=return_code,
        timed_out=timed_out,
        failure_reason=failure_reason,
        elapsed_sec=elapsed_sec,
        stream_stats=done_payload.get("stream_stats", {}),
        noise_stats=done_payload.get("noise_stats", {}),
    )
    yield from yield_events
    yield_events.clear()
    yield ("done", done_payload)


def _extract_codex_raw_fatal_reason(line: str) -> str:
    for pattern in CODEX_RAW_FATAL_PATTERNS:
        match = pattern.search(line)
        if match:
            return match.group("reason") or "unknown"
    return ""


def _extract_codex_rollout_missing_thread(line: str) -> str:
    match = CODEX_RAW_ROLLOUT_MISSING_PATTERN.search(line)
    if not match:
        return ""
    return match.group("thread") or ""


def _is_forbidden_codex_command(command: str) -> bool:
    cmd = command or ""
    return any(p.search(cmd) for p in CODEX_FORBIDDEN_COMMAND_PATTERNS)


def _merge_no_proxy(existing: str, add_hosts: List[str]) -> str:
    parts = [p.strip() for p in (existing or "").split(",") if p.strip()]
    for host in add_hosts:
        if host not in parts:
            parts.append(host)
    return ",".join(parts)


def _build_codex_subprocess_env(proxy: str, unset_proxy: bool) -> Dict[str, str]:
    env = dict(os.environ)
    if proxy:
        for key in CODEX_PROXY_ENV_KEYS:
            env[key] = proxy
        env["NO_PROXY"] = _merge_no_proxy(env.get("NO_PROXY", ""), ["127.0.0.1", "localhost"])
        env["no_proxy"] = _merge_no_proxy(env.get("no_proxy", ""), ["127.0.0.1", "localhost"])
    if unset_proxy:
        for key in CODEX_PROXY_ENV_KEYS:
            env.pop(key, None)
    return env


def _build_text_report(items: List[Dict[str, Any]]) -> str:
    def sort_items(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

    lines: List[str] = []
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
    top_k: int = Query(30, ge=1, le=50),
    max_workers: int = Query(150, ge=1, le=150),
    score_threshold: float = Query(SCORE_MIN, ge=0.0, le=10.0),
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
                if quote and score > 0 and score >= score_threshold:
                    result["must_read"] = score >= SCORE_MUST
                    result["tier"] = "core" if score >= SCORE_RECOMMEND else "extended"
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
                    elif score <= 0:
                        msg = f"skip: {title} (zero score)"
                    else:
                        msg = f"skip: {title} (score={score:.1f} < {score_threshold})"
                    _log(msg)
                    yield _format_sse("log_skip", msg)

        hits.sort(key=lambda item: item.get("score", 0), reverse=True)
        final_hits = hits
        text_report = _build_text_report(final_hits)
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
                "text_report": text_report,
            },
        )

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/stream_codex_research")
def stream_codex_research(
    query: str = Query(..., min_length=1),
    context: str = Query("", min_length=0),
    timeout_sec: int = Query(CODEX_TIMEOUT_SEC, ge=30, le=7200),
    sandbox_mode: str = Query("workspace-write", pattern="^(read-only|workspace-write|danger-full-access)$"),
    # Default to `danger` to avoid Codex LandlockRestrict preventing local command execution.
    privilege_mode: str = Query("danger", pattern="^(default|full-auto|danger)$"),
    proxy: str = Query("", min_length=0),
    unset_proxy: bool = Query(False),
) -> StreamingResponse:
    def event_generator() -> Iterable[str]:
        for event, data in _iter_codex_research_events(
            query=query,
            context=context,
            timeout_sec=timeout_sec,
            sandbox_mode=sandbox_mode,
            privilege_mode=privilege_mode,
            proxy=proxy,
            unset_proxy=unset_proxy,
        ):
            yield _format_sse(event, data)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/deep_research")
def api_deep_research(req: DeepResearchRequest) -> Dict[str, Any]:
    _require_api_token(req.token)
    done_payload: Dict[str, Any] = {}
    for event, data in _iter_codex_research_events(
        query=req.query,
        context=req.context,
        timeout_sec=CODEX_TIMEOUT_SEC,
        sandbox_mode="workspace-write",
        privilege_mode="danger",
        proxy="",
        unset_proxy=False,
    ):
        if event == "done" and isinstance(data, dict):
            done_payload = data
    if not done_payload:
        raise HTTPException(status_code=500, detail="missing done payload")
    # API contract: text_report is the primary answer.
    return {
        "query": done_payload.get("query", ""),
        "text_report": done_payload.get("text_report", ""),
        "results": done_payload.get("results", []),
        "elapsed_sec": done_payload.get("elapsed_sec", 0),
        "exit_code": done_payload.get("exit_code", -1),
        "timed_out": done_payload.get("timed_out", False),
        "failure_reason": done_payload.get("failure_reason", ""),
        "usage": done_payload.get("usage", {}),
        "final_message": done_payload.get("final_message", ""),
    }
