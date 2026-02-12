#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import selectors
import subprocess
import time
from typing import Any, Dict, List, Tuple

HEARTBEAT_SEC = 5.0
RAW_FATAL_PATTERNS = [
    re.compile(r"codex_api::endpoint::responses:\s*error=(?P<reason>[\w\-]+)"),
]
ROLLOUT_MISSING_PATTERN = re.compile(
    r"codex_core::rollout::list:\s*state db missing rollout path for thread\s+(?P<thread>[0-9a-f\-]+)"
)
FORBIDDEN_COMMAND_PATTERNS = [
    re.compile(r"\b(curl|wget)\b", re.IGNORECASE),
    re.compile(r"\b(web_search|open_page|find_in_page)\b", re.IGNORECASE),
    re.compile(r"\b(playwright|chromium)\b", re.IGNORECASE),
    re.compile(r"\b(pip3?|conda|npm|yarn|apt(-get)?|brew)\b", re.IGNORECASE),
    re.compile(r"https?://", re.IGNORECASE),
]

FINAL_JSON_RE = re.compile(r"<FINAL_JSON>\\s*(?P<json>\\{.*\\})\\s*</FINAL_JSON>", re.DOTALL)


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def remove_nul_bytes(text: str) -> Tuple[str, int]:
    if "\x00" not in text:
        return text, 0
    count = text.count("\x00")
    return text.replace("\x00", ""), count


def build_prompt(query: str, context: str) -> str:
    clean_query = query.strip()
    clean_context = context.strip() if context.strip() else "（无）"
    # This repo's local sources are fixed under data/.
    local_db = "data/ps_2026-01-07.json"
    return (
        "你是 Cabinet 的深度研究代理。请围绕用户问题进行主动、深度、全面、迭代检索，"
        "目标是找到可核验的原文章证据，像博物馆导览员一样给出高光。\n\n"
        "执行要求：\n"
        f"0) 本任务只允许使用本地信源：{local_db}。\n"
        "   - 数据格式：JSON 数组，共约 3355 条；文件约 16MB。\n"
        "   - 字段：id/title/question/content/url/publishedAt/updatedAt/proofread。\n"
        "   - 单条示例结构：{\"id\":\"...\",\"title\":\"...\",\"question\":\"...\",\"content\":\"...\",\"url\":\"...\"}\n"
        "   - content 中有大量换行；长度中位数约 867 字符，最大约 19502 字符。\n"
        "   禁止任何网络访问/网页工具/HTTP 请求/下载依赖。\n"
        "   你必须通过命令行工具在本地文件中完成检索。\n"
        "   你最终给出的所有引用证据，必须来自该数据源的 content 原文子串。\n"
        "   强烈建议只使用本仓库提供的本地检索 CLI（避免你自己反复 json.load / 打印大段文本导致上下文膨胀/超时）：\n"
        "   - `scripts/ps stats`\n"
        "   - `scripts/ps question-grep --contains '...' --topk 20`\n"
        "   - `scripts/ps search --query '...' --topk 10`\n"
        "   - `scripts/ps substring-scan --phrase '...' --phrase '...' --topk 30`\n"
        "   - `scripts/ps sentence-grep --id <id> --contains '...' --max-results 20`（默认只返回 offsets + display）\n"
        "   - `scripts/ps slice --id <id> --start <quote_start> --end <quote_end>`（取回可直接引用的原句）\n"
        "   - `scripts/ps locate --id <id> --quote '原句' --normalize-quotes`\n"
        "   输出控制：任何时候都不要 `cat` 整个 data 文件，也不要打印整篇 content；只看上述工具返回的 snippet/preview。\n"
        "1) 先做研究计划：拆出子问题、检索关键词和潜在盲区。\n"
        "2) 至少进行两轮迭代：每轮说明新发现和剩余缺口。\n"
        "3) 证据优先给“原句摘录 + 出处（id/title/url）”；不能确认原句时明确标注。\n"
        "   每条引用必须是 content 的原文子串，并尽量给出 quote_start/quote_end（content 中的字符偏移）。\n"
        "4) 最后输出三个部分：\n"
        "   A. 研究结论（简洁）\n"
        "   B. 证据清单（按重要性排序）\n"
        "   C. 未覆盖问题与下一步建议\n"
        "5) 全程尽量调用工具完成检索，不要只给泛化建议。\n\n"
        "最终输出格式（很重要）：\n"
        "6) 你可以在正文中自由说明，但在最后必须额外附上一段可机器解析的 JSON，且必须放在标记之间：\n"
        "   <FINAL_JSON>\n"
        "   {\"results\": [{\"id\": \"...\", \"quote\": \"...\", \"quote_start\": 0, \"quote_end\": 0, \"score\": 0-10}], \"notes\": \"...\"}\n"
        "   </FINAL_JSON>\n"
        "   约束：\n"
        "   - `results` 最多 16 条。\n"
        "   - `quote` 必须是 content 的原文子串（可包含换行），且 (quote_start, quote_end) 必须精确对应。\n"
        "   - `score` 用于排序（10 最强）。\n\n"
        f"用户问题：{clean_query}\n"
        f"用户上下文：{clean_context}\n"
    )


def parse_json_line(text: str) -> Dict[str, Any] | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def extract_final_json_block(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    match = FINAL_JSON_RE.search(text)
    if not match:
        return {}
    raw = match.group("json").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def extract_raw_fatal_reason(line: str) -> str:
    for pattern in RAW_FATAL_PATTERNS:
        match = pattern.search(line)
        if match:
            return match.group("reason") or "unknown"
    return ""


def extract_rollout_thread(line: str) -> str:
    match = ROLLOUT_MISSING_PATTERN.search(line)
    if not match:
        return ""
    return match.group("thread") or ""


def is_forbidden_command(command: str) -> bool:
    cmd = command or ""
    return any(p.search(cmd) for p in FORBIDDEN_COMMAND_PATTERNS)


def emit(task_id: str, event: str, status: str, message: str = "", **extra: Any) -> None:
    payload: Dict[str, Any] = {
        "event": event,
        "id": task_id,
        "status": status,
        "ts": utc_now_iso(),
    }
    if message:
        payload["message"] = message
    for key, value in extra.items():
        payload[key] = value
    print(json.dumps(payload, ensure_ascii=False), flush=True)

PROXY_ENV_KEYS = [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
]


def _merge_no_proxy(existing: str, add_hosts: List[str]) -> str:
    parts = [p.strip() for p in (existing or "").split(",") if p.strip()]
    for host in add_hosts:
        if host not in parts:
            parts.append(host)
    return ",".join(parts)


def build_env(args: argparse.Namespace) -> Dict[str, str]:
    env = dict(os.environ)
    if args.proxy:
        for key in PROXY_ENV_KEYS:
            env[key] = args.proxy
        env["NO_PROXY"] = _merge_no_proxy(env.get("NO_PROXY", ""), ["127.0.0.1", "localhost"])
        env["no_proxy"] = _merge_no_proxy(env.get("no_proxy", ""), ["127.0.0.1", "localhost"])
    if args.unset_proxy:
        for key in PROXY_ENV_KEYS:
            env.pop(key, None)
    return env


def build_codex_cmd(args: argparse.Namespace, prompt: str) -> List[str]:
    cmd: List[str] = ["codex", "-C", args.cwd]
    if args.model:
        cmd.extend(["--model", args.model])
    if args.privilege_mode == "danger":
        cmd.append("--dangerously-bypass-approvals-and-sandbox")
    elif args.privilege_mode == "full-auto":
        cmd.append("--full-auto")
    else:
        cmd.extend(["--sandbox", args.sandbox_mode, "--ask-for-approval", "never"])
    cmd.extend(["exec", "--json", "--skip-git-repo-check", prompt])
    return cmd


def build_claude_cmd(args: argparse.Namespace, prompt: str) -> List[str]:
    # Claude Code requires --verbose when streaming JSON.
    cmd: List[str] = ["claude"]
    if args.model:
        cmd.extend(["--model", args.model])
    cmd.extend(
        [
            "--print",
            "--verbose",
            "--output-format",
            "stream-json",
            "--include-partial-messages",
            "--permission-mode",
            "bypassPermissions",
            "--tools",
            args.claude_tools,
            "--",
            prompt,
        ]
    )
    return cmd


def run_once_codex(args: argparse.Namespace, task_id: str, query: str, context: str) -> Dict[str, Any]:
    started = time.time()
    prompt = build_prompt(query, context)
    cmd = build_codex_cmd(args, prompt)
    env = build_env(args)

    emit(
        task_id,
        "start",
        "running",
        "codex process preparing",
        query=query,
        context_present=bool(context.strip()),
        proxy_enabled=bool(args.proxy) and not args.unset_proxy,
        proxy_url=args.proxy if (args.proxy and not args.unset_proxy) else "",
        privilege_mode=args.privilege_mode,
        sandbox_mode=args.sandbox_mode,
        timeout_sec=args.timeout_sec,
        cwd=args.cwd,
        cmd=cmd,
    )

    try:
        process = subprocess.Popen(
            cmd,
            cwd=args.cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
    except Exception as exc:
        emit(task_id, "error", "failed", f"spawn failed: {exc}", error_type="spawn_failed")
        return {
            "exit_code": -1,
            "timed_out": False,
            "failure_reason": "spawn_failed",
            "final_message": "",
            "messages": [],
            "reasoning": [],
            "commands": [],
            "stream_stats": {"events": 0, "json_lines": 0, "raw_lines": 0},
            "noise_stats": {"rollout_missing_count": 0, "rollout_missing_threads_sample": []},
        }

    if process.stdout is None:
        emit(
            task_id,
            "error",
            "failed",
            "failed to capture codex output",
            error_type="stdout_unavailable",
        )
        return {
            "exit_code": -1,
            "timed_out": False,
            "failure_reason": "stdout_unavailable",
            "final_message": "",
            "messages": [],
            "reasoning": [],
            "commands": [],
            "stream_stats": {"events": 0, "json_lines": 0, "raw_lines": 0},
            "noise_stats": {"rollout_missing_count": 0, "rollout_missing_threads_sample": []},
        }

    emit(task_id, "progress", "running", "codex process started")

    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)

    final_messages: List[str] = []
    reasoning_messages: List[str] = []
    command_steps: List[Dict[str, Any]] = []
    command_map: Dict[str, Dict[str, Any]] = {}
    usage: Dict[str, Any] = {}

    timed_out = False
    turn_completed = False
    failure_reason = ""
    raw_line_count = 0
    json_line_count = 0
    event_count = 0
    rollout_missing_count = 0
    rollout_missing_threads_sample: List[str] = []
    last_heartbeat_ts = started
    should_stop = False

    try:
        while True:
            elapsed = time.time() - started
            if elapsed > args.timeout_sec:
                timed_out = True
                process.kill()
                failure_reason = "service_timeout"
                emit(
                    task_id,
                    "error",
                    "timeout",
                    f"codex timeout after {args.timeout_sec}s",
                    elapsed_sec=round(elapsed, 2),
                    timeout_sec=args.timeout_sec,
                )
                break

            ready = selector.select(timeout=0.5)
            if not ready:
                if process.poll() is not None:
                    break
                now = time.time()
                if now - last_heartbeat_ts >= HEARTBEAT_SEC:
                    emit(
                        task_id,
                        "progress",
                        "heartbeat",
                        "codex heartbeat",
                        elapsed_sec=round(now - started, 1),
                        events=event_count,
                        raw_lines=raw_line_count,
                        rollout_missing=rollout_missing_count,
                    )
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
                parsed = parse_json_line(text_line)
                if parsed is None:
                    rollout_thread = extract_rollout_thread(text_line)
                    if rollout_thread:
                        rollout_missing_count += 1
                        if (
                            rollout_thread not in rollout_missing_threads_sample
                            and len(rollout_missing_threads_sample) < 12
                        ):
                            rollout_missing_threads_sample.append(rollout_thread)
                        if rollout_missing_count <= 2:
                            emit(
                                task_id,
                                "progress",
                                "noise",
                                text_line,
                                raw_index=raw_line_count,
                            )
                        elif rollout_missing_count % 20 == 0:
                            emit(
                                task_id,
                                "progress",
                                "noise_summary",
                                "rollout-missing suppressed",
                                rollout_missing_count=rollout_missing_count,
                                latest_thread=rollout_thread,
                            )
                        continue

                    emit(
                        task_id,
                        "progress",
                        "raw",
                        text_line,
                        raw_index=raw_line_count,
                    )
                    raw_fatal_reason = extract_raw_fatal_reason(text_line)
                    if raw_fatal_reason:
                        failure_reason = f"codex_api_{raw_fatal_reason}"
                        emit(
                            task_id,
                            "error",
                            "failed",
                            f"codex fatal raw detected: {failure_reason}",
                            raw_reason=raw_fatal_reason,
                        )
                        process.kill()
                        should_stop = True
                    continue

                json_line_count += 1
                event_count += 1
                event_type = str(parsed.get("type", ""))
                emit(
                    task_id,
                    "progress",
                    "event",
                    "codex event",
                    event_type=event_type,
                    event_index=event_count,
                )

                if event_type == "turn.completed":
                    usage = parsed.get("usage", {}) if isinstance(parsed, dict) else {}
                    turn_completed = True
                    emit(
                        task_id,
                        "progress",
                        "turn_completed",
                        "codex turn completed",
                        usage=usage,
                    )
                    should_stop = True
                    continue

                if event_type not in {"item.started", "item.completed"}:
                    continue

                item = parsed.get("item", {})
                if not isinstance(item, dict):
                    continue
                item_type = str(item.get("type", ""))
                item_id = str(item.get("id", ""))

                if item_type == "reasoning" and event_type == "item.completed":
                    text = str(item.get("text", "")).strip()
                    if text:
                        reasoning_messages.append(text)
                        emit(task_id, "progress", "thought", text, item_id=item_id)
                    continue

                if item_type == "agent_message" and event_type == "item.completed":
                    text = str(item.get("text", "")).strip()
                    if text:
                        final_messages.append(text)
                        emit(task_id, "progress", "response", text, item_id=item_id)
                    continue

                if item_type != "command_execution":
                    continue

                record = command_map.get(item_id)
                if not record:
                    record = {
                        "id": item_id,
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
                    record["output"] = output

                if is_forbidden_command(record.get("command", "")):
                    failure_reason = "forbidden_command"
                    emit(
                        task_id,
                        "error",
                        "failed",
                        "forbidden command detected; aborting",
                        item_id=item_id,
                        command=record.get("command", ""),
                    )
                    process.kill()
                    should_stop = True
                    continue

                emit(
                    task_id,
                    "progress",
                    "call",
                    record.get("command", ""),
                    item_id=item_id,
                    command_status=record.get("status", ""),
                    exit_code=record.get("exit_code"),
                )

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
            final_message = f"codex run failed: {failure_reason}"
        elif timed_out:
            final_message = "codex run timed out"
        else:
            final_message = "codex returned no final message"

    if not failure_reason and (timed_out or return_code != 0):
        failure_reason = "non_zero_exit"

    final_json = extract_final_json_block(final_message)
    if final_json:
        emit(task_id, "progress", "final_json", "parsed FINAL_JSON", final_json=final_json)

    return {
        "engine": "codex",
        "exit_code": return_code,
        "timed_out": timed_out,
        "failure_reason": failure_reason,
        "elapsed_sec": elapsed_sec,
        "final_message": final_message,
        "final_json": final_json,
        "messages": final_messages,
        "reasoning": reasoning_messages,
        "commands": command_steps,
        "usage": usage,
        "stream_stats": {
            "events": event_count,
            "json_lines": json_line_count,
            "raw_lines": raw_line_count,
        },
        "noise_stats": {
            "rollout_missing_count": rollout_missing_count,
            "rollout_missing_threads_sample": rollout_missing_threads_sample,
        },
    }


def _extract_claude_text_from_message(message: Dict[str, Any]) -> str:
    content = message.get("content", [])
    if not isinstance(content, list):
        return ""
    parts: List[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block.get("text", ""))
    return "".join(parts).strip()


def _extract_claude_tool_command(event: Dict[str, Any]) -> Tuple[str, str, str]:
    # Returns (tool_use_id, tool_name, command).
    if not isinstance(event, dict):
        return "", "", ""
    if event.get("type") != "content_block_start":
        return "", "", ""
    block = event.get("content_block", {})
    if not isinstance(block, dict):
        return "", "", ""
    if block.get("type") != "tool_use":
        return "", "", ""
    tool_use_id = str(block.get("id", "") or "")
    tool_name = str(block.get("name", "") or "")
    tool_input = block.get("input", {})
    if not isinstance(tool_input, dict):
        return tool_use_id, tool_name, ""
    command = str(tool_input.get("command", "") or "")
    return tool_use_id, tool_name, command


def _is_allowed_claude_tool(tool_name: str) -> bool:
    return tool_name.strip() == "Bash"


def run_once_claude(args: argparse.Namespace, task_id: str, query: str, context: str) -> Dict[str, Any]:
    started = time.time()
    prompt = build_prompt(query, context)
    cmd = build_claude_cmd(args, prompt)
    env = build_env(args)

    emit(
        task_id,
        "start",
        "running",
        "claude process preparing",
        query=query,
        context_present=bool(context.strip()),
        proxy_enabled=bool(args.proxy) and not args.unset_proxy,
        proxy_url=args.proxy if (args.proxy and not args.unset_proxy) else "",
        permission_mode="bypassPermissions",
        tools=args.claude_tools,
        timeout_sec=args.timeout_sec,
        cwd=args.cwd,
        cmd=cmd,
    )

    try:
        process = subprocess.Popen(
            cmd,
            cwd=args.cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
    except Exception as exc:
        emit(task_id, "error", "failed", f"spawn failed: {exc}", error_type="spawn_failed")
        return {
            "engine": "claude",
            "exit_code": -1,
            "timed_out": False,
            "failure_reason": "spawn_failed",
            "final_message": "",
            "messages": [],
            "reasoning": [],
            "commands": [],
            "stream_stats": {"events": 0, "json_lines": 0, "raw_lines": 0},
            "noise_stats": {},
        }

    if process.stdout is None:
        emit(
            task_id,
            "error",
            "failed",
            "failed to capture claude output",
            error_type="stdout_unavailable",
        )
        return {
            "engine": "claude",
            "exit_code": -1,
            "timed_out": False,
            "failure_reason": "stdout_unavailable",
            "final_message": "",
            "messages": [],
            "reasoning": [],
            "commands": [],
            "stream_stats": {"events": 0, "json_lines": 0, "raw_lines": 0},
            "noise_stats": {},
        }

    emit(task_id, "progress", "running", "claude process started")

    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)

    final_messages: List[str] = []
    command_steps: List[Dict[str, Any]] = []
    command_map: Dict[str, Dict[str, Any]] = {}
    usage: Dict[str, Any] = {}

    timed_out = False
    failure_reason = ""
    raw_line_count = 0
    json_line_count = 0
    event_count = 0
    last_heartbeat_ts = started
    should_stop = False

    partial_buf: List[str] = []
    last_partial_emit = started

    def maybe_emit_partial(force: bool = False) -> None:
        nonlocal last_partial_emit
        if not partial_buf:
            return
        now = time.time()
        size = sum(len(x) for x in partial_buf)
        if not force and (now - last_partial_emit) < 0.6 and size < 240:
            return
        text = "".join(partial_buf)
        partial_buf.clear()
        last_partial_emit = now
        emit(task_id, "progress", "response", text, partial=True)

    try:
        while True:
            elapsed = time.time() - started
            if elapsed > args.timeout_sec:
                timed_out = True
                process.kill()
                failure_reason = "service_timeout"
                emit(
                    task_id,
                    "error",
                    "timeout",
                    f"claude timeout after {args.timeout_sec}s",
                    elapsed_sec=round(elapsed, 2),
                    timeout_sec=args.timeout_sec,
                )
                break

            ready = selector.select(timeout=0.5)
            if not ready:
                if process.poll() is not None:
                    break
                now = time.time()
                if now - last_heartbeat_ts >= HEARTBEAT_SEC:
                    emit(
                        task_id,
                        "progress",
                        "heartbeat",
                        "claude heartbeat",
                        elapsed_sec=round(now - started, 1),
                        events=event_count,
                        raw_lines=raw_line_count,
                    )
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
                parsed = parse_json_line(text_line)
                if parsed is None:
                    emit(task_id, "progress", "raw", text_line, raw_index=raw_line_count)
                    continue

                json_line_count += 1
                event_count += 1
                emit(
                    task_id,
                    "progress",
                    "event",
                    "claude event",
                    event_type=str(parsed.get("type", "")),
                    event_index=event_count,
                )

                typ = str(parsed.get("type", ""))
                if typ == "stream_event":
                    event = parsed.get("event", {})
                    if isinstance(event, dict):
                        tool_use_id, tool_name, command = _extract_claude_tool_command(event)
                        if tool_use_id or tool_name or command:
                            rec_id = tool_use_id or f"tool_{event_count}"
                            record = command_map.get(rec_id)
                            if not record:
                                record = {
                                    "id": rec_id,
                                    "tool": tool_name,
                                    "command": command,
                                    "status": "in_progress",
                                    "exit_code": None,
                                    "output": "",
                                }
                                command_map[rec_id] = record
                                command_steps.append(record)
                            record["tool"] = tool_name or record.get("tool", "")
                            record["command"] = command or record.get("command", "")

                            if record.get("tool") and not _is_allowed_claude_tool(str(record.get("tool", ""))):
                                failure_reason = "forbidden_tool"
                                emit(
                                    task_id,
                                    "error",
                                    "failed",
                                    "forbidden tool detected; aborting",
                                    item_id=rec_id,
                                    tool=record.get("tool", ""),
                                )
                                process.kill()
                                should_stop = True
                                continue

                            if is_forbidden_command(record.get("command", "")):
                                failure_reason = "forbidden_command"
                                emit(
                                    task_id,
                                    "error",
                                    "failed",
                                    "forbidden command detected; aborting",
                                    item_id=rec_id,
                                    command=record.get("command", ""),
                                )
                                process.kill()
                                should_stop = True
                                continue

                            # The full Bash command is often only present in the later `assistant` tool_use message.
                            if record.get("command"):
                                emit(
                                    task_id,
                                    "progress",
                                    "call",
                                    record.get("command", ""),
                                    item_id=rec_id,
                                    tool=record.get("tool", ""),
                                )
                            continue

                        if event.get("type") == "content_block_delta":
                            delta = event.get("delta", {})
                            if isinstance(delta, dict) and delta.get("type") == "text_delta":
                                text = str(delta.get("text", ""))
                                if text:
                                    partial_buf.append(text)
                                    maybe_emit_partial(force=False)
                            continue

                        if event.get("type") == "message_stop":
                            # End of a message, not necessarily end of the overall run.
                            emit(task_id, "progress", "event", "claude message_stop")
                            continue

                if typ == "assistant":
                    message = parsed.get("message", {})
                    if isinstance(message, dict):
                        content = message.get("content", [])
                        if isinstance(content, list):
                            for block in content:
                                if not isinstance(block, dict):
                                    continue
                                if block.get("type") != "tool_use":
                                    continue
                                tool_use_id = str(block.get("id", "") or "")
                                tool_name = str(block.get("name", "") or "")
                                tool_input = block.get("input", {}) if isinstance(block.get("input", {}), dict) else {}
                                cmd_text = str(tool_input.get("command", "") or "")

                                rec_id = tool_use_id or f"tool_{event_count}"
                                record = command_map.get(rec_id)
                                if not record:
                                    record = {
                                        "id": rec_id,
                                        "tool": tool_name,
                                        "command": cmd_text,
                                        "status": "in_progress",
                                        "exit_code": None,
                                        "output": "",
                                    }
                                    command_map[rec_id] = record
                                    command_steps.append(record)
                                record["tool"] = tool_name or record.get("tool", "")
                                record["command"] = cmd_text or record.get("command", "")

                                if record.get("tool") and not _is_allowed_claude_tool(str(record.get("tool", ""))):
                                    failure_reason = "forbidden_tool"
                                    emit(
                                        task_id,
                                        "error",
                                        "failed",
                                        "forbidden tool detected; aborting",
                                        item_id=rec_id,
                                        tool=record.get("tool", ""),
                                    )
                                    process.kill()
                                    should_stop = True
                                    break

                                if cmd_text and is_forbidden_command(cmd_text):
                                    failure_reason = "forbidden_command"
                                    emit(
                                        task_id,
                                        "error",
                                        "failed",
                                        "forbidden command detected; aborting",
                                        item_id=rec_id,
                                        command=cmd_text,
                                    )
                                    process.kill()
                                    should_stop = True
                                    break

                                if record.get("command"):
                                    emit(
                                        task_id,
                                        "progress",
                                        "call",
                                        record.get("command", ""),
                                        item_id=rec_id,
                                        tool=record.get("tool", ""),
                                    )
                            if should_stop:
                                continue

                        text = _extract_claude_text_from_message(message)
                        if text:
                            final_messages.append(text)
                            emit(task_id, "progress", "response", text, partial=False)
                    continue

                if typ == "user":
                    message = parsed.get("message", {})
                    if isinstance(message, dict):
                        content = message.get("content", [])
                        if isinstance(content, list):
                            for block in content:
                                if not isinstance(block, dict):
                                    continue
                                if block.get("type") != "tool_result":
                                    continue
                                tool_use_id = str(block.get("tool_use_id", "") or "")
                                output = str(block.get("content", "") or "")
                                if not tool_use_id:
                                    continue
                                record = command_map.get(tool_use_id)
                                if record is None:
                                    continue
                                record["status"] = "completed"
                                record["output"] = output
                                emit(
                                    task_id,
                                    "progress",
                                    "call",
                                    record.get("command", ""),
                                    item_id=tool_use_id,
                                    tool=record.get("tool", ""),
                                    command_status="completed",
                                )
                    continue

                if typ == "result":
                    if isinstance(parsed.get("usage", {}), dict):
                        usage = parsed.get("usage", {})
                    if parsed.get("is_error"):
                        failure_reason = "claude_result_error"
                    emit(
                        task_id,
                        "progress",
                        "turn_completed",
                        "claude result received",
                        is_error=bool(parsed.get("is_error")),
                    )
                    should_stop = True
                    continue

            if should_stop:
                break
    finally:
        selector.close()
        if process.poll() is None:
            if timed_out or failure_reason in {"service_timeout", "forbidden_command", "forbidden_tool"}:
                process.kill()
            else:
                # We received a terminal event (e.g. `result`) and can let the CLI exit cleanly.
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        process.kill()

    return_code = process.wait()
    maybe_emit_partial(force=True)
    elapsed_sec = round(time.time() - started, 2)
    final_message = "\n\n".join(final_messages).strip()
    if not final_message:
        if failure_reason:
            final_message = f"claude run failed: {failure_reason}"
        elif timed_out:
            final_message = "claude run timed out"
        else:
            final_message = "claude returned no final message"

    if not failure_reason and (timed_out or return_code != 0):
        failure_reason = "non_zero_exit"

    final_json = extract_final_json_block(final_message)
    if final_json:
        emit(task_id, "progress", "final_json", "parsed FINAL_JSON", final_json=final_json)

    return {
        "engine": "claude",
        "exit_code": return_code,
        "timed_out": timed_out,
        "failure_reason": failure_reason,
        "elapsed_sec": elapsed_sec,
        "final_message": final_message,
        "final_json": final_json,
        "messages": final_messages,
        "reasoning": [],
        "commands": command_steps,
        "usage": usage,
        "stream_stats": {"events": event_count, "json_lines": json_line_count, "raw_lines": raw_line_count},
        "noise_stats": {},
    }


def run_with_retries(args: argparse.Namespace, query: str, context: str) -> int:
    task_base = args.task_id or f"deep-cli-{int(time.time() * 1000)}"
    attempts = max(1, args.retries + 1)
    last_result: Dict[str, Any] = {}

    for attempt in range(1, attempts + 1):
        task_id = f"{task_base}-a{attempt}"
        emit(task_id, "progress", "attempt", f"starting attempt {attempt}/{attempts}")
        if args.engine == "claude":
            result = run_once_claude(args, task_id, query, context)
        else:
            result = run_once_codex(args, task_id, query, context)
        last_result = result
        complete_status = (
            "success"
            if result.get("exit_code") == 0 and not result.get("timed_out")
            else "failed"
        )
        emit(
            task_id,
            "complete",
            complete_status,
            "deep run finished",
            engine=result.get("engine", args.engine),
            exit_code=result.get("exit_code"),
            timed_out=result.get("timed_out"),
            failure_reason=result.get("failure_reason"),
            elapsed_sec=result.get("elapsed_sec"),
            stream_stats=result.get("stream_stats", {}),
            noise_stats=result.get("noise_stats", {}),
            usage=result.get("usage", {}),
            final_message=result.get("final_message", ""),
        )

        should_retry = (
            attempt < attempts
            and result.get("failure_reason") in {"codex_api_timeout", "service_timeout"}
        )
        if not should_retry:
            break
        emit(
            task_id,
            "progress",
            "retrying",
            "retrying after timeout-like failure",
            next_attempt=attempt + 1,
            reason=result.get("failure_reason"),
        )

    if args.print_final and last_result.get("final_message"):
        print(last_result.get("final_message", ""), flush=True)

    return 0 if last_result.get("exit_code") == 0 and not last_result.get("timed_out") else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deep research from CLI (codex or claude) and stream structured JSON logs."
    )
    parser.add_argument("query", help="User query.")
    parser.add_argument("--context", default="", help="Extra context.")
    parser.add_argument("--cwd", default=os.getcwd(), help="Working directory for Codex.")
    parser.add_argument("--timeout-sec", type=int, default=1200, help="Per-attempt timeout seconds.")
    parser.add_argument(
        "--engine",
        default="codex",
        choices=["codex", "claude"],
        help="Deep engine to run: codex or claude.",
    )
    parser.add_argument(
        "--proxy",
        default="",
        help="Optional proxy URL, e.g. http://127.0.0.1:7890 (sets HTTP(S)/ALL_PROXY for codex subprocess).",
    )
    parser.add_argument(
        "--unset-proxy",
        action="store_true",
        help="Unset proxy env vars for the codex subprocess even if present in parent shell.",
    )
    parser.add_argument(
        "--sandbox-mode",
        default="workspace-write",
        choices=["read-only", "workspace-write", "danger-full-access"],
        help="Sandbox mode when privilege_mode=default.",
    )
    parser.add_argument(
        "--privilege-mode",
        # Default to `danger` to avoid Codex LandlockRestrict preventing local command execution.
        default="danger",
        choices=["default", "full-auto", "danger"],
        help="Privilege preset. danger bypasses sandbox to allow local command execution.",
    )
    parser.add_argument(
        "--claude-tools",
        default="Bash",
        help="Claude --tools value when engine=claude (default: Bash).",
    )
    parser.add_argument("--model", default="", help="Optional model name passed to codex --model.")
    parser.add_argument("--retries", type=int, default=0, help="Retry count for timeout-like failures.")
    parser.add_argument("--task-id", default="", help="Optional stable task id prefix.")
    parser.add_argument(
        "--print-final",
        action="store_true",
        help="Print final assistant message as plain text after stream logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query, query_nul = remove_nul_bytes(args.query)
    context, context_nul = remove_nul_bytes(args.context)
    if query_nul or context_nul:
        task_id = args.task_id or f"codex-cli-{int(time.time() * 1000)}"
        emit(
            task_id,
            "progress",
            "sanitized",
            "removed nul bytes from prompt inputs",
            query_nul_removed=query_nul,
            context_nul_removed=context_nul,
        )
    exit_code = run_with_retries(args, query, context)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
