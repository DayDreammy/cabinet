from __future__ import annotations

import json
import re
import urllib.request
from typing import Any, Dict, Tuple

DEFAULT_CHAT_URL = "http://127.0.0.1:8000/v1/chat/completions"


def _post_json(url: str, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _extract_message_content(response: Dict[str, Any]) -> str:
    choices = response.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    if isinstance(message, dict):
        return message.get("content", "") or ""
    return choices[0].get("text", "") or ""


def _extract_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _locate_quote(content: str, quote: str) -> Tuple[int, int, str]:
    if not quote:
        return -1, -1, ""
    start = content.find(quote)
    if start != -1:
        return start, start + len(quote), quote

    trimmed = quote.strip()
    if trimmed:
        start = content.find(trimmed)
        if start != -1:
            return start, start + len(trimmed), trimmed

    return -1, -1, ""


def review_doc(doc: Dict[str, Any], query: str, chat_url: str) -> Dict[str, Any]:
    title = doc.get("title", "")
    content = doc.get("content", "") or ""

    prompt = (
        "You are a reviewer. Only quote exact sentences from the article. "
        "If the article does not answer the question, return an empty quote.\n\n"
        f"Question: {query}\n\n"
        f"Title: {title}\n"
        f"Article: {content}\n\n"
        "Return JSON only: {\"quote\": \"...\", \"score\": 0-10}"
    )

    payload = {
        "model": "GLM-4-Flash",
        "agentic": False,
        "temperature": 0.2,
        "max_tokens": 512,
        "messages": [
            {
                "role": "system",
                "content": "You are a reviewer. Use only exact quotes from the article.",
            },
            {"role": "user", "content": prompt},
        ],
    }

    error = ""
    try:
        response = _post_json(chat_url, payload)
    except Exception as exc:
        error = str(exc)
        response = {}

    content_text = _extract_message_content(response)
    parsed = _extract_json(content_text)

    quote = str(parsed.get("quote", "")) if parsed else ""
    score = parsed.get("score", 0) if parsed else 0
    try:
        score_value = float(score)
    except (TypeError, ValueError):
        score_value = 0.0

    start, end, quote = _locate_quote(content, quote)
    if start == -1:
        quote = ""
        start = 0
        end = 0
        score_value = 0.0

    return {
        "id": doc.get("id", ""),
        "title": title,
        "url": doc.get("url", ""),
        "quote": quote,
        "quote_start": start,
        "quote_end": end,
        "score": score_value,
        "error": error,
    }
