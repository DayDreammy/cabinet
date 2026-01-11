from __future__ import annotations

import json
import re
import urllib.request
from typing import Any, Dict, Tuple

DEFAULT_CHAT_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL_NAME = "GLM-4-Flash"

_QUOTE_NORMALIZE_BASE = str.maketrans(
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
_QUOTE_NORMALIZE_ALL = str.maketrans(
    {
        ord('"'): '"',
        ord("'"): '"',
        ord("“"): '"',
        ord("”"): '"',
        ord("„"): '"',
        ord("‟"): '"',
        ord("«"): '"',
        ord("»"): '"',
        ord("‹"): '"',
        ord("›"): '"',
        ord("‘"): '"',
        ord("’"): '"',
        ord("‚"): '"',
        ord("‛"): '"',
        ord("「"): '"',
        ord("」"): '"',
        ord("『"): '"',
        ord("』"): '"',
        ord("《"): '"',
        ord("》"): '"',
        ord("〈"): '"',
        ord("〉"): '"',
        ord("＂"): '"',
        ord("＇"): '"',
    }
)


def post_json(url: str, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_review_prompt(doc: Dict[str, Any], query: str) -> str:
    title = doc.get("title", "")
    content = doc.get("content", "") or ""
    return (
        "You are a reviewer. Only quote exact sentences from the article. "
        "If the article does not answer the question, return an empty quote.\n\n"
        f"Question: {query}\n\n"
        f"Title: {title}\n"
        f"Article: {content}\n\n"
        'Return JSON only: {"quote": "...", "score": 0-10}'
    )


def build_keyword_prompt(query: str, max_keywords: int) -> str:
    return (
        "You are a research assistant. Extract core search keywords from the question.\n"
        f"- Return 3-5 keywords if possible, up to {max_keywords} max.\n"
        "- Prefer concrete terms and key concepts; avoid filler words.\n"
        "- Output JSON only.\n\n"
        f"Question: {query}\n\n"
        'Return JSON only: {"keywords": ["...", "..."]}'
    )


def build_review_payload(
    doc: Dict[str, Any], query: str, model: str = MODEL_NAME
) -> Dict[str, Any]:
    prompt = build_review_prompt(doc, query)
    return {
        "model": model,
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


def build_keyword_payload(
    query: str, max_keywords: int, model: str = MODEL_NAME
) -> Dict[str, Any]:
    prompt = build_keyword_prompt(query, max_keywords)
    return {
        "model": model,
        "agentic": False,
        "temperature": 0.2,
        "max_tokens": 256,
        "messages": [
            {
                "role": "system",
                "content": "Extract search keywords and return JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
    }


def extract_message_content(response: Dict[str, Any]) -> str:
    choices = response.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    if isinstance(message, dict):
        return message.get("content", "") or ""
    return choices[0].get("text", "") or ""


def extract_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def parse_keywords_response(
    response: Dict[str, Any], max_keywords: int
) -> Dict[str, Any]:
    content_text = extract_message_content(response)
    parsed = extract_json(content_text)
    keywords = []
    if isinstance(parsed, dict):
        raw_list = parsed.get("keywords") or parsed.get("key_terms") or []
        if isinstance(raw_list, list):
            keywords = [str(item).strip() for item in raw_list if str(item).strip()]

    seen = set()
    deduped = []
    for item in keywords:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
        if len(deduped) >= max_keywords:
            break

    return {
        "keywords": deduped,
        "raw_text": content_text,
        "parsed": parsed,
    }


def _find_with_map(content: str, quote: str, trans: Dict[int, str]) -> int:
    if not quote:
        return -1
    norm_content = content.translate(trans)
    norm_quote = quote.translate(trans)
    return norm_content.find(norm_quote)


def _normalize_wo_space(text: str, trans: Dict[int, str]) -> Tuple[str, List[int]]:
    normalized = []
    index_map: List[int] = []
    for idx, ch in enumerate(text):
        if ch.isspace():
            continue
        norm_ch = ch.translate(trans)
        if norm_ch.isspace():
            continue
        normalized.append(norm_ch)
        index_map.append(idx)
    return "".join(normalized), index_map


def locate_quote(content: str, quote: str) -> Tuple[int, int, str, str]:
    if not quote:
        return -1, -1, "", "empty"
    start = content.find(quote)
    if start != -1:
        return start, start + len(quote), quote, "exact"

    trimmed = quote.strip()
    if trimmed and trimmed != quote:
        start = content.find(trimmed)
        if start != -1:
            return start, start + len(trimmed), trimmed, "trimmed"

    start = _find_with_map(content, quote, _QUOTE_NORMALIZE_BASE)
    if start != -1:
        return start, start + len(quote), content[start : start + len(quote)], "normalize_curly"

    start = _find_with_map(content, quote, _QUOTE_NORMALIZE_ALL)
    if start != -1:
        return start, start + len(quote), content[start : start + len(quote)], "normalize_all"

    norm_content, index_map = _normalize_wo_space(content, _QUOTE_NORMALIZE_BASE)
    norm_quote, _ = _normalize_wo_space(quote, _QUOTE_NORMALIZE_BASE)
    if norm_quote:
        pos = norm_content.find(norm_quote)
        if pos != -1:
            start = index_map[pos]
            end = index_map[pos + len(norm_quote) - 1] + 1
            return start, end, content[start:end], "normalize_ws"

    norm_content, index_map = _normalize_wo_space(content, _QUOTE_NORMALIZE_ALL)
    norm_quote, _ = _normalize_wo_space(quote, _QUOTE_NORMALIZE_ALL)
    if norm_quote:
        pos = norm_content.find(norm_quote)
        if pos != -1:
            start = index_map[pos]
            end = index_map[pos + len(norm_quote) - 1] + 1
            return start, end, content[start:end], "normalize_ws_all"

    return -1, -1, "", "not_found"


def parse_review_response(doc: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
    content_text = extract_message_content(response)
    parsed = extract_json(content_text)

    quote_raw = str(parsed.get("quote", "")) if parsed else ""
    score = parsed.get("score", 0) if parsed else 0
    try:
        score_value = float(score)
    except (TypeError, ValueError):
        score_value = 0.0

    content = doc.get("content", "") or ""
    start, end, quote, strategy = locate_quote(content, quote_raw)
    if start == -1:
        quote = ""
        start = 0
        end = 0
        score_value = 0.0

    return {
        "quote_raw": quote_raw,
        "quote": quote,
        "quote_start": start,
        "quote_end": end,
        "score": score_value,
        "match_strategy": strategy,
        "raw_text": content_text,
        "parsed": parsed,
    }


def review_doc(doc: Dict[str, Any], query: str, chat_url: str) -> Dict[str, Any]:
    payload = build_review_payload(doc, query, model=MODEL_NAME)
    error = ""
    try:
        response = post_json(chat_url, payload)
    except Exception as exc:
        error = str(exc)
        response = {}

    parsed = parse_review_response(doc, response)

    return {
        "id": doc.get("id", ""),
        "title": doc.get("title", ""),
        "url": doc.get("url", ""),
        "quote": parsed.get("quote", ""),
        "quote_start": parsed.get("quote_start", 0),
        "quote_end": parsed.get("quote_end", 0),
        "score": parsed.get("score", 0.0),
        "error": error,
    }


def extract_keywords(query: str, chat_url: str, max_keywords: int = 10) -> Dict[str, Any]:
    payload = build_keyword_payload(query, max_keywords, model=MODEL_NAME)
    error = ""
    try:
        response = post_json(chat_url, payload)
    except Exception as exc:
        error = str(exc)
        response = {}

    parsed = parse_keywords_response(response, max_keywords=max_keywords)
    return {
        "query": query,
        "keywords": parsed.get("keywords", []),
        "raw_text": parsed.get("raw_text", ""),
        "parsed": parsed.get("parsed", {}),
        "payload": payload,
        "error": error,
    }
