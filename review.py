from __future__ import annotations

import json
import re
import urllib.request
from typing import Any, Dict, Tuple

DEFAULT_CHAT_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL_NAME = "GLM-4-Flash"
KEYWORD_MODEL_NAME = "glm-4.7"

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
        "你是一个研究助理，需要把问题扩展成用于检索的概念集合。\n"
        "- 提取核心本质、隐含约束、外延相关概念。\n"
        "- 可包含：基本概念、相关领域、对照/反义概念（如有帮助）。\n"
        "- 若是关系/情感类问题，优先包含 1-2 个核心情感/价值概念（如：爱、信任、尊重、安全感、亲密、边界），选择与问题最相关者。\n"
        f"- 正常返回 3-5 个关键词，最多 {max_keywords} 个。\n"
        "- 优先具体概念，避免空泛词；关键词以中文为主。\n"
        "- 优先从下方“候选关键词库”中选取最相关项；必要时可补充少量新词。\n"
        "- 只输出 JSON。\n\n"
        "候选关键词库：\n"
        "策略、怀疑的艺术、人际交往、社会化、亲密关系、爱、子女教育、自然法、社会伦理、历史、中国、会计、价值观、风险管理、亲子关系、可持续性、职业伦理、组织伦理、职场、学习、反脆弱、功夫、文化、补弱/补强、科普、财富观、反抑郁、社会治理、自我认知、安全、世界史、语文、国际政治、国际关系、反自欺、人际关系、政治、成熟、责任、自强、方法论、沟通、婚姻、信仰、权柄、信源管理、技术、技术解决、礼仪、艺术、净输出、商业、伦理、女性、教育、神性享乐、两性关系、人己权界、恋爱、尊重、悲观/乐观、傲慢、市场、自由、POETIC IRONY、驱动力、SOP、事业、影响力、人生观、能力、法律、情绪、职业规划、管理、投资、独立、表达、家族、旷野、企管、错得对、择善固执、领导、宗教、元技能、劳动、理想、分手、组织管理、态度、快乐、美、商业伦理、战争、隐私、谦卑、军事、科学、美国。\n\n"
        "示例：\n"
        "问题：恋人在争吵的冷静期间，适合做些什么？\n"
        "只输出 JSON：{\"keywords\": [\"爱\", \"冷静\", \"情绪管理\", \"沟通\", \"亲密关系\"]}\n\n"
        f"问题：{query}\n\n"
        '只输出 JSON：{"keywords": ["...", "..."]}'
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
        "thinking": {"type": "disabled"},
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
    payload = build_keyword_payload(query, max_keywords, model=KEYWORD_MODEL_NAME)
    error = ""
    primary_response: Dict[str, Any] = {}
    fallback_response: Dict[str, Any] = {}
    try:
        response = post_json(chat_url, payload)
    except Exception as exc:
        error = str(exc)
        response = {}

    primary_response = response
    parsed = parse_keywords_response(response, max_keywords=max_keywords)
    if not parsed.get("raw_text"):
        fallback_payload = build_keyword_payload(query, max_keywords, model=MODEL_NAME)
        try:
            response = post_json(chat_url, fallback_payload)
        except Exception as exc:
            if not error:
                error = str(exc)
        else:
            parsed = parse_keywords_response(response, max_keywords=max_keywords)
            payload = fallback_payload
            fallback_response = response
    return {
        "query": query,
        "keywords": parsed.get("keywords", []),
        "raw_text": parsed.get("raw_text", ""),
        "parsed": parsed.get("parsed", {}),
        "payload": payload,
        "response": response,
        "response_primary": primary_response,
        "response_fallback": fallback_response,
        "keyword_model": payload.get("model", ""),
        "error": error,
    }
