"""Structured extraction of independent long-term user facts.

This module only builds extraction prompts and validates model output.
It never calls a model or writes memories/indexes itself.
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


FACT_CATEGORIES = frozenset({
    "preferences",
    "plans",
    "technical_config",
    "projects",
    "decisions",
    "experiences",
})

MAX_FACTS = 20
MAX_FACT_CHARS = 1200


def build_extraction_prompt(user_context: str) -> str:
    """Build a prompt that separates explicit user facts from speculation."""
    context = user_context.strip() if isinstance(user_context, str) else ""
    if not context:
        return ""

    return f"""请从下面的用户消息中提取值得长期保存的独立事实。

要求：
1. 只提取用户明确陈述或明确确认的事实，不要把助手的建议当成用户事实。
2. 一条事实只表达一个主要信息；不同的偏好、计划、配置、项目进展和决定应分别列出。
3. 保留重要的具体名称、版本、数值、路径、限制条件和时间。
4. 不要推断，不要把问题、假设、未确认的建议改写成既定事实。
5. 临时问题、普通闲聊、没有长期价值的内容不要提取。
6. 同一事实在上下文中重复出现时只保留一次；明确更正时以用户最新明确表述为准。
7. 项目相关的独立事实不能仅因属于同一项目就合并。
8. category 必须是以下之一：preferences、plans、technical_config、projects、decisions、experiences。
9. 如果没有符合条件的事实，返回空数组。

只返回 JSON，不要添加 Markdown 或其他解释，格式如下：
{{"facts":[{{"category":"projects","fact":"独立且明确的事实"}}]}}

用户消息上下文：
<context>
{context}
</context>
"""


def parse_extracted_facts(
    model_output: str,
    *,
    max_facts: int = MAX_FACTS,
    max_fact_chars: int = MAX_FACT_CHARS,
) -> list[dict[str, str]]:
    """Parse and validate extractor JSON; malformed output yields no facts."""
    if not isinstance(model_output, str) or max_facts < 1 or max_fact_chars < 1:
        return []

    text = model_output.strip()
    if text.startswith("\`\`\`"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("\`\`\`"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "\`\`\`":
            lines = lines[:-1]
        text = "\\n".join(lines).strip()

    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return []

    if not isinstance(payload, Mapping):
        return []
    raw_facts = payload.get("facts")
    if not isinstance(raw_facts, list):
        return []

    result: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for item in raw_facts:
        if len(result) >= max_facts:
            break
        if not isinstance(item, Mapping):
            continue

        category = item.get("category")
        fact = item.get("fact")
        if not isinstance(category, str) or category not in FACT_CATEGORIES:
            continue
        if not isinstance(fact, str):
            continue

        fact = " ".join(fact.split())
        if not fact:
            continue
        if len(fact) > max_fact_chars:
            continue

        key = (category, fact.casefold())
        if key in seen:
            continue

        seen.add(key)
        result.append({"category": category, "fact": fact})

    return result


# Local inference is deliberately separate from memory writing.
import os
import urllib.error
import urllib.request


LM_STUDIO_URL = os.environ.get(
    "AI_MEMORY_CAPTURE_URL",
    "http://127.0.0.1:1234/v1/chat/completions",
)
MODEL = os.environ.get("AI_MEMORY_CAPTURE_MODEL", "openai/gpt-oss-20b")
MODEL_TIMEOUT = 120


class FactExtractionError(RuntimeError):
    """Local model request failed or returned an unusable response."""


def extract_facts_with_local_model(user_context: str) -> list[dict[str, str]]:
    """Extract facts through LM Studio without writing memory or indexes.

    Raises FactExtractionError on transport, API, or response-format failures.
    """
    prompt = build_extraction_prompt(user_context)
    if not prompt:
        return []

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是长期记忆事实提取器。只依据用户提供的上下文，"
                    "严格输出指定 JSON，不得推断，也不得写入任何数据。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
    }
    request = urllib.request.Request(
        LM_STUDIO_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=MODEL_TIMEOUT) as response:
            response_data = json.loads(response.read().decode("utf-8"))
        content = response_data["choices"][0]["message"]["content"]
    except (
        urllib.error.URLError,
        TimeoutError,
        OSError,
        ValueError,
        KeyError,
        IndexError,
        TypeError,
    ) as exc:
        raise FactExtractionError(
            f"Local model request or response failed: {type(exc).__name__}"
        ) from exc

    facts = parse_extracted_facts(content)
    if not isinstance(content, str):
        raise FactExtractionError("Local model returned non-text content.")
    if not facts and content.strip() not in (
        '{"facts":[]}',
        "\`\`\`json\\n{\\\"facts\\\":[]}\\n\`\`\`",
    ):
        # Do not treat malformed or unusable model output as a successful
        # extraction. A valid empty facts array is allowed.
        try:
            parsed = json.loads(content.strip())
        except (json.JSONDecodeError, TypeError):
            raise FactExtractionError("Local model returned invalid fact JSON.")
        if not isinstance(parsed, dict) or parsed.get("facts") != []:
            raise FactExtractionError("Local model returned no valid facts.")

    return facts


def deduplicate_facts(
    facts: list[dict[str, str]],
    *,
    max_facts: int = MAX_FACTS,
    max_fact_chars: int = MAX_FACT_CHARS,
) -> list[dict[str, str]]:
    """Normalize and deduplicate a batch without accessing persistent storage.

    Deduplication is case-insensitive and whitespace-normalized within the
    same category. Facts from different categories remain independent.
    Invalid items are ignored, and input order is preserved.
    """
    if not isinstance(facts, list) or max_facts < 1 or max_fact_chars < 1:
        return []

    result: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for item in facts:
        if len(result) >= max_facts:
            break
        if not isinstance(item, Mapping):
            continue

        category = item.get("category")
        fact = item.get("fact")
        if not isinstance(category, str) or category not in FACT_CATEGORIES:
            continue
        if not isinstance(fact, str):
            continue

        fact = " ".join(fact.split())
        if not fact or len(fact) > max_fact_chars:
            continue

        key = (category, fact.casefold())
        if key in seen:
            continue

        seen.add(key)
        result.append({"category": category, "fact": fact})

    return result


def extract_fact_batch(user_context: str) -> list[dict[str, str]]:
    """Extract and deduplicate facts without writing persistent memory.

    Model/API failures propagate as FactExtractionError. An empty context
    returns an empty batch without making a network request.
    """
    facts = extract_facts_with_local_model(user_context)
    return deduplicate_facts(facts)
