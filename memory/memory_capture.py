"""Safe, context-aware capture preparation for long-term user memories.

This module prepares user-authored context only. It does not call models,
write memory files, or update any database/index.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


LONG_TERM_CATEGORIES = (
    "preferences",
    "plans",
    "technical_config",
    "projects",
    "decisions",
    "experiences",
)

DEFAULT_MAX_MESSAGES = 12
DEFAULT_MAX_CHARS = 6000


@dataclass(frozen=True)
class CaptureMessage:
    role: str
    content: str


def _text(content: object) -> str:
    """Normalize a plain-text message; reject non-text payloads safely."""
    if not isinstance(content, str):
        return ""
    return content.strip()


def select_user_context(
    messages: Iterable[Mapping[str, object]],
    *,
    max_messages: int = DEFAULT_MAX_MESSAGES,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[CaptureMessage]:
    """Select recent user-authored messages only, preserving chronological order.

    Assistant/system/tool messages are deliberately excluded to avoid treating
    generated claims as user-confirmed facts. Limits apply to the final context.
    """
    if max_messages < 1 or max_chars < 1:
        return []

    selected: list[CaptureMessage] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        if message.get("role") != "user":
            continue
        content = _text(message.get("content"))
        if content:
            selected.append(CaptureMessage("user", content))

    selected = selected[-max_messages:]
    bounded: list[CaptureMessage] = []
    remaining = max_chars
    for message in reversed(selected):
        if remaining <= 0:
            break
        content = message.content
        if len(content) > remaining:
            content = content[-remaining:]
        bounded.append(CaptureMessage("user", content))
        remaining -= len(content)

    return list(reversed(bounded))


def build_capture_input(
    messages: Sequence[Mapping[str, object]],
    *,
    max_messages: int = DEFAULT_MAX_MESSAGES,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """Build a bounded, explicitly user-sourced input for the existing Gate."""
    selected = select_user_context(
        messages,
        max_messages=max_messages,
        max_chars=max_chars,
    )
    if not selected:
        return ""

    sections = [
        "以下内容全部来自用户消息，按时间顺序排列。",
        "只提取用户明确陈述或确认的长期事实；不要把助手生成的内容视为用户事实。",
        "区分新事实、对旧事实的补充、明确更正和临时信息；不确定时不要推断。",
        "",
    ]
    for index, message in enumerate(selected, 1):
        sections.append(f"[用户消息 {index}]\n{message.content}")
    return "\n".join(sections)
