"""Stage 8: evidence sufficiency gate for memory-grounded answers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def evaluate_evidence(
    query: str,
    memories: Sequence[Mapping[str, Any]],
    *,
    min_score: float = 0.45,
    min_supporting_memories: int = 1,
) -> dict[str, Any]:
    """
    Evaluate whether retrieved active memories provide a minimally
    sufficient basis for answering a memory-grounded question.

    This is a conservative retrieval gate, not a semantic entailment
    verifier. A high score alone cannot prove that a memory supports
    every claim in the answer.
    """
    threshold = max(0.0, min(1.0, float(min_score)))
    eligible = []

    for memory in memories:
        try:
            score = float(memory.get("score", 0.0))
        except (TypeError, ValueError):
            continue

        metadata = memory.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            continue

        if metadata.get("status") != "active":
            continue

        if score < threshold:
            continue

        content = memory.get("content")
        if not isinstance(content, str) or not content.strip():
            continue

        eligible.append({
            "id": memory.get("id"),
            "score": score,
            "content": content,
            "path": str(memory.get("path", "")),
        })

    required = max(1, int(min_supporting_memories))
    sufficient = len(eligible) >= required

    return {
        "answerable_from_retrieval": sufficient,
        "query": query,
        "reason": (
            "eligible_evidence_found"
            if sufficient
            else "insufficient_retrieval_evidence"
        ),
        "evidence_count": len(eligible),
        "min_score": threshold,
        "evidence": eligible,
        "warning": (
            "Retrieval eligibility does not establish that evidence "
            "entails a proposed answer. Validate claims before answering."
        ),
    }


def refusal_message(result: Mapping[str, Any]) -> str:
    """Return a clear refusal when the retrieval gate finds no evidence."""
    if result.get("answerable_from_retrieval"):
        return ""
    return "我在现有记忆中没有找到足够的证据来可靠回答这个问题。"
