"""Safe orchestration for extracting and writing multiple memory facts."""

from memory.memory_fact_extractor import extract_fact_batch


def write_fact_batch(user_context, writer=None, extractor=None):
    """Extract facts and process them sequentially through the existing writer.

    Dependencies can be injected for isolated tests.
    Stops immediately after index synchronization failure.
    Does not modify the writer's existing decision or validation logic.
    """
    if extractor is None:
        extractor = extract_fact_batch

    if writer is None:
        from memory_writer_v2 import execute
        writer = execute

    facts = extractor(user_context)
    results = []
    memory_ids = []

    for fact in facts:
        category = fact["category"]
        text = fact["fact"]
        writer_input = (
            f"待判断的用户长期事实（类别：{category}）：{text}"
        )

        try:
            result = writer(writer_input, expected_type=category)
        except Exception as exc:
            result = {
                "status": "error",
                "action": "UNKNOWN",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

        if not isinstance(result, dict):
            result = {
                "status": "error",
                "action": "UNKNOWN",
                "error": "Writer returned a non-dictionary result",
            }

        item = {
            "category": category,
            "fact": text,
            "status": result.get("status", "error"),
            "action": result.get("action", "UNKNOWN"),
            "memory_ids": result.get("memory_ids", []),
        }

        # Preserve diagnostic details for failed items without changing
        # the writer's status or the batch's retry/stop behavior.
        if item["status"] not in ("success", "ignored"):
            for key in ("error_type", "error", "reason"):
                if result.get(key) is not None:
                    item[key] = result[key]
        results.append(item)

        if item["status"] in ("success", "index_sync_failed"):
            memory_ids.extend(item["memory_ids"])

        if item["status"] == "index_sync_failed":
            break

    statuses = [item["status"] for item in results]

    if not statuses:
        overall_status = "empty"
    elif "index_sync_failed" in statuses:
        overall_status = "index_sync_failed"
    elif all(status in ("success", "ignored") for status in statuses):
        overall_status = "success"
    elif any(status == "success" for status in statuses):
        overall_status = "partial"
    else:
        overall_status = "error"

    return {
        "status": overall_status,
        "processed": len(results),
        "extracted": len(facts),
        "memory_ids": list(dict.fromkeys(memory_ids)),
        "results": results,
    }
