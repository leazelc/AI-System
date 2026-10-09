"""Conservative repair policy. This module never executes repairs."""

POLICY_VERSION = 1

# These issue types may be considered for a future, restricted repair.
# Inclusion here does NOT authorize execution.
REPAIR_CANDIDATES = {
    "stale_embedding": {
        "match": "正文哈希与向量索引不一致",
        "proposed_operation": "rebuild_embedding",
        "risk": "medium",
    },
    "missing_embedding": {
        "match": "active Markdown 记忆缺少向量记录",
        "proposed_operation": "create_embedding",
        "risk": "medium",
    },
    "invalid_embedding": {
        "match": "向量数据无效",
        "proposed_operation": "rebuild_embedding",
        "risk": "medium",
    },
}


def classify_repair_issue(problem):
    """Classify an issue without authorizing or executing any operation."""
    problem = str(problem or "")

    # Ambiguous identity or structural problems must remain manual-only.
    manual_only_markers = (
        "相同 ID",
        "重复记录",
        "路径",
        "不存在",
        "无法读取",
        "缺少有效元数据",
        "缺少 ID",
        "未登记",
        "不应保留",
        "不是该 ID 对应的 active 记忆",
        "检查执行失败",
    )

    if any(marker in problem for marker in manual_only_markers):
        return {
            "category": "manual_review",
            "proposed_operation": None,
            "risk": "high",
            "manual_review_required": True,
            "execution_allowed": False,
        }

    for name, rule in REPAIR_CANDIDATES.items():
        if rule["match"] in problem:
            return {
                "category": name,
                "proposed_operation": rule["proposed_operation"],
                "risk": rule["risk"],
                "manual_review_required": True,
                "execution_allowed": False,
            }

    return {
        "category": "unknown",
        "proposed_operation": None,
        "risk": "high",
        "manual_review_required": True,
        "execution_allowed": False,
    }
