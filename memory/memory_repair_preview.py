"""Generate repair plans without modifying memory files or databases."""


def _build_plan(issue):
    """Build a conservative, non-executable repair proposal."""
    problem = str(issue.get("problem", "未知问题"))
    memory_id = issue.get("memory_id") or issue.get("id")
    paths = list(issue.get("related_paths") or [])

    base = {
        "memory_id": memory_id,
        "problem": problem,
        "related_paths": paths,
        "source_check": issue.get("check", "未指定"),
        "severity": issue.get("severity", issue.get("level", "error")),
        "executed": False,
    }

    # Duplicate IDs must never be resolved automatically.
    if "多个 Markdown 文件使用相同 ID" in problem:
        return {
            **base,
            "proposed_action": (
                "人工比较所有关联文件，确认记忆归属后制定独立方案；"
                "不得自动删除、覆盖或重新分配 ID。"
            ),
            "risk": "high",
            "manual_review_required": True,
        }

    # Unknown problems remain manual-only, even if the diagnostic report
    # mistakenly labels them repairable.
    if "正文哈希与向量索引不一致" in problem:
        return {
            **base,
            "proposed_action": (
                "核对 Markdown 正文及其元数据；确认源内容正确后，"
                "再考虑重新生成对应向量并复检。当前仅预览，不执行。"
            ),
            "risk": "medium",
            "manual_review_required": True,
        }

    if "缺少向量记录" in problem:
        return {
            **base,
            "proposed_action": (
                "确认该文件属于需要建立向量索引的 active 记忆后，"
                "再考虑补建向量并复检。当前仅预览，不执行。"
            ),
            "risk": "medium",
            "manual_review_required": True,
        }

    if "向量数据无效" in problem:
        return {
            **base,
            "proposed_action": (
                "检查源文件、向量维度和数据格式；确认源内容正确后，"
                "再考虑重新生成向量并复检。当前仅预览，不执行。"
            ),
            "risk": "medium",
            "manual_review_required": True,
        }

    if "Markdown 文件未登记在结构化索引中" in problem:
        return {
            **base,
            "proposed_action": (
                "核对 Markdown 元数据和索引同步规则，"
                "确认记录合法后再考虑补建索引。当前仅预览，不执行。"
            ),
            "risk": "medium",
            "manual_review_required": True,
        }

    return {
        **base,
        "proposed_action": (
            "暂不提出自动修复操作。请人工核对问题、源文件和相关索引，"
            "确认原因后再制定方案。"
        ),
        "risk": "high",
        "manual_review_required": True,
    }


def build_repair_preview(diagnostic_report):
    """Create a reviewable plan; never perform writes or repairs."""
    issues = diagnostic_report.get("issues", [])

    plans = [
        _build_plan(issue)
        for issue in issues
    ]

    return {
        "plan_count": len(plans),
        "plans": plans,
        "preview_only": True,
        "executed": False,
        "read_only": True,
    }


def format_repair_preview(preview):
    """Format a repair preview for terminal display."""
    lines = [
        "========================================",
        "本地 AI 记忆系统：修复预览",
        "========================================",
        f"发现问题：{preview['plan_count']}",
        "实际执行：否",
        "数据修改：未执行（只读预览）",
        "",
    ]

    if not preview["plans"]:
        lines.append("没有需要审查的修复方案。")
        return "\n".join(lines)

    for number, plan in enumerate(preview["plans"], start=1):
        lines.extend([
            f"[{number}] {plan['risk'].upper()}",
            f"检查项目：{plan.get('source_check', '未指定')}",
            f"记忆 ID：{plan.get('memory_id') or '无'}",
            f"问题：{plan.get('problem', '未知问题')}",
        ])

        paths = plan.get("related_paths", [])
        if paths:
            lines.append("相关路径：")
            lines.extend(f"  - {path}" for path in paths)

        lines.extend([
            f"建议操作：{plan['proposed_action']}",
            "需要人工审核："
            + ("是" if plan["manual_review_required"] else "否"),
            "执行状态：尚未执行",
            "",
        ])

    return "\n".join(lines)


def run_repair_preview():
    """Run diagnostics and display a repair preview without applying it."""
    from memory.memory_diagnostics import (
        diagnose_memory,
        format_diagnostic_report,
    )

    diagnostic_report = diagnose_memory()

    print(format_diagnostic_report(diagnostic_report))
    print()
    preview = build_repair_preview(diagnostic_report)
    print(format_repair_preview(preview))

    return preview


if __name__ == "__main__":
    run_repair_preview()
