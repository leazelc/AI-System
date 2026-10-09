"""Read-only diagnostic reports for the local AI memory system."""

from pathlib import Path

from memory.memory_consistency import (
    CORE_DB_PATH,
    EMBEDDING_DB_PATH,
    MEMORY_ROOT,
    run_consistency_check,
)


def _classify_issue(issue):
    """Return a conservative recommendation and repairability assessment."""
    problem = str(issue.get("problem", "未知问题"))

    # Duplicate IDs require a human to determine which memory is authoritative.
    if "多个 Markdown 文件使用相同 ID" in problem:
        return (
            "检查关联文件的内容与来源，人工确定正确的 ID 分配；"
            "确认前不要删除或覆盖任何文件。",
            False,
        )

    if "多个 active Markdown 文件使用相同 ID" in problem:
        return (
            "检查所有关联文件，确认是否为重复记忆或 ID 冲突；"
            "人工确认前不要自动修改。",
            False,
        )

    if "正文哈希与向量索引不一致" in problem:
        return (
            "核对 Markdown 正文是否为最新版本；确认内容正确后，"
            "再考虑重新生成对应向量，并在修复后重新检查。",
            True,
        )

    if "缺少向量记录" in problem:
        return (
            "确认该文件确实是应建立索引的 active 记忆，"
            "然后考虑重新生成向量并验证索引。",
            True,
        )

    if "向量数据无效" in problem:
        return (
            "检查向量维度、字节长度及数值有效性；"
            "确认源文件正确后，再考虑重新生成向量。",
            True,
        )

    if "Markdown 文件未登记在结构化索引中" in problem:
        return (
            "核对 Markdown 元数据与索引同步规则；"
            "确认记录完整、合法后，再考虑重建对应索引。",
            True,
        )

    if "数据库记录指向的 Markdown 文件不存在" in problem:
        return (
            "检查文件是否被移动、归档或删除，并核对数据库路径；"
            "不要仅凭文件缺失就直接删除数据库记录。",
            False,
        )

    if "向量记录对应的 Markdown 文件不存在" in problem:
        return (
            "检查源文件是否被移动、归档或删除；"
            "确认其真实状态后，再决定如何处理向量记录。",
            False,
        )

    if "数据库路径位于记忆目录之外" in problem:
        return (
            "检查数据库中的路径是否正确，并确认目标文件的真实位置；"
            "确认前不要自动移动文件或改写数据库。",
            False,
        )

    if "向量记录的路径位于记忆目录之外" in problem:
        return (
            "核对向量记录中的文件路径与记忆目录；"
            "确认源文件位置后再决定如何处理。",
            False,
        )

    if "缺少有效元数据" in problem or "缺少 ID" in problem:
        return (
            "检查 Markdown 文件的 frontmatter 和必需字段；"
            "先确认记忆身份及状态，再决定是否补充元数据。",
            False,
        )

    if "不一致" in problem or "实际路径" in problem:
        return (
            "对照 Markdown 文件和相关数据库记录，确认哪一方是权威来源；"
            "确认前不要直接覆盖任何一方。",
            False,
        )

    return (
        "暂时无法安全地自动判断修复方式。请检查原始问题、相关文件"
        "及数据库记录，再决定下一步。",
        False,
    )


def build_diagnostic_report(raw_report):
    """Enrich an existing consistency report without changing source data."""
    raw_issues = raw_report.get("issues", [])
    issues = []

    for raw_issue in raw_issues:
        issue = dict(raw_issue)
        recommendation, repairable = _classify_issue(issue)

        related_paths = issue.get("paths", [])
        if not related_paths and issue.get("path"):
            related_paths = [issue["path"]]

        issue["severity"] = issue.get("level", "error")
        issue["memory_id"] = issue.get("id")
        issue["related_paths"] = list(related_paths)
        issue["recommendation"] = recommendation
        issue["repairable"] = repairable

        issues.append(issue)

    error_count = sum(
        issue.get("severity") == "error"
        for issue in issues
    )
    warning_count = sum(
        issue.get("severity") == "warning"
        for issue in issues
    )

    return {
        "checks": list(raw_report.get("checks", [])),
        "issues": issues,
        "issue_count": len(issues),
        "error_count": error_count,
        "warning_count": warning_count,
        "repairable_count": sum(
            issue["repairable"] for issue in issues
        ),
        "read_only": True,
    }


def diagnose_memory(
    memory_root=MEMORY_ROOT,
    core_db_path=CORE_DB_PATH,
    embedding_db_path=EMBEDDING_DB_PATH,
):
    """Run consistency checks and return an enriched read-only report."""
    try:
        raw_report = run_consistency_check(
            memory_root,
            core_db_path,
            embedding_db_path,
        )
        return build_diagnostic_report(raw_report)

    except Exception as exc:
        issue = {
            "level": "error",
            "severity": "error",
            "check": "诊断器",
            "problem": (
                f"一致性检查无法完成：{type(exc).__name__}: {exc}"
            ),
            "recommendation": (
                "检查数据库文件、目录权限和异常详情；"
                "解决原因前不要执行自动修复。"
            ),
            "memory_id": None,
            "related_paths": [],
            "repairable": False,
        }

        return {
            "checks": [{
                "name": "一致性检查",
                "status": "failed",
                "issue_count": 1,
            }],
            "issues": [issue],
            "issue_count": 1,
            "error_count": 1,
            "warning_count": 0,
            "repairable_count": 0,
            "read_only": True,
        }


def format_diagnostic_report(report):
    """Format a diagnostic report as human-readable text."""
    lines = [
        "========================================",
        "本地 AI 记忆系统：故障诊断报告",
        "========================================",
        f"问题总数：{report['issue_count']}",
        f"错误：{report['error_count']}",
        f"警告：{report['warning_count']}",
        f"存在候选修复方案的问题：{report['repairable_count']}",
        "数据修改：未执行（只读诊断）",
        "",
    ]

    if not report["issues"]:
        lines.append("未发现问题。")
        return "\n".join(lines)

    for number, issue in enumerate(report["issues"], start=1):
        lines.extend([
            f"[{number}] {issue.get('severity', 'error').upper()}",
            f"检查项目：{issue.get('check', '未指定')}",
            f"记忆 ID：{issue.get('memory_id') or '无'}",
            f"问题：{issue.get('problem', '未知问题')}",
        ])

        paths = issue.get("related_paths", [])
        if paths:
            lines.append("相关路径：")
            lines.extend(f"  - {path}" for path in paths)

        lines.append(
            f"处理建议：{issue.get('recommendation', '请人工检查。')}"
        )
        lines.append(
            "存在候选修复方案："
            + ("是（尚未执行）" if issue.get("repairable") else "否")
        )
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    report = diagnose_memory()
    print(format_diagnostic_report(report))
