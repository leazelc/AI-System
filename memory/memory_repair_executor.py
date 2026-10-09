"""Conservative repair executor.

Default behavior is preview-only.
Only explicitly approved embedding repairs may be considered.
Ambiguous identity, path, metadata, or database issues are rejected.
"""

from pathlib import Path
import math

from memory.memory_repair_policy import classify_repair_issue
from memory.memory_repair_backup import (
    create_repair_backup,
    verify_repair_backup,
)


ALLOWED_OPERATIONS = {
    "stale_embedding": "rebuild_embedding",
    "missing_embedding": "create_embedding",
    "invalid_embedding": "rebuild_embedding",
}


def _refuse(reason, issue=None):
    return {
        "success": False,
        "executed": False,
        "refused": True,
        "reason": reason,
        "memory_id": (issue or {}).get("memory_id")
        or (issue or {}).get("id"),
    }


def _resolve_source(issue, memory_root):
    """Resolve and validate a single unambiguous active Markdown source."""
    memory_root = Path(memory_root).expanduser().resolve()
    paths = issue.get("related_paths") or []

    if not paths and issue.get("path"):
        paths = [issue["path"]]

    if len(paths) != 1:
        raise ValueError("必须且只能有一个明确的源文件路径")

    source = Path(paths[0]).expanduser().resolve(strict=True)

    try:
        source.relative_to(memory_root)
    except ValueError:
        raise ValueError("源文件位于记忆目录之外")

    if not source.is_file() or source.suffix.lower() != ".md":
        raise ValueError("源文件必须是记忆目录内的 Markdown 文件")

    if "archive" in source.relative_to(memory_root).parts:
        raise ValueError("归档目录中的文件不允许自动修复")

    text = source.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError("源文件缺少 frontmatter")

    parts = text.split("---", 2)
    if len(parts) != 3:
        raise ValueError("源文件 frontmatter 格式无效")

    metadata = {}
    for line in parts[1].strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()

    memory_id = issue.get("memory_id") or issue.get("id")
    if not memory_id or metadata.get("id") != memory_id:
        raise ValueError("问题 ID 与 Markdown 内部 ID 不一致")

    if metadata.get("status") != "active":
        raise ValueError("源记忆不是 active 状态")

    return source, memory_id, text, parts[2].strip()


def _read_embedding_rows(db_path):
    """Read relevant index rows using SQLite read-only mode."""
    import sqlite3

    db_path = Path(db_path).expanduser().resolve(strict=True)
    uri = db_path.as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        return conn.execute(
            """
            SELECT id, path, content_hash, dimension, vector, updated_at
            FROM embeddings
            ORDER BY id
            """
        ).fetchall()
    finally:
        conn.close()


def _check_identity(issue, source, memory_id, db_path):
    """Reject duplicate IDs, path collisions, or ambiguous index state."""
    import sqlite3

    rows = _read_embedding_rows(db_path)
    id_rows = [row for row in rows if row[0] == memory_id]
    resolved_source = source.resolve()

    path_rows = []
    for row in rows:
        try:
            if Path(row[1]).expanduser().resolve() == resolved_source:
                path_rows.append(row)
        except (OSError, RuntimeError):
            return "索引中存在无法安全解析的路径"

    if len(id_rows) > 1:
        return "向量索引中存在重复 ID"

    if len(path_rows) > 1:
        return "向量索引中存在重复路径"

    if any(row[0] != memory_id for row in path_rows):
        return "源文件路径已关联其他记忆 ID"

    if any(Path(row[1]).expanduser().resolve() != resolved_source
           for row in id_rows):
        return "同一 ID 指向其他源文件"

    try:
        conn = sqlite3.connect(
            Path(db_path).expanduser().resolve().as_uri() + "?mode=ro",
            uri=True,
        )
        try:
            duplicate = conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE id = ?",
                (memory_id,),
            ).fetchone()[0]
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return f"无法读取向量索引：{exc}"

    if duplicate > 1:
        return "向量索引 ID 冲突"

    return None


def execute_repair(
    issue,
    *,
    execute=False,
    approved=False,
    memory_root=None,
    embedding_db_path=None,
    backup_root=None,
):
    """Preview by default; never repair without explicit approval."""
    from memory.memory_consistency import (
        MEMORY_ROOT,
        EMBEDDING_DB_PATH,
    )

    memory_root = Path(memory_root or MEMORY_ROOT).expanduser().resolve()
    embedding_db_path = Path(
        embedding_db_path or EMBEDDING_DB_PATH
    ).expanduser().resolve()

    problem = str(issue.get("problem", "未知问题"))
    policy = classify_repair_issue(problem)

    if not execute:
        return {
            "success": True,
            "executed": False,
            "preview_only": True,
            "memory_id": issue.get("memory_id") or issue.get("id"),
            "category": policy["category"],
            "proposed_operation": policy["proposed_operation"],
            "approval_required": True,
        }

    if not approved:
        return _refuse("执行操作必须显式提供 approved=True", issue)

    if not policy["execution_allowed"]:
        return _refuse("修复策略未授权此问题类型", issue)

    category = policy["category"]
    operation = policy["proposed_operation"]

    if category not in ALLOWED_OPERATIONS:
        return _refuse("问题类型不在执行器白名单中", issue)

    if operation != ALLOWED_OPERATIONS[category]:
        return _refuse("策略操作与执行器白名单不一致", issue)

    try:
        source, memory_id, source_text, content = _resolve_source(
            issue, memory_root
        )
    except (ValueError, OSError, UnicodeError) as exc:
        return _refuse(f"源文件验证失败：{exc}", issue)

    if not embedding_db_path.is_file():
        return _refuse("向量数据库不存在", issue)

    identity_error = _check_identity(
        issue, source, memory_id, embedding_db_path
    )
    if identity_error:
        return _refuse(identity_error, issue)

    # Re-diagnose immediately before mutation. The issue must still exist
    # and match the same category, memory ID, and source path.
    from memory.memory_diagnostics import diagnose_memory

    try:
        current_report = diagnose_memory(
            memory_root=memory_root,
            core_db_path=memory_root / "index" / "memory.sqlite",
            embedding_db_path=embedding_db_path,
        )
    except Exception as exc:
        return _refuse(
            f"修复前重新诊断失败，拒绝执行："
            f"{type(exc).__name__}: {exc}",
            issue,
        )

    if (
        not isinstance(current_report, dict)
        or not isinstance(current_report.get("issues"), list)
        or not isinstance(current_report.get("error_count", 0), int)
        or isinstance(current_report.get("error_count", 0), bool)
        or current_report.get("error_count", 0) < 0
    ):
        return _refuse("修复前诊断报告无效，拒绝执行", issue)

    if any(
        not isinstance(item, dict)
        for item in current_report["issues"]
    ):
        return _refuse("修复前诊断问题列表格式无效，拒绝执行", issue)

    current_matches = []
    for current in current_report.get("issues", []):
        current_id = current.get("memory_id") or current.get("id")
        current_paths = current.get("related_paths") or []
        if not current_paths and current.get("path"):
            current_paths = [current["path"]]

        if current_id == memory_id and str(current.get("problem", "")) == problem:
            if len(current_paths) == 1:
                try:
                    if Path(current_paths[0]).expanduser().resolve() == source:
                        current_matches.append(current)
                except (OSError, RuntimeError):
                    pass

    if len(current_matches) != 1:
        return _refuse("重新诊断未能唯一确认原问题仍存在", issue)

    if current_report.get("error_count", 0) < 0:
        return _refuse("诊断报告状态无效", issue)

    # Back up the source Markdown and a consistent SQLite snapshot.
    # SQLite's backup API avoids copying a live database file directly.
    import sqlite3
    import tempfile
    import hashlib
    import os

    temporary_snapshot = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="embedding_snapshot_",
            suffix=".sqlite",
            delete=False,
        ) as temp_file:
            temporary_snapshot = Path(temp_file.name)

        src_conn = sqlite3.connect(
            embedding_db_path.as_uri() + "?mode=ro", uri=True
        )
        dst_conn = sqlite3.connect(temporary_snapshot)
        try:
            src_conn.backup(dst_conn)
            dst_conn.commit()
        finally:
            dst_conn.close()
            src_conn.close()

        backup_result = create_repair_backup(
            [source, temporary_snapshot],
            backup_root=backup_root,
        )

        verified = verify_repair_backup(backup_result["backup_dir"])
        if not backup_result.get("success") or not verified.get("valid"):
            return _refuse("修复前备份验证失败，拒绝执行", issue)

        # Re-read the source and re-check its identity after backup.
        if source.read_text(encoding="utf-8") != source_text:
            return _refuse("备份期间源文件发生变化，拒绝执行", issue)

        from memory.embedding_index import get_embedding, vector_to_blob
        from memory.memory_consistency import markdown_content_hash

        new_vector = get_embedding(content)
        if not isinstance(new_vector, list) or not new_vector:
            return _refuse("Embedding 服务返回空向量", issue)

        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in new_vector
        ):
            return _refuse("Embedding 服务返回无效数值", issue)

        current_hash = markdown_content_hash(source)

        from datetime import datetime, timezone
        from memory.memory_repair_transaction import (
            apply_embedding_repair,
            restore_embedding_repair,
        )

        vector_blob = vector_to_blob(new_vector)
        repair_result = apply_embedding_repair(
            embedding_db_path,
            operation=operation,
            memory_id=memory_id,
            source_path=str(source),
            content_hash=current_hash,
            dimension=len(new_vector),
            vector_blob=vector_blob,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

        # Re-run consistency checks after the committed transaction.
        # If diagnostics raise or return an invalid report, treat this as
        # a post-commit failure so the targeted recovery path still runs.
        try:
            after_report = diagnose_memory(
                memory_root=memory_root,
                core_db_path=memory_root / "index" / "memory.sqlite",
                embedding_db_path=embedding_db_path,
            )

            if (
                not isinstance(after_report, dict)
                or not isinstance(after_report.get("issues"), list)
            ):
                raise ValueError("复检返回了无效的诊断报告")

            after_issues = after_report["issues"]
            diagnostic_failed = any(
                "检查无法完成" in str(item.get("problem", ""))
                or "检查执行失败" in str(item.get("problem", ""))
                for item in after_issues
            )

        except Exception as diagnostic_exc:
            diagnostic_failed = True
            after_issues = [{
                "memory_id": memory_id,
                "problem": (
                    f"检查执行失败："
                    f"{type(diagnostic_exc).__name__}: {diagnostic_exc}"
                ),
            }]

        remaining = [
            item for item in after_issues
            if (item.get("memory_id") or item.get("id")) == memory_id
            and (
                "正文哈希与向量索引不一致" in str(item.get("problem", ""))
                or "缺少向量记录" in str(item.get("problem", ""))
                or "向量数据无效" in str(item.get("problem", ""))
            )
        ]

        if diagnostic_failed or remaining:
            try:
                restore_result = restore_embedding_repair(
                    embedding_db_path, repair_result
                )
                recovery = {
                    "attempted": True,
                    "success": bool(restore_result.get("success")),
                    "reason": None,
                }
            except Exception as recovery_exc:
                recovery = {
                    "attempted": True,
                    "success": False,
                    "reason": (
                        f"{type(recovery_exc).__name__}: {recovery_exc}"
                    ),
                }

            return {
                "success": False,
                "executed": True,
                "preview_only": False,
                "memory_id": memory_id,
                "operation": operation,
                "backup_dir": backup_result["backup_dir"],
                "post_check_passed": False,
                "remaining_related_issues": remaining,
                "recovery": recovery,
                "reason": (
                    "复检失败，已尝试恢复目标记录；"
                    "请检查 recovery 字段确认恢复结果。"
                ),
                "repairs_executed": True,
            }

        return {
            "success": True,
            "executed": True,
            "preview_only": False,
            "memory_id": memory_id,
            "operation": operation,
            "backup_dir": backup_result["backup_dir"],
            "post_check_passed": True,
            "remaining_related_issues": [],
            "recovery": {
                "attempted": False,
                "success": None,
                "reason": None,
            },
            "repairs_executed": True,
        }

    except Exception as exc:
        # A successful transaction result means the write was committed.
        # Never report "not executed" after that point.
        committed_result = locals().get("repair_result")

        if (
            isinstance(committed_result, dict)
            and committed_result.get("success") is True
        ):
            try:
                from memory.memory_repair_transaction import (
                    restore_embedding_repair,
                )

                restore_result = restore_embedding_repair(
                    embedding_db_path,
                    committed_result,
                )
                recovery = {
                    "attempted": True,
                    "success": bool(restore_result.get("success")),
                    "reason": None,
                }
            except Exception as recovery_exc:
                recovery = {
                    "attempted": True,
                    "success": False,
                    "reason": (
                        f"{type(recovery_exc).__name__}: {recovery_exc}"
                    ),
                }

            return {
                "success": False,
                "executed": True,
                "preview_only": False,
                "refused": False,
                "memory_id": memory_id,
                "operation": operation,
                "backup_dir": (
                    backup_result.get("backup_dir")
                    if isinstance(locals().get("backup_result"), dict)
                    else None
                ),
                "post_check_passed": False,
                "remaining_related_issues": [],
                "recovery": recovery,
                "reason": (
                    f"修复提交后发生意外异常："
                    f"{type(exc).__name__}: {exc}；"
                    "已尝试安全恢复，请检查 recovery 字段。"
                ),
                "repairs_executed": True,
            }

        return {
            "success": False,
            "executed": False,
            "refused": True,
            "reason": f"修复流程失败：{type(exc).__name__}: {exc}",
            "memory_id": memory_id,
        }
    finally:
        if temporary_snapshot is not None:
            try:
                temporary_snapshot.unlink(missing_ok=True)
            except OSError:
                pass


def math_is_finite(value):
    """Compatibility helper for finite numeric values."""
    import math
    return math.isfinite(value)
