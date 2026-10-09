"""Atomic, single-record SQLite operations for memory repair.

This module does not generate embeddings or decide whether a repair is allowed.
Callers must independently validate policy, source identity, and backups.
"""

import sqlite3
from pathlib import Path


COLUMNS = (
    "id", "path", "content_hash", "dimension", "vector", "updated_at"
)


def _row_dict(row):
    return dict(zip(COLUMNS, row)) if row is not None else None


def _fetch_target_rows(conn, memory_id, source_path):
    rows = conn.execute(
        """
        SELECT id, path, content_hash, dimension, vector, updated_at
        FROM embeddings
        WHERE id = ? OR path = ?
        ORDER BY id
        """,
        (memory_id, source_path),
    ).fetchall()
    return [_row_dict(row) for row in rows]


def _same_row(left, right):
    if left is None or right is None:
        return left is right
    return all(left.get(key) == right.get(key) for key in COLUMNS)


def apply_embedding_repair(
    db_path,
    *,
    operation,
    memory_id,
    source_path,
    content_hash,
    dimension,
    vector_blob,
    updated_at,
):
    """Insert a missing vector or rebuild one existing vector atomically.

    The operation is deliberately narrow: only one verified ID/path pair may
    be inserted or updated. Returns before/after snapshots for guarded restore.
    """
    db_path = Path(db_path).expanduser().resolve(strict=True)
    source_path = str(Path(source_path).expanduser().resolve())

    if operation not in {"create_embedding", "rebuild_embedding"}:
        raise ValueError("不支持的向量修复操作")
    if not memory_id or not source_path:
        raise ValueError("记忆 ID 和源路径不能为空")
    if not isinstance(content_hash, str) or not content_hash:
        raise ValueError("正文哈希无效")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("向量维度无效")
    if not isinstance(vector_blob, (bytes, bytearray)) or not vector_blob:
        raise ValueError("向量字节数据无效")

    conn = sqlite3.connect(db_path, timeout=10)
    try:
        conn.execute("BEGIN IMMEDIATE")
        rows = _fetch_target_rows(conn, memory_id, source_path)

        if operation == "create_embedding":
            if rows:
                raise RuntimeError("目标 ID 或路径已存在，拒绝重复插入")
            before = None
            conn.execute(
                """
                INSERT INTO embeddings
                (id, path, content_hash, dimension, vector, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id, source_path, content_hash, dimension,
                    bytes(vector_blob), updated_at,
                ),
            )
        else:
            if len(rows) != 1:
                raise RuntimeError("目标向量记录不存在或不唯一")
            before = rows[0]
            if before["id"] != memory_id or before["path"] != source_path:
                raise RuntimeError("目标 ID 与路径不完全匹配，拒绝更新")

            conn.execute(
                """
                UPDATE embeddings
                SET content_hash = ?, dimension = ?, vector = ?, updated_at = ?
                WHERE id = ? AND path = ?
                """,
                (
                    content_hash, dimension, bytes(vector_blob), updated_at,
                    memory_id, source_path,
                ),
            )
            if conn.execute("SELECT changes()").fetchone()[0] != 1:
                raise RuntimeError("目标更新数量不是 1")

        after_rows = _fetch_target_rows(conn, memory_id, source_path)
        if len(after_rows) != 1:
            raise RuntimeError("写入后目标记录数量异常")
        after = after_rows[0]
        conn.commit()
        return {
            "success": True,
            "operation": operation,
            "memory_id": memory_id,
            "source_path": source_path,
            "before": before,
            "after": after,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def restore_embedding_repair(db_path, repair_result):
    """Restore only the target row if it still exactly matches our write.

    For an insertion, remove that row. For an update, restore its prior values.
    Any concurrent change or identity mismatch causes a safe refusal.
    """
    if not isinstance(repair_result, dict) or not repair_result.get("success"):
        raise ValueError("缺少有效的修复结果快照")

    db_path = Path(db_path).expanduser().resolve(strict=True)
    memory_id = repair_result["memory_id"]
    source_path = repair_result["source_path"]
    before = repair_result["before"]
    after = repair_result["after"]

    conn = sqlite3.connect(db_path, timeout=10)
    try:
        conn.execute("BEGIN IMMEDIATE")
        rows = _fetch_target_rows(conn, memory_id, source_path)

        if len(rows) != 1 or not _same_row(rows[0], after):
            raise RuntimeError("目标记录在修复后已发生变化，拒绝覆盖")

        if before is None:
            conn.execute(
                "DELETE FROM embeddings WHERE id = ? AND path = ?",
                (memory_id, source_path),
            )
            if conn.execute("SELECT changes()").fetchone()[0] != 1:
                raise RuntimeError("回滚删除数量不是 1")
            restored = None
        else:
            conn.execute(
                """
                UPDATE embeddings
                SET content_hash = ?, dimension = ?, vector = ?, updated_at = ?
                WHERE id = ? AND path = ?
                """,
                (
                    before["content_hash"], before["dimension"],
                    before["vector"], before["updated_at"],
                    memory_id, source_path,
                ),
            )
            if conn.execute("SELECT changes()").fetchone()[0] != 1:
                raise RuntimeError("回滚更新数量不是 1")
            restored_rows = _fetch_target_rows(conn, memory_id, source_path)
            if len(restored_rows) != 1 or not _same_row(restored_rows[0], before):
                raise RuntimeError("恢复后的目标记录校验失败")
            restored = restored_rows[0]

        conn.commit()
        return {
            "success": True,
            "memory_id": memory_id,
            "restored": restored,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
