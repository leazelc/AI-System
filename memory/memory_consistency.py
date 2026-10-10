"""Read-only consistency checks for the local AI memory system."""

from pathlib import Path
import math
import sqlite3
import struct


MEMORY_ROOT = Path.home() / "AI-System" / "memory"
CORE_DB_PATH = MEMORY_ROOT / "index" / "memory.sqlite"
EMBEDDING_DB_PATH = MEMORY_ROOT / "index" / "embeddings.sqlite"


def open_readonly_db(path):
    """Open an existing SQLite database without allowing writes."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Database not found: {path}")

    uri = path.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


def read_markdown_metadata(path):
    """Read simple key-value metadata from a Markdown frontmatter block."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    if not text.startswith("---"):
        return {}

    parts = text.split("---", 2)

    if len(parts) != 3:
        return {}

    metadata = {}

    for line in parts[1].strip().splitlines():
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()

    return metadata


def decode_vector(blob, dimension):
    """Decode a float32 vector, rejecting malformed or non-finite values."""
    if not isinstance(dimension, int) or dimension <= 0:
        raise ValueError(f"Invalid vector dimension: {dimension}")

    if blob is None or len(blob) != dimension * 4:
        raise ValueError(
            f"Vector byte length mismatch: dimension={dimension}, "
            f"bytes={0 if blob is None else len(blob)}, "
            f"expected={dimension * 4}"
        )

    vector = struct.unpack(f"{dimension}f", blob)

    if not all(math.isfinite(value) for value in vector):
        raise ValueError("Vector contains NaN or infinity")

    return vector


def scan_markdown_files(memory_root=MEMORY_ROOT):
    """Find Markdown files without modifying the filesystem."""
    memory_root = Path(memory_root).expanduser().resolve()

    if not memory_root.is_dir():
        raise FileNotFoundError(f"Memory directory not found: {memory_root}")

    files = []

    for path in memory_root.rglob("*.md"):
        if not path.is_file():
            continue

        relative_path = path.relative_to(memory_root)

        # SQLite indexes and other index-related files are not memory documents.
        if relative_path.parts[0] == "index":
            continue

        files.append(path.resolve())

    return sorted(files)


def find_duplicate_markdown_ids(files):
    """Return Markdown IDs that occur in more than one file."""
    from collections import defaultdict

    id_to_paths = defaultdict(list)

    for path in files:
        metadata = read_markdown_metadata(path)
        memory_id = metadata.get("id")

        if memory_id:
            id_to_paths[memory_id].append(str(Path(path).resolve()))

    return {
        memory_id: paths
        for memory_id, paths in id_to_paths.items()
        if len(paths) > 1
    }


def read_core_index(db_path=CORE_DB_PATH):
    """Read all metadata-index records without modifying the database."""
    conn = open_readonly_db(db_path)

    try:
        rows = conn.execute(
            """
            SELECT id, path, type, status, importance, confidence,
                   created_at, updated_at
            FROM memories
            ORDER BY id
            """
        ).fetchall()

        columns = (
            "id",
            "path",
            "type",
            "status",
            "importance",
            "confidence",
            "created_at",
            "updated_at",
        )

        return [dict(zip(columns, row)) for row in rows]
    finally:
        conn.close()


def check_core_index_consistency(
    memory_root=MEMORY_ROOT,
    db_path=CORE_DB_PATH,
):
    """Check core-index records against their Markdown files."""
    memory_root = Path(memory_root).expanduser().resolve()
    rows = read_core_index(db_path)
    issues = []

    for row in rows:
        path = Path(row["path"]).expanduser().resolve()

        try:
            path.relative_to(memory_root)
        except ValueError:
            issues.append({
                "level": "error",
                "id": row["id"],
                "problem": "数据库路径位于记忆目录之外",
                "path": str(path),
            })
            continue

        if not path.is_file():
            issues.append({
                "level": "error",
                "id": row["id"],
                "problem": "数据库记录指向的 Markdown 文件不存在",
                "path": str(path),
            })
            continue

        try:
            metadata = read_markdown_metadata(path)
        except (OSError, UnicodeError) as exc:
            issues.append({
                "level": "error",
                "id": row["id"],
                "problem": f"无法读取 Markdown 文件：{exc}",
                "path": str(path),
            })
            continue

        if not metadata:
            issues.append({
                "level": "error",
                "id": row["id"],
                "problem": "Markdown 文件缺少有效元数据",
                "path": str(path),
            })
            continue

        for field in ("id", "status", "type"):
            if metadata.get(field) != row[field]:
                issues.append({
                    "level": "error",
                    "id": row["id"],
                    "problem": (
                        f"{field} 不一致：数据库={row[field]!r}，"
                        f"文件={metadata.get(field)!r}"
                    ),
                    "path": str(path),
                })

    return issues


def check_markdown_index_coverage(
    memory_root=MEMORY_ROOT,
    db_path=CORE_DB_PATH,
):
    """Check that Markdown files are represented correctly in the core index."""
    memory_root = Path(memory_root).expanduser().resolve()
    files = scan_markdown_files(memory_root)
    rows = read_core_index(db_path)

    rows_by_id = {}
    for row in rows:
        rows_by_id.setdefault(row["id"], []).append(row)

    issues = []

    for path in files:
        try:
            metadata = read_markdown_metadata(path)
        except (OSError, UnicodeError) as exc:
            issues.append({
                "level": "error",
                "problem": f"无法读取 Markdown 文件：{exc}",
                "path": str(path),
            })
            continue

        memory_id = metadata.get("id")
        relative_path = path.relative_to(memory_root)
        parts = relative_path.parts

        if not memory_id:
            level = "error" if parts[0] not in ("archive", "candidates") else "warning"
            issues.append({
                "level": level,
                "problem": "Markdown 文件缺少 ID",
                "path": str(path),
            })
            continue

        matching_rows = rows_by_id.get(memory_id, [])

        if not matching_rows:
            issues.append({
                "level": "error",
                "id": memory_id,
                "problem": "Markdown 文件未登记在结构化索引中",
                "path": str(path),
            })
            continue

        matching_paths = {
            Path(row["path"]).expanduser().resolve()
            for row in matching_rows
        }

        if path not in matching_paths:
            issues.append({
                "level": "error",
                "id": memory_id,
                "problem": "数据库中存在该 ID，但没有记录此文件的实际路径",
                "path": str(path),
                "indexed_paths": sorted(str(p) for p in matching_paths),
            })

        if len(matching_rows) > 1:
            issues.append({
                "level": "error",
                "id": memory_id,
                "problem": "结构化索引中存在多个相同 ID 的记录",
                "path": str(path),
            })

    return issues


def read_embedding_index(db_path=EMBEDDING_DB_PATH):
    """Read embedding-index metadata without modifying the database."""
    conn = open_readonly_db(db_path)

    try:
        rows = conn.execute(
            """
            SELECT id, path, content_hash, dimension, vector, updated_at
            FROM embeddings
            ORDER BY id
            """
        ).fetchall()

        columns = (
            "id",
            "path",
            "content_hash",
            "dimension",
            "vector",
            "updated_at",
        )

        return [dict(zip(columns, row)) for row in rows]
    finally:
        conn.close()


def markdown_content_hash(path):
    """Calculate the SHA-256 hash using the embedding index's parsing rules."""
    import hashlib

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    content = text

    if text.startswith("---"):
        parts = text.split("---", 2)

        if len(parts) == 3:
            content = parts[2].strip()

    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def check_embedding_consistency(
    memory_root=MEMORY_ROOT,
    db_path=EMBEDDING_DB_PATH,
):
    """Check embedding records against active Markdown memories."""
    memory_root = Path(memory_root).expanduser().resolve()
    rows = read_embedding_index(db_path)
    issues = []

    for row in rows:
        memory_id = row["id"]
        path = Path(row["path"]).expanduser().resolve()

        try:
            path.relative_to(memory_root)
        except ValueError:
            issues.append({
                "level": "error",
                "id": memory_id,
                "problem": "向量记录的路径位于记忆目录之外",
                "path": str(path),
            })
            continue

        if not path.is_file():
            issues.append({
                "level": "error",
                "id": memory_id,
                "problem": "向量记录对应的 Markdown 文件不存在",
                "path": str(path),
            })
            continue

        try:
            metadata = read_markdown_metadata(path)
            current_hash = markdown_content_hash(path)
        except (OSError, UnicodeError) as exc:
            issues.append({
                "level": "error",
                "id": memory_id,
                "problem": f"读取 Markdown 失败：{exc}",
                "path": str(path),
            })
            continue

        if metadata.get("id") != memory_id:
            issues.append({
                "level": "error",
                "id": memory_id,
                "problem": "向量记录 ID 与 Markdown 内部 ID 不一致",
                "path": str(path),
            })

        if metadata.get("status") != "active":
            issues.append({
                "level": "error",
                "id": memory_id,
                "problem": "向量对应的 Markdown 不是 active 状态",
                "path": str(path),
            })

        if current_hash != row["content_hash"]:
            issues.append({
                "code": "EMBEDDING_HASH_MISMATCH",
                "level": "error",
                "id": memory_id,
                "problem": "正文哈希与向量索引不一致，向量可能已过期",
                "path": str(path),
            })

        try:
            decode_vector(row["vector"], row["dimension"])
        except (ValueError, TypeError, struct.error) as exc:
            issues.append({
                "level": "error",
                "id": memory_id,
                "problem": f"向量数据无效：{exc}",
                "path": str(path),
            })

    return issues


def check_embedding_coverage(
    memory_root=MEMORY_ROOT,
    db_path=EMBEDDING_DB_PATH,
):
    """Check for missing, stale, duplicate, or unexpected embedding records."""
    from collections import defaultdict

    memory_root = Path(memory_root).expanduser().resolve()
    files = scan_markdown_files(memory_root)
    rows = read_embedding_index(db_path)
    issues = []

    rows_by_id = defaultdict(list)
    rows_by_path = defaultdict(list)

    for row in rows:
        rows_by_id[row["id"]].append(row)
        resolved_path = str(Path(row["path"]).expanduser().resolve())
        rows_by_path[resolved_path].append(row)

    # Detect duplicate IDs and paths in the embedding database.
    for memory_id, matches in rows_by_id.items():
        if len(matches) > 1:
            issues.append({
                "level": "error",
                "id": memory_id,
                "problem": f"向量索引中存在 {len(matches)} 条相同 ID 的记录",
            })

    for indexed_path, matches in rows_by_path.items():
        if len(matches) > 1:
            issues.append({
                "level": "error",
                "problem": f"向量索引中存在 {len(matches)} 条指向同一路径的记录",
                "path": indexed_path,
            })

    expected_by_id = {}

    # Build the expected set using the embedding sync rules:
    # active Markdown files outside any archive directory.
    for path in files:
        relative = path.relative_to(memory_root)

        if "archive" in relative.parts:
            continue

        try:
            metadata = read_markdown_metadata(path)
        except (OSError, UnicodeError) as exc:
            issues.append({
                "level": "error",
                "problem": f"读取 Markdown 失败：{exc}",
                "path": str(path),
            })
            continue

        if metadata.get("status") != "active":
            continue

        memory_id = metadata.get("id")

        if not memory_id:
            issues.append({
                "level": "error",
                "problem": "应建立向量索引的 active 记忆缺少 ID",
                "path": str(path),
            })
            continue

        if memory_id in expected_by_id:
            issues.append({
                "level": "error",
                "id": memory_id,
                "problem": "多个 active Markdown 文件使用相同 ID",
                "path": str(path),
            })
            continue

        expected_by_id[memory_id] = path.resolve()

    # Every expected active memory must have a matching embedding.
    for memory_id, expected_path in expected_by_id.items():
        matches = rows_by_id.get(memory_id, [])

        if not matches:
            issues.append({
                "level": "error",
                "id": memory_id,
                "problem": "active Markdown 记忆缺少向量记录",
                "path": str(expected_path),
            })
            continue

        if not any(
            Path(row["path"]).expanduser().resolve() == expected_path
            for row in matches
        ):
            issues.append({
                "level": "error",
                "id": memory_id,
                "problem": "向量记录路径与 active Markdown 文件不一致",
                "path": str(expected_path),
            })

    # Every embedding must correspond to an expected active memory.
    for row in rows:
        memory_id = row["id"]
        indexed_path = Path(row["path"]).expanduser().resolve()
        expected_path = expected_by_id.get(memory_id)

        if expected_path is None:
            issues.append({
                "level": "error",
                "id": memory_id,
                "problem": "存在不应保留的向量记录：找不到对应的 active 非归档记忆",
                "path": str(indexed_path),
            })
        elif indexed_path != expected_path:
            issues.append({
                "level": "error",
                "id": memory_id,
                "problem": "向量记录指向的文件不是该 ID 对应的 active 记忆",
                "path": str(indexed_path),
            })

    return issues


def check_database_duplicates(db_path, table, columns):
    """Check duplicate values using read-only SQL."""
    if table not in ("memories", "embeddings"):
        raise ValueError("Unsupported table")

    allowed_columns = {
        "memories": {"id", "path"},
        "embeddings": {"id", "path"},
    }

    if any(column not in allowed_columns[table] for column in columns):
        raise ValueError("Unsupported column")

    conn = open_readonly_db(db_path)
    issues = []

    try:
        for column in columns:
            query = (
                f"SELECT {column}, COUNT(*) FROM {table} "
                f"GROUP BY {column} HAVING COUNT(*) > 1"
            )

            for value, count in conn.execute(query):
                issues.append({
                    "level": "error",
                    "problem": (
                        f"{table} 表的 {column} 存在重复值："
                        f"{value!r}，共 {count} 条"
                    ),
                })
    finally:
        conn.close()

    return issues


def run_consistency_check(
    memory_root=MEMORY_ROOT,
    core_db_path=CORE_DB_PATH,
    embedding_db_path=EMBEDDING_DB_PATH,
):
    """Run all consistency checks and return a unified read-only report."""
    memory_root = Path(memory_root).expanduser().resolve()

    checks = (
        (
            "Markdown ID 重复",
            lambda: [
                {
                    "level": "error",
                    "id": memory_id,
                    "problem": "多个 Markdown 文件使用相同 ID",
                    "paths": paths,
                }
                for memory_id, paths in find_duplicate_markdown_ids(
                    scan_markdown_files(memory_root)
                ).items()
            ],
        ),
        (
            "结构化索引一致性",
            lambda: check_core_index_consistency(
                memory_root, core_db_path
            ),
        ),
        (
            "结构化索引覆盖率",
            lambda: check_markdown_index_coverage(
                memory_root, core_db_path
            ),
        ),
        (
            "结构化索引重复记录",
            lambda: check_database_duplicates(
                core_db_path, "memories", ("id", "path")
            ),
        ),
        (
            "向量记录一致性",
            lambda: check_embedding_consistency(
                memory_root, embedding_db_path
            ),
        ),
        (
            "向量索引覆盖率及残留记录",
            lambda: check_embedding_coverage(
                memory_root, embedding_db_path
            ),
        ),
        (
            "向量索引重复记录",
            lambda: check_database_duplicates(
                embedding_db_path, "embeddings", ("id", "path")
            ),
        ),
    )

    report = {
        "checks": [],
        "issues": [],
        "error_count": 0,
        "warning_count": 0,
    }

    for name, check in checks:
        try:
            issues = check()
            report["checks"].append({
                "name": name,
                "status": "passed" if not issues else "issues_found",
                "issue_count": len(issues),
            })

            for issue in issues:
                issue = dict(issue)
                issue["check"] = name
                report["issues"].append(issue)

        except Exception as exc:
            issue = {
                "level": "error",
                "check": name,
                "problem": f"检查执行失败：{type(exc).__name__}: {exc}",
            }
            report["issues"].append(issue)
            report["checks"].append({
                "name": name,
                "status": "failed",
                "issue_count": 1,
            })

    report["error_count"] = sum(
        issue.get("level") == "error"
        for issue in report["issues"]
    )
    report["warning_count"] = sum(
        issue.get("level") == "warning"
        for issue in report["issues"]
    )

    return report
