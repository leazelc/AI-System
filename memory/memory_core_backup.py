from pathlib import Path
import sqlite3

MEMORY_ROOT = Path.home() / "AI-System" / "memory"
INDEX_DB = MEMORY_ROOT / "index" / "memory.sqlite"


def get_db():
    conn = sqlite3.connect(INDEX_DB)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            type TEXT,
            status TEXT,
            importance REAL,
            confidence REAL,
            created_at TEXT,
            updated_at TEXT
        )
    """)

    conn.commit()
    return conn


def read_metadata(path):
    text = path.read_text(encoding="utf-8")

    if not text.startswith("---"):
        return {}

    parts = text.split("---", 2)

    if len(parts) < 3:
        return {}

    metadata = {}

    for line in parts[1].strip().splitlines():
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()

    return metadata


def scan():
    conn = get_db()

    current_ids = set()

    # 扫描 SSD 上真实存在的记忆文件
    for path in MEMORY_ROOT.rglob("*.md"):
        metadata = read_metadata(path)

        memory_id = metadata.get("id")

        if not memory_id:
            continue

        current_ids.add(memory_id)

        conn.execute("""
            INSERT INTO memories
            (id, path, type, status, importance, confidence, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)

            ON CONFLICT(id) DO UPDATE SET
                path=excluded.path,
                type=excluded.type,
                status=excluded.status,
                importance=excluded.importance,
                confidence=excluded.confidence,
                created_at=excluded.created_at,
                updated_at=excluded.updated_at
        """, (
            memory_id,
            str(path),
            metadata.get("type"),
            metadata.get("status"),
            float(metadata.get("importance", 0)),
            float(metadata.get("confidence", 0)),
            metadata.get("created_at"),
            metadata.get("updated_at"),
        ))

    # 删除 SQLite 中已经不存在于 SSD 的记忆
    rows = conn.execute("SELECT id, path FROM memories").fetchall()

    for memory_id, path in rows:
        if memory_id not in current_ids or not Path(path).exists():
            conn.execute(
                "DELETE FROM memories WHERE id = ?",
                (memory_id,)
            )

    conn.commit()
    conn.close()


def list_memories():
    conn = get_db()

    rows = conn.execute("""
        SELECT id, path, type, status, importance, confidence
        FROM memories
        ORDER BY updated_at DESC
    """).fetchall()

    conn.close()

    return rows


if __name__ == "__main__":
    scan()

    print("Memory Core initialized.")
    print()

    memories = list_memories()

    if not memories:
        print("No memories found.")
    else:
        print(f"Memories indexed: {len(memories)}")
        print()

        for row in memories:
            print(f"ID: {row[0]}")
            print(f"Path: {row[1]}")
            print(f"Type: {row[2]}")
            print(f"Status: {row[3]}")
            print(f"Importance: {row[4]}")
            print(f"Confidence: {row[5]}")
            print()
