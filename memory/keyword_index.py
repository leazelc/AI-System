from pathlib import Path
import re
import sqlite3

from memory.memory_core import INDEX_DB, MEMORY_ROOT


def get_db():
    conn = sqlite3.connect(INDEX_DB)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_keyword_fts
        USING fts5(
            id UNINDEXED,
            content,
            tokenize='unicode61'
        )
    """)
    return conn


def read_body(path):
    text = path.read_text(encoding="utf-8")

    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            text = parts[2]

    return text.strip()


def make_terms(text):
    terms = []
    current = []
    current_is_chinese = None

    def flush():
        if not current:
            return

        token = "".join(current)

        if current_is_chinese:
            terms.extend(
                token[index:index + 2]
                for index in range(len(token) - 1)
            )
            if len(token) == 1:
                terms.append(token)
        else:
            terms.append(token)

        current.clear()

    for char in text.lower():
        is_chinese = "\u3400" <= char <= "\u9fff"
        is_word = char.isalnum() or char == "_"

        if not is_word:
            flush()
            current_is_chinese = None
            continue

        if current and is_chinese != current_is_chinese:
            flush()

        current_is_chinese = is_chinese
        current.append(char)

    flush()
    return " ".join(terms)


def rebuild():
    conn = get_db()
    savepoint_created = False

    try:
        rows = conn.execute("""
            SELECT id, path
            FROM memories
            WHERE status = 'active'
        """).fetchall()

        documents = []
        failures = []

        for memory_id, raw_path in rows:
            path = Path(raw_path)

            if not path.is_file():
                failures.append(f"{memory_id}: 文件不存在 ({path})")
                continue

            try:
                body = read_body(path)
            except (OSError, UnicodeError) as exc:
                failures.append(f"{memory_id}: 文件读取失败 ({exc})")
                continue

            documents.append((memory_id, make_terms(body)))

        if failures:
            details = "; ".join(failures[:5])
            if len(failures) > 5:
                details += f"; 另有 {len(failures) - 5} 个错误"
            raise RuntimeError(
                f"关键词索引重建中止：{len(failures)} 个 active 记忆文件无法读取。"
                f"原索引未被清空。详情：{details}"
            )

        if len(documents) != len(rows):
            raise RuntimeError(
                f"关键词索引数量不一致：active {len(rows)} 条，"
                f"可索引 {len(documents)} 条。"
            )

        conn.execute("SAVEPOINT keyword_rebuild")
        savepoint_created = True

        conn.execute("DELETE FROM memory_keyword_fts")
        conn.executemany("""
            INSERT INTO memory_keyword_fts (id, content)
            VALUES (?, ?)
        """, documents)

        conn.execute("RELEASE keyword_rebuild")
        savepoint_created = False
        conn.commit()

        return len(documents)

    except Exception:
        if savepoint_created:
            conn.execute("ROLLBACK TO keyword_rebuild")
            conn.execute("RELEASE keyword_rebuild")
        raise

    finally:
        conn.close()

def search(query, limit=5):
    if not query or not query.strip() or limit < 1:
        return []

    terms = list(dict.fromkeys(make_terms(query).split()))

    if not terms:
        return []

    match_query = " OR ".join(f'"{term}"' for term in terms)

    conn = get_db()

    try:
        rows = conn.execute("""
            SELECT f.id, m.path, bm25(memory_keyword_fts) AS rank
            FROM memory_keyword_fts AS f
            JOIN memories AS m ON m.id = f.id
            WHERE memory_keyword_fts MATCH ?
              AND m.status = 'active'
            ORDER BY rank
            LIMIT ?
        """, (match_query, limit)).fetchall()

        return [
            {
                "id": memory_id,
                "path": Path(path),
                "score": -rank,
            }
            for memory_id, path, rank in rows
        ]

    finally:
        conn.close()
