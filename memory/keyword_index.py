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

    try:
        rows = conn.execute("""
            SELECT id, path
            FROM memories
            WHERE status = 'active'
        """).fetchall()

        documents = []

        for memory_id, raw_path in rows:
            path = Path(raw_path)

            if not path.is_file():
                continue

            try:
                body = read_body(path)
            except (OSError, UnicodeError):
                continue

            documents.append((memory_id, make_terms(body)))

        conn.execute("SAVEPOINT keyword_rebuild")
        conn.execute("DELETE FROM memory_keyword_fts")
        conn.executemany("""
            INSERT INTO memory_keyword_fts (id, content)
            VALUES (?, ?)
        """, documents)
        conn.execute("RELEASE keyword_rebuild")
        conn.commit()

        return len(documents)

    except Exception:
        if conn.in_transaction:
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
