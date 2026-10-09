from pathlib import Path
import sqlite3
import json
import hashlib
import urllib.request
import math
import struct

MEMORY_ROOT = Path.home() / "AI-System" / "memory"
DB_PATH = MEMORY_ROOT / "index" / "embeddings.sqlite"

LM_STUDIO_URL = "http://127.0.0.1:1234/v1/embeddings"
EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            content_hash TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            vector BLOB NOT NULL,
            updated_at TEXT
        )
    """)
    conn.commit()
    return conn


def parse_memory(path):
    text = path.read_text(encoding="utf-8")

    metadata = {}
    content = text

    if text.startswith("---"):
        parts = text.split("---", 2)

        if len(parts) == 3:
            frontmatter = parts[1].strip()
            content = parts[2].strip()

            for line in frontmatter.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip()

    return metadata, content


def get_embedding(text):
    payload = json.dumps({
        "model": EMBEDDING_MODEL,
        "input": text
    }).encode("utf-8")

    request = urllib.request.Request(
        LM_STUDIO_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))

    return data["data"][0]["embedding"]


def vector_to_blob(vector):
    return struct.pack(f"{len(vector)}f", *vector)


def blob_to_vector(blob, dimension):
    return list(struct.unpack(f"{dimension}f", blob))


def content_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sync():
    conn = get_db()
    current_ids = set()

    files = list(MEMORY_ROOT.rglob("*.md"))

    print("Scanning memory files...")
    print(f"Found {len(files)} markdown files.\n")

    for path in files:
        # Archive is historical storage only.
        # Superseded/deleted memories must never participate
        # in the normal semantic index.
        if "archive" in path.parts:
            continue

        try:
            metadata, content = parse_memory(path)
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue

        memory_id = metadata.get("id")

        if not memory_id:
            continue

        if metadata.get("status") != "active":
            continue

        current_ids.add(memory_id)

        digest = content_hash(content)

        existing = conn.execute(
            "SELECT content_hash FROM embeddings WHERE id = ?",
            (memory_id,)
        ).fetchone()

        if existing and existing[0] == digest:
            print(f"[UNCHANGED] {memory_id}")
            continue

        print(f"[EMBED] {memory_id}")
        print(f"         {path}")

        try:
            vector = get_embedding(content)
        except Exception as e:
            print(f"         ERROR: {e}")
            continue

        blob = vector_to_blob(vector)

        conn.execute("""
            INSERT INTO embeddings
            (id, path, content_hash, dimension, vector, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                path = excluded.path,
                content_hash = excluded.content_hash,
                dimension = excluded.dimension,
                vector = excluded.vector,
                updated_at = excluded.updated_at
        """, (
            memory_id,
            str(path),
            digest,
            len(vector),
            blob,
            metadata.get("updated_at", "")
        ))

        conn.commit()

        print(f"         dimension: {len(vector)}")

    rows = conn.execute("SELECT id, path FROM embeddings").fetchall()

    for memory_id, path in rows:
        if memory_id not in current_ids or not Path(path).exists():
            print(f"[DELETE] {memory_id}")
            conn.execute(
                "DELETE FROM embeddings WHERE id = ?",
                (memory_id,)
            )

    conn.commit()

    total = conn.execute(
        "SELECT COUNT(*) FROM embeddings"
    ).fetchone()[0]

    conn.close()

    print(f"\nEmbedding index ready.")
    print(f"Vectors stored on SSD: {total}")


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def search(query, limit=5):
    conn = get_db()

    query_vector = get_embedding(query)

    rows = conn.execute("""
        SELECT id, path, dimension, vector
        FROM embeddings
    """).fetchall()

    results = []

    for memory_id, path, dimension, blob in rows:
        vector = blob_to_vector(blob, dimension)

        score = cosine_similarity(
            query_vector,
            vector
        )

        results.append({
            "id": memory_id,
            "path": path,
            "score": score
        })

    conn.close()

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:limit]


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        sync()
        sys.exit(0)

    query = " ".join(sys.argv[1:])

    print(f"\nSemantic search: {query}\n")

    results = search(query)

    if not results:
        print("No semantic memories found.")
        sys.exit(0)

    for i, result in enumerate(results, 1):
        print("=" * 60)
        print(f"[{i}] Similarity: {result['score']:.4f}")
        print(f"ID: {result['id']}")
        print(f"File: {result['path']}")
