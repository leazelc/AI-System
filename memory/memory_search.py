from pathlib import Path
import sys

MEMORY_ROOT = Path.home() / "AI-System" / "memory"

# 允许直接从 memory 目录导入 embedding_index
sys.path.insert(0, str(MEMORY_ROOT))

from embedding_index import search as semantic_search


def parse_memory(path):
    """
    读取一个 Markdown 长期记忆文件。

    返回：
        metadata: frontmatter 元数据
        content: 记忆正文
    """
    text = path.read_text(encoding="utf-8")

    if text.startswith("---"):
        parts = text.split("---", 2)

        if len(parts) == 3:
            frontmatter = parts[1].strip()
            content = parts[2].strip()
        else:
            frontmatter = ""
            content = text
    else:
        frontmatter = ""
        content = text

    metadata = {}

    for line in frontmatter.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()

    return metadata, content


def search(query, limit=10, min_score=None):
    """
    使用 Nomic 语义检索长期记忆。

    只读取 Top-K 命中的 Markdown 文件，
    不扫描并读取整个 memory 目录。

    返回：
        [
            {
                "score": float,
                "path": Path,
                "metadata": dict,
                "content": str
            }
        ]
    """

    if min_score is None:
        import os
        try:
            min_score = float(os.environ.get("AI_MEMORY_MIN_SCORE", "0.45"))
        except ValueError:
            min_score = 0.45

        min_score = max(0.0, min(1.0, min_score))

    candidates = [
        item
        for item in semantic_search(query, limit=limit)
        if item["score"] >= min_score
    ]

    results = []

    for item in candidates:
        path = Path(item["path"])

        # 文件不存在则跳过
        if not path.exists():
            continue

        try:
            metadata, content = parse_memory(path)
        except Exception:
            continue

        # 安全检查：只允许 active 记忆进入正常检索结果
        if metadata.get("status") != "active":
            continue

        results.append({
            "id": item["id"],
            "score": item["score"],
            "path": path,
            "metadata": metadata,
            "content": content,
        })

    return results


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]).strip()

    if not query:
        print('用法：python3 ~/AI-System/memory/memory_search.py "你的问题"')
        raise SystemExit

    results = search(query, limit=5)

    print(f"找到 {len(results)} 条候选记忆\n")

    for i, item in enumerate(results, 1):
        print("=" * 70)
        print(f"[{i}] similarity: {item['score']:.4f}")
        print(f"ID: {item['metadata'].get('id', 'unknown')}")
        print(f"PATH: {item['path']}")

        print("\n--- memory content ---")
        print(item["content"])
