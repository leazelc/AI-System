from pathlib import Path
import re

MEMORY_ROOT = Path.home() / "AI-System" / "memory"

def parse_memory(path):
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


def search(query, limit=10):
    query_terms = re.findall(r"[\w\u4e00-\u9fff]+", query.lower())

    results = []

    for path in MEMORY_ROOT.rglob("*.md"):
        try:
            metadata, content = parse_memory(path)
        except Exception:
            continue

        if metadata.get("status") == "deleted":
            continue

        searchable = (
            path.name.lower()
            + " "
            + content.lower()
            + " "
            + " ".join(metadata.values()).lower()
        )

        score = 0

        for term in query_terms:
            if term in searchable:
                score += 1

        if score > 0:
            results.append({
                "score": score,
                "path": path,
                "metadata": metadata,
                "content": content,
            })

    results.sort(
        key=lambda x: (
            -x["score"],
            -float(x["metadata"].get("importance", 0))
        )
    )

    return results[:limit]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法：")
        print("python3 memory_search.py \"搜索内容\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    print(f"\n搜索：{query}\n")

    results = search(query)

    if not results:
        print("没有找到相关记忆。")
        sys.exit(0)

    print(f"找到 {len(results)} 条相关记忆：\n")

    for i, result in enumerate(results, 1):
        metadata = result["metadata"]

        print("=" * 60)
        print(f"[{i}] Score: {result['score']}")
        print(f"ID: {metadata.get('id', 'unknown')}")
        print(f"Type: {metadata.get('type', 'unknown')}")
        print(f"Status: {metadata.get('status', 'unknown')}")
        print(f"Importance: {metadata.get('importance', 'unknown')}")
        print(f"File: {result['path']}")
        print()
        print(result["content"])
        print()
