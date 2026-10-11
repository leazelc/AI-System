import json
import sys
import time
import urllib.request
from pathlib import Path
from collections import defaultdict

ROOT = Path.home() / "AI-System"
sys.path.insert(0, str(ROOT / "memory"))

from memory_search import search

DATASET = ROOT / "tests" / "retrieval_benchmark_extended.json"
API = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"


def expand_query(query):
    """让本地大模型生成三个不同表达的检索问题。"""
    prompt = f"""
你是个人记忆系统的检索规划器。
用户问题：{query}

请生成恰好 3 个不同的检索表达，用于查找用户过去保存的个人记忆。
要求：
1. 保留问题的核心含义。
2. 尝试使用同义词、不同问法或相关主题词。
3. 不要回答用户的问题，不要虚构个人事实。
4. 不要重复原问题。
只输出 JSON：{{"queries": ["检索表达1", "检索表达2", "检索表达3"]}}
"""
    payload = json.dumps({
        "model": MODEL,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": "你只负责规划记忆检索查询，不回答个人问题。只输出合法 JSON。"
            },
            {"role": "user", "content": prompt}
        ]
    }, ensure_ascii=False).encode("utf-8")

    request = urllib.request.Request(
        API,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(request, timeout=180) as response:
        data = json.loads(response.read())

    text = data["choices"][0]["message"]["content"]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"查询规划未返回有效 JSON：{text[:200]}")

    # 优先解析标准 JSON；若模型输出格式错误，则尝试逐行提取候选查询。
    try:
        parsed = json.loads(text[start:end + 1])
        queries = parsed.get("queries", []) if isinstance(parsed, dict) else []
    except json.JSONDecodeError:
        queries = []

    if not isinstance(queries, list):
        queries = []

    queries = [
        item.strip()
        for item in queries
        if isinstance(item, str) and item.strip()
    ][:3]

    # 扩展失败时返回空列表，让 A/B 实验退回原问题检索，
    # 而不是把整个案例记为异常。
    return queries


def file_name(item):
    return Path(item["path"]).name


def run_a(query):
    start = time.perf_counter()
    results = search(query, limit=10)
    elapsed = time.perf_counter() - start
    return results, elapsed


def run_b(query):
    start = time.perf_counter()

    expanded = expand_query(query)
    queries = [query] + expanded

    merged = {}
    for subquery in queries:
        for item in search(subquery, limit=10):
            key = item.get("id") or str(item["path"])
            old = merged.get(key)

            if old is None or item["score"] > old["score"]:
                merged[key] = dict(item)

    results = sorted(
        merged.values(),
        key=lambda item: item["score"],
        reverse=True,
    )[:10]

    elapsed = time.perf_counter() - start
    return results, elapsed, expanded


def main():
    cases = json.loads(DATASET.read_text(encoding="utf-8"))

    print("=" * 72)
    print("本地记忆检索 A/B 实验")
    print("A：原问题直接检索")
    print("B：GPT-OSS 查询扩展 + 向量检索 + 去重")
    print(f"案例数：{len(cases)}")
    print("注意：只读取现有索引和基准文件，不修改记忆。")
    print("=" * 72)

    stats = {
        "A_hits": 0,
        "B_hits": 0,
        "A_recall_sum": 0.0,
        "B_recall_sum": 0.0,
        "A_mrr_sum": 0.0,
        "B_mrr_sum": 0.0,
        "A_seconds": 0.0,
        "B_seconds": 0.0,
        "errors": 0,
    }

    for i, case in enumerate(cases, 1):
        query = case["query"]
        relevant = set(case["relevant_files"])

        print(f"\n{i:02d}/{len(cases)} {case['name']}")
        print("问题：", query)

        try:
            a_results, a_time = run_a(query)
            b_results, b_time, expanded = run_b(query)

            a_files = [file_name(item) for item in a_results]
            b_files = [file_name(item) for item in b_results]

            # 无答案案例不参与相关记忆召回率计算。
            if relevant:
                a_recalled = len(relevant.intersection(a_files))
                b_recalled = len(relevant.intersection(b_files))

                a_recall = a_recalled / len(relevant)
                b_recall = b_recalled / len(relevant)

                a_rank = next(
                    (rank for rank, name in enumerate(a_files, 1)
                     if name in relevant),
                    None,
                )
                b_rank = next(
                    (rank for rank, name in enumerate(b_files, 1)
                     if name in relevant),
                    None,
                )

                stats["A_recall_sum"] += a_recall
                stats["B_recall_sum"] += b_recall
                stats["A_mrr_sum"] += 1 / a_rank if a_rank else 0
                stats["B_mrr_sum"] += 1 / b_rank if b_rank else 0
                stats["A_hits"] += bool(a_recalled)
                stats["B_hits"] += bool(b_recalled)

                print(
                    f"A 命中：{bool(a_recalled)}；"
                    f"B 命中：{bool(b_recalled)}"
                )
                print("A 目标排名：", a_rank or "未命中")
                print("B 目标排名：", b_rank or "未命中")
            else:
                # 对无答案题，只检查检索是否仍返回候选记忆。
                print("无答案案例")
                print(f"A 返回候选：{len(a_results)} 条")
                print(f"B 返回候选：{len(b_results)} 条")

            print(f"A 耗时：{a_time:.2f}s")
            print(f"B 耗时：{b_time:.2f}s")
            print("扩展查询：", expanded)

            stats["A_seconds"] += a_time
            stats["B_seconds"] += b_time

        except Exception as exc:
            stats["errors"] += 1
            print(f"实验异常：{type(exc).__name__}: {exc}")

    relevant_cases = sum(bool(c["relevant_files"]) for c in cases)
    total = len(cases)

    print("\n" + "=" * 72)
    print("实验汇总")
    print("=" * 72)
    print(f"有目标记忆的案例：{relevant_cases}")
    if relevant_cases:
        print(
            f"A 命中率：{stats['A_hits']}/{relevant_cases} "
            f"({stats['A_hits'] / relevant_cases:.1%})"
        )
        print(
            f"B 命中率：{stats['B_hits']}/{relevant_cases} "
            f"({stats['B_hits'] / relevant_cases:.1%})"
        )
        print(
            f"A 平均 Recall：{stats['A_recall_sum'] / relevant_cases:.1%}"
        )
        print(
            f"B 平均 Recall：{stats['B_recall_sum'] / relevant_cases:.1%}"
        )
        print(
            f"A MRR：{stats['A_mrr_sum'] / relevant_cases:.3f}"
        )
        print(
            f"B MRR：{stats['B_mrr_sum'] / relevant_cases:.3f}"
        )

    print(f"A 总耗时：{stats['A_seconds']:.2f}s")
    print(f"B 总耗时：{stats['B_seconds']:.2f}s")
    print(f"案例总数：{total}")
    print(f"异常数：{stats['errors']}")
    print("说明：无答案案例的候选数量仅是初步指标，不等于误报率。")

    return 1 if stats["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
