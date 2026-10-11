import json
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path.home() / "AI-System"
K = 5
RRF_K = 60

sys.path.insert(0, str(ROOT / "memory"))
sys.path.insert(0, str(ROOT))

from memory_search import search as semantic_search
from memory import keyword_index


def filename(item):
    return Path(item["path"]).name


def reciprocal_rank(results, relevant):
    for rank, item in enumerate(results, 1):
        if filename(item) in relevant:
            return 1 / rank
    return 0.0


def evaluate(cases, methods):
    summaries = {}

    for method_name, retrieve in methods.items():
        precision_hits = 0
        relevant_hits = 0
        relevant_total = 0
        reciprocal_ranks = []
        answerable_count = 0
        no_answer_count = 0
        false_positives = 0
        details = []

        for case in cases:
            relevant = set(case["relevant_files"])
            results = retrieve(case["query"])[:K]
            names = [filename(item) for item in results]
            hits = sum(name in relevant for name in names)
            rr = reciprocal_rank(results, relevant)

            if relevant:
                answerable_count += 1
                precision_hits += hits
                relevant_hits += hits
                relevant_total += len(relevant)
                reciprocal_ranks.append(rr)
            else:
                no_answer_count += 1
                false_positives += bool(results)

            details.append({
                "name": case["name"],
                "query": case["query"],
                "relevant": sorted(relevant),
                "results": names,
                "hits": hits,
                "rr": rr,
                "false_positive": not relevant and bool(results),
            })

        summaries[method_name] = {
            "precision": precision_hits / (answerable_count * K),
            "recall": relevant_hits / relevant_total if relevant_total else 0,
            "mrr": sum(reciprocal_ranks) / answerable_count
            if answerable_count else 0,
            "false_positive_rate": false_positives / no_answer_count
            if no_answer_count else 0,
            "details": details,
        }

    return summaries


def main():
    cases_path = ROOT / "tests/retrieval_benchmark_extended.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))


    source_db = ROOT / "memory/index/memory.sqlite"

    with tempfile.TemporaryDirectory(prefix="ai_memory_benchmark_") as temp:
        temp_db = Path(temp) / "memory.sqlite"
        shutil.copy2(source_db, temp_db)

        keyword_index.INDEX_DB = temp_db
        indexed = keyword_index.rebuild()

        with sqlite3.connect(temp_db) as conn:
            active_count = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE status = 'active'"
            ).fetchone()[0]

        if indexed != active_count:
            raise RuntimeError(
                f"关键词索引数量异常：索引 {indexed} 条，"
                f"active 记忆 {active_count} 条"
            )

        def semantic(query):
            return semantic_search(query, limit=K)

        def keyword(query):
            return keyword_index.search(query, limit=K)

        def rrf(query):
            semantic_results = semantic(query)
            keyword_results = keyword(query)
            scores = {}
            items = {}

            for ranked in (semantic_results, keyword_results):
                for rank, item in enumerate(ranked, 1):
                    key = item["id"]
                    scores[key] = scores.get(key, 0.0) + 1 / (
                        RRF_K + rank
                    )
                    items[key] = item

            fused = sorted(
                items.values(),
                key=lambda item: scores[item["id"]],
                reverse=True,
            )

            return [
                {**item, "score": scores[item["id"]]}
                for item in fused[:K]
            ]

        methods = {
            "语义检索": semantic,
            "关键词检索": keyword,
            "RRF 融合": rrf,
        }

        summaries = evaluate(cases, methods)

    print("\n" + "=" * 72)
    print("扩展检索基准测试")
    print("=" * 72)
    print(f"测试题总数：{len(cases)}")
    print(f"可回答问题：{sum(bool(c['relevant_files']) for c in cases)}")
    print(f"无答案问题：{sum(not c['relevant_files'] for c in cases)}")
    print(f"Top-K：{K} | RRF 常数：{RRF_K}")
    print("\n汇总指标")
    print("-" * 72)
    print(f"{'方法':<12}{'Precision@5':>14}{'Recall@5':>12}"
          f"{'MRR':>10}{'无答案误报率':>16}")

    for name, result in summaries.items():
        print(
            f"{name:<12}"
            f"{result['precision']:>14.3f}"
            f"{result['recall']:>12.3f}"
            f"{result['mrr']:>10.3f}"
            f"{result['false_positive_rate']:>16.3f}"
        )

    print("\n逐题结果")
    print("-" * 72)

    for index, case in enumerate(cases, 1):
        print(f"\n{index:02d}. {case['name']}")
        print(f"查询：{case['query']}")

        for method_name, result in summaries.items():
            detail = result["details"][index - 1]
            ranked = detail["results"]
            relevant = set(detail["relevant"])

            first_hit = next(
                (
                    rank
                    for rank, name in enumerate(ranked, 1)
                    if name in relevant
                ),
                None,
            )

            print(
                f"  {method_name}: "
                f"首个命中={first_hit or '无'}，"
                f"误检={'是' if detail['false_positive'] else '否'}，"
                f"结果={ranked}"
            )

    print("\n测试完成。临时关键词数据库已自动清理。")
    print("正式记忆、正式关键词索引和向量索引未被主动修改。")


if __name__ == "__main__":
    main()
