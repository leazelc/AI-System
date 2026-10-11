import json
import shutil
import sys
import sqlite3
import tempfile
from pathlib import Path

ROOT = Path.home() / "AI-System"
sys.path.insert(0, str(ROOT / "memory"))
sys.path.insert(0, str(ROOT))
K = 5
CANDIDATE_LIMIT = 10
RRF_K = 60

from memory_search import search as semantic_search
from memory import keyword_index


def filename(item):
    return Path(item["path"]).name


def evaluate(cases, retrieve):
    answerable = [c for c in cases if c["relevant_files"]]
    unanswered = [c for c in cases if not c["relevant_files"]]

    hits = 0
    total_relevant = sum(len(c["relevant_files"]) for c in answerable)
    reciprocal_ranks = []

    for case in answerable:
        results = retrieve(case["query"])[:K]
        relevant = set(case["relevant_files"])
        names = [filename(x) for x in results]
        hits += sum(name in relevant for name in names)

        rr = next(
            (1 / rank for rank, name in enumerate(names, 1)
             if name in relevant),
            0.0,
        )
        reciprocal_ranks.append(rr)

    false_positives = sum(bool(retrieve(c["query"])[:K]) for c in unanswered)

    return {
        "Recall@5": hits / total_relevant if total_relevant else 0,
        "MRR": sum(reciprocal_ranks) / len(answerable) if answerable else 0,
        "无答案误报率": (
            false_positives / len(unanswered) if unanswered else 0
        ),
    }


def main():
    cases = json.loads(
        (ROOT / "tests/retrieval_benchmark_extended.json").read_text(
            encoding="utf-8"
        )
    )

    source_db = ROOT / "memory/index/memory.sqlite"

    with tempfile.TemporaryDirectory(prefix="ai_memory_gate_") as temp:
        temp_db = Path(temp) / "memory.sqlite"
        shutil.copy2(source_db, temp_db)

        keyword_index.INDEX_DB = temp_db
        indexed = keyword_index.rebuild()

        with sqlite3.connect(temp_db) as conn:
            active_count = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE status='active'"
            ).fetchone()[0]

        if indexed != active_count:
            raise RuntimeError(
                f"关键词索引数量异常：{indexed}/{active_count}"
            )

        cache = {}

        def get_candidates(query):
            if query in cache:
                return cache[query]

            semantic = semantic_search(query, limit=CANDIDATE_LIMIT)
            keyword = keyword_index.search(query, limit=CANDIDATE_LIMIT)

            sem_rank = {x["id"]: rank for rank, x in enumerate(semantic, 1)}
            sem_items = {x["id"]: x for x in semantic}
            key_rank = {x["id"]: rank for rank, x in enumerate(keyword, 1)}
            items = {x["id"]: x for x in semantic + keyword}

            rrf_scores = {}
            for results in (semantic, keyword):
                for rank, item in enumerate(results, 1):
                    rrf_scores[item["id"]] = (
                        rrf_scores.get(item["id"], 0)
                        + 1 / (RRF_K + rank)
                    )

            fused = sorted(
                items.values(),
                key=lambda x: rrf_scores[x["id"]],
                reverse=True,
            )

            sem_gap = (
                semantic[0]["score"] - semantic[1]["score"]
                if len(semantic) >= 2 else 0
            )

            cache[query] = {
                "semantic": semantic,
                "keyword": keyword,
                "sem_rank": sem_rank,
                "key_rank": key_rank,
                "sem_items": sem_items,
                "fused": fused,
                "gap": sem_gap,
                "top_sem": semantic[0] if semantic else None,
            }
            return cache[query]

        def rrf(query):
            return get_candidates(query)["fused"]

        def intersection(query):
            c = get_candidates(query)
            common = set(c["sem_rank"]) & set(c["key_rank"])
            return [
                item for item in c["fused"]
                if item["id"] in common
            ]

        def gated(threshold, margin):
            def retrieve(query):
                c = get_candidates(query)
                common = set(c["sem_rank"]) & set(c["key_rank"])

                eligible = set(common)
                top = c["top_sem"]

                # 仅作实验：关键词未命中时，测试是否可用
                # 语义第一名的分数和领先幅度作为补充条件。
                if (
                    top is not None
                    and top["score"] >= threshold
                    and c["gap"] >= margin
                ):
                    eligible.add(top["id"])

                return [
                    item for item in c["fused"]
                    if item["id"] in eligible
                ]
            return retrieve

        methods = [
            ("RRF 原始", rrf),
            ("双路交集", intersection),
        ]

        for threshold in (0.60, 0.65, 0.70, 0.75):
            for margin in (0.02, 0.03, 0.04):
                methods.append((
                    f"门槛{threshold:.2f}+差值{margin:.2f}",
                    gated(threshold, margin),
                ))

        print("=" * 78)
        print("混合检索过滤实验（只读正式记忆；关键词索引使用临时数据库）")
        print(f"测试题：{len(cases)}；有答案："
              f"{sum(bool(c['relevant_files']) for c in cases)}；无答案："
              f"{sum(not c['relevant_files'] for c in cases)}")
        print("-" * 78)
        print(f"{'方案':<24}{'Recall@5':>12}{'MRR':>10}{'无答案误报率':>18}")

        results = {}
        for name, retrieve in methods:
            metrics = evaluate(cases, retrieve)
            results[name] = metrics
            print(
                f"{name:<24}{metrics['Recall@5']:>12.3f}"
                f"{metrics['MRR']:>10.3f}"
                f"{metrics['无答案误报率']:>18.3f}"
            )

        print("\n========== 无答案题逐题检查 ==========")
        for case in cases:
            if case["relevant_files"]:
                continue

            print(f"\n{case['name']}：{case['query']}")
            for name, retrieve in methods:
                result = retrieve(case["query"])[:K]
                names = [filename(x) for x in result]
                print(f"  {name}: {'有结果' if names else '拒绝返回'}"
                      + (f"；首条={names[0]}" if names else ""))

        print("\n实验结束：没有修改正式记忆、正式关键词索引或向量索引。")
        print("注意：语义分数差值只是待验证的启发式规则，不是可靠性证明。")


if __name__ == "__main__":
    main()
