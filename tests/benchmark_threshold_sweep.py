import json
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path.home() / "AI-System"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "memory"))

from embedding_index import search as vector_search
from memory import keyword_index

DATASET = ROOT / "tests/retrieval_benchmark_extended.json"
SOURCE_DB = ROOT / "memory/index/memory.sqlite"

cases = json.loads(DATASET.read_text(encoding="utf-8"))

if not cases:
    raise RuntimeError("测试数据为空")

with sqlite3.connect(
    f"file:{SOURCE_DB}?mode=ro", uri=True
) as conn:
    active_count = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE status='active'"
    ).fetchone()[0]

with tempfile.TemporaryDirectory(prefix="threshold_sweep_") as temp:
    temp_db = Path(temp) / "memory.sqlite"
    shutil.copy2(SOURCE_DB, temp_db)

    original_db = keyword_index.INDEX_DB

    try:
        keyword_index.INDEX_DB = temp_db
        indexed = keyword_index.rebuild()

        if indexed != active_count:
            raise RuntimeError(
                f"关键词索引数量异常：{indexed}，"
                f"正式数据库 active 数量：{active_count}"
            )

        print(f"测试题数量：{len(cases)}")
        print(f"有效记忆数量：{active_count}")
        print("正式数据库：只读；关键词索引：临时数据库")
        print("\n正在检索并缓存测试结果……")

        cached = []

        for i, case in enumerate(cases, 1):
            query = case["query"]
            relevant = set(case["relevant_files"])

            semantic = vector_search(query, limit=50)
            keyword = keyword_index.search(query, limit=50)

            cached.append({
                "name": case.get("name", str(i)),
                "query": query,
                "relevant": relevant,
                "semantic": semantic,
                "keyword_ids": {x["id"] for x in keyword},
            })

            print(f"[{i}/{len(cases)}] 完成：{query}")

        strategies = {}

        for threshold in (0.35, 0.50, 0.64, 0.70, 0.78):
            strategies[f"语义阈值 {threshold:.2f}"] = (
                lambda row, t=threshold: [
                    x for x in row["semantic"]
                    if x["score"] >= t
                ]
            )

        strategies["关键词"] = (
            lambda row: [
                x for x in keyword_index.search(
                    row["query"], limit=50
                )
            ]
        )

        strategies["混合规则 0.64/0.78"] = (
            lambda row: [
                x for x in row["semantic"]
                if x["score"] >= 0.78
                or (
                    x["score"] >= 0.64
                    and x["id"] in row["keyword_ids"]
                )
            ]
        )

        print("\n========== 策略对比 ==========")
        print(
            f"{'策略':<22}"
            f"{'可回答命中':<14}"
            f"{'无答案候选率':<16}"
            f"{'实际Precision':<16}"
            f"{'平均候选数':<12}"
        )

        for name, retrieve in strategies.items():
            answerable = 0
            answer_hits = 0
            no_answer = 0
            no_answer_with_candidates = 0
            total_relevant = 0
            total_returned = 0
            candidate_sum = 0

            for row in cached:
                results = retrieve(row)
                relevant = row["relevant"]

                names = {
                    Path(x["path"]).name for x in results
                }

                hits = len(names & relevant)
                candidate_sum += len(results)
                total_returned += len(results)
                total_relevant += hits

                if relevant:
                    answerable += 1
                    if hits:
                        answer_hits += 1
                else:
                    no_answer += 1
                    if results:
                        no_answer_with_candidates += 1

            hit_rate = (
                answer_hits / answerable if answerable else 0
            )
            false_candidate_rate = (
                no_answer_with_candidates / no_answer
                if no_answer else 0
            )
            precision = (
                total_relevant / total_returned
                if total_returned else 0
            )
            average = candidate_sum / len(cached)

            print(
                f"{name:<22}"
                f"{answer_hits}/{answerable} ({hit_rate:.0%})".ljust(14)
                + f"{no_answer_with_candidates}/{no_answer} "
                  f"({false_candidate_rate:.0%})".ljust(16)
                + f"{precision:.1%}".ljust(16)
                + f"{average:.2f}"
            )

        print("\n评测完成。")
        print("注意：无答案候选率不等于模型最终回答的幻觉率。")

    finally:
        keyword_index.INDEX_DB = original_db
