import json
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path.home() / "AI-System"
sys.path.insert(0, str(ROOT / "memory"))
sys.path.insert(0, str(ROOT))

from memory_search import search as semantic_search
from memory import keyword_index


def main():
    cases = json.loads(
        (ROOT / "tests/retrieval_benchmark_extended.json").read_text(
            encoding="utf-8"
        )
    )

    source_db = ROOT / "memory/index/memory.sqlite"

    with tempfile.TemporaryDirectory(prefix="hybrid_filter_") as temp:
        temp_db = Path(temp) / "memory.sqlite"
        shutil.copy2(source_db, temp_db)
        keyword_index.INDEX_DB = temp_db

        indexed_count = keyword_index.rebuild()
        with sqlite3.connect(temp_db) as conn:
            active_count = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE status = 'active'"
            ).fetchone()[0]

        if indexed_count != active_count:
            raise RuntimeError(
                f"关键词索引数量异常：索引 {indexed_count} 条，"
                f"active 记忆 {active_count} 条"
            )

        results_by_case = []

        for case in cases:
            semantic = semantic_search(case["query"], limit=5)
            keyword = keyword_index.search(case["query"], limit=5)
            keyword_ids = {item["id"] for item in keyword}

            hybrid = []
            for item in semantic:
                score = item["score"]

                # 高置信度保留；灰区要求关键词提供辅助证据
                if score >= 0.78 or (
                    score >= 0.64 and item["id"] in keyword_ids
                ):
                    hybrid.append(item)

            results_by_case.append((case, semantic, keyword, hybrid))

        print("规则：分数 >= 0.78 保留；0.64–0.78 要求关键词命中")
        print("\n逐题结果")

        answer_hit = 0
        answer_total = 0
        no_answer_false = 0
        no_answer_total = 0

        for i, (case, semantic, keyword, hybrid) in enumerate(
            results_by_case, 1
        ):
            relevant = set(case["relevant_files"])
            names = [Path(item["path"]).name for item in hybrid]
            hit = any(name in relevant for name in names)

            if relevant:
                answer_total += 1
                answer_hit += hit
            else:
                no_answer_total += 1
                no_answer_false += bool(hybrid)

            print(
                f"{i:02d}. {'可回答' if relevant else '无答案'} | "
                f"语义={len(semantic)} | 关键词={len(keyword)} | "
                f"过滤后={len(hybrid)} | "
                f"相关命中={'是' if hit else '否'} | "
                f"{case['query']}"
            )

        print("\n========== 汇总 ==========")
        print(f"可回答问题命中：{answer_hit}/{answer_total}")
        print(f"无答案误报：{no_answer_false}/{no_answer_total}")
        print("正式数据库未修改；关键词索引使用临时数据库。")


if __name__ == "__main__":
    main()
