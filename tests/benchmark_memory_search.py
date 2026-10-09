import json
import sys
from pathlib import Path

ROOT = Path.home() / "AI-System"
K = 5
sys.path.insert(0, str(ROOT / "memory"))

from memory_search import search


def main():
    cases = json.loads(
        (ROOT / "tests/retrieval_benchmark.json").read_text(
            encoding="utf-8"
        )
    )

    answerable = []
    unanswerable = []
    precision_hits = 0
    reciprocal_ranks = []
    retrieved_relevant = 0
    total_relevant = 0

    for index, case in enumerate(cases, 1):
        results = search(case["query"], limit=K)
        relevant = set(case["relevant_files"])
        ranked = [Path(item["path"]).name for item in results]
        hits = sum(name in relevant for name in ranked)
        rank = next(
            (i for i, name in enumerate(ranked, 1) if name in relevant),
            None,
        )

        print(f"[{index}/{len(cases)}] {case['name']}")
        print(f"查询：{case['query']}")
        print(f"结果数：{len(results)}")
        print(f"相关记忆命中数：{hits}")
        print(f"首个相关结果排名：{rank or '未找到'}")

        for position, item in enumerate(results, 1):
            filename = Path(item["path"]).name
            marker = " [相关]" if filename in relevant else ""
            print(
                f"  {position}. {item['score']:.4f} "
                f"{filename}{marker}"
            )
        print()

        if relevant:
            answerable.append(case)
            precision_hits += hits
            retrieved_relevant += hits
            total_relevant += len(relevant)
            reciprocal_ranks.append(1 / rank if rank else 0)
        else:
            unanswerable.append(bool(results))

    count = len(answerable)
    retrieved_slots = count * K

    precision = precision_hits / retrieved_slots if retrieved_slots else 0
    recall = retrieved_relevant / total_relevant if total_relevant else 0
    mrr = sum(reciprocal_ranks) / count if count else 0
    false_positive_rate = (
        sum(unanswerable) / len(unanswerable) if unanswerable else 0
    )

    print("=" * 60)
    print("检索质量基线")
    print("=" * 60)
    print(f"可回答问题：{count}")
    print(f"无答案问题：{len(unanswerable)}")
    print(f"Precision@{K}：{precision:.3f}")
    print(f"Recall@{K}：   {recall:.3f}")
    print(f"MRR：          {mrr:.3f}")
    print(f"无答案误报率： {false_positive_rate:.3f}")
    print()
    print("注意：无答案问题的误报，表示检索返回了至少一条记忆，")
    print("不等于最终回答一定错误；还需结合后续回答流程判断。")


if __name__ == "__main__":
    main()
