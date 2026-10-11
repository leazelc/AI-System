import json
import sys
import importlib.util
from collections import Counter
from pathlib import Path

ROOT = Path.home() / "AI-System"
sys.path.insert(0, str(ROOT / "memory"))

from memory_search import search
from answer_validator import validate_answer

PIPELINE_PATH = ROOT / "tests" / "test_memory_answer_pipeline.py"
DATASET_PATH = ROOT / "tests" / "retrieval_benchmark_extended.json"

ABSTENTION = "根据本次检索到的记忆，无法确认这个问题的答案。"


def load_pipeline():
    spec = importlib.util.spec_from_file_location(
        "experiment_answer_pipeline_20",
        PIPELINE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_evidence(memories):
    parts = []

    for index, memory in enumerate(memories, 1):
        content = memory["content"].strip()
        if content:
            parts.append(f"检索记忆 {index}：\n{content}")

    if not parts:
        return "检索状态：本次语义检索未返回可用的 active 记忆。"

    return "\n\n---\n\n".join(parts)


def main():
    pipeline = load_pipeline()
    cases = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    stats = Counter()

    print("=" * 72)
    print("20 题本地记忆问答与独立证据校验实验")
    print(f"模型：{pipeline.MODEL}")
    print(f"题数：{len(cases)}")
    print("流程：语义检索 Top 10 → 生成答案 → 独立证据校验")
    print("模式：只读实验，不修改记忆、数据库或索引")
    print("=" * 72)

    for index, case in enumerate(cases, 1):
        question = case["query"]
        expected_files = set(case.get("relevant_files", []))
        expected_answerable = bool(expected_files)

        stats["total"] += 1
        stats["answerable_total" if expected_answerable else "unanswerable_total"] += 1

        print(f"\n{'-' * 72}")
        print(f"[{index:02d}/{len(cases)}] {case['name']}")
        print(f"问题：{question}")
        print(f"预期：{'有答案' if expected_answerable else '无答案'}")

        try:
            results = search(question, limit=10)

            memories = [
                {
                    "id": item["id"],
                    "content": pipeline.read_memory(item["path"]),
                }
                for item in results
            ]

            retrieved_files = {
                Path(item["path"]).name for item in results
            }
            target_hit = bool(expected_files & retrieved_files)

            if expected_answerable:
                stats["retrieval_target_total"] += 1
                stats["retrieval_target_hit"] += target_hit

            print(f"检索记忆数：{len(memories)}")
            print(f"目标记忆是否进入 Top 10：{'是' if target_hit else '否'}")

            generated = pipeline.ask_model(question, memories)
            predicted = generated["answerable"]
            answerability_ok = predicted == expected_answerable

            if answerability_ok:
                stats["answerability_correct"] += 1

            if predicted and not expected_answerable:
                stats["false_positive"] += 1
            elif not predicted and expected_answerable:
                stats["false_negative"] += 1

            answer = (
                generated["answer"].strip()
                if predicted
                else ABSTENTION
            )

            evidence = build_evidence(memories)
            validation = validate_answer(
                evidence=evidence,
                answer=answer,
            )

            verdict = validation["verdict"]
            reason = validation["reason"]

            if verdict == "PASS":
                stats["validator_pass"] += 1
            if answerability_ok and verdict == "PASS":
                stats["combined_pass"] += 1

            print(f"模型判断有答案：{'是' if predicted else '否'}")
            print(f"可回答性判断：{'正确' if answerability_ok else '错误'}")
            print(f"回答：{generated['answer'] if predicted else '拒答'}")

            if predicted:
                print(f"模型引用证据：{generated['evidence']}")

            print(f"独立校验：{verdict}")
            print(f"校验理由：{reason}")
            print(
                "本题综合结果：",
                "通过" if answerability_ok and verdict == "PASS" else "未通过",
            )

        except Exception as exc:
            stats["execution_errors"] += 1
            print(f"执行异常：{type(exc).__name__}: {exc}")
            print("本题不予放行。")

    total = stats["total"]
    answerable_total = stats["answerable_total"]
    unanswerable_total = stats["unanswerable_total"]
    target_total = stats["retrieval_target_total"]

    def rate(numerator, denominator):
        return f"{numerator / denominator:.1%}" if denominator else "N/A"

    print("\n" + "=" * 72)
    print("20 题实验汇总")
    print("=" * 72)
    print(f"总题数：{total}")
    print(f"预期有答案：{answerable_total}")
    print(f"预期无答案：{unanswerable_total}")
    print(
        f"目标记忆 Top 10 命中率："
        f"{stats['retrieval_target_hit']}/{target_total} "
        f"({rate(stats['retrieval_target_hit'], target_total)})"
    )
    print(
        f"可回答性判断准确率："
        f"{stats['answerability_correct']}/{total} "
        f"({rate(stats['answerability_correct'], total)})"
    )
    print(
        f"独立校验 PASS：{stats['validator_pass']}/{total} "
        f"({rate(stats['validator_pass'], total)})"
    )
    print(
        f"综合通过：{stats['combined_pass']}/{total} "
        f"({rate(stats['combined_pass'], total)})"
    )
    print(
        f"无答案误答：{stats['false_positive']}/{unanswerable_total} "
        f"({rate(stats['false_positive'], unanswerable_total)})"
    )
    print(
        f"有答案误拒：{stats['false_negative']}/{answerable_total} "
        f"({rate(stats['false_negative'], answerable_total)})"
    )
    print(f"执行异常：{stats['execution_errors']}")
    print()
    print("注意：目标记忆命中不等于答案一定正确；校验 PASS 也不等于答案完整。")
    print("本实验不修改正式记忆、数据库、索引或检索逻辑。")

    return 1 if stats["execution_errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
