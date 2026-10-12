import json
import sys
from pathlib import Path
from collections import Counter
import importlib.util

ROOT = Path.home() / "AI-System"
sys.path.insert(0, str(ROOT / "memory"))

from memory_search import search
from answer_validator import validate_answer

from answer_validator import AnswerValidationError

PIPELINE_PATH = ROOT / "tests" / "test_memory_answer_pipeline.py"
DATASET_PATH = ROOT / "tests" / "answer_benchmark_pilot.json"

ABSTENTION = "根据本次检索到的记忆，无法确认这个问题的答案。"


def load_pipeline():
    spec = importlib.util.spec_from_file_location(
        "answer_pipeline_validation_test",
        PIPELINE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_evidence(memories):
    """
    仅使用本次实际检索到的记忆正文。
    如果没有命中，则明确记录本次检索没有返回可用记忆。
    """
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
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    cases = dataset["cases"]

    stats = Counter()

    print("=" * 68)
    print("记忆问答 + 独立答案校验试验")
    print(f"模型：{pipeline.MODEL}")
    print(f"试验题数：{len(cases)}")
    print("流程：语义检索 Top 10 → 生成答案 → 独立证据校验")
    print("=" * 68)

    for index, case in enumerate(cases, 1):
        question = case["query"]
        expected_answerable = case["expected_answerable"]

        print(f"\n{'-' * 68}")
        print(f"{index:02d}/{len(cases)}：{case['name']}")
        print("问题：", question)

        try:
            # 只读取检索结果，不修改记忆或索引。
            results = search(question, limit=10)

            memories = [
                {
                    "id": item["id"],
                    "content": pipeline.read_memory(item["path"]),
                }
                for item in results
            ]

            evidence = build_evidence(memories)

            # 复用现有问答流程。
            generated = pipeline.ask_model(question, memories)
            predicted_answerable = generated["answerable"]

            answerability_ok = (
                predicted_answerable == expected_answerable
            )

            # 模型拒答时，也校验拒答是否合理。
            answer_to_validate = (
                generated["answer"].strip()
                if predicted_answerable
                else ABSTENTION
            )

            # 独立校验器只读取 evidence 和 answer，
            # 不访问记忆库，也不会写入记忆。
            validation = validate_answer(
                evidence=evidence,
                answer=answer_to_validate,
            )

            validator_verdict = validation["verdict"]
            validator_reason = validation["reason"]

            validator_pass = validator_verdict == "PASS"
            combined_pass = answerability_ok and validator_pass

            stats["total"] += 1
            stats["answerability_correct"] += answerability_ok
            stats["validator_pass"] += validator_pass
            stats["combined_pass"] += combined_pass

            if predicted_answerable:
                print("模型回答：", generated["answer"])
                print("模型引用的证据：", generated["evidence"])
            else:
                print("模型回答：拒答")

            print("本次检索到的记忆数：", len(memories))
            print(
                "可回答性判断：",
                "正确" if answerability_ok else "错误",
            )
            print("独立校验结果：", validator_verdict)
            print("校验理由：", validator_reason)
            print(
                "综合结果：",
                "通过" if combined_pass else "未通过",
            )

        except (AnswerValidationError, Exception) as exc:
            stats["total"] += 1
            stats["execution_errors"] += 1

            print("执行异常：", type(exc).__name__, str(exc))
            print("本题不予放行。")

    total = stats["total"]

    print("\n" + "=" * 68)
    print("试验汇总")
    print("=" * 68)
    failed = total - stats["combined_pass"]

    print(f"可回答性判断正确：{stats['answerability_correct']}/{total}")
    print(f"独立校验 PASS：{stats['validator_pass']}/{total}")
    print(f"综合通过：{stats['combined_pass']}/{total}")
    print(f"综合未通过：{failed}")
    print(f"执行异常：{stats['execution_errors']}")
    print()
    print("注意：PASS 只代表校验器认为答案与所提供证据一致，")
    print("不代表检索覆盖率、答案完整性或整个系统的准确率已得到证明。")
    print("本试验不会调用记忆写入器，不修改正式记忆、数据库或索引。")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
