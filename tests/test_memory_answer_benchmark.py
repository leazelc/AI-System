import json
import re
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "memory"))

from memory_search import search

import importlib.util

spec = importlib.util.spec_from_file_location(
    "answer_pipeline",
    ROOT / "tests" / "test_memory_answer_pipeline.py",
)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)

DATASET = ROOT / "tests" / "answer_benchmark_pilot.json"


def normalize(text):
    return re.sub(r"\s+", "", str(text)).lower()


def contains_required_term(answer, term):
    text = normalize(answer)
    target = normalize(term)

    if target in text:
        return True

    # 兼容预算金额的常见书写方式。
    if term == "2万元":
        compact = re.sub(r"[,，]", "", text)
        return any(
            variant in compact
            for variant in (
                "20000元",
                "20000人民币",
                "人民币20000元",
                "人民币20000",
                "人民币两万元",
                "两万元人民币",
                "2万人民币",
            )
        )

    return False


def main():
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    cases = dataset["cases"]

    stats = Counter()
    details = []

    print("=" * 60)
    print("个人记忆问答准确率试验")
    print(f"模型：{pipeline.MODEL}")
    print(f"案例数：{len(cases)}")
    print("=" * 60)

    for index, case in enumerate(cases, 1):
        question = case["query"]
        expected_answerable = case["expected_answerable"]

        try:
            results = search(question, limit=10)

            memories = [
                {
                    "id": item["id"],
                    "content": pipeline.read_memory(item["path"]),
                }
                for item in results
            ]

            verdict = pipeline.ask_model(question, memories)
            predicted = verdict["answerable"]

            answerability_ok = predicted == expected_answerable

            if expected_answerable and predicted:
                answer = verdict["answer"]
                content_ok = all(
                    contains_required_term(answer, term)
                    for term in case["required_answer_terms"]
                )

                evidence = normalize(verdict["evidence"])
                evidence_ok = bool(evidence) and any(
                    evidence in normalize(memory["content"])
                    for memory in memories
                )
            elif not expected_answerable and not predicted:
                answer = ""
                content_ok = True
                evidence_ok = True
            else:
                answer = verdict.get("answer", "")
                content_ok = False
                evidence_ok = False

            target_file = case.get("evidence_file")
            retrieved_files = {
                Path(item["path"]).name for item in results
            }
            retrieval_applicable = target_file is not None
            retrieval_ok = (
                target_file in retrieved_files
                if retrieval_applicable
                else True
            )

            stats["total"] += 1
            stats["answerability_correct"] += answerability_ok
            stats["content_correct"] += content_ok
            stats["evidence_valid"] += evidence_ok
            if retrieval_applicable:
                stats["retrieval_applicable"] += 1
                stats["retrieval_covered"] += retrieval_ok

            passed = (
                answerability_ok
                and content_ok
                and evidence_ok
                and retrieval_ok
            )
            stats["full_pass"] += passed

            details.append({
                "name": case["name"],
                "query": question,
                "answerable": predicted,
                "answerability_correct": answerability_ok,
                "content_correct": content_ok,
                "evidence_valid": evidence_ok,
                "retrieval_covered": retrieval_ok,
                "full_pass": passed,
                "answer": answer,
                "evidence": verdict.get("evidence", ""),
            })

            print(f"\n{index:02d}/{len(cases)} [{'通过' if passed else '未通过'}] {case['name']}")
            print("问题：", question)
            print("可回答性：", "正确" if answerability_ok else "错误")
            print("内容核验：", "通过" if content_ok else "未通过")
            print("证据核验：", "通过" if evidence_ok else "未通过")
            print("目标记忆：", "已检索" if retrieval_ok else "未检索")

            if predicted:
                print("回答：", verdict["answer"])
                print("证据：", verdict["evidence"])

        except Exception as exc:
            stats["total"] += 1
            stats["execution_errors"] += 1
            print(f"\n{index:02d}/{len(cases)} [执行异常] {question}")
            print(f"原因：{type(exc).__name__}: {exc}")

    total = stats["total"]

    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    print(f"可回答性正确：{stats['answerability_correct']}/{total}")
    print(f"答案内容正确：{stats['content_correct']}/{total}")
    print(f"证据有效：{stats['evidence_valid']}/{total}")
    print(
        f"目标记忆检索覆盖：{stats['retrieval_covered']}/"
        f"{stats['retrieval_applicable']}（仅统计有目标记忆的题目）"
    )
    failed = total - stats["full_pass"]

    print(f"完整通过：{stats['full_pass']}/{total}")
    print(f"完整未通过：{failed}")
    print(f"执行异常：{stats['execution_errors']}")

    print("\n说明：这是 5 道题的初步试验，不能据此推断整个系统的总体准确率。")
    print("测试没有修改正式记忆、数据库或索引。")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
