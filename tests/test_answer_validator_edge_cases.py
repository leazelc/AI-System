import sys
from pathlib import Path
from collections import Counter

ROOT = Path.home() / "AI-System"
sys.path.insert(0, str(ROOT / "memory"))

from answer_validator import validate_answer, AnswerValidationError

CASES = [
    {
        "name": "历史计划：只陈述计划",
        "evidence": "我计划去莫斯科国立师范大学读本科。",
        "answer": "我计划去莫斯科国立师范大学读本科。",
        "expected": {"PASS"},
    },
    {
        "name": "历史计划：错误断言已经入学",
        "evidence": "我计划去莫斯科国立师范大学读本科。",
        "answer": "我已经正式入学莫斯科国立师范大学。",
        "expected": {"FLAG", "UNCERTAIN"},
    },
    {
        "name": "计划日期已过：错误断言计划失败",
        "evidence": (
            "记录中写着：我计划于2026年10月1日抵达莫斯科。"
            "没有任何后续记录说明实际是否抵达。"
        ),
        "answer": "我没有抵达莫斯科，计划已经失败。",
        "expected": {"FLAG", "UNCERTAIN"},
    },
    {
        "name": "记忆冲突：把其中一条直接当作当前事实",
        "evidence": (
            "记忆A：电脑预算上限为20000元。"
            "记忆B：电脑预算后来调整为15000元。"
            "没有更多信息能够确认哪条预算仍然有效。"
        ),
        "answer": "我目前的电脑预算上限确定是20000元。",
        "expected": {"FLAG", "UNCERTAIN"},
    },
    {
        "name": "证据只支持部分内容：答案过度推断",
        "evidence": "我组装电脑时优先关注SSD和显存。",
        "answer": "我只在乎SSD和显存，其他所有硬件都不重要。",
        "expected": {"FLAG", "UNCERTAIN"},
    },
]


def main():
    stats = Counter()

    print("=" * 64)
    print("Stage 8：历史计划与证据边界专项测试")
    print(f"测试数量：{len(CASES)}")
    print("=" * 64)

    for i, case in enumerate(CASES, 1):
        print(f"\n{i}/{len(CASES)}：{case['name']}")

        try:
            result = validate_answer(
                evidence=case["evidence"],
                answer=case["answer"],
            )

            verdict = result["verdict"]
            reason = result["reason"]
            passed = verdict in case["expected"]

            stats["total"] += 1
            stats["passed"] += passed
            if not passed:
                stats["failed"] += 1

            print("预期结果：", " / ".join(sorted(case["expected"])))
            print("实际结果：", verdict)
            print("校验理由：", reason)
            print("测试结果：", "通过" if passed else "未通过")

        except (AnswerValidationError, Exception) as exc:
            stats["total"] += 1
            stats["errors"] += 1
            print("测试结果：执行异常")
            print(f"异常：{type(exc).__name__}: {exc}")

    print("\n" + "=" * 64)
    print("专项测试汇总")
    print("=" * 64)
    print(f"符合预期：{stats['passed']}/{stats['total']}")
    print(f"结果不符：{stats['failed']}")
    print(f"执行异常：{stats['errors']}")
    print("本测试只调用校验器，不修改正式记忆、数据库或索引。")
    print("测试通过不代表校验器在所有场景下都可靠。")

    return 1 if stats["failed"] or stats["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
