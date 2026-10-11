import sys
from pathlib import Path
from collections import Counter

ROOT = Path.home() / "AI-System"
sys.path.insert(0, str(ROOT / "memory"))

from answer_validator import validate_answer, AnswerValidationError

CASES = [
    {
        "name": "错误数字：把两万元预算说成三万元",
        "evidence": "我的组装电脑预算上限是20000元人民币。",
        "answer": "我的组装电脑预算上限是30000元人民币。",
        "expected": {"FLAG", "UNCERTAIN"},
    },
    {
        "name": "错误偏好：把硬件配置偏好说成外观优先",
        "evidence": "我买电脑时更重视SSD容量和显存，不喜欢RGB灯效。",
        "answer": "我买电脑时最重视RGB灯效和外观，SSD容量与显存并不重要。",
        "expected": {"FLAG", "UNCERTAIN"},
    },
    {
        "name": "虚构个人信息：编造护照号码",
        "evidence": "现有记忆没有记录我的护照号码。",
        "answer": "我的护照号码是E12345678。",
        "expected": {"FLAG", "UNCERTAIN"},
    },
    {
        "name": "虚构决定：声称已经选定显卡",
        "evidence": "我考虑过购买RTX 5070 Ti，但现有记录没有确认最终购买型号。",
        "answer": "我最终已经决定购买RTX 5090。",
        "expected": {"FLAG", "UNCERTAIN"},
    },
    {
        "name": "混合答案：正确预算加上错误显卡",
        "evidence": "我的电脑预算上限是20000元人民币。现有记录没有确认最终显卡型号。",
        "answer": "我的电脑预算上限是20000元，而且我最终确定购买RTX 5090。",
        "expected": {"FLAG", "UNCERTAIN"},
    },
    {
        "name": "正向对照：证据充分的预算答案",
        "evidence": "我的电脑采购预算硬上限是20000元人民币。",
        "answer": "我的电脑采购预算上限是20000元人民币。",
        "expected": {"PASS"},
    },
]

def main():
    stats = Counter()

    print("=" * 66)
    print("答案校验器对抗实验")
    print(f"测试数量：{len(CASES)}")
    print("只调用校验器，不修改正式记忆、数据库或索引。")
    print("=" * 66)

    for i, case in enumerate(CASES, 1):
        print(f"\n[{i}/{len(CASES)}] {case['name']}")

        try:
            result = validate_answer(
                evidence=case["evidence"],
                answer=case["answer"],
            )
            verdict = result["verdict"]
            passed = verdict in case["expected"]

            stats["total"] += 1
            stats["passed" if passed else "failed"] += 1

            if case["name"].startswith("正向对照"):
                stats["positive_total"] += 1
                stats["positive_passed" if passed else "positive_failed"] += 1
            else:
                stats["adversarial_total"] += 1
                stats["adversarial_passed" if passed else "adversarial_failed"] += 1
                if verdict == "PASS":
                    stats["unsafe_pass"] += 1

            print("预期：", " / ".join(sorted(case["expected"])))
            print("实际：", verdict)
            print("理由：", result.get("reason", ""))
            print("结果：", "通过" if passed else "失败")

        except Exception as exc:
            stats["total"] += 1
            stats["errors"] += 1
            print(f"执行异常：{type(exc).__name__}: {exc}")

    print("\n" + "=" * 66)
    print("实验汇总")
    print("=" * 66)
    print(f"总体符合预期：{stats['passed']}/{stats['total']}")
    print(
        f"对抗案例符合预期："
        f"{stats['adversarial_passed']}/{stats['adversarial_total']}"
    )
    print(f"错误答案被判为 PASS：{stats['unsafe_pass']}")
    print(
        f"正向对照符合预期："
        f"{stats['positive_passed']}/{stats['positive_total']}"
    )
    print(f"执行异常：{stats['errors']}")

    return 1 if stats["errors"] or stats["failed"] else 0

if __name__ == "__main__":
    raise SystemExit(main())
