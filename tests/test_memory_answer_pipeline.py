import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path.home() / "AI-System"
sys.path.insert(0, str(ROOT / "memory"))

from memory_search import search

DATASET = (
    Path(sys.argv[1]).expanduser()
    if __name__ == "__main__" and len(sys.argv) > 1
    else ROOT / "tests/retrieval_benchmark_extended.json"
)
CASES = json.loads(DATASET.read_text(encoding="utf-8"))

API = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"


def read_memory(path):
    text = Path(path).read_text(encoding="utf-8")
    parts = text.split("---", 2)
    if text.startswith("---") and len(parts) == 3:
        text = parts[2]
    return text.strip()[:2000]


def ask_model(question, memories):
    prompt = {
        "question": question,
        "candidate_memories": memories,
        "instructions": [
            "只根据候选记忆回答，不得使用外部知识或猜测。",
            "如果候选记忆能够回答问题，就必须根据记忆回答。",
            "允许语义相近、措辞不同的记忆作为答案依据。",
            "有证据时用中文简洁回答，并引用对应记忆原文。",
            "没有充分证据时，answerable 必须为 false，answer 和 evidence 置空。",
            "只输出 JSON，不要输出 Markdown。"
        ],
        "required_format": {
            "answerable": True,
            "answer": "简洁答案；无答案时为空字符串",
            "evidence": "支持答案的记忆原文；无答案时为空字符串"
        }
    }

    payload = json.dumps({
        "model": MODEL,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": "你是个人记忆问答器。只能根据提供的记忆回答。如果记忆包含答案，就必须回答；只有确实没有证据时才拒答。只输出 JSON。"
            },
            {
                "role": "user",
                "content": json.dumps(prompt, ensure_ascii=False)
            }
        ]
    }, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(
        API,
        data=payload,
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req, timeout=180) as response:
        data = json.loads(response.read())

    text = data["choices"][0]["message"]["content"]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"模型未返回有效 JSON：{text[:200]}")

    result = json.loads(text[start:end + 1])

    if not isinstance(result.get("answerable"), bool):
        raise ValueError("answerable 必须是布尔值")

    if result["answerable"] and not (
        result.get("answer", "").strip()
        and result.get("evidence", "").strip()
    ):
        raise ValueError("模型声称有答案，但没有提供答案或证据")

    return result



def main():
    correct = 0
    answerable_total = 0
    unanswerable_total = 0
    false_positive = 0
    false_negative = 0
    failures = 0

    print("=" * 58)
    print("本地记忆问答完整流程测试")
    print("语义检索 Top 10 → 证据核验 → 答案提取")
    print("=" * 58)

    for i, case in enumerate(CASES, 1):
        question = case["query"]
        expected = bool(case["relevant_files"])

        try:
            results = search(question, limit=10)
            memories = [
                {
                    "id": item["id"],
                    "content": read_memory(item["path"])
                }
                for item in results
            ]

            verdict = ask_model(question, memories)
            predicted = verdict["answerable"]
            passed = predicted == expected
            correct += passed

            if expected:
                answerable_total += 1
                false_negative += not predicted
            else:
                unanswerable_total += 1
                false_positive += predicted

            status = "正确" if passed else "错误"
            print(f"\n{i:02d}/20 [{status}] {question}")

            if predicted:
                print("回答：", verdict["answer"])
                print("证据：", verdict["evidence"])
            else:
                print("回答：现有记忆中没有找到足够证据。")

        except Exception as exc:
            failures += 1
            print(f"\n{i:02d}/20 [测试失败] {question}")
            print(f"原因：{exc}")

    print("\n" + "=" * 58)
    print("测试汇总")
    print("=" * 58)
    print(f"判定正确：{correct}/{len(CASES)}")
    print(f"有答案漏判：{false_negative}/{answerable_total}")
    print(f"无答案误报：{false_positive}/{unanswerable_total}")
    print(f"执行失败：{failures}")
    print("正式记忆与索引未修改。")

    return 1 if correct != len(CASES) or failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
