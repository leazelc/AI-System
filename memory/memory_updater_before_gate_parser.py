from pathlib import Path
from datetime import datetime
import json
import re
import sys
import urllib.request

from memory_gate import ask_gate
from embedding_index import search as semantic_search


MEMORY_ROOT = Path.home() / "AI-System" / "memory"
LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"


def ask_updater(prompt):
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": """你是一个严格的长期记忆更新决策器。

你的任务不是判断“新信息和旧信息是不是同一个主题”，
而是判断新信息是否改变了旧记忆的实际含义、适用条件或行为规则。

必须区分：

IGNORE：
- 只是换一种说法
- 同义改写
- 完全重复
- 没有增加任何实际信息
- 没有改变行为规则

UPDATE：
- 改变旧记忆的政策或偏好
- 增加新的限制条件
- 删除原有限制条件
- 增加或删除例外条件
- 改变优先级
- 改变适用范围
- 改变“什么时候允许/不允许”
- 把模糊偏好变成明确规则
- 把原本宽泛的允许条件收紧
- 把原本严格的要求放宽

MERGE：
- 新信息和多个已有记忆共同构成一个更完整、统一的规则
- 合并后可以减少重复或互相分散的记忆

NEW：
- 没有任何已有记忆真正表达这个长期信息

特别注意：

“语义相似”不等于“重复”。

例如：

旧：
“个人 AI 数据优先保存在本地，必要时可以使用云端服务。”

新：
“个人 AI 数据默认优先保存在本地，但只有本地模型无法完成任务时才可以使用云端 AI。”

这不是简单重复。
新信息增加了“只有本地模型无法完成任务”这一限制，
因此必须判断为 UPDATE。

另一个例子：

旧：
“我喜欢 Mac。”

新：
“我喜欢 Mac，但以后购买电脑时优先考虑内存容量，而不是芯片性能。”

这是新增决策规则，应当 UPDATE 或 MERGE，而不是简单 IGNORE。

你必须严格输出 JSON，不要输出 Markdown，不要解释 JSON 之外的内容。

格式：

{
  "action": "NEW|UPDATE|MERGE|IGNORE",
  "target_id": "已有记忆ID或null",
  "memory": "新的完整记忆内容或null",
  "reason": "简短原因"
}
"""
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1
    }

    data_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    request = urllib.request.Request(
        LM_STUDIO_URL,
        data=data_bytes,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


def parse_frontmatter(text):
    if not text.startswith("---"):
        return {}

    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}

    result = {}

    for line in parts[1].splitlines():
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        result[key.strip()] = value.strip()

    return result


def read_memory_file(path):
    return path.read_text(
        encoding="utf-8",
        errors="ignore"
    )


def strip_code_fences(text):
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE
        )
        text = re.sub(
            r"\s*```$",
            "",
            text
        )

    return text.strip()


def decide_update(user_text):
    gate_raw = ask_gate(user_text)

    try:
        gate = json.loads(gate_raw)
    except json.JSONDecodeError:
        return {
            "action": "IGNORE",
            "target_id": None,
            "type": None,
            "importance": 0.0,
            "confidence": 0.0,
            "memory": None,
            "reason": "Memory Gate returned invalid JSON."
        }

    if not gate.get("should_remember"):
        return {
            "action": "IGNORE",
            "target_id": None,
            "type": None,
            "importance": 0.0,
            "confidence": 0.0,
            "memory": None,
            "reason": gate.get("reason", "Memory Gate rejected.")
        }

    results = semantic_search(
        user_text,
        limit=5
    )

    candidates = []

    for item in results:
        path = Path(item["path"])

        # Archive versions are history only.
        # They must never participate in normal update decisions.
        if "archive" in path.parts:
            continue

        try:
            text = read_memory_file(path)
            meta = parse_frontmatter(text)

            if meta.get("status") != "active":
                continue

            candidates.append({
                "id": item["id"],
                "path": str(path),
                "similarity": item["score"],
                "content": text
            })

        except Exception:
            continue

    candidate_text = ""

    for i, item in enumerate(candidates, 1):
        candidate_text += (
            f"\n[{i}] ID: {item['id']}\n"
            f"Similarity: {item['similarity']:.4f}\n"
            f"File: {item['path']}\n"
            f"---\n{item['content']}\n"
        )

    prompt = f"""
用户新信息：

{user_text}

Memory Gate 判断：

{json.dumps(gate, ensure_ascii=False, indent=2)}

已有相关长期记忆：

{candidate_text if candidate_text else "没有找到相关 active 记忆。"}

请判断新信息应该：

NEW / UPDATE / MERGE / IGNORE

重点检查：

1. 是否改变旧记忆的实际政策？
2. 是否增加或删除限制条件？
3. 是否改变允许或禁止的条件？
4. 是否增加例外？
5. 是否改变优先级？
6. 是否扩大或缩小适用范围？
7. 是否只是同义改写？

如果新信息只是“同主题”，但行为规则发生变化，必须 UPDATE。

如果新信息没有任何实际变化，才 IGNORE。

如果选择 UPDATE，必须填写最相关的 target_id。

如果选择 NEW，target_id 必须为 null。

如果选择 IGNORE，memory 必须为 null。

请只返回 JSON。
"""

    raw = ask_updater(prompt)
    raw = strip_code_fences(raw)

    try:
        decision = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "action": "IGNORE",
            "target_id": None,
            "type": gate.get("type"),
            "importance": gate.get("importance", 0.0),
            "confidence": gate.get("confidence", 0.0),
            "memory": None,
            "reason": "Updater returned invalid JSON."
        }

    action = decision.get("action", "IGNORE")

    if action not in {
        "NEW",
        "UPDATE",
        "MERGE",
        "IGNORE"
    }:
        action = "IGNORE"

    target_id = decision.get("target_id")

    if action in {"UPDATE", "MERGE"} and target_id:
        valid_ids = {item["id"] for item in candidates}

        if target_id not in valid_ids:
            action = "NEW"
            target_id = None

    if action in {"NEW", "UPDATE", "MERGE"}:
        memory = decision.get("memory") or gate.get("memory")
    else:
        memory = None

    return {
        "action": action,
        "target_id": target_id,
        "type": gate.get("type"),
        "importance": gate.get("importance", 0.0),
        "confidence": gate.get("confidence", 0.0),
        "memory": memory,
        "reason": decision.get("reason", "")
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 memory_updater.py \"用户新信息\"")
        sys.exit(1)

    user_text = sys.argv[1]

    print()
    print("Memory Updater V3")
    print("=" * 70)

    print("用户新信息：")
    print(user_text)

    print()
    print("[1] Memory Gate")
    print("=" * 70)

    gate_raw = ask_gate(user_text)

    try:
        gate = json.loads(gate_raw)
    except json.JSONDecodeError:
        return {
            "action": "IGNORE",
            "target_id": None,
            "type": None,
            "importance": 0.0,
            "confidence": 0.0,
            "memory": None,
            "reason": "Memory Gate returned invalid JSON."
        }
    print(json.dumps(
        gate,
        ensure_ascii=False,
        indent=2
    ))

    print()
    print("[2] Semantic Search")
    print("=" * 70)

    results = semantic_search(
        user_text,
        limit=5
    )

    if not results:
        print("No semantic matches found.")
    else:
        for i, item in enumerate(results, 1):
            print(
                f"\n[{i}] Similarity: {item['score']:.4f}\n"
                f"ID: {item['id']}\n"
                f"File: {item['path']}"
            )

    print()
    print("[3] Final Decision")
    print("=" * 70)

    decision = decide_update(user_text)

    print(json.dumps(
        decision,
        ensure_ascii=False,
        indent=2
    ))


if __name__ == "__main__":
    main()
