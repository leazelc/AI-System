import json
import subprocess
import sys
from pathlib import Path
import urllib.request

MEMORY_ROOT = Path.home() / "AI-System" / "memory"
GATE_SCRIPT = MEMORY_ROOT / "memory_gate.py"
SEARCH_SCRIPT = MEMORY_ROOT / "embedding_index.py"

LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"


def run_gate(user_text):
    result = subprocess.run(
        ["python3", str(GATE_SCRIPT), user_text],
        capture_output=True,
        text=True,
        check=True,
    )

    output = result.stdout
    marker = "解析结果："

    if marker not in output:
        raise RuntimeError("无法解析 Memory Gate 结果")

    return json.loads(output.split(marker, 1)[1].strip())


def run_semantic_search(user_text):
    result = subprocess.run(
        ["python3", str(SEARCH_SCRIPT), user_text],
        capture_output=True,
        text=True,
        check=True,
    )

    return result.stdout


def extract_results(search_output):
    """
    从 embedding_index.py 的输出中提取搜索到的记忆 ID 和文件路径。
    """
    results = []

    lines = search_output.splitlines()

    current_id = None
    current_file = None
    current_similarity = None

    for line in lines:
        line = line.strip()

        if line.startswith("[") and "] Similarity:" in line:
            try:
                current_similarity = float(
                    line.split("Similarity:", 1)[1].strip()
                )
            except ValueError:
                current_similarity = None

        elif line.startswith("ID:"):
            current_id = line.split("ID:", 1)[1].strip()

        elif line.startswith("File:"):
            current_file = line.split("File:", 1)[1].strip()

            if current_id and current_file:
                results.append({
                    "id": current_id,
                    "file": current_file,
                    "similarity": current_similarity,
                })

                current_id = None
                current_file = None
                current_similarity = None

    return results


def read_memory_file(path):
    path = Path(path)

    if not path.exists():
        return None

    return path.read_text(encoding="utf-8")


def ask_updater(user_text, gate_result, candidates):
    system_prompt = """
你是一个本地 AI 长期记忆更新判断器。

你的任务是比较：

1. 用户刚刚表达的新信息
2. 已有长期记忆的完整内容

然后决定应该如何处理。

你只能选择：

NEW
UPDATE
MERGE
IGNORE

==================================================
四种操作的严格定义
==================================================

NEW：
新信息具有长期价值，但与已有记忆没有足够强的关系。
应该创建新的独立记忆。

UPDATE：
新信息改变、修正、撤销或替代已有记忆。

特别注意以下表达：

“不再……”
“以后可以……”
“以后不要求……”
“改成……”
“我现在接受……”
“我决定……”
“取消之前……”
“从现在开始……”
“以前……现在……”

如果新信息与旧记忆属于同一个偏好/政策/决定，
但立场发生变化，应优先 UPDATE。

例如：

旧：
“用户希望个人 AI 数据尽可能保存在本地。”

新：
“以后个人 AI 可以使用云端服务，不再要求所有数据都必须本地保存。”

这不是重复。

这是对原有偏好的修改。

应该：

UPDATE

==================================================

MERGE：
新信息与旧记忆属于同一个主题，
但不是简单重复，也没有明确冲突，
两者可以合并成一个更完整的长期记忆。

例如：

旧：
“用户喜欢本地 AI。”

新：
“用户特别重视本地 AI 的隐私。”

可以 MERGE。

==================================================

IGNORE：
只有以下情况才使用：

1. 新信息与已有记忆本质完全相同
2. 新信息只是重复表达
3. 新信息没有增加任何新的长期信息

==================================================
非常重要
==================================================

语义相似度高 ≠ 内容重复。

不要仅根据 embedding similarity 判断。

必须阅读已有 Markdown 的完整正文，
比较双方表达的实际立场。

尤其注意：

“尽可能本地”
“不允许云端”
“允许使用云端”
“不再要求全部本地”

这些表达可能属于同一个主题，
但它们的政策立场不同。

如果立场发生变化，必须 UPDATE。

不要删除旧记忆。

UPDATE 只是告诉上层系统：
“应该更新哪一条记忆”。

当前阶段绝对不要执行文件修改。

如果无法确定，优先 NEW，
不要错误地 UPDATE 或 IGNORE。

只返回 JSON，不要输出任何解释。

格式：

{
  "action": "NEW",
  "target_id": null,
  "memory": "新的长期记忆",
  "reason": "原因"
}

UPDATE：

{
  "action": "UPDATE",
  "target_id": "已有记忆ID",
  "memory": "更新后的完整长期记忆",
  "reason": "原因"
}

MERGE：

{
  "action": "MERGE",
  "target_id": "已有记忆ID",
  "memory": "合并后的完整长期记忆",
  "reason": "原因"
}

IGNORE：

{
  "action": "IGNORE",
  "target_id": null,
  "memory": null,
  "reason": "原因"
}
"""

    candidate_text = []

    for i, candidate in enumerate(candidates, 1):
        candidate_text.append(
            f"""
===== EXISTING MEMORY {i} =====

ID:
{candidate["id"]}

Similarity:
{candidate["similarity"]}

File:
{candidate["file"]}

Full Markdown Content:
{candidate["content"]}

===== END MEMORY {i} =====
"""
        )

    user_prompt = f"""
用户新信息：

{user_text}

Memory Gate：

{json.dumps(gate_result, ensure_ascii=False, indent=2)}

相关已有长期记忆：

{"".join(candidate_text) if candidate_text else "没有找到相关已有记忆。"}

请严格按照规则判断应该执行 NEW、UPDATE、MERGE 还是 IGNORE。
"""

    payload = {
        "model": MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    request = urllib.request.Request(
        LM_STUDIO_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result["choices"][0]["message"]["content"]



def decide_update(user_text):
    """
    给 Writer V2 使用的统一接口。

    返回：
    {
        action,
        target_id,
        type,
        importance,
        confidence,
        memory,
        reason
    }

    不修改任何 Markdown 文件。
    """

    gate_result = run_gate(user_text)

    if not gate_result.get("should_remember"):
        return {
            "action": "IGNORE",
            "target_id": None,
            "type": None,
            "importance": 0.0,
            "confidence": 0.0,
            "memory": None,
            "reason": "Memory Gate 判断该信息没有长期记忆价值。",
        }

    search_result = run_semantic_search(user_text)

    candidates = extract_results(search_result)

    valid_candidates = []

    for candidate in candidates:
        content = read_memory_file(candidate["file"])

        if content is None:
            continue

        candidate["content"] = content
        valid_candidates.append(candidate)

    raw = ask_updater(
        user_text,
        gate_result,
        valid_candidates,
    )

    # GPT-OSS 可能返回 Markdown 代码块，
    # 这里先清理代码块，再解析 JSON。
    cleaned = raw.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

    try:
        decision = json.loads(cleaned)
    except json.JSONDecodeError:
        # 尝试从文本中提取第一个 JSON 对象
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError(
                "GPT-OSS 返回的结果不是有效 JSON：\n" + raw
            )

        decision = json.loads(cleaned[start:end + 1])

    action = decision.get("action")

    if action not in {"NEW", "UPDATE", "MERGE", "IGNORE"}:
        raise ValueError(
            f"GPT-OSS 返回了未知 action: {action}"
        )

    return {
        "action": action,
        "target_id": decision.get("target_id"),
        "type": decision.get("type") or gate_result.get("type"),
        "importance": decision.get(
            "importance",
            gate_result.get("importance", 0.5)
        ),
        "confidence": decision.get(
            "confidence",
            gate_result.get("confidence", 0.5)
        ),
        "memory": decision.get("memory"),
        "reason": decision.get("reason", ""),
    }

def main():
    if len(sys.argv) < 2:
        print('用法：python3 memory_updater.py "用户说的话"')
        return

    user_text = " ".join(sys.argv[1:])

    print("\nMemory Updater V2")
    print("=" * 70)

    print("用户新信息：")
    print(user_text)

    print("\n[1] Memory Gate")
    print("=" * 70)

    gate_result = run_gate(user_text)
    print(json.dumps(gate_result, ensure_ascii=False, indent=2))

    if not gate_result.get("should_remember"):
        print("\n→ 不需要长期记忆。")
        return

    print("\n[2] Semantic Search")
    print("=" * 70)

    search_result = run_semantic_search(user_text)
    print(search_result)

    candidates = extract_results(search_result)

    print("\n[3] Existing Memory Contents")
    print("=" * 70)

    valid_candidates = []

    for candidate in candidates:
        content = read_memory_file(candidate["file"])

        if content is None:
            print(f"WARNING: 文件不存在：{candidate['file']}")
            continue

        candidate["content"] = content
        valid_candidates.append(candidate)

        print(f"\nID: {candidate['id']}")
        print(f"Similarity: {candidate['similarity']}")
        print(f"File: {candidate['file']}")
        print("-" * 60)
        print(content)

    print("\n[4] GPT-OSS Update Decision")
    print("=" * 70)

    raw = ask_updater(
        user_text,
        gate_result,
        valid_candidates,
    )

    print(raw)

    print("\n[5] Parsed Decision")
    print("=" * 70)

    try:
        parsed = json.loads(raw)
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
    except json.JSONDecodeError:
        print("WARNING: GPT-OSS 没有返回有效 JSON。")


if __name__ == "__main__":
    main()
