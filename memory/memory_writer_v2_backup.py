from pathlib import Path
from datetime import datetime
import re
import subprocess

from memory_updater import decide_update


MEMORY_ROOT = Path.home() / "AI-System" / "memory"


def now_date():
    return datetime.now().strftime("%Y-%m-%d")


def now_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def clean_memory_body(text):
    """
    清理 GPT 返回的 memory 内容。
    防止把 YAML frontmatter 或重复标题写进正文。
    """
    if not text:
        return ""

    text = text.strip()

    # 删除可能返回的 YAML frontmatter
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            text = parts[2].strip()

    # 删除 GPT 可能附带的标题
    text = re.sub(r"^#+\s*长期记忆\s*\n*", "", text)

    return text.strip()


def find_memory_file(memory_id):
    """
    根据 memory ID 查找真实 Markdown 文件。
    Markdown 文件才是真正的数据源。
    """
    for path in MEMORY_ROOT.rglob("*.md"):
        text = path.read_text(encoding="utf-8", errors="ignore")

        match = re.search(
            r"^id:\s*(.+?)\s*$",
            text,
            re.MULTILINE
        )

        if match and match.group(1).strip() == memory_id:
            return path

    return None


def parse_frontmatter(text):
    if not text.startswith("---"):
        return {}

    parts = text.split("---", 2)

    if len(parts) < 3:
        return {}

    front = parts[1]
    result = {}

    for line in front.splitlines():
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        result[key.strip()] = value.strip()

    return result


def write_new_memory(memory):
    memory_type = memory["type"]

    target_dir = MEMORY_ROOT / memory_type
    target_dir.mkdir(parents=True, exist_ok=True)

    memory_id = f"mem_{now_stamp()}"

    path = target_dir / f"{memory_id}.md"

    content = f"""---
id: {memory_id}
type: {memory_type}
status: active
importance: {memory["importance"]}
confidence: {memory["confidence"]}
created_at: {now_date()}
updated_at: {now_date()}
---

# 长期记忆

{clean_memory_body(memory["memory"])}
"""

    path.write_text(content, encoding="utf-8")

    print(f"[NEW] {memory_id}")
    print(f"      {path}")

    return path


def update_memory(memory_id, memory):
    path = find_memory_file(memory_id)

    if not path:
        print(f"[ERROR] Memory not found: {memory_id}")
        return None

    old_text = path.read_text(encoding="utf-8")
    metadata = parse_frontmatter(old_text)

    memory_type = memory.get("type") or metadata.get("type", "preferences")

    importance = memory.get("importance")
    confidence = memory.get("confidence")

    if importance is None:
        importance = metadata.get("importance", "0.5")

    if confidence is None:
        confidence = metadata.get("confidence", "0.5")

    created_at = metadata.get("created_at", now_date())

    content = f"""---
id: {memory_id}
type: {memory_type}
status: active
importance: {importance}
confidence: {confidence}
created_at: {created_at}
updated_at: {now_date()}
---

# 长期记忆

{clean_memory_body(memory["memory"])}
"""

    path.write_text(content, encoding="utf-8")

    print(f"[UPDATE] {memory_id}")
    print(f"         {path}")

    return path


def execute(user_input):
    print("Running Memory Writer V2...\n")

    decision = decide_update(user_input)

    if not decision:
        print("[ERROR] No decision returned.")
        return

    action = decision.get("action")
    target_id = decision.get("target_id")
    memory_text = decision.get("memory")

    print(f"Action: {action}")

    if action == "IGNORE":
        print("Nothing written.")
        return

    if action == "NEW":
        if not memory_text:
            print("[ERROR] NEW without memory content.")
            return

        memory = {
            "type": decision.get("type", "preferences"),
            "importance": decision.get("importance", 0.5),
            "confidence": decision.get("confidence", 0.5),
            "memory": memory_text,
        }

        write_new_memory(memory)

    elif action in ("UPDATE", "MERGE"):
        if not target_id:
            print(f"[ERROR] {action} without target_id.")
            return

        if not memory_text:
            print(f"[ERROR] {action} without memory content.")
            return

        memory = {
            "type": decision.get("type"),
            "importance": decision.get("importance"),
            "confidence": decision.get("confidence"),
            "memory": memory_text,
        }

        update_memory(target_id, memory)

    else:
        print(f"[ERROR] Unknown action: {action}")
        return

    print("\nWriter operation completed.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print('python3 memory_writer_v2.py "用户输入"')
        raise SystemExit(1)

    user_input = " ".join(sys.argv[1:])

    execute(user_input)
