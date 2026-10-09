from pathlib import Path
from datetime import datetime
import re

from memory_updater import decide_update


MEMORY_ROOT = Path.home() / "AI-System" / "memory"
ARCHIVE_ROOT = MEMORY_ROOT / "archive"


def now_date():
    return datetime.now().strftime("%Y-%m-%d")


def now_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def clean_memory_body(text):
    if not text:
        return ""

    text = text.strip()

    # 删除可能返回的 YAML frontmatter
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            text = parts[2].strip()

    # 删除可能返回的标题
    text = re.sub(r"^#+\s*长期记忆\s*\n*", "", text)

    return text.strip()


def find_memory_file(memory_id):
    """
    只搜索当前 active memory。
    archive 不参与正常目标查找。
    """
    for path in MEMORY_ROOT.rglob("*.md"):
        try:
            relative = path.relative_to(MEMORY_ROOT)

            if relative.parts and relative.parts[0] == "archive":
                continue

            text = path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            match = re.search(
                r"^id:\s*(.+?)\s*$",
                text,
                re.MULTILINE
            )

            if match and match.group(1).strip() == memory_id:
                return path

        except Exception:
            continue

    return None


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


def write_new_memory(memory):
    memory_type = memory.get("type") or "preferences"

    target_dir = MEMORY_ROOT / memory_type
    target_dir.mkdir(parents=True, exist_ok=True)

    memory_id = f"mem_{now_stamp()}"

    path = target_dir / f"{memory_id}.md"

    content = f"""---
id: {memory_id}
type: {memory_type}
status: active
importance: {memory.get("importance", 0.5)}
confidence: {memory.get("confidence", 0.5)}
created_at: {now_date()}
updated_at: {now_date()}
---

# 长期记忆

{clean_memory_body(memory.get("memory"))}
"""

    path.write_text(content, encoding="utf-8")

    print(f"[NEW] {memory_id}")
    print(f"      {path}")

    return path


def archive_existing_memory(path, memory_id):
    """Archive the current active memory without losing history."""

    if not path.exists():
        return None

    old_text = path.read_text(encoding="utf-8")

    # Generate ONE version ID and reuse it everywhere.
    version_id = f"{memory_id}__v{now_stamp()}"

    # Replace ID
    lines = old_text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("id:"):
            lines[i] = f"id: {version_id}"
            break

    # Mark as superseded
    for i, line in enumerate(lines):
        if line.startswith("status:"):
            lines[i] = "status: superseded"
            break

    # Update timestamp
    for i, line in enumerate(lines):
        if line.startswith("updated_at:"):
            lines[i] = f"updated_at: {now_date()}"
            break

    archived_text = "\n".join(lines) + "\n"

    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)

    archive_path = ARCHIVE_ROOT / f"{version_id}.md"
    archive_path.write_text(archived_text, encoding="utf-8")

    print(f"[ARCHIVE] {memory_id}")
    print(f"          {archive_path}")

    return archive_path


def update_memory(memory_id, memory):
    path = find_memory_file(memory_id)

    if not path:
        print(f"[ERROR] Memory not found: {memory_id}")
        return None

    old_text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    metadata = parse_frontmatter(old_text)

    # 先归档旧版本
    archive_existing_memory(path, memory_id)

    memory_type = (
        memory.get("type")
        or metadata.get("type")
        or "preferences"
    )

    importance = (
        memory.get("importance")
        if memory.get("importance") is not None
        else metadata.get("importance", "0.5")
    )

    confidence = (
        memory.get("confidence")
        if memory.get("confidence") is not None
        else metadata.get("confidence", "0.5")
    )

    created_at = metadata.get(
        "created_at",
        now_date()
    )

    new_content = f"""---
id: {memory_id}
type: {memory_type}
status: active
importance: {importance}
confidence: {confidence}
created_at: {created_at}
updated_at: {now_date()}
---

# 长期记忆

{clean_memory_body(memory.get("memory"))}
"""

    path.write_text(
        new_content,
        encoding="utf-8"
    )

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

    print(f"Action: {action}")

    if action == "IGNORE":
        print("Nothing written.")
        return

    if action == "NEW":
        if not decision.get("memory"):
            print("[ERROR] NEW without memory content.")
            return

        write_new_memory({
            "type": decision.get("type"),
            "importance": decision.get("importance"),
            "confidence": decision.get("confidence"),
            "memory": decision.get("memory"),
        })

    elif action in ("UPDATE", "MERGE"):
        target_id = decision.get("target_id")

        if not target_id:
            print(f"[ERROR] {action} without target_id.")
            return

        if not decision.get("memory"):
            print(f"[ERROR] {action} without memory content.")
            return

        update_memory(
            target_id,
            {
                "type": decision.get("type"),
                "importance": decision.get("importance"),
                "confidence": decision.get("confidence"),
                "memory": decision.get("memory"),
            }
        )

    else:
        print(f"[ERROR] Unknown action: {action}")
        return

    print("\nWriter operation completed.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print('用法：python3 memory_writer_v2.py "用户说的话"')
        raise SystemExit(1)

    user_input = " ".join(sys.argv[1:])

    execute(user_input)
