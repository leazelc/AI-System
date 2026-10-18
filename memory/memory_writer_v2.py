from pathlib import Path
from datetime import datetime
import threading
import re

from memory_updater import decide_update
from memory_core import scan as scan_core
from embedding_index import sync as sync_embeddings
from memory import keyword_index
from memory_validation import validate_decision, VALID_TYPES


from aisystem_paths import MEMORY_ROOT, ARCHIVE_ROOT, CANDIDATE_ROOT
CANDIDATE_ARCHIVE_ROOT = CANDIDATE_ROOT / "archive"


def write_candidate(memory):
    """
    Write a memory candidate into the intermediate memory layer.
    Candidates are not active long-term memories.
    Never overwrite an existing candidate file.
    """
    CANDIDATE_ROOT.mkdir(parents=True, exist_ok=True)

    while True:
        memory_id = f"candidate_{now_stamp()}"
        path = CANDIDATE_ROOT / f"{memory_id}.md"

        content = f"""---
id: {memory_id}
type: {memory.get("type") or "unknown"}
status: candidate
importance: {memory.get("importance", 0.5)}
confidence: {memory.get("confidence", 0.5)}
created_at: {now_date()}
updated_at: {now_date()}
---

# 候选记忆

{clean_memory_body(memory.get("memory"))}
"""

        try:
            with path.open("x", encoding="utf-8") as file:
                try:
                    file.write(content)
                except BaseException:
                    file.close()
                    path.unlink(missing_ok=True)
                    raise
        except FileExistsError:
            continue

        break

    print(f"[CANDIDATE] {memory_id}")
    print(f"            {path}")

    return path

def archive_candidate(path):
    """
    Move a processed candidate into the candidate archive.
    """
    if not path or not path.exists():
        return None

    CANDIDATE_ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)

    archive_path = CANDIDATE_ARCHIVE_ROOT / path.name
    path.replace(archive_path)

    print(f"[CANDIDATE ARCHIVE] {archive_path}")

    return archive_path


def sync_indexes():
    """
    Synchronize metadata, keyword, and semantic indexes
    after a successful memory write.
    """
    print("\nSyncing memory indexes...")

    try:
        scan_core()
        print("[CORE] SQLite metadata index synchronized.")
    except Exception as e:
        print(f"[CORE ERROR] {e}")
        return False

    try:
        keyword_count = keyword_index.rebuild()
        print(
            f"[KEYWORD] Keyword index synchronized "
            f"({keyword_count} active memories)."
        )
    except Exception as e:
        print(f"[KEYWORD ERROR] {e}")
        return False

    try:
        embeddings_synced = sync_embeddings()

        if embeddings_synced is False:
            print(
                "[EMBEDDING ERROR] "
                "Semantic index synchronization was incomplete."
            )
            return False

        print("[EMBEDDING] Semantic index synchronized.")
    except Exception as e:
        print(f"[EMBEDDING ERROR] {e}")
        return False

    print("Memory indexes synchronized.")
    return True


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
    """
    Create a new active memory without overwriting existing files.
    """
    memory_type = memory.get("type") or "preferences"

    target_dir = MEMORY_ROOT / memory_type
    target_dir.mkdir(parents=True, exist_ok=True)

    while True:
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

        try:
            with path.open("x", encoding="utf-8") as file:
                try:
                    file.write(content)
                except BaseException:
                    file.close()
                    path.unlink(missing_ok=True)
                    raise
        except FileExistsError:
            continue

        break

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

    import os
    import tempfile

    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)

    archive_path = ARCHIVE_ROOT / f"{version_id}.md"
    temp_path = None

    try:
        # Prepare the complete archive before publishing its final name.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(ARCHIVE_ROOT),
            prefix=f".{version_id}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(archived_text)
            temp_file.flush()
            os.fsync(temp_file.fileno())

        # Atomic publication without overwriting existing history.
        # The temporary file and archive must be on the same filesystem.
        os.link(str(temp_path), str(archive_path))

        print(f"[ARCHIVE] {memory_id}")
        print(f"          {archive_path}")

        return archive_path

    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError as exc:
                print(
                    f"[WARNING] Could not clean up temporary file: "
                    f"{temp_path} ({exc})"
                )


def _update_memory_unlocked(memory_id, memory):
    path = find_memory_file(memory_id)

    if not path:
        print(f"[ERROR] Memory not found: {memory_id}")
        return None

    old_text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    metadata = parse_frontmatter(old_text)

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

    # Write the complete new version to a temporary file first.
    import os
    import tempfile

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(new_content)
            temp_file.flush()
            os.fsync(temp_file.fileno())

        # Archive the original only after the new version is ready.
        archive_path = archive_existing_memory(path, memory_id)

        if not archive_path:
            print(f"[ERROR] Could not archive memory: {memory_id}")
            return None

        # Replace the active file atomically on the same filesystem.
        os.replace(str(temp_path), str(path))
        temp_path = None

    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError as exc:
                print(
                    f"[WARNING] Could not clean up temporary file: "
                    f"{temp_path} ({exc})"
                )

    print(f"[UPDATE] {memory_id}")
    print(f"         {path}")

    return path



# Serialize updates to the same memory within this Python process.
_UPDATE_LOCKS = {}
_UPDATE_LOCKS_GUARD = threading.Lock()


def _get_update_lock(memory_id):
    key = str(memory_id)

    with _UPDATE_LOCKS_GUARD:
        lock = _UPDATE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _UPDATE_LOCKS[key] = lock
        return lock


def update_memory(memory_id, memory):
    """
    Update one memory atomically with respect to other threads in this process.
    Different memory IDs can still be updated concurrently.
    """
    lock = _get_update_lock(memory_id)

    with lock:
        return _update_memory_unlocked(memory_id, memory)


def execute(user_input, expected_type=None):
    print("Running Memory Writer V2...\\n")

    if expected_type is not None and expected_type not in VALID_TYPES:
        print(f"[REJECTED] Invalid expected_type: {expected_type!r}")
        return {"status": "error", "action": "REJECTED"}

    decision = decide_update(user_input, expected_type=expected_type)

    if not decision:
        print("[ERROR] No decision returned.")
        return {"status": "error", "action": "UNKNOWN"}

    if not isinstance(decision, dict):
        print("[REJECTED] Invalid memory decision: expected a dictionary.")
        return {"status": "error", "action": "REJECTED"}

    action = decision.get("action")

    if action == "NEW" and expected_type is not None:
        decision["type"] = expected_type

    is_valid, reason = validate_decision(decision)
    if not is_valid:
        print(f"[REJECTED] Invalid memory decision: {reason}")
        print("No candidate or active memory was written.")
        return {"status": "error", "action": "REJECTED"}

    action = decision.get("action")
    print(f"Action: {action}")

    if action == "IGNORE":
        print("Nothing written.")
        return {"status": "ignored", "action": "IGNORE"}

    if action not in ("NEW", "UPDATE", "MERGE"):
        print(f"[ERROR] Unknown action: {action}")
        return {"status": "error", "action": action or "UNKNOWN"}

    target_id = decision.get("target_id")
    if action in ("UPDATE", "MERGE"):
        target_path = find_memory_file(target_id)
        if not target_path:
            print(f"[REJECTED] Target memory does not exist: {target_id}")
            print("No candidate or active memory was written.")
            return {"status": "error", "action": "REJECTED"}

        try:
            target_text = target_path.read_text(encoding="utf-8", errors="ignore")
            target_metadata = parse_frontmatter(target_text)
            target_type = target_metadata.get("type")
        except Exception as exc:
            print(f"[REJECTED] Could not verify target memory type: {exc}")
            return {"status": "error", "action": "REJECTED"}

        if target_type not in VALID_TYPES:
            print(f"[REJECTED] Target memory has invalid or missing type: {target_type!r}")
            print("No candidate or active memory was written.")
            return {"status": "error", "action": "REJECTED"}

        if expected_type is not None and target_type != expected_type:
            print(
                f"[REJECTED] Category mismatch: expected {expected_type!r}, "
                f"but target memory type is {target_type!r}."
            )
            print("No candidate or active memory was written.")
            return {"status": "error", "action": "REJECTED"}

        # Existing memory categories are immutable during UPDATE/MERGE.
        decision["type"] = target_type

    if not decision.get("memory"):
        print(f"[ERROR] {action} without memory content.")
        return {"status": "error", "action": action}

    candidate_path = write_candidate({
        "type": decision.get("type"),
        "importance": decision.get("importance"),
        "confidence": decision.get("confidence"),
        "memory": decision.get("memory"),
    })

    if action == "NEW":
        core_path = write_new_memory({
            "type": decision.get("type"),
            "importance": decision.get("importance"),
            "confidence": decision.get("confidence"),
            "memory": decision.get("memory"),
        })
    else:
        core_path = update_memory(
            target_id,
            {
                "type": decision.get("type"),
                "importance": decision.get("importance"),
                "confidence": decision.get("confidence"),
                "memory": decision.get("memory"),
            }
        )

    if not core_path:
        return {"status": "error", "action": action}

    if candidate_path:
        try:
            archive_candidate(candidate_path)
        except Exception as exc:
            print(f"[WARNING] Candidate archival failed: {exc}")

    memory_id = core_path.stem if action == "NEW" else target_id

    # Synchronize both indexes BEFORE reporting success to the proxy.
    indexes_synced = sync_indexes()
    if not indexes_synced:
        print(
            "\\n[WARNING] Memory was written, but index synchronization failed."
        )
        return {
            "status": "index_sync_failed",
            "action": action,
            "memory_ids": [memory_id],
        }

    print("\\nWriter operation completed.")
    return {
        "status": "success",
        "action": action,
        "memory_ids": [memory_id],
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print('用法：python3 memory_writer_v2.py "用户说的话"')
        raise SystemExit(1)

    user_input = " ".join(sys.argv[1:])

    execute(user_input)
