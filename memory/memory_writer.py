import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

MEMORY_ROOT = Path.home() / "AI-System" / "memory"
GATE_SCRIPT = MEMORY_ROOT / "memory_gate.py"

VALID_TYPES = {
    "core",
    "preferences",
    "projects",
    "experiences",
    "decisions",
}


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
        raise RuntimeError("无法找到 Memory Gate 解析结果")

    json_text = output.split(marker, 1)[1].strip()
    return json.loads(json_text)


def make_memory_file(gate_result):
    memory_type = gate_result["type"]
    memory_text = gate_result["memory"]
    importance = gate_result["importance"]
    confidence = gate_result["confidence"]

    if memory_type not in VALID_TYPES:
        raise ValueError(f"无效 memory type: {memory_type}")

    if not memory_text:
        raise ValueError("memory 内容为空")

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    memory_id = f"mem_{date_str}_{now.strftime('%H%M%S_%f')[:-3]}"

    directory = MEMORY_ROOT / memory_type
    directory.mkdir(parents=True, exist_ok=True)

    filename = f"{memory_id}.md"
    path = directory / filename

    content = f"""---
id: {memory_id}
type: {memory_type}
status: active
importance: {importance}
confidence: {confidence}
created_at: {date_str}
updated_at: {date_str}
---

# 长期记忆

{memory_text}

"""

    path.write_text(content, encoding="utf-8")

    return path


def main():
    if len(sys.argv) < 2:
        print('用法：python3 memory_writer.py "用户说的话"')
        return

    user_text = " ".join(sys.argv[1:])

    print("\nMemory Writer V1")
    print("=" * 60)
    print("用户内容：")
    print(user_text)
    print("=" * 60)

    gate_result = run_gate(user_text)

    print("Memory Gate：")
    print(json.dumps(gate_result, ensure_ascii=False, indent=2))

    if not gate_result.get("should_remember"):
        print("\n→ 不保存。")
        return

    path = make_memory_file(gate_result)

    print("\n→ 已创建长期记忆：")
    print(path)


if __name__ == "__main__":
    main()
