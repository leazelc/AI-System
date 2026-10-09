import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

BASE_DIR = Path(__file__).resolve().parents[1]
MEMORY_DIR = BASE_DIR / "memory"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if str(MEMORY_DIR) not in sys.path:
    sys.path.insert(0, str(MEMORY_DIR))

from memory.memory_updater import decide_update


class TestMemoryUpdaterDecision(unittest.TestCase):
    def test_invalid_update_target_must_not_become_new_memory(self):
        """
        如果模型选择了不存在于候选列表中的 UPDATE 目标，
        系统必须安全拒绝，不能自动转换成 NEW。
        """
        gate_response = {
            "should_remember": True,
            "type": "preferences",
            "importance": 0.8,
            "confidence": 0.95,
            "memory": "电脑预算上限调整为 15000 元。",
            "reason": "长期预算发生变化。",
        }

        updater_response = {
            "action": "UPDATE",
            "target_id": "mem_id_that_does_not_exist",
            "memory": "电脑预算上限调整为 15000 元。",
            "reason": "更新预算。",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            memory_file = Path(temp_dir) / "mem_budget.md"
            memory_file.write_text(
                """---
id: mem_budget
type: preferences
status: active
importance: 0.8
confidence: 0.9
created_at: 2026-10-01
updated_at: 2026-10-01
---

电脑预算上限为 20000 元。
""",
                encoding="utf-8",
            )

            with patch(
                "memory.memory_updater.ask_gate",
                return_value=json.dumps(
                    gate_response,
                    ensure_ascii=False,
                ),
            ), patch(
                "memory.memory_updater.semantic_search",
                return_value=[
                    {
                        "id": "mem_budget",
                        "path": str(memory_file),
                        "score": 0.95,
                    }
                ],
            ), patch(
                "memory.memory_updater.ask_updater",
                return_value=json.dumps(
                    updater_response,
                    ensure_ascii=False,
                ),
            ):
                result = decide_update(
                    "我现在购买电脑的预算上限改成 15000 元。"
                )

        self.assertEqual(
            result["action"],
            "IGNORE",
            "无效的 UPDATE 目标不能自动变成 NEW，否则可能产生冲突记忆。",
        )
        self.assertIsNone(result["target_id"])
        self.assertIsNone(result["memory"])
        self.assertIn("target", result["reason"].lower())


if __name__ == "__main__":
    unittest.main()
