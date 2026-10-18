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
    def test_invalid_gate_json_raises_decision_output_error(self):
        from memory.memory_updater import DecisionOutputError

        with patch(
            "memory.memory_updater.ask_gate",
            return_value="not valid JSON",
        ):
            with self.assertRaisesRegex(
                DecisionOutputError, "Memory Gate returned invalid JSON"
            ):
                decide_update("一条合成测试事实")

    def test_invalid_updater_json_raises_decision_output_error(self):
        from memory.memory_updater import DecisionOutputError

        gate_response = {
            "should_remember": True,
            "type": "preferences",
            "importance": 0.8,
            "confidence": 0.9,
            "memory": "合成测试事实",
            "reason": "测试",
        }

        with patch(
            "memory.memory_updater.ask_gate",
            return_value=json.dumps(gate_response, ensure_ascii=False),
        ), patch(
            "memory.memory_updater.semantic_search",
            return_value=[],
        ), patch(
            "memory.memory_updater.ask_updater",
            return_value="not valid JSON",
        ):
            with self.assertRaisesRegex(
                DecisionOutputError, "Updater returned invalid JSON"
            ):
                decide_update("一条合成测试事实")


    def test_missing_updater_action_raises_decision_output_error(self):
        from memory.memory_updater import DecisionOutputError

        gate_response = {
            "should_remember": True,
            "type": "preferences",
            "importance": 0.8,
            "confidence": 0.9,
            "memory": "合成测试事实",
            "reason": "测试",
        }

        with patch(
            "memory.memory_updater.ask_gate",
            return_value=json.dumps(gate_response, ensure_ascii=False),
        ), patch(
            "memory.memory_updater.semantic_search",
            return_value=[],
        ), patch(
            "memory.memory_updater.ask_updater",
            return_value=json.dumps({"reason": "缺少 action"}),
        ):
            with self.assertRaisesRegex(
                DecisionOutputError, "invalid action"
            ):
                decide_update("一条合成测试事实")

    def test_unknown_updater_action_raises_decision_output_error(self):
        from memory.memory_updater import DecisionOutputError

        gate_response = {
            "should_remember": True,
            "type": "preferences",
            "importance": 0.8,
            "confidence": 0.9,
            "memory": "合成测试事实",
            "reason": "测试",
        }

        with patch(
            "memory.memory_updater.ask_gate",
            return_value=json.dumps(gate_response, ensure_ascii=False),
        ), patch(
            "memory.memory_updater.semantic_search",
            return_value=[],
        ), patch(
            "memory.memory_updater.ask_updater",
            return_value=json.dumps({"action": "UNKNOWN"}),
        ):
            with self.assertRaisesRegex(
                DecisionOutputError, "invalid action"
            ):
                decide_update("一条合成测试事实")

    def test_non_object_updater_json_raises_decision_output_error(self):
        from memory.memory_updater import DecisionOutputError

        gate_response = {
            "should_remember": True,
            "type": "preferences",
            "importance": 0.8,
            "confidence": 0.9,
            "memory": "合成测试事实",
            "reason": "测试",
        }

        with patch(
            "memory.memory_updater.ask_gate",
            return_value=json.dumps(gate_response, ensure_ascii=False),
        ), patch(
            "memory.memory_updater.semantic_search",
            return_value=[],
        ), patch(
            "memory.memory_updater.ask_updater",
            return_value='["not", "an", "object"]',
        ):
            with self.assertRaisesRegex(
                DecisionOutputError, "must be a JSON object"
            ):
                decide_update("一条合成测试事实")

    def test_invalid_gate_shape_raises_decision_output_error(self):
        from memory.memory_updater import DecisionOutputError

        with patch(
            "memory.memory_updater.ask_gate",
            return_value='["not", "an", "object"]',
        ):
            with self.assertRaisesRegex(
                DecisionOutputError, "must be a JSON object"
            ):
                decide_update("一条合成测试事实")

    def test_missing_memory_text_raises_decision_output_error(self):
        from memory.memory_updater import DecisionOutputError

        gate_response = {
            "should_remember": True,
            "type": "preferences",
            "importance": 0.8,
            "confidence": 0.9,
            "memory": None,
            "reason": "测试",
        }

        with patch(
            "memory.memory_updater.ask_gate",
            return_value=json.dumps(gate_response, ensure_ascii=False),
        ), patch(
            "memory.memory_updater.semantic_search",
            return_value=[],
        ), patch(
            "memory.memory_updater.ask_updater",
            return_value=json.dumps({
                "action": "NEW",
                "target_id": None,
                "memory": None,
                "reason": "测试",
            }),
        ):
            with self.assertRaisesRegex(
                DecisionOutputError, "without valid memory text"
            ):
                decide_update("一条合成测试事实")

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
                from memory.memory_updater import DecisionOutputError

                with self.assertRaisesRegex(
                    DecisionOutputError, "Invalid UPDATE target"
                ):
                    decide_update(
                        "我现在购买电脑的预算上限改成 15000 元。"
                    )


    def test_expected_type_overrides_gate_type_for_new_memory(self):
        gate_response = {
            "should_remember": True,
            "type": "preferences",
            "importance": 0.8,
            "confidence": 0.9,
            "memory": "正在开发本地记忆系统",
            "reason": "长期项目事实",
        }
        updater_response = {
            "action": "NEW",
            "target_id": None,
            "memory": "正在开发本地记忆系统",
            "reason": "新的独立项目事实",
        }

        with patch(
            "memory.memory_updater.ask_gate",
            return_value=json.dumps(gate_response, ensure_ascii=False),
        ), patch(
            "memory.memory_updater.semantic_search",
            return_value=[],
        ), patch(
            "memory.memory_updater.ask_updater",
            return_value=json.dumps(updater_response, ensure_ascii=False),
        ):
            result = decide_update(
                "正在开发本地记忆系统",
                expected_type="projects",
            )

        self.assertEqual(result["action"], "NEW")
        self.assertEqual(result["type"], "projects")

    def test_invalid_expected_type_is_rejected_before_gate_call(self):
        from memory.memory_updater import DecisionOutputError

        with patch("memory.memory_updater.ask_gate") as ask_gate:
            with self.assertRaisesRegex(
                DecisionOutputError, "expected_type"
            ):
                decide_update(
                    "一条合成测试事实",
                    expected_type="unknown",
                )

        ask_gate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
