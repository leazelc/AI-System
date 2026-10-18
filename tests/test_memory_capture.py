import unittest

from memory.memory_capture import build_capture_input, select_user_context


class TestMemoryCapture(unittest.TestCase):
    def test_only_user_messages_are_selected(self):
        messages = [
            {"role": "user", "content": "我正在开发一个本地优先的 AI 项目。"},
            {"role": "assistant", "content": "项目使用了某某框架。"},
            {"role": "system", "content": "internal prompt"},
            {"role": "user", "content": "以后优先保持本地处理。"},
        ]
        result = select_user_context(messages)
        self.assertEqual(
            [item.content for item in result],
            ["我正在开发一个本地优先的 AI 项目。", "以后优先保持本地处理。"],
        )

    def test_message_and_character_limits_are_respected(self):
        messages = [
            {"role": "user", "content": "A" * 10},
            {"role": "user", "content": "B" * 10},
            {"role": "user", "content": "C" * 10},
        ]
        result = select_user_context(messages, max_messages=2, max_chars=12)
        self.assertLessEqual(sum(len(item.content) for item in result), 12)
        self.assertLessEqual(len(result), 2)

    def test_empty_or_non_text_messages_are_ignored(self):
        self.assertEqual(
            select_user_context([
                {"role": "user", "content": ""},
                {"role": "user", "content": {"text": "not plain text"}},
                {"role": "assistant", "content": "not user-authored"},
            ]),
            [],
        )

    def test_capture_input_labels_source_and_does_not_include_assistant_text(self):
        result = build_capture_input([
            {"role": "user", "content": "我决定继续维护 AI-System。"},
            {"role": "assistant", "content": "用户决定使用云服务。"},
        ])
        self.assertIn("全部来自用户消息", result)
        self.assertIn("我决定继续维护 AI-System。", result)
        self.assertNotIn("用户决定使用云服务", result)

    def test_invalid_limits_return_empty_context(self):
        messages = [{"role": "user", "content": "长期偏好"}]
        self.assertEqual(select_user_context(messages, max_messages=0), [])
        self.assertEqual(select_user_context(messages, max_chars=0), [])


if __name__ == "__main__":
    unittest.main()
