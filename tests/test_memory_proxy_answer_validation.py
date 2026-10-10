import io
import json
import unittest
from unittest.mock import patch

import memory_proxy


class TestMemoryProxyAnswerValidation(unittest.TestCase):

    def setUp(self):
        self.handler = memory_proxy.MemoryProxyHandler.__new__(
            memory_proxy.MemoryProxyHandler
        )
        self.responses = []
        self.handler.send_json = lambda status, data: (
            self.responses.append((status, data))
        )
        self.upstream = io.StringIO(json.dumps({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "用户之前计划学习俄语。"
                }
            }]
        }))
        self.question = "我们之前讨论过什么计划？"
        self.evidence = "用户计划继续学习俄语。"

    def get_answer(self):
        return self.responses[-1][1]["choices"][0]["message"]["content"]

    @patch("memory_proxy.call_writer")
    @patch("memory_proxy.validate_answer")
    def test_pass_preserves_answer(self, validate, writer):
        validate.return_value = {
            "verdict": "PASS",
            "reason": "证据支持回答"
        }

        self.handler.handle_non_stream(
            self.upstream, "test-model", self.question,
            self.evidence, True
        )

        self.assertEqual(self.responses[-1][0], 200)
        self.assertEqual(self.get_answer(), "用户之前计划学习俄语。")
        validate.assert_called_once_with(self.evidence, "用户之前计划学习俄语。")
        writer.assert_called_once_with(self.question)

    @patch("memory_proxy.call_writer")
    @patch("memory_proxy.validate_answer")
    def test_flag_replaces_unsupported_answer(self, validate, writer):
        validate.return_value = {
            "verdict": "FLAG",
            "reason": "存在无证据支持的个人事实"
        }

        self.handler.handle_non_stream(
            self.upstream, "test-model", self.question,
            self.evidence, True
        )

        self.assertEqual(self.responses[-1][0], 200)
        self.assertEqual(
            self.get_answer(),
            "根据本次检索到的记忆，无法确认这个问题的答案。"
        )
        writer.assert_called_once_with(self.question)

    @patch("memory_proxy.call_writer")
    @patch("memory_proxy.validate_answer")
    def test_missing_evidence_refuses_to_guess(self, validate, writer):
        self.handler.handle_non_stream(
            self.upstream, "test-model", self.question, "", False
        )

        self.assertEqual(self.responses[-1][0], 200)
        self.assertEqual(
            self.get_answer(),
            "我在现有记忆中没有找到足够的证据来可靠回答这个问题。"
        )
        validate.assert_not_called()
        writer.assert_called_once_with(self.question)

    @patch("memory_proxy.call_writer")
    @patch("memory_proxy.validate_answer")
    def test_validation_exception_fails_closed(self, validate, writer):
        validate.side_effect = RuntimeError("模拟校验器故障")

        self.handler.handle_non_stream(
            self.upstream, "test-model", self.question,
            self.evidence, True
        )

        self.assertEqual(self.responses[-1][0], 502)
        self.assertEqual(
            self.responses[-1][1]["error"]["type"],
            "answer_validation_error"
        )
        writer.assert_not_called()

    @patch("memory_proxy.call_writer")
    @patch("memory_proxy.validate_answer")
    def test_general_question_is_not_memory_validated(
        self, validate, writer
    ):
        self.handler.handle_non_stream(
            self.upstream, "test-model", "什么是俄语语法？",
            self.evidence, True
        )

        self.assertEqual(self.responses[-1][0], 200)
        self.assertEqual(self.get_answer(), "用户之前计划学习俄语。")
        validate.assert_not_called()
        writer.assert_called_once_with("什么是俄语语法？")


if __name__ == "__main__":
    unittest.main()
