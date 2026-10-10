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


class TestMemoryProxyStreamingValidation(unittest.TestCase):
    def setUp(self):
        import io
        from unittest.mock import patch

        self.io = io
        self.patch = patch

        self.handler_class = next(
            obj for obj in vars(memory_proxy).values()
            if isinstance(obj, type) and hasattr(obj, "handle_stream")
        )

        self.handler = self.handler_class.__new__(self.handler_class)
        self.output = io.BytesIO()
        self.handler.wfile = self.output

        self.status_codes = []
        self.headers = []
        self.json_errors = []

        self.handler.send_response = lambda status: self.status_codes.append(status)
        self.handler.send_header = lambda key, value: self.headers.append((key, value))
        self.handler.end_headers = lambda: None
        self.handler.send_json = lambda status, body: self.json_errors.append(
            (status, body)
        )

        import json
        payload = json.dumps(
            {"choices": [{"delta": {"content": "测试答案"}}]},
            ensure_ascii=False,
        ).encode("utf-8")
        self.upstream = io.BytesIO(
            b"data: " + payload + b"\n\n"
            + b"data: [DONE]\n\n"
        )

        self.writer_patch = patch(
            "memory_proxy.call_writer"
        ).start()
        self.addCleanup(patch.stopall)

    def run_stream(self, question="我之前说过什么？",
                   evidence="已有记忆证据", available=True):
        self.handler.handle_stream(
            self.upstream,
            "test-model",
            question,
            evidence,
            available,
        )
        return self.output.getvalue().decode("utf-8")

    def test_pass_preserves_original_stream_answer(self):
        with self.patch.object(
            memory_proxy, "is_personal_memory_question", return_value=True
        ), self.patch.object(
            memory_proxy, "validate_answer",
            return_value={"verdict": "PASS", "reason": "有证据支持"}
        ) as validator:
            result = self.run_stream()

        validator.assert_called_once_with("已有记忆证据", "测试答案")
        self.assertIn("测试答案", result)
        self.assertEqual(self.status_codes, [200])

    def test_flag_blocks_original_answer(self):
        with self.patch.object(
            memory_proxy, "is_personal_memory_question", return_value=True
        ), self.patch.object(
            memory_proxy, "validate_answer",
            return_value={"verdict": "FLAG", "reason": "无证据支持"}
        ):
            result = self.run_stream()

        self.assertNotIn("测试答案", result)
        self.assertIn("无法确认", result)

    def test_missing_evidence_blocks_answer(self):
        with self.patch.object(
            memory_proxy, "is_personal_memory_question", return_value=True
        ), self.patch.object(
            memory_proxy, "validate_answer"
        ) as validator:
            result = self.run_stream(evidence="", available=False)

        validator.assert_not_called()
        self.assertNotIn("测试答案", result)
        self.assertIn("没有找到足够的证据", result)

    def test_validator_exception_fails_closed(self):
        with self.patch.object(
            memory_proxy, "is_personal_memory_question", return_value=True
        ), self.patch.object(
            memory_proxy, "validate_answer",
            side_effect=RuntimeError("模拟异常")
        ):
            self.run_stream()

        self.assertEqual(self.json_errors[0][0], 502)
        self.assertEqual(self.status_codes, [])
        self.assertEqual(self.output.getvalue(), b"")

    def test_general_question_skips_memory_validation(self):
        with self.patch.object(
            memory_proxy, "is_personal_memory_question", return_value=False
        ), self.patch.object(
            memory_proxy, "validate_answer"
        ) as validator:
            result = self.run_stream(question="什么是机器学习？")

        validator.assert_not_called()
        self.assertIn("测试答案", result)
        self.assertEqual(self.status_codes, [200])
