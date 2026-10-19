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


    def test_stream_missing_done_returns_502(self):
        import io

        self.upstream = io.BytesIO(
            b'data: {"choices":[{"delta":{"content":"partial"}}]}\\n\\n'
        )

        with self.patch.object(
            memory_proxy, "is_personal_memory_question",
            return_value=False,
        ):
            self.run_stream()

        self.assertEqual(self.json_errors[0][0], 502)
        self.assertEqual(self.status_codes, [])
        self.assertTrue(self.upstream.closed)

    def test_stream_read_error_returns_502_and_closes_upstream(self):
        class BrokenUpstream:
            closed = False

            def readline(self):
                raise OSError("simulated upstream read failure")

            def close(self):
                self.closed = True

        upstream = BrokenUpstream()
        self.handler.handle_stream(
            upstream, "test-model", "什么是机器学习？", "", False
        )

        self.assertEqual(self.json_errors[0][0], 502)
        self.assertEqual(self.status_codes, [])
        self.assertTrue(upstream.closed)

    def test_stream_client_disconnect_does_not_send_second_response(self):
        class BrokenOutput:
            def write(self, data):
                raise BrokenPipeError("simulated client disconnect")

            def flush(self):
                raise BrokenPipeError("simulated client disconnect")

        self.handler.wfile = BrokenOutput()

        with self.patch.object(
            memory_proxy, "is_personal_memory_question",
            return_value=False,
        ):
            self.run_stream()

        self.assertEqual(self.status_codes, [200])
        self.assertEqual(self.json_errors, [])
        self.assertTrue(self.upstream.closed)


class TestMemoryProxyHTTPInputSafety(unittest.TestCase):
    """新增 HTTP 请求边界安全测试。"""

    def make_handler(self, body, content_length=None):
        handler = memory_proxy.MemoryProxyHandler.__new__(
            memory_proxy.MemoryProxyHandler
        )
        handler.path = "/v1/chat/completions"
        raw = body if isinstance(body, bytes) else body.encode("utf-8")
        length = str(len(raw)) if content_length is None else str(content_length)
        handler.headers = {"Content-Length": length}
        handler.rfile = io.BytesIO(raw)
        handler.responses = []
        handler.send_json = lambda status, payload: handler.responses.append(
            (status, payload)
        )
        return handler

    def test_invalid_json_returns_400(self):
        handler = self.make_handler("{invalid json")
        handler.do_POST()
        self.assertEqual(handler.responses[0][0], 400)

    def test_json_array_returns_400(self):
        handler = self.make_handler('[]')
        handler.do_POST()
        self.assertEqual(handler.responses[0][0], 400)

    def test_messages_string_returns_400(self):
        handler = self.make_handler(
            json.dumps({"messages": "not-a-list"})
        )
        handler.do_POST()
        self.assertEqual(handler.responses[0][0], 400)

    def test_non_object_message_returns_400(self):
        handler = self.make_handler(
            json.dumps({"messages": ["invalid-message"]})
        )
        handler.do_POST()
        self.assertEqual(handler.responses[0][0], 400)

    def test_oversized_body_returns_413_without_reading(self):
        handler = self.make_handler("{}", content_length=10 * 1024 * 1024 + 1)
        handler.do_POST()
        self.assertEqual(handler.responses[0][0], 413)

    def test_internal_exception_details_are_not_returned(self):
        handler = self.make_handler(
            json.dumps({
                "messages": [
                    {"role": "user", "content": "hello"}
                ]
            })
        )
        with patch.object(
            memory_proxy,
            "build_memory_context",
            side_effect=RuntimeError("SECRET /private/internal/path"),
        ):
            handler.do_POST()

        status, payload = handler.responses[0]
        self.assertEqual(status, 500)
        self.assertEqual(payload["error"], "Internal server error")
        self.assertNotIn("SECRET", json.dumps(payload))
        self.assertNotIn("/private/internal/path", json.dumps(payload))

    def test_upstream_http_error_body_is_hidden(self):
        handler = self.make_handler(
            json.dumps({
                "messages": [
                    {"role": "user", "content": "hello"}
                ]
            })
        )
        upstream_error = __import__("urllib.error", fromlist=["HTTPError"]).HTTPError(
            "http://127.0.0.1:1234/v1/chat/completions",
            503,
            "Service Unavailable",
            {},
            io.BytesIO(b"SECRET upstream traceback /internal/path"),
        )

        with patch.object(
            memory_proxy,
            "build_memory_context",
            return_value=("", [], False),
        ), patch.object(
            memory_proxy.urllib.request,
            "urlopen",
            side_effect=upstream_error,
        ):
            handler.do_POST()

        status, payload = handler.responses[0]
        self.assertEqual(status, 503)
        self.assertEqual(payload["error"], "Upstream model request failed")
        self.assertNotIn("SECRET", json.dumps(payload))
        self.assertNotIn("/internal/path", json.dumps(payload))

    def test_upstream_http_error_status_is_preserved(self):
        handler = self.make_handler(
            json.dumps({
                "messages": [
                    {"role": "user", "content": "hello"}
                ]
            })
        )
        upstream_error = __import__("urllib.error", fromlist=["HTTPError"]).HTTPError(
            "http://127.0.0.1:1234/v1/chat/completions",
            429,
            "Too Many Requests",
            {},
            io.BytesIO(b"rate limit details"),
        )

        with patch.object(
            memory_proxy,
            "build_memory_context",
            return_value=("", [], False),
        ), patch.object(
            memory_proxy.urllib.request,
            "urlopen",
            side_effect=upstream_error,
        ):
            handler.do_POST()

        self.assertEqual(handler.responses[0][0], 429)

    def test_upstream_connection_failure_returns_502_without_details(self):
        handler = self.make_handler(json.dumps({
            "messages": [{"role": "user", "content": "hello"}]
        }))

        with patch.object(
            memory_proxy,
            "build_memory_context",
            return_value=("", [], False),
        ), patch.object(
            memory_proxy.urllib.request,
            "urlopen",
            side_effect=__import__(
                "urllib.error", fromlist=["URLError"]
            ).URLError("SECRET /private/internal/path"),
        ):
            handler.do_POST()

        status, payload = handler.responses[0]
        self.assertEqual(status, 502)
        self.assertEqual(
            payload["error"],
            "Upstream model unavailable or timed out",
        )
        self.assertNotIn("SECRET", json.dumps(payload))
        self.assertNotIn("/private/internal/path", json.dumps(payload))

    def test_upstream_timeout_returns_502_without_details(self):
        handler = self.make_handler(json.dumps({
            "messages": [{"role": "user", "content": "hello"}]
        }))

        with patch.object(
            memory_proxy,
            "build_memory_context",
            return_value=("", [], False),
        ), patch.object(
            memory_proxy.urllib.request,
            "urlopen",
            side_effect=TimeoutError(
                "SECRET /private/internal/path"
            ),
        ):
            handler.do_POST()

        status, payload = handler.responses[0]
        self.assertEqual(status, 502)
        self.assertEqual(
            payload["error"],
            "Upstream model unavailable or timed out",
        )
        self.assertNotIn("SECRET", json.dumps(payload))
        self.assertNotIn("/private/internal/path", json.dumps(payload))

    def test_string_stream_value_returns_400(self):
        handler = self.make_handler(json.dumps({
            "model": "test-model",
            "stream": "false",
            "messages": [{"role": "user", "content": "hello"}],
        }))

        with patch.object(
            memory_proxy, "build_memory_context",
            return_value=("", [], False),
        ):
            handler.do_POST()

        self.assertEqual(handler.responses[0][0], 400)

    def test_non_boolean_stream_value_returns_400(self):
        handler = self.make_handler(json.dumps({
            "stream": 1,
            "messages": [{"role": "user", "content": "hello"}],
        }))

        handler.do_POST()
        self.assertEqual(handler.responses[0][0], 400)

    def test_non_object_content_part_returns_400(self):
        handler = self.make_handler(json.dumps({
            "messages": [{
                "role": "user",
                "content": ["invalid-part"],
            }],
        }))

        handler.do_POST()
        self.assertEqual(handler.responses[0][0], 400)

    def test_text_content_part_with_non_string_text_returns_400(self):
        handler = self.make_handler(json.dumps({
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": 123}],
            }],
        }))

        handler.do_POST()
        self.assertEqual(handler.responses[0][0], 400)

    def test_valid_multimodal_content_part_is_not_rejected(self):
        handler = self.make_handler(json.dumps({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "请描述这张图片"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.invalid/image.png"},
                    },
                ],
            }],
        }))

        with patch.object(
            memory_proxy, "build_memory_context",
            return_value=("", [], False),
        ), patch.object(
            memory_proxy.urllib.request,
            "urlopen",
            side_effect=RuntimeError("stop before upstream"),
        ):
            handler.do_POST()

        # 输入结构已通过校验；随后模拟的上游异常产生 500。
        self.assertEqual(handler.responses[0][0], 500)
        self.assertEqual(
            handler.responses[0][1]["error"],
            "Internal server error",
        )

class TestMemoryRetrievalOrchestratorIntegration(unittest.TestCase):
    def setUp(self):
        self.question = "我们之前讨论过什么计划？"
        self.memory = {
            "id": "integration-memory-1",
            "content": "用户计划继续学习俄语。",
            "score": 0.9,
            "metadata": {"status": "active"},
        }
        self.decision = {
            "evidence": [{
                "content": self.memory["content"],
                "score": 0.9,
            }],
            "answerable_from_retrieval": True,
        }

    def test_orchestrated_memories_reach_evidence_gate(self):
        with patch(
            "memory_proxy.retrieve_memories",
            return_value={"memories": [self.memory]},
        ) as orchestrator, patch(
            "memory_proxy.search"
        ) as legacy_search, patch(
            "memory_proxy.evaluate_evidence",
            return_value=self.decision,
        ) as evidence_gate:
            context, evidence, answerable = (
                memory_proxy.build_memory_context(
                    self.question, return_details=True
                )
            )

        orchestrator.assert_called_once_with(
            self.question,
            query_limit=memory_proxy.MEMORY_QUERY_LIMIT,
            candidate_limit=10,
        )
        legacy_search.assert_not_called()
        evidence_gate.assert_called_once_with(
            self.question,
            [self.memory],
            min_score=memory_proxy._memory_min_score(),
        )
        self.assertTrue(answerable)
        self.assertEqual(evidence, self.memory["content"])
        self.assertIn(self.memory["content"], context)

    def test_orchestrator_failure_falls_back_through_evidence_gate(self):
        fallback_memories = [self.memory]
        with patch(
            "memory_proxy.retrieve_memories",
            side_effect=RuntimeError("simulated retrieval failure"),
        ), patch(
            "memory_proxy.search",
            return_value=fallback_memories,
        ) as legacy_search, patch(
            "memory_proxy.evaluate_evidence",
            return_value=self.decision,
        ) as evidence_gate:
            context, evidence, answerable = (
                memory_proxy.build_memory_context(
                    self.question, return_details=True
                )
            )

        legacy_search.assert_called_once_with(self.question, limit=10)
        evidence_gate.assert_called_once_with(
            self.question,
            fallback_memories,
            min_score=memory_proxy._memory_min_score(),
        )
        self.assertTrue(answerable)
        self.assertEqual(evidence, self.memory["content"])
        self.assertIn(self.memory["content"], context)
if __name__ == "__main__":
    unittest.main()
