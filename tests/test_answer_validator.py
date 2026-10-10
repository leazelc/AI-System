import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "memory"))

from answer_validator import (
    AnswerValidationError,
    _extract_json,
    validate_answer,
)


class TestAnswerValidator(unittest.TestCase):

    def test_reject_non_string_verdict_values(self):
        for verdict in (None, 1, True, [], {}):
            with self.subTest(verdict=repr(verdict)):
                raw = json.dumps({
                    "verdict": verdict,
                    "reason": "测试理由",
                })
                with self.assertRaises(AnswerValidationError):
                    _extract_json(raw)

    def test_reject_non_object_json_values(self):
        for raw in ("null", "[]", "123", "true", '"text"'):
            with self.subTest(raw=raw):
                with self.assertRaises(AnswerValidationError):
                    _extract_json(raw)

    def test_reject_non_string_reason(self):
        for reason in (None, 123, True, [], {}):
            with self.subTest(reason=repr(reason)):
                raw = json.dumps({
                    "verdict": "PASS",
                    "reason": reason,
                })
                with self.assertRaises(AnswerValidationError):
                    _extract_json(raw)

    def test_reject_unknown_control_token_after_json(self):
        raw = (
            '{"verdict":"PASS","reason":"证据支持"}'
            '<|unknown_token|>'
        )
        with self.assertRaises(AnswerValidationError):
            _extract_json(raw)

    def test_reject_leading_plain_text_before_json(self):
        raw = (
            '答案如下：'
            '{"verdict":"PASS","reason":"证据支持"}'
        )
        with self.assertRaises(AnswerValidationError):
            _extract_json(raw)

    def test_accept_whitespace_around_valid_json(self):
        result = _extract_json(
            '  {"verdict":"PASS","reason":"证据支持"}  '
        )
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["reason"], "证据支持")

    def test_parse_all_valid_verdicts(self):
        for verdict in ("PASS", "FLAG", "UNCERTAIN"):
            with self.subTest(verdict=verdict):
                result = _extract_json(json.dumps({
                    "verdict": verdict,
                    "reason": "测试理由",
                }))
                self.assertEqual(result["verdict"], verdict)

    def test_parse_model_control_tokens(self):
        result = _extract_json(
            '<|channel|>final <|constrain|>JSON<|message|>'
            '{"verdict":"FLAG","reason":"证据不足"}'
        )
        self.assertEqual(result["verdict"], "FLAG")

    def test_reject_invalid_json(self):
        with self.assertRaises(AnswerValidationError):
            _extract_json("不是 JSON")

    def test_reject_json_followed_by_unexpected_text(self):
        with self.assertRaises(AnswerValidationError):
            _extract_json(
                '{"verdict":"PASS","reason":"看似正确"}额外内容'
            )

    def test_reject_multiple_json_objects(self):
        with self.assertRaises(AnswerValidationError):
            _extract_json(
                '{"verdict":"PASS","reason":"第一个"}'
                '{"verdict":"FLAG","reason":"第二个"}'
            )

    def test_allow_trailing_model_control_token(self):
        result = _extract_json(
            '{"verdict":"PASS","reason":"证据支持"}<|end|>'
        )
        self.assertEqual(result["verdict"], "PASS")

    def test_invalid_utf8_api_response_fails_closed(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b"\\xff"

        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(),
        ):
            with self.assertRaises(AnswerValidationError):
                validate_answer("历史计划", "目前无法确认")

    def test_reject_invalid_verdict(self):
        with self.assertRaises(AnswerValidationError):
            _extract_json(
                '{"verdict":"ALLOW","reason":"不合法结果"}'
            )

    def test_reject_missing_reason(self):
        with self.assertRaises(AnswerValidationError):
            _extract_json('{"verdict":"PASS","reason":""}')

    def test_empty_evidence_rejected_before_network(self):
        with self.assertRaises(ValueError):
            validate_answer("", "目前无法确认")

    def test_empty_answer_rejected_before_network(self):
        with self.assertRaises(ValueError):
            validate_answer("只有一项历史计划", " ")

    def test_mocked_successful_validation(self):
        api_response = {
            "choices": [{
                "message": {
                    "content": (
                        '{"verdict":"FLAG",'
                        '"reason":"答案断言了证据未支持的事实"}'
                    )
                }
            }]
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(api_response).encode("utf-8")

        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(),
        ) as mocked:
            result = validate_answer(
                "用户计划未来几天入学。",
                "你已经正式入学。",
            )

        self.assertEqual(result["verdict"], "FLAG")
        mocked.assert_called_once()

        request = mocked.call_args.args[0]
        sent = json.loads(request.data)
        self.assertEqual(sent["temperature"], 0)
        self.assertFalse(sent["stream"])

    def test_invalid_model_response_fails_closed(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"choices":[{"message":{"content":"oops"}}]}'

        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(),
        ):
            with self.assertRaises(AnswerValidationError):
                validate_answer("一项历史计划", "你已经入学。")

    def test_missing_choices_fails_closed(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"unexpected":[]}'

        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(),
        ):
            with self.assertRaises(AnswerValidationError):
                validate_answer("历史计划", "已经完成")

    def test_non_string_model_content_fails_closed(self):
        response_data = {
            "choices": [{
                "message": {
                    "content": {"verdict": "PASS"}
                }
            }]
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(response_data).encode("utf-8")

        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(),
        ):
            with self.assertRaises(AnswerValidationError):
                validate_answer("历史计划", "已经完成")

    def test_zero_timeout_rejected_before_network(self):
        with patch("urllib.request.urlopen") as mocked:
            with self.assertRaises(ValueError):
                validate_answer(
                    "历史计划",
                    "已经完成",
                    timeout=0,
                )

        mocked.assert_not_called()

    def test_invalid_timeout_values_rejected_before_network(self):
        invalid_values = [
            True,
            False,
            0,
            -1,
            float("nan"),
            float("inf"),
            float("-inf"),
        ]

        for timeout in invalid_values:
            with self.subTest(timeout=repr(timeout)):
                with patch("urllib.request.urlopen") as mocked:
                    with self.assertRaises(ValueError):
                        validate_answer(
                            "历史计划",
                            "已经完成",
                            timeout=timeout,
                        )
                    mocked.assert_not_called()

    def test_valid_positive_timeout_accepted_by_validation(self):
        # 参数通过前置检查后，模拟网络层失败，确认没有被误判为 ValueError。
        with patch(
            "urllib.request.urlopen",
            side_effect=TimeoutError("test timeout"),
        ):
            with self.assertRaises(AnswerValidationError):
                validate_answer(
                    "历史计划",
                    "已经完成",
                    timeout=0.5,
                )

    def test_response_read_runtime_error_fails_closed(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                raise RuntimeError("simulated read failure")

        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(),
        ):
            with self.assertRaises(AnswerValidationError):
                validate_answer("历史计划", "目前无法确认")

    def test_response_json_array_fails_closed(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b"[]"

        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(),
        ):
            with self.assertRaises(AnswerValidationError):
                validate_answer("历史计划", "目前无法确认")

    def test_response_json_null_fails_closed(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b"null"

        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(),
        ):
            with self.assertRaises(AnswerValidationError):
                validate_answer("历史计划", "目前无法确认")

    def test_response_read_runtime_error_fails_closed(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                raise RuntimeError("simulated read failure")

        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(),
        ):
            with self.assertRaises(AnswerValidationError):
                validate_answer("历史计划", "目前无法确认")

    def test_response_json_array_fails_closed(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b"[]"

        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(),
        ):
            with self.assertRaises(AnswerValidationError):
                validate_answer("历史计划", "目前无法确认")

    def test_response_json_null_fails_closed(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b"null"

        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(),
        ):
            with self.assertRaises(AnswerValidationError):
                validate_answer("历史计划", "目前无法确认")

    def test_response_read_runtime_error_fails_closed(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                raise RuntimeError("simulated read failure")

        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(),
        ):
            with self.assertRaises(AnswerValidationError):
                validate_answer("历史计划", "目前无法确认")

    def test_response_json_array_fails_closed(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b"[]"

        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(),
        ):
            with self.assertRaises(AnswerValidationError):
                validate_answer("历史计划", "目前无法确认")

    def test_response_json_null_fails_closed(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b"null"

        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(),
        ):
            with self.assertRaises(AnswerValidationError):
                validate_answer("历史计划", "目前无法确认")

    def test_network_failure_raises_validation_error(self):
        with patch(
            "urllib.request.urlopen",
            side_effect=TimeoutError("test timeout"),
        ):
            with self.assertRaises(AnswerValidationError):
                validate_answer("一项历史计划", "你已经入学。")


if __name__ == "__main__":
    unittest.main(verbosity=2)
