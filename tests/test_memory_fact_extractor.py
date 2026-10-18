import unittest

from memory.memory_fact_extractor import (
    FACT_CATEGORIES,
    build_extraction_prompt,
    parse_extracted_facts,
)


class MemoryFactExtractorTests(unittest.TestCase):
    def test_prompt_requires_independent_user_facts(self):
        prompt = build_extraction_prompt("我决定使用 GPT-OSS-20B。")
        self.assertIn("不要把助手的建议当成用户事实", prompt)
        self.assertIn("technical_config", prompt)
        self.assertIn("我决定使用 GPT-OSS-20B。", prompt)

    def test_empty_context_returns_empty_prompt(self):
        self.assertEqual(build_extraction_prompt("  "), "")

    def test_parses_multiple_independent_categories(self):
        output = (
            '{"facts":['
            '{"category":"projects","fact":"正在开发 AI-System"},'
            '{"category":"technical_config","fact":"使用 GPT-OSS-20B"},'
            '{"category":"decisions","fact":"优先本地处理"}]}'
        )
        facts = parse_extracted_facts(output)
        self.assertEqual(len(facts), 3)
        self.assertEqual(
            {fact["category"] for fact in facts},
            {"projects", "technical_config", "decisions"},
        )

    def test_deduplicates_same_category_and_fact(self):
        output = (
            '{"facts":['
            '{"category":"plans","fact":"准备发布新版本"},'
            '{"category":"plans","fact":"准备发布新版本"},'
            '{"category":"projects","fact":"准备发布新版本"}]}'
        )
        facts = parse_extracted_facts(output)
        self.assertEqual(len(facts), 2)

    def test_rejects_invalid_categories_and_empty_facts(self):
        output = (
            '{"facts":['
            '{"category":"unknown","fact":"不应保留"},'
            '{"category":"plans","fact":"   "},'
            '{"category":"plans","fact":"有效计划"}]}'
        )
        facts = parse_extracted_facts(output)
        self.assertEqual(facts, [{"category": "plans", "fact": "有效计划"}])

    def test_invalid_json_fails_closed(self):
        self.assertEqual(parse_extracted_facts("not json"), [])
        self.assertEqual(parse_extracted_facts('{"facts":"not a list"}'), [])

    def test_limits_fact_count_and_length(self):
        output = (
            '{"facts":['
            '{"category":"plans","fact":"A"},'
            '{"category":"plans","fact":"B"},'
            '{"category":"plans","fact":"' + ("x" * 20) + '"}]}'
        )
        self.assertEqual(parse_extracted_facts(output, max_facts=1), [
            {"category": "plans", "fact": "A"}
        ])
        self.assertEqual(parse_extracted_facts(output, max_fact_chars=10), [
            {"category": "plans", "fact": "A"},
            {"category": "plans", "fact": "B"},
        ])

    def test_all_supported_categories_are_explicit(self):
        self.assertEqual(len(FACT_CATEGORIES), 6)



class LocalModelCallTests(unittest.TestCase):
    def test_local_model_success(self):
        import json
        from unittest.mock import MagicMock, patch
        from memory.memory_fact_extractor import extract_facts_with_local_model

        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "choices": [{
                "message": {
                    "content": (
                        '{"facts":[{"category":"projects",'
                        '"fact":"正在开发 AI-System"}]}'
                    )
                }
            }]
        }).encode("utf-8")

        with patch(
            "memory.memory_fact_extractor.urllib.request.urlopen",
            return_value=response,
        ) as urlopen:
            facts = extract_facts_with_local_model("我正在开发 AI-System。")

        self.assertEqual(facts, [
            {"category": "projects", "fact": "正在开发 AI-System"}
        ])
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 120)

    def test_empty_context_skips_network(self):
        from unittest.mock import patch
        from memory.memory_fact_extractor import extract_facts_with_local_model

        with patch(
            "memory.memory_fact_extractor.urllib.request.urlopen"
        ) as urlopen:
            self.assertEqual(extract_facts_with_local_model(" "), [])
            urlopen.assert_not_called()

    def test_network_failure_raises_safe_error(self):
        from unittest.mock import patch
        from urllib.error import URLError
        from memory.memory_fact_extractor import (
            FactExtractionError,
            extract_facts_with_local_model,
        )

        with patch(
            "memory.memory_fact_extractor.urllib.request.urlopen",
            side_effect=URLError("simulated"),
        ):
            with self.assertRaises(FactExtractionError):
                extract_facts_with_local_model("我正在开发 AI-System。")

    def test_invalid_model_json_raises_safe_error(self):
        import json
        from unittest.mock import MagicMock, patch
        from memory.memory_fact_extractor import (
            FactExtractionError,
            extract_facts_with_local_model,
        )

        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "choices": [{"message": {"content": "not JSON"}}]
        }).encode("utf-8")

        with patch(
            "memory.memory_fact_extractor.urllib.request.urlopen",
            return_value=response,
        ):
            with self.assertRaises(FactExtractionError):
                extract_facts_with_local_model("我正在开发 AI-System。")


class BatchDeduplicationTests(unittest.TestCase):
    def test_batch_deduplication(self):
        from memory.memory_fact_extractor import deduplicate_facts

        facts = [
            {"category": "projects", "fact": " 正在开发   AI-System "},
            {"category": "projects", "fact": "正在开发 AI-System"},
            {"category": "plans", "fact": "正在开发 AI-System"},
        ]
        self.assertEqual(deduplicate_facts(facts), [
            {"category": "projects", "fact": "正在开发 AI-System"},
            {"category": "plans", "fact": "正在开发 AI-System"},
        ])

    def test_batch_deduplication_is_case_insensitive(self):
        from memory.memory_fact_extractor import deduplicate_facts

        facts = [
            {"category": "technical_config", "fact": "GPT-OSS-20B"},
            {"category": "technical_config", "fact": "gpt-oss-20b"},
        ]
        self.assertEqual(len(deduplicate_facts(facts)), 1)

    def test_batch_deduplication_rejects_invalid_items(self):
        from memory.memory_fact_extractor import deduplicate_facts

        facts = [
            None,
            {"category": "unknown", "fact": "invalid"},
            {"category": "plans", "fact": "   "},
            {"category": "plans", "fact": "valid plan"},
        ]
        self.assertEqual(deduplicate_facts(facts), [
            {"category": "plans", "fact": "valid plan"}
        ])

    def test_batch_deduplication_respects_limit(self):
        from memory.memory_fact_extractor import deduplicate_facts

        facts = [
            {"category": "plans", "fact": "A"},
            {"category": "plans", "fact": "B"},
        ]
        self.assertEqual(deduplicate_facts(facts, max_facts=1), [
            {"category": "plans", "fact": "A"}
        ])


class ExtractFactBatchTests(unittest.TestCase):
    def test_extract_fact_batch_deduplicates_model_output(self):
        from unittest.mock import patch
        from memory.memory_fact_extractor import extract_fact_batch

        model_facts = [
            {"category": "projects", "fact": "正在开发 AI-System"},
            {"category": "projects", "fact": " 正在开发   AI-System "},
            {"category": "plans", "fact": "下周测试"},
        ]

        with patch(
            "memory.memory_fact_extractor.extract_facts_with_local_model",
            return_value=model_facts,
        ):
            result = extract_fact_batch("测试上下文")

        self.assertEqual(result, [
            {"category": "projects", "fact": "正在开发 AI-System"},
            {"category": "plans", "fact": "下周测试"},
        ])

    def test_extract_fact_batch_empty_context(self):
        from unittest.mock import patch
        from memory.memory_fact_extractor import extract_fact_batch

        with patch(
            "memory.memory_fact_extractor.extract_facts_with_local_model",
            return_value=[],
        ) as extract:
            self.assertEqual(extract_fact_batch(""), [])
            extract.assert_called_once_with("")

    def test_extract_fact_batch_propagates_model_failure(self):
        from unittest.mock import patch
        from memory.memory_fact_extractor import (
            FactExtractionError,
            extract_fact_batch,
        )

        with patch(
            "memory.memory_fact_extractor.extract_facts_with_local_model",
            side_effect=FactExtractionError("simulated"),
        ):
            with self.assertRaises(FactExtractionError):
                extract_fact_batch("测试上下文")


if __name__ == "__main__":
    unittest.main()
