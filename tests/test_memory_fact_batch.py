import unittest
from unittest.mock import Mock

from memory.memory_fact_batch import write_fact_batch


class MemoryFactBatchTests(unittest.TestCase):
    def setUp(self):
        self.facts = [
            {"category": "preferences", "fact": "偏好本地处理数据"},
            {"category": "projects", "fact": "正在开发本地记忆系统"},
            {"category": "plans", "fact": "计划测试新版本"},
        ]

    def test_processes_each_fact_sequentially(self):
        extractor = Mock(return_value=self.facts)
        writer = Mock(side_effect=[
            {"status": "success", "action": "NEW",
             "memory_ids": ["m1"]},
            {"status": "ignored", "action": "IGNORE"},
            {"status": "success", "action": "NEW",
             "memory_ids": ["m2"]},
        ])

        result = write_fact_batch("sample context", writer, extractor)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["processed"], 3)
        self.assertEqual(result["memory_ids"], ["m1", "m2"])
        self.assertEqual(writer.call_count, 3)
        self.assertIn("preferences", writer.call_args_list[0].args[0])

    def test_passes_each_extracted_category_as_expected_type(self):
        writer = Mock(return_value={
            "status": "success",
            "action": "NEW",
            "memory_ids": ["m1"],
        })
        extractor = Mock(return_value=self.facts)

        result = write_fact_batch("sample context", writer, extractor)

        self.assertEqual(result["status"], "success")
        self.assertEqual(
            [call.kwargs.get("expected_type") for call in writer.call_args_list],
            ["preferences", "projects", "plans"],
        )


    def test_stops_after_index_sync_failure(self):
        writer = Mock(side_effect=[
            {"status": "success", "action": "NEW",
             "memory_ids": ["m1"]},
            {"status": "index_sync_failed", "action": "UPDATE",
             "memory_ids": ["m2"]},
            {"status": "success", "action": "NEW",
             "memory_ids": ["m3"]},
        ])

        result = write_fact_batch(
            "sample context", writer, Mock(return_value=self.facts)
        )

        self.assertEqual(result["status"], "index_sync_failed")
        self.assertEqual(result["processed"], 2)
        self.assertEqual(writer.call_count, 2)
        self.assertEqual(result["memory_ids"], ["m1", "m2"])

    def test_continues_after_ordinary_error(self):
        writer = Mock(side_effect=[
            {"status": "error", "action": "REJECTED"},
            {"status": "success", "action": "NEW",
             "memory_ids": ["m2"]},
        ])

        result = write_fact_batch(
            "sample context", writer,
            Mock(return_value=self.facts[:2])
        )

        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["processed"], 2)
        self.assertEqual(writer.call_count, 2)

    def test_preserves_writer_error_details(self):
        writer = Mock(side_effect=[
            {
                "status": "error",
                "action": "UNKNOWN",
                "error_type": "DecisionOutputError",
                "error": "Updater returned invalid JSON.",
            },
            {"status": "success", "action": "NEW",
             "memory_ids": ["m2"]},
        ])

        result = write_fact_batch(
            "sample context", writer,
            Mock(return_value=self.facts[:2])
        )

        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["results"][0]["status"], "error")
        self.assertEqual(
            result["results"][0]["error_type"], "DecisionOutputError"
        )
        self.assertEqual(
            result["results"][0]["error"], "Updater returned invalid JSON."
        )
        self.assertEqual(result["memory_ids"], ["m2"])

    def test_preserves_raised_exception_details(self):
        writer = Mock(side_effect=[
            RuntimeError("local model unavailable"),
            {"status": "success", "action": "NEW",
             "memory_ids": ["m2"]},
        ])

        result = write_fact_batch(
            "sample context", writer,
            Mock(return_value=self.facts[:2])
        )

        self.assertEqual(result["status"], "partial")
        self.assertEqual(
            result["results"][0]["error_type"], "RuntimeError"
        )
        self.assertEqual(
            result["results"][0]["error"], "local model unavailable"
        )
        self.assertEqual(result["memory_ids"], ["m2"])

    def test_empty_batch_does_not_call_writer(self):
        writer = Mock()
        result = write_fact_batch(
            "sample context", writer, Mock(return_value=[])
        )

        self.assertEqual(result["status"], "empty")
        self.assertEqual(result["processed"], 0)
        writer.assert_not_called()

    def test_extraction_failure_prevents_writes(self):
        writer = Mock()
        extractor = Mock(side_effect=RuntimeError("extraction failed"))

        with self.assertRaisesRegex(RuntimeError, "extraction failed"):
            write_fact_batch("sample context", writer, extractor)

        writer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
