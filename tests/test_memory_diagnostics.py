import unittest
from unittest.mock import patch

from memory.memory_diagnostics import (
    build_diagnostic_report,
    diagnose_memory,
)


class TestMemoryDiagnostics(unittest.TestCase):

    def test_error_is_enriched_with_location_and_recommendation(self):
        raw_report = {
            "checks": [],
            "issues": [
                {
                    "level": "error",
                    "id": "memory-001",
                    "problem": "正文哈希与向量索引不一致，向量可能已过期",
                    "path": "/tmp/memory/preferences/example.md",
                    "check": "向量记录一致性",
                }
            ],
            "error_count": 1,
            "warning_count": 0,
        }

        report = build_diagnostic_report(raw_report)

        self.assertEqual(report["issue_count"], 1)
        issue = report["issues"][0]

        self.assertEqual(issue["severity"], "error")
        self.assertEqual(issue["memory_id"], "memory-001")
        self.assertEqual(
            issue["path"],
            "/tmp/memory/preferences/example.md",
        )
        self.assertTrue(issue["recommendation"])
        self.assertIsInstance(issue["repairable"], bool)

    def test_duplicate_memory_id_is_not_marked_safe_to_repair(self):
        raw_report = {
            "checks": [],
            "issues": [
                {
                    "level": "error",
                    "id": "memory-002",
                    "problem": "多个 Markdown 文件使用相同 ID",
                    "paths": [
                        "/tmp/memory/a.md",
                        "/tmp/memory/b.md",
                    ],
                    "check": "Markdown ID 重复",
                }
            ],
            "error_count": 1,
            "warning_count": 0,
        }

        report = build_diagnostic_report(raw_report)
        issue = report["issues"][0]

        self.assertFalse(issue["repairable"])
        self.assertEqual(len(issue["related_paths"]), 2)

    def test_warning_level_is_preserved(self):
        raw_report = {
            "checks": [],
            "issues": [
                {
                    "level": "warning",
                    "problem": "Markdown 文件缺少 ID",
                    "path": "/tmp/memory/candidates/example.md",
                    "check": "结构化索引覆盖率",
                }
            ],
            "error_count": 0,
            "warning_count": 1,
        }

        report = build_diagnostic_report(raw_report)

        self.assertEqual(report["warning_count"], 1)
        self.assertEqual(report["issues"][0]["severity"], "warning")

    def test_unknown_problem_is_not_assumed_repairable(self):
        raw_report = {
            "checks": [],
            "issues": [
                {
                    "level": "error",
                    "id": "memory-003",
                    "problem": "未知类型的异常问题",
                    "path": "/tmp/memory/unknown.md",
                    "check": "未知检查",
                }
            ],
            "error_count": 1,
            "warning_count": 0,
        }

        report = build_diagnostic_report(raw_report)

        self.assertFalse(report["issues"][0]["repairable"])

    @patch("memory.memory_diagnostics.run_consistency_check")
    def test_diagnose_memory_uses_existing_checker(self, mock_check):
        mock_check.return_value = {
            "checks": [],
            "issues": [],
            "error_count": 0,
            "warning_count": 0,
        }

        report = diagnose_memory(
            memory_root="/tmp/memory",
            core_db_path="/tmp/memory/index/memory.sqlite",
            embedding_db_path="/tmp/memory/index/embeddings.sqlite",
        )

        mock_check.assert_called_once_with(
            "/tmp/memory",
            "/tmp/memory/index/memory.sqlite",
            "/tmp/memory/index/embeddings.sqlite",
        )
        self.assertEqual(report["issue_count"], 0)

    def test_diagnose_memory_preserves_checker_exception(self):
        with patch(
            "memory.memory_diagnostics.run_consistency_check",
            side_effect=RuntimeError("模拟检查失败"),
        ):
            report = diagnose_memory()

        self.assertEqual(report["error_count"], 1)
        self.assertEqual(report["issue_count"], 1)
        self.assertIn("模拟检查失败", report["issues"][0]["problem"])
        self.assertFalse(report["issues"][0]["repairable"])


if __name__ == "__main__":
    unittest.main()
