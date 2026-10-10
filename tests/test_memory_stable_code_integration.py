import copy
import unittest

from memory.memory_diagnostics import build_diagnostic_report
from memory.memory_repair_preview import build_repair_preview


class TestStableCodeIntegration(unittest.TestCase):

    def test_embedding_hash_mismatch_flows_through_diagnostics_and_preview(self):
        raw_report = {
            "checks": [],
            "issues": [
                {
                    "code": "EMBEDDING_HASH_MISMATCH",
                    "level": "error",
                    "id": "integration-memory-001",
                    "problem": "正文哈希与向量索引不一致，向量可能已过期",
                    "path": "/tmp/memory/preferences/example.md",
                    "check": "向量记录一致性",
                }
            ],
            "error_count": 1,
            "warning_count": 0,
        }

        original_report = copy.deepcopy(raw_report)

        # Stage 1: Build the diagnostic report.
        diagnostic_report = build_diagnostic_report(raw_report)
        diagnostic_issue = diagnostic_report["issues"][0]

        self.assertEqual(
            diagnostic_issue["code"],
            "EMBEDDING_HASH_MISMATCH",
        )
        self.assertTrue(diagnostic_issue["repairable"])
        self.assertEqual(
            diagnostic_issue["memory_id"],
            "integration-memory-001",
        )

        # Stage 2: Build a repair preview without executing repairs.
        preview = build_repair_preview(diagnostic_report)

        self.assertEqual(len(preview["plans"]), 1)
        plan = preview["plans"][0]

        self.assertEqual(
            plan["memory_id"],
            "integration-memory-001",
        )
        self.assertEqual(plan["risk"], "medium")
        self.assertTrue(plan["manual_review_required"])
        self.assertFalse(plan["executed"])
        self.assertTrue(preview["preview_only"])
        self.assertTrue(preview["read_only"])

        # Neither stage should mutate the original report.
        self.assertEqual(raw_report, original_report)

    def test_unknown_code_stays_conservative_through_diagnostics_and_preview(self):
        raw_report = {
            "checks": [],
            "issues": [
                {
                    "code": "UNRECOGNIZED_FUTURE_CODE",
                    "level": "error",
                    "id": "integration-memory-002",
                    "problem": "正文哈希与向量索引不一致",
                    "path": "/tmp/memory/preferences/unknown.md",
                }
            ],
            "error_count": 1,
            "warning_count": 0,
        }

        diagnostic_report = build_diagnostic_report(raw_report)
        issue = diagnostic_report["issues"][0]

        self.assertEqual(issue["code"], "UNRECOGNIZED_FUTURE_CODE")
        self.assertFalse(issue["repairable"])

        preview = build_repair_preview(diagnostic_report)
        plan = preview["plans"][0]

        self.assertEqual(plan["risk"], "high")
        self.assertTrue(plan["manual_review_required"])
        self.assertFalse(plan["executed"])
        self.assertTrue(preview["preview_only"])
        self.assertTrue(preview["read_only"])


if __name__ == "__main__":
    unittest.main()
