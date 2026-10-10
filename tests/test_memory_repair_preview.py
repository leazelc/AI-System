import unittest

from memory.memory_repair_preview import (
    build_repair_preview,
    format_repair_preview,
)


class TestMemoryRepairPreview(unittest.TestCase):

    def test_clean_report_produces_empty_preview(self):
        report = {
            "checks": [],
            "issues": [],
            "issue_count": 0,
            "error_count": 0,
            "warning_count": 0,
            "repairable_count": 0,
            "read_only": True,
        }

        preview = build_repair_preview(report)

        self.assertEqual(preview["plan_count"], 0)
        self.assertEqual(preview["plans"], [])
        self.assertTrue(preview["preview_only"])
        self.assertFalse(preview["executed"])

    def test_stale_embedding_produces_rebuild_preview(self):
        report = {
            "checks": [],
            "issues": [
                {
                    "severity": "error",
                    "memory_id": "memory-101",
                    "problem": "正文哈希与向量索引不一致，向量可能已过期",
                    "related_paths": ["/tmp/memory/preferences/example.md"],
                    "check": "向量记录一致性",
                    "repairable": True,
                    "recommendation": "确认正文后重新生成向量。",
                }
            ],
            "issue_count": 1,
            "error_count": 1,
            "warning_count": 0,
            "repairable_count": 1,
            "read_only": True,
        }

        preview = build_repair_preview(report)
        plan = preview["plans"][0]

        self.assertEqual(preview["plan_count"], 1)
        self.assertEqual(plan["memory_id"], "memory-101")
        self.assertEqual(plan["risk"], "medium")
        self.assertIn("向量", plan["proposed_action"])
        self.assertFalse(plan["executed"])

    def test_duplicate_ids_require_manual_review(self):
        report = {
            "checks": [],
            "issues": [
                {
                    "severity": "error",
                    "memory_id": "memory-202",
                    "problem": "多个 Markdown 文件使用相同 ID",
                    "related_paths": [
                        "/tmp/memory/a.md",
                        "/tmp/memory/b.md",
                    ],
                    "check": "Markdown ID 重复",
                    "repairable": False,
                    "recommendation": "人工确认正确的 ID 分配。",
                }
            ],
            "issue_count": 1,
            "error_count": 1,
            "warning_count": 0,
            "repairable_count": 0,
            "read_only": True,
        }

        preview = build_repair_preview(report)
        plan = preview["plans"][0]

        self.assertEqual(plan["risk"], "high")
        self.assertTrue(plan["manual_review_required"])
        self.assertIn("人工", plan["proposed_action"])

    def test_unknown_issue_is_never_assumed_safe(self):
        report = {
            "checks": [],
            "issues": [
                {
                    "severity": "error",
                    "memory_id": "memory-303",
                    "problem": "未知类型的问题",
                    "related_paths": ["/tmp/memory/unknown.md"],
                    "check": "未知检查",
                    "repairable": True,
                    "recommendation": "需要检查。",
                }
            ],
            "issue_count": 1,
            "error_count": 1,
            "warning_count": 0,
            "repairable_count": 1,
            "read_only": True,
        }

        preview = build_repair_preview(report)
        plan = preview["plans"][0]

        self.assertEqual(plan["risk"], "high")
        self.assertTrue(plan["manual_review_required"])

    def test_preview_does_not_mutate_input_report(self):
        issue = {
            "severity": "error",
            "memory_id": "memory-404",
            "problem": "向量数据无效",
            "related_paths": ["/tmp/memory/example.md"],
            "check": "向量记录一致性",
            "repairable": True,
            "recommendation": "检查向量。",
        }
        report = {
            "checks": [],
            "issues": [issue],
            "issue_count": 1,
            "error_count": 1,
            "warning_count": 0,
            "repairable_count": 1,
            "read_only": True,
        }
        original_problem = issue["problem"]
        original_paths = list(issue["related_paths"])

        build_repair_preview(report)

        self.assertEqual(issue["problem"], original_problem)
        self.assertEqual(issue["related_paths"], original_paths)
        self.assertTrue(report["read_only"])


    def test_format_preview_shows_plan_and_read_only_status(self):
        preview = {
            "plan_count": 1,
            "preview_only": True,
            "executed": False,
            "read_only": True,
            "plans": [
                {
                    "memory_id": "memory-101",
                    "problem": "向量数据无效",
                    "source_check": "向量记录一致性",
                    "related_paths": ["/tmp/memory/example.md"],
                    "proposed_action": "检查向量并考虑重新生成。",
                    "risk": "medium",
                    "manual_review_required": False,
                    "executed": False,
                }
            ],
        }

        output = format_repair_preview(preview)

        self.assertIn("修复预览", output)
        self.assertIn("memory-101", output)
        self.assertIn("/tmp/memory/example.md", output)
        self.assertIn("实际执行：否", output)
        self.assertIn("执行状态：尚未执行", output)

    def test_format_empty_preview(self):
        preview = {
            "plan_count": 0,
            "plans": [],
            "preview_only": True,
            "executed": False,
            "read_only": True,
        }

        output = format_repair_preview(preview)

        self.assertIn("没有需要审查的修复方案", output)
        self.assertIn("数据修改：未执行", output)


    def test_repairable_flag_never_triggers_execution(self):
        report = {
            "checks": [],
            "issues": [
                {
                    "severity": "error",
                    "memory_id": "memory-505",
                    "problem": "正文哈希与向量索引不一致",
                    "related_paths": ["/tmp/memory/example.md"],
                    "check": "向量记录一致性",
                    "repairable": True,
                    "recommendation": "重新生成向量。",
                }
            ],
            "issue_count": 1,
            "error_count": 1,
            "warning_count": 0,
            "repairable_count": 1,
            "read_only": True,
        }

        preview = build_repair_preview(report)

        self.assertTrue(preview["preview_only"])
        self.assertTrue(preview["read_only"])
        self.assertFalse(preview["executed"])

        for plan in preview["plans"]:
            self.assertFalse(plan["executed"])
            self.assertNotIn("execute", plan)
            self.assertNotIn("execute_function", plan)

    def test_duplicate_active_ids_require_manual_review(self):
        report = {
            "checks": [],
            "issues": [
                {
                    "severity": "error",
                    "memory_id": "memory-606",
                    "problem": "多个 active Markdown 文件使用相同 ID",
                    "related_paths": [
                        "/tmp/memory/active/a.md",
                        "/tmp/memory/active/b.md",
                    ],
                    "check": "Markdown ID 重复",
                    "repairable": True,
                    "recommendation": "人工检查。",
                }
            ],
            "issue_count": 1,
            "error_count": 1,
            "warning_count": 0,
            "repairable_count": 1,
            "read_only": True,
        }

        preview = build_repair_preview(report)
        plan = preview["plans"][0]

        self.assertEqual(plan["risk"], "high")
        self.assertTrue(plan["manual_review_required"])
        self.assertFalse(plan["executed"])


    def test_all_known_repair_plans_require_manual_review(self):
        problems = [
            "多个 Markdown 文件使用相同 ID",
            "正文哈希与向量索引不一致",
            "缺少向量记录",
            "向量数据无效",
            "Markdown 文件未登记在结构化索引中",
            "未知类型的问题",
        ]

        report = {
            "issues": [
                {
                    "problem": problem,
                    "repairable": True,
                }
                for problem in problems
            ]
        }

        preview = build_repair_preview(report)

        self.assertEqual(preview["plan_count"], len(problems))

        for plan in preview["plans"]:
            with self.subTest(problem=plan["problem"]):
                self.assertTrue(plan["manual_review_required"])
                self.assertFalse(plan["executed"])


    def test_stable_code_selects_plan_independently_of_problem_text(self):
        report = {
            "issues": [
                {
                    "code": "EMBEDDING_HASH_MISMATCH",
                    "problem": "描述文字已经改变",
                    "memory_id": "memory-code-003",
                    "related_paths": ["/tmp/memory/preferences/example.md"],
                    "severity": "error",
                }
            ]
        }

        preview = build_repair_preview(report)
        plan = preview["plans"][0]

        self.assertEqual(plan["memory_id"], "memory-code-003")
        self.assertEqual(plan["risk"], "medium")
        self.assertTrue(plan["manual_review_required"])
        self.assertFalse(plan["executed"])


if __name__ == "__main__":
    unittest.main()
