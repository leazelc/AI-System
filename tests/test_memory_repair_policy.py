import unittest

from memory.memory_repair_policy import (
    REPAIR_CANDIDATES,
    classify_repair_issue,
)


class TestMemoryRepairPolicy(unittest.TestCase):

    def test_known_candidates_require_manual_review(self):
        cases = [
            "正文哈希与向量索引不一致，向量可能已过期",
            "active Markdown 记忆缺少向量记录",
            "向量数据无效：维度错误",
        ]

        for problem in cases:
            with self.subTest(problem=problem):
                result = classify_repair_issue(problem)
                self.assertIn(result["category"], REPAIR_CANDIDATES)
                self.assertTrue(result["manual_review_required"])
                self.assertFalse(result["execution_allowed"])
                self.assertIsNotNone(result["proposed_operation"])

    def test_duplicate_ids_are_manual_only(self):
        problems = [
            "多个 Markdown 文件使用相同 ID",
            "多个 active Markdown 文件使用相同 ID",
            "结构化索引中存在多个相同 ID 的记录",
            "向量索引中存在 2 条相同 ID 的记录",
        ]

        for problem in problems:
            with self.subTest(problem=problem):
                result = classify_repair_issue(problem)
                self.assertEqual(result["category"], "manual_review")
                self.assertIsNone(result["proposed_operation"])
                self.assertTrue(result["manual_review_required"])
                self.assertFalse(result["execution_allowed"])

    def test_path_and_source_errors_are_manual_only(self):
        problems = [
            "数据库路径位于记忆目录之外",
            "数据库记录指向的 Markdown 文件不存在",
            "无法读取 Markdown 文件：Permission denied",
            "Markdown 文件缺少有效元数据",
            "Markdown 文件缺少 ID",
            "检查执行失败：RuntimeError",
        ]

        for problem in problems:
            with self.subTest(problem=problem):
                result = classify_repair_issue(problem)
                self.assertEqual(result["category"], "manual_review")
                self.assertTrue(result["manual_review_required"])
                self.assertFalse(result["execution_allowed"])

    def test_unknown_problem_is_never_executable(self):
        result = classify_repair_issue("一个全新的未知故障")

        self.assertEqual(result["category"], "unknown")
        self.assertIsNone(result["proposed_operation"])
        self.assertTrue(result["manual_review_required"])
        self.assertFalse(result["execution_allowed"])

    def test_empty_problem_is_never_executable(self):
        result = classify_repair_issue("")

        self.assertEqual(result["category"], "unknown")
        self.assertTrue(result["manual_review_required"])
        self.assertFalse(result["execution_allowed"])

    def test_candidate_policy_does_not_authorize_execution(self):
        for name, rule in REPAIR_CANDIDATES.items():
            with self.subTest(candidate=name):
                self.assertIn(
                    rule["proposed_operation"],
                    {"create_embedding", "rebuild_embedding"},
                )
                self.assertNotIn("execution_allowed", rule)


if __name__ == "__main__":
    unittest.main()
