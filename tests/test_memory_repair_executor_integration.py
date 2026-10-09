import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from memory.memory_repair_executor import execute_repair
from memory.memory_repair_transaction import restore_embedding_repair as real_restore


class TestMemoryRepairExecutorIntegration(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.memory_root = self.root / "memory"
        self.memory_root.mkdir()
        self.source = self.memory_root / "test-memory.md"
        self.source.write_text(
            "---\nid: test-001\nstatus: active\n---\n\nIntegration test memory.\n",
            encoding="utf-8",
        )

        self.index_dir = self.memory_root / "index"
        self.index_dir.mkdir()
        self.embedding_db = self.index_dir / "embeddings.sqlite"

        with sqlite3.connect(self.embedding_db) as conn:
            conn.execute("""
                CREATE TABLE embeddings (
                    id TEXT PRIMARY KEY,
                    path TEXT NOT NULL UNIQUE,
                    content_hash TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    updated_at TEXT
                )
            """)

        self.backup_root = self.root / "backups"
        self.issue = {
            "id": "test-001",
            "memory_id": "test-001",
            "path": str(self.source),
            "problem": "active Markdown 记忆缺少向量记录",
        }

        self.authorized_test_policy = {
            "category": "missing_embedding",
            "proposed_operation": "create_embedding",
            "risk": "medium",
            "manual_review_required": True,
            "execution_allowed": True,
        }

    def tearDown(self):
        self.temp.cleanup()

    def run_executor(self, diagnosis_side_effect, extra_patches=None):
        patches = [
            patch(
                "memory.memory_repair_executor.classify_repair_issue",
                return_value=self.authorized_test_policy,
            ),
            patch(
                "memory.memory_diagnostics.diagnose_memory",
                side_effect=diagnosis_side_effect,
            ),
            patch(
                "memory.embedding_index.get_embedding",
                return_value=[0.1, 0.2],
            ),
            patch(
                "memory.memory_consistency.markdown_content_hash",
                return_value="fresh-test-hash",
            ),
        ]

        if extra_patches:
            patches.extend(extra_patches)

        started = []
        try:
            for item in patches:
                started.append(item)
                item.start()

            return execute_repair(
                self.issue,
                execute=True,
                approved=True,
                memory_root=self.memory_root,
                embedding_db_path=self.embedding_db,
                backup_root=self.backup_root,
            )
        finally:
            for item in reversed(started):
                item.stop()

    def diagnosis_issue(self):
        return {
            "issues": [dict(self.issue)],
            "error_count": 1,
        }

    def clean_diagnosis(self):
        return {"issues": [], "error_count": 0}

    def read_rows(self):
        with sqlite3.connect(self.embedding_db) as conn:
            return conn.execute(
                "SELECT id, path, content_hash, dimension, vector "
                "FROM embeddings"
            ).fetchall()

    def test_successful_repair_writes_and_passes_post_check(self):
        result = self.run_executor([
            self.diagnosis_issue(),
            self.clean_diagnosis(),
        ])

        self.assertTrue(result["success"], result)
        self.assertTrue(result["executed"], result)
        self.assertTrue(result["post_check_passed"], result)

        rows = self.read_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "test-001")
        self.assertEqual(rows[0][1], str(self.source.resolve()))
        self.assertEqual(rows[0][2], "fresh-test-hash")
        self.assertEqual(rows[0][3], 2)
        self.assertTrue(rows[0][4])

        self.assertTrue(Path(result["backup_dir"]).exists())

    def test_failed_post_check_restores_inserted_record(self):
        result = self.run_executor([
            self.diagnosis_issue(),
            self.diagnosis_issue(),
        ])

        self.assertFalse(result["success"], result)
        self.assertFalse(result["post_check_passed"], result)
        self.assertTrue(result["recovery"]["attempted"], result)
        self.assertTrue(result["recovery"]["success"], result)
        self.assertEqual(self.read_rows(), [])

    def test_recovery_refuses_to_overwrite_concurrent_change(self):
        def mutate_then_restore(db_path, repair_result):
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "UPDATE embeddings SET content_hash = ? WHERE id = ?",
                    ("concurrent-change", "test-001"),
                )
            return real_restore(db_path, repair_result)

        result = self.run_executor(
            [
                self.diagnosis_issue(),
                self.diagnosis_issue(),
            ],
            extra_patches=[
                patch(
                    "memory.memory_repair_transaction.restore_embedding_repair",
                    side_effect=mutate_then_restore,
                )
            ],
        )

        self.assertFalse(result["success"], result)
        self.assertTrue(result["recovery"]["attempted"], result)
        self.assertFalse(result["recovery"]["success"], result)

        rows = self.read_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][2], "concurrent-change")


    def test_post_check_exception_restores_inserted_record(self):
        result = self.run_executor([
            self.diagnosis_issue(),
            RuntimeError("模拟复检服务突然崩溃"),
        ])

        self.assertFalse(result["success"], result)
        self.assertTrue(result["executed"], result)
        self.assertFalse(result["post_check_passed"], result)

        self.assertTrue(result["recovery"]["attempted"], result)
        self.assertTrue(
            result["recovery"]["success"],
            f"复检异常后未能恢复：{result}",
        )

        self.assertEqual(
            self.read_rows(),
            [],
            "复检抛出异常后，目标记录应恢复到修复前状态",
        )


    def test_pre_repair_diagnosis_exception_does_not_modify_database(self):
        before_rows = self.read_rows()

        result = self.run_executor([
            RuntimeError("模拟修复前诊断失败"),
        ])

        self.assertFalse(
            result["success"],
            f"诊断异常不应被当作修复成功：{result}",
        )
        self.assertFalse(
            result.get("executed", False),
            f"修复前诊断失败，不应开始修复：{result}",
        )
        self.assertEqual(
            self.read_rows(),
            before_rows,
            "修复前诊断失败时，数据库必须保持不变",
        )


    def test_unexpected_post_commit_exception_reports_execution_and_recovers(self):
        class IssuesRaiseOnSecondIteration(list):
            def __init__(self, items):
                super().__init__(items)
                self.iterations = 0

            def __iter__(self):
                self.iterations += 1
                if self.iterations == 2:
                    raise RuntimeError("模拟提交后处理阶段意外崩溃")
                return super().__iter__()

        malformed_timing_issues = IssuesRaiseOnSecondIteration([
            {
                "memory_id": "unrelated-memory",
                "problem": "无关问题",
            }
        ])

        result = self.run_executor([
            self.diagnosis_issue(),
            {
                "issues": malformed_timing_issues,
                "error_count": 0,
            },
        ])

        self.assertFalse(result["success"], result)
        self.assertTrue(
            result["executed"],
            f"事务已提交，必须如实报告已执行：{result}",
        )
        self.assertTrue(result["repairs_executed"], result)
        self.assertTrue(result["recovery"]["attempted"], result)
        self.assertTrue(
            result["recovery"]["success"],
            f"意外异常后应成功恢复：{result}",
        )
        self.assertEqual(
            self.read_rows(),
            [],
            "异常处理完成后，数据库应恢复到修复前状态",
        )


if __name__ == "__main__":
    unittest.main()
