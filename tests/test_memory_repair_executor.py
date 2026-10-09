import tempfile
import unittest
from pathlib import Path

from memory.memory_repair_executor import execute_repair
from memory.memory_repair_backup import DEFAULT_BACKUP_ROOT


class TestMemoryRepairExecutor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.memory_root = self.root / "memory"
        self.memory_root.mkdir()
        self.source = self.memory_root / "sample.md"
        self.source.write_text(
            "---\\nid: test-001\\nstatus: active\\n---\\n测试正文\\n",
            encoding="utf-8",
        )
        self.db_path = self.root / "embeddings.sqlite"

        import sqlite3
        conn = sqlite3.connect(self.db_path)
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
        conn.commit()
        conn.close()

        self.issue = {
            "id": "test-001",
            "problem": "正文哈希与向量索引不一致，向量可能已过期",
            "path": str(self.source),
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_is_preview_only(self):
        before = self.source.read_bytes()
        result = execute_repair(
            self.issue,
            memory_root=self.memory_root,
            embedding_db_path=self.db_path,
        )
        self.assertTrue(result["success"])
        self.assertFalse(result["executed"])
        self.assertTrue(result["preview_only"])
        self.assertEqual(self.source.read_bytes(), before)

    def test_execution_without_approval_is_refused(self):
        result = execute_repair(
            self.issue,
            execute=True,
            approved=False,
            memory_root=self.memory_root,
            embedding_db_path=self.db_path,
        )
        self.assertFalse(result["success"])
        self.assertFalse(result["executed"])
        self.assertTrue(result["refused"])

    def test_policy_still_blocks_real_writes(self):
        before = self.source.read_bytes()
        result = execute_repair(
            self.issue,
            execute=True,
            approved=True,
            memory_root=self.memory_root,
            embedding_db_path=self.db_path,
        )
        self.assertFalse(result["success"])
        self.assertFalse(result["executed"])
        self.assertTrue(result["refused"])
        self.assertEqual(self.source.read_bytes(), before)

        import sqlite3
        conn = sqlite3.connect(self.db_path)
        count = conn.execute(
            "SELECT COUNT(*) FROM embeddings"
        ).fetchone()[0]
        conn.close()
        self.assertEqual(count, 0)

    def test_default_backup_root_is_outside_memory_tree(self):
        memory_root = Path.home() / "AI-System" / "memory"
        self.assertFalse(
            DEFAULT_BACKUP_ROOT.resolve().is_relative_to(
                memory_root.resolve()
            )
        )


if __name__ == "__main__":
    unittest.main()
