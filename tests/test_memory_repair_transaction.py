import sqlite3
import tempfile
import unittest
from pathlib import Path

from memory.memory_repair_transaction import (
    apply_embedding_repair,
    restore_embedding_repair,
)


class TestMemoryRepairTransaction(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.db = self.root / "embeddings.sqlite"
        self.source = str((self.root / "memory.md").resolve())

        conn = sqlite3.connect(self.db)
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
        conn.execute(
            "INSERT INTO embeddings VALUES (?, ?, ?, ?, ?, ?)",
            ("other", str((self.root / "other.md").resolve()),
             "hash-other", 2, b"other-vector", "old"),
        )
        conn.commit()
        conn.close()

    def tearDown(self):
        self.temp_dir.cleanup()

    def rows(self):
        conn = sqlite3.connect(self.db)
        try:
            return conn.execute(
                "SELECT id, path, content_hash, dimension, vector, updated_at "
                "FROM embeddings ORDER BY id"
            ).fetchall()
        finally:
            conn.close()

    def repair(self, operation, memory_id="target"):
        return apply_embedding_repair(
            self.db,
            operation=operation,
            memory_id=memory_id,
            source_path=self.source,
            content_hash="new-hash",
            dimension=2,
            vector_blob=b"new-vector",
            updated_at="new",
        )

    def test_create_and_restore_removes_only_inserted_row(self):
        result = self.repair("create_embedding")
        self.assertTrue(result["success"])
        self.assertEqual(len(self.rows()), 2)

        restored = restore_embedding_repair(self.db, result)
        self.assertTrue(restored["success"])
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(self.rows()[0][0], "other")

    def test_rebuild_and_restore_restores_previous_row(self):
        conn = sqlite3.connect(self.db)
        conn.execute(
            "INSERT INTO embeddings VALUES (?, ?, ?, ?, ?, ?)",
            ("target", self.source, "old-hash", 2, b"old-vector", "old"),
        )
        conn.commit()
        conn.close()
        before = self.rows()

        result = self.repair("rebuild_embedding")
        self.assertEqual(result["before"]["content_hash"], "old-hash")
        restore_embedding_repair(self.db, result)
        self.assertEqual(self.rows(), before)

    def test_create_refuses_existing_path_collision(self):
        conn = sqlite3.connect(self.db)
        conn.execute(
            "INSERT INTO embeddings VALUES (?, ?, ?, ?, ?, ?)",
            ("existing-id", self.source, "old-hash", 2, b"old-vector", "old"),
        )
        conn.commit()
        conn.close()

        before = self.rows()
        with self.assertRaises(RuntimeError):
            self.repair("create_embedding", memory_id="different-id")
        self.assertEqual(self.rows(), before)

    def test_rebuild_refuses_missing_target(self):
        before = self.rows()
        with self.assertRaises(RuntimeError):
            self.repair("rebuild_embedding")
        self.assertEqual(self.rows(), before)

    def test_restore_refuses_concurrent_change(self):
        result = self.repair("create_embedding")
        conn = sqlite3.connect(self.db)
        conn.execute(
            "UPDATE embeddings SET content_hash = ? WHERE id = ?",
            ("concurrent-change", "target"),
        )
        conn.commit()
        conn.close()

        before_restore = self.rows()
        with self.assertRaises(RuntimeError):
            restore_embedding_repair(self.db, result)
        self.assertEqual(self.rows(), before_restore)

    def test_other_rows_are_preserved(self):
        original_other = self.rows()[0]
        result = self.repair("create_embedding")
        restore_embedding_repair(self.db, result)
        self.assertEqual(self.rows()[0], original_other)

    def test_unknown_operation_is_rejected(self):
        before = self.rows()
        with self.assertRaises(ValueError):
            self.repair("delete_everything")
        self.assertEqual(self.rows(), before)


if __name__ == "__main__":
    unittest.main()
