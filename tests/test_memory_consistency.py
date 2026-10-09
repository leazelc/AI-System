import sqlite3
import struct
import tempfile
import unittest
from pathlib import Path

from memory.memory_consistency import (
    open_readonly_db,
    decode_vector,
    markdown_content_hash,
    check_core_index_consistency,
    check_markdown_index_coverage,
    check_embedding_consistency,
    check_embedding_coverage,
    run_consistency_check,
)


class TestMemoryConsistency(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "memory"
        self.docs = self.root / "preferences"
        self.index = self.root / "index"

        self.docs.mkdir(parents=True)
        self.index.mkdir(parents=True)

        self.core_db = self.index / "memory.sqlite"
        self.embedding_db = self.index / "embeddings.sqlite"

        self.md = self.docs / "test.md"
        self.md.write_text(
            "---\n"
            "id: test-001\n"
            "type: preference\n"
            "status: active\n"
            "importance: 0.8\n"
            "confidence: 0.9\n"
            "---\n"
            "测试记忆正文\n",
            encoding="utf-8",
        )

        with sqlite3.connect(self.core_db) as conn:
            conn.execute("""
                CREATE TABLE memories (
                    id TEXT, path TEXT, type TEXT, status TEXT,
                    importance REAL, confidence REAL,
                    created_at TEXT, updated_at TEXT
                )
            """)
            conn.execute(
                "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "test-001", str(self.md), "preference", "active",
                    0.8, 0.9, "2026-01-01", "2026-01-01",
                ),
            )

        with sqlite3.connect(self.embedding_db) as conn:
            conn.execute("""
                CREATE TABLE embeddings (
                    id TEXT, path TEXT, content_hash TEXT,
                    dimension INTEGER, vector BLOB, updated_at TEXT
                )
            """)
            conn.execute(
                "INSERT INTO embeddings VALUES (?, ?, ?, ?, ?, ?)",
                (
                    "test-001",
                    str(self.md),
                    markdown_content_hash(self.md),
                    2,
                    struct.pack("2f", 0.1, 0.2),
                    "2026-01-01",
                ),
            )

    def tearDown(self):
        self.temp.cleanup()

    def diagnose(self):
        return run_consistency_check(
            self.root, self.core_db, self.embedding_db
        )

    def test_valid_memory_passes_all_checks(self):
        report = self.diagnose()
        self.assertEqual(report["error_count"], 0, report["issues"])
        self.assertEqual(report["warning_count"], 0, report["issues"])

    def test_database_connection_is_read_only(self):
        conn = open_readonly_db(self.core_db)
        try:
            with self.assertRaises(sqlite3.OperationalError):
                conn.execute(
                    "UPDATE memories SET status = 'deleted'"
                )
        finally:
            conn.close()

    def test_invalid_vector_is_rejected(self):
        with self.assertRaises(ValueError):
            decode_vector(b"\x00\x00", 2)

        with self.assertRaises(ValueError):
            decode_vector(struct.pack("f", float("nan")), 1)

    def test_stale_embedding_hash_is_detected(self):
        with sqlite3.connect(self.embedding_db) as conn:
            conn.execute(
                "UPDATE embeddings SET content_hash = ?",
                ("incorrect-hash",),
            )

        issues = check_embedding_consistency(
            self.root, self.embedding_db
        )
        self.assertTrue(
            any("哈希" in issue["problem"] for issue in issues),
            issues,
        )

    def test_missing_embedding_is_detected(self):
        with sqlite3.connect(self.embedding_db) as conn:
            conn.execute("DELETE FROM embeddings")

        issues = check_embedding_coverage(
            self.root, self.embedding_db
        )
        self.assertTrue(
            any("缺少向量记录" in issue["problem"] for issue in issues),
            issues,
        )

    def test_orphan_embedding_is_detected(self):
        with sqlite3.connect(self.embedding_db) as conn:
            conn.execute(
                "INSERT INTO embeddings VALUES (?, ?, ?, ?, ?, ?)",
                (
                    "orphan-001",
                    str(self.docs / "deleted.md"),
                    "fake-hash",
                    2,
                    struct.pack("2f", 0.1, 0.2),
                    "2026-01-01",
                ),
            )

        issues = check_embedding_coverage(
            self.root, self.embedding_db
        )
        self.assertTrue(
            any("不应保留" in issue["problem"] for issue in issues),
            issues,
        )

    def test_duplicate_markdown_ids_are_detected(self):
        duplicate = self.docs / "duplicate.md"
        duplicate.write_text(
            self.md.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

        report = self.diagnose()
        self.assertTrue(
            any("多个 Markdown 文件使用相同 ID" in issue["problem"]
                for issue in report["issues"]),
            report["issues"],
        )

    def test_core_index_path_mismatch_is_detected(self):
        with sqlite3.connect(self.core_db) as conn:
            conn.execute(
                "UPDATE memories SET path = ? WHERE id = ?",
                (str(self.docs / "missing.md"), "test-001"),
            )

        issues = check_core_index_consistency(
            self.root, self.core_db
        )
        self.assertTrue(issues, "应当发现结构化索引路径异常")

    def test_markdown_without_core_index_is_detected(self):
        with sqlite3.connect(self.core_db) as conn:
            conn.execute("DELETE FROM memories")

        issues = check_markdown_index_coverage(
            self.root, self.core_db
        )
        self.assertTrue(
            any("未登记" in issue["problem"] for issue in issues),
            issues,
        )


if __name__ == "__main__":
    unittest.main()
