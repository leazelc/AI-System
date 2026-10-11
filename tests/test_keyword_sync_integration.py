import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from memory import memory_writer_v2


class TestKeywordSyncIntegration(unittest.TestCase):
    def test_keyword_rebuild_runs_after_metadata_sync(self):
        with patch.object(memory_writer_v2, "scan_core") as scan, \
             patch.object(memory_writer_v2.keyword_index, "rebuild",
                          return_value=13) as rebuild, \
             patch.object(memory_writer_v2, "sync_embeddings",
                          return_value=True) as embeddings:
            self.assertTrue(memory_writer_v2.sync_indexes())

        scan.assert_called_once_with()
        rebuild.assert_called_once_with()
        embeddings.assert_called_once_with()

    def test_keyword_rebuild_failure_is_not_reported_as_success(self):
        with patch.object(memory_writer_v2, "scan_core"), \
             patch.object(memory_writer_v2.keyword_index, "rebuild",
                          side_effect=RuntimeError("test failure")), \
             patch.object(memory_writer_v2, "sync_embeddings") as embeddings:
            self.assertFalse(memory_writer_v2.sync_indexes())

        embeddings.assert_not_called()

    def test_missing_active_file_preserves_existing_keyword_index(self):
        from memory import keyword_index

        original_db = keyword_index.INDEX_DB
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "memory.sqlite"
            valid_path = Path(temp_dir) / "valid.md"
            valid_path.write_text("有效的本地记忆", encoding="utf-8")

            with sqlite3.connect(db_path) as conn:
                conn.execute("""
                    CREATE TABLE memories (
                        id TEXT PRIMARY KEY,
                        path TEXT NOT NULL UNIQUE,
                        status TEXT NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE VIRTUAL TABLE memory_keyword_fts
                    USING fts5(id UNINDEXED, content, tokenize='unicode61')
                """)
                conn.execute(
                    "INSERT INTO memories VALUES (?, ?, ?)",
                    ("valid", str(valid_path), "active"),
                )
                conn.execute(
                    "INSERT INTO memories VALUES (?, ?, ?)",
                    ("missing", str(Path(temp_dir) / "missing.md"), "active"),
                )
                conn.execute(
                    "INSERT INTO memory_keyword_fts VALUES (?, ?)",
                    ("old", "原有索引内容"),
                )

            try:
                keyword_index.INDEX_DB = db_path

                with self.assertRaisesRegex(RuntimeError, "原索引未被清空"):
                    keyword_index.rebuild()

                with sqlite3.connect(db_path) as conn:
                    rows = conn.execute(
                        "SELECT id FROM memory_keyword_fts"
                    ).fetchall()

                self.assertEqual(rows, [("old",)])
            finally:
                keyword_index.INDEX_DB = original_db

    def test_keyword_index_rebuild_covers_all_active_memories(self):
        from memory import keyword_index

        original_db = keyword_index.INDEX_DB
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "memory.sqlite"
            memory_dir = Path(temp_dir) / "memories"
            memory_dir.mkdir()

            conn = sqlite3.connect(db_path)
            conn.execute("""
                CREATE TABLE memories (
                    id TEXT PRIMARY KEY,
                    path TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL
                )
            """)
            for memory_id, body in (
                ("a", "本地人工智能记忆"),
                ("b", "关键词索引同步"),
            ):
                path = memory_dir / f"{memory_id}.md"
                path.write_text(body, encoding="utf-8")
                conn.execute(
                    "INSERT INTO memories (id, path, status) VALUES (?, ?, ?)",
                    (memory_id, str(path), "active"),
                )
            conn.commit()
            conn.close()

            try:
                keyword_index.INDEX_DB = db_path
                self.assertEqual(keyword_index.rebuild(), 2)
                with sqlite3.connect(db_path) as conn:
                    count = conn.execute(
                        "SELECT COUNT(*) FROM memory_keyword_fts"
                    ).fetchone()[0]
                self.assertEqual(count, 2)
            finally:
                keyword_index.INDEX_DB = original_db


if __name__ == "__main__":
    unittest.main(verbosity=2)
