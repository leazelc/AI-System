import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout
from io import StringIO

from memory import embedding_index


class TestEmbeddingIndexSafety(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.memory_dir = self.root / "memory"
        self.memory_dir.mkdir()

        self.db_path = self.root / "embeddings.sqlite"

        with patch.object(embedding_index, "DB_PATH", self.db_path):
            conn = embedding_index.get_db()
            conn.close()

    def tearDown(self):
        self.temp_dir.cleanup()

    def insert_old_vector(self, memory_id, path, content):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT INTO embeddings
            (id, path, content_hash, dimension, vector, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                str(path),
                embedding_index.content_hash(content),
                2,
                embedding_index.vector_to_blob([0.1, 0.2]),
                "2026-10-01",
            ),
        )
        conn.commit()
        conn.close()

    def read_vector_ids(self):
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT id FROM embeddings ORDER BY id"
        ).fetchall()
        conn.close()
        return [row[0] for row in rows]

    def test_embedding_failure_is_reported_and_old_vector_preserved(self):
        memory_file = self.memory_dir / "memory.md"
        memory_file.write_text(
            "---\n"
            "id: mem_embedding_failure\n"
            "status: active\n"
            "updated_at: 2026-10-09\n"
            "---\n"
            "更新后的记忆内容。\n",
            encoding="utf-8",
        )

        self.insert_old_vector(
            "mem_embedding_failure",
            memory_file,
            "旧的记忆内容。",
        )

        output = StringIO()

        with patch.object(
            embedding_index, "MEMORY_ROOT", self.memory_dir
        ), patch.object(
            embedding_index, "DB_PATH", self.db_path
        ), patch.object(
            embedding_index,
            "get_embedding",
            side_effect=RuntimeError("模拟向量生成失败"),
        ), redirect_stdout(output):
            result = embedding_index.sync()

        self.assertEqual(
            self.read_vector_ids(),
            ["mem_embedding_failure"],
            "向量生成失败时必须保留已有向量。",
        )
        self.assertIs(
            result,
            False,
            "向量生成失败时，sync() 必须明确返回 False。",
        )
        self.assertIn("ERROR", output.getvalue())

    def test_successful_sync_returns_true_and_stores_vector(self):
        memory_file = self.memory_dir / "memory.md"
        memory_file.write_text(
            "---\n"
            "id: mem_sync_success\n"
            "status: active\n"
            "updated_at: 2026-10-09\n"
            "---\n"
            "正常同步测试内容。\n",
            encoding="utf-8",
        )

        output = StringIO()

        with patch.object(
            embedding_index, "MEMORY_ROOT", self.memory_dir
        ), patch.object(
            embedding_index, "DB_PATH", self.db_path
        ), patch.object(
            embedding_index,
            "get_embedding",
            return_value=[0.1, 0.2],
        ), redirect_stdout(output):
            result = embedding_index.sync()

        self.assertIs(
            result,
            True,
            "正常同步完成时，sync() 必须返回 True。",
        )
        self.assertEqual(
            self.read_vector_ids(),
            ["mem_sync_success"],
            "正常同步后应将向量写入临时数据库。",
        )


    def test_parse_failure_preserves_existing_vector(self):
        memory_file = self.memory_dir / "memory.md"
        memory_file.write_text(
            "这是一份暂时无法读取的记忆文件。",
            encoding="utf-8",
        )

        self.insert_old_vector(
            "mem_parse_failure",
            memory_file,
            "之前已建立索引的记忆内容。",
        )

        output = StringIO()

        with patch.object(
            embedding_index, "MEMORY_ROOT", self.memory_dir
        ), patch.object(
            embedding_index, "DB_PATH", self.db_path
        ), patch.object(
            embedding_index,
            "parse_memory",
            side_effect=ValueError("模拟解析失败"),
        ), redirect_stdout(output):
            result = embedding_index.sync()

        self.assertIs(
            result,
            False,
            "Markdown 解析失败时，sync() 必须明确返回 False。",
        )
        self.assertEqual(
            self.read_vector_ids(),
            ["mem_parse_failure"],
            "文件解析失败时不能直接删除已有向量。",
        )
        self.assertIn("Skipping", output.getvalue())


if __name__ == "__main__":
    unittest.main()
