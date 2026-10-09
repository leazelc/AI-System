import sqlite3
import tempfile
import unittest
from pathlib import Path

from memory import keyword_index


class TestKeywordIndex(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.db_path = self.root / "test.sqlite"
        self.memory_dir = self.root / "memories"
        self.memory_dir.mkdir()

        self.original_db = keyword_index.INDEX_DB
        keyword_index.INDEX_DB = self.db_path

        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE memories (
                id TEXT PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL
            )
        """)

        self.add_memory(
            conn,
            "budget",
            "我的电脑预算上限固定为2万元人民币。",
        )
        self.add_memory(
            conn,
            "hardware",
            "购买电脑时，我优先关注SSD和显存。",
        )
        self.add_memory(
            conn,
            "model",
            "选择本地AI模型时，我看重实际运行体验。"
            "我也会使用GPT-OSS 20B。",
        )
        self.add_memory(
            conn,
            "candidate",
            "这是候选记忆，不应进入正式检索结果。",
            status="candidate",
        )

        conn.commit()
        conn.close()

        self.assertEqual(keyword_index.rebuild(), 3)

    def add_memory(self, conn, memory_id, body, status="active"):
        path = self.memory_dir / f"{memory_id}.md"
        path.write_text(body, encoding="utf-8")
        conn.execute(
            "INSERT INTO memories (id, path, status) VALUES (?, ?, ?)",
            (memory_id, str(path), status),
        )

    def tearDown(self):
        keyword_index.INDEX_DB = self.original_db
        self.temp_dir.cleanup()

    def test_empty_query(self):
        self.assertEqual(keyword_index.search(""), [])
        self.assertEqual(keyword_index.search("   "), [])
        self.assertEqual(keyword_index.search(None), [])

    def test_invalid_limit(self):
        self.assertEqual(keyword_index.search("电脑预算", limit=0), [])
        self.assertEqual(keyword_index.search("电脑预算", limit=-1), [])

    def test_chinese_query(self):
        results = keyword_index.search("电脑预算上限")
        self.assertTrue(results)
        self.assertEqual(results[0]["id"], "budget")

    def test_mixed_chinese_and_english(self):
        results = keyword_index.search("本地 AI 模型 GPT-OSS 20B")
        self.assertTrue(results)
        self.assertEqual(results[0]["id"], "model")

    def test_special_characters_do_not_break_search(self):
        results = keyword_index.search('电脑预算 " OR * ')
        self.assertIsInstance(results, list)

    def test_unrelated_query_returns_no_results(self):
        results = keyword_index.search("火星海豚")
        self.assertEqual(results, [])

    def test_candidate_is_excluded(self):
        results = keyword_index.search("候选记忆")
        self.assertNotIn("candidate", [item["id"] for item in results])


if __name__ == "__main__":
    unittest.main(verbosity=2)
