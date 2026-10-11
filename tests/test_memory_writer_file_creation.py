import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

BASE_DIR = Path(__file__).resolve().parents[1]
MEMORY_DIR = BASE_DIR / "memory"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if str(MEMORY_DIR) not in sys.path:
    sys.path.insert(0, str(MEMORY_DIR))

from memory import memory_writer_v2 as writer


class TestMemoryWriterFileCreation(unittest.TestCase):

    def test_new_memory_collision_does_not_overwrite_existing_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target_dir = root / "preferences"
            target_dir.mkdir()

            existing = target_dir / "mem_collision.md"
            original_content = "原始记忆，绝对不能覆盖。"
            existing.write_text(original_content, encoding="utf-8")

            with patch.object(writer, "MEMORY_ROOT", root), \
                 patch.object(
                     writer,
                     "now_stamp",
                     side_effect=["collision", "replacement"],
                 ):
                result = writer.write_new_memory({
                    "type": "preferences",
                    "importance": 0.8,
                    "confidence": 0.9,
                    "memory": "一条新的测试记忆。",
                })

            self.assertEqual(
                existing.read_text(encoding="utf-8"),
                original_content,
                "创建新记忆不得覆盖已有文件。",
            )
            self.assertNotEqual(result, existing)
            self.assertTrue(result.exists())
            self.assertIn("replacement", result.name)

    def test_candidate_collision_does_not_overwrite_existing_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            root.mkdir(exist_ok=True)

            existing = root / "candidate_collision.md"
            original_content = "原始候选记忆，绝对不能覆盖。"
            existing.write_text(original_content, encoding="utf-8")

            with patch.object(writer, "CANDIDATE_ROOT", root), \
                 patch.object(
                     writer,
                     "now_stamp",
                     side_effect=["collision", "replacement"],
                 ):
                result = writer.write_candidate({
                    "type": "preferences",
                    "importance": 0.7,
                    "confidence": 0.9,
                    "memory": "一条新的候选记忆。",
                })

            self.assertEqual(
                existing.read_text(encoding="utf-8"),
                original_content,
                "创建候选记忆不得覆盖已有文件。",
            )
            self.assertNotEqual(result, existing)
            self.assertTrue(result.exists())
            self.assertIn("replacement", result.name)


if __name__ == "__main__":
    unittest.main()
