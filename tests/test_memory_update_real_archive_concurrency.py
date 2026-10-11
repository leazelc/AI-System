import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

BASE_DIR = Path(__file__).resolve().parents[1]
MEMORY_DIR = BASE_DIR / "memory"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if str(MEMORY_DIR) not in sys.path:
    sys.path.insert(0, str(MEMORY_DIR))

import memory.memory_writer_v2 as writer


class TestMemoryUpdateRealArchiveConcurrency(unittest.TestCase):
    def test_concurrent_updates_preserve_intermediate_version_in_archive(self):
        original_content = """---
id: mem_real_archive_test
type: preferences
status: active
importance: 0.8
confidence: 0.9
created_at: 2026-10-01
updated_at: 2026-10-01
---

# 长期记忆

原始记忆内容。
"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            memory_file = root / "mem_real_archive_test.md"
            archive_dir = root / "archive"
            memory_file.write_text(original_content, encoding="utf-8")

            update_a = {
                "type": "preferences",
                "importance": 0.9,
                "confidence": 0.95,
                "memory": "并发更新 A 的内容。",
            }
            update_b = {
                "type": "preferences",
                "importance": 0.9,
                "confidence": 0.95,
                "memory": "并发更新 B 的内容。",
            }

            with patch.object(
                writer, "ARCHIVE_ROOT", archive_dir
            ), patch.object(
                writer,
                "find_memory_file",
                return_value=memory_file,
            ):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_a = executor.submit(
                        writer.update_memory,
                        "mem_real_archive_test",
                        update_a,
                    )
                    future_b = executor.submit(
                        writer.update_memory,
                        "mem_real_archive_test",
                        update_b,
                    )

                    result_a = future_a.result(timeout=15)
                    result_b = future_b.result(timeout=15)

            self.assertEqual(result_a, memory_file)
            self.assertEqual(result_b, memory_file)

            final_content = memory_file.read_text(encoding="utf-8")
            archives = list(archive_dir.glob("*.md"))
            archive_contents = [
                path.read_text(encoding="utf-8")
                for path in archives
            ]

            self.assertEqual(
                len(archives),
                2,
                "两个成功的更新应当各自产生一个归档版本。",
            )

            # 两个归档版本应分别保留：最初版本，以及中间更新版本。
            self.assertEqual(
                sum("原始记忆内容。" in content for content in archive_contents),
                1,
            )

            self.assertEqual(
                sum(
                    "并发更新 A 的内容。" in content
                    for content in archive_contents
                ),
                1 if "并发更新 B 的内容。" in final_content else 0,
            )
            self.assertEqual(
                sum(
                    "并发更新 B 的内容。" in content
                    for content in archive_contents
                ),
                1 if "并发更新 A 的内容。" in final_content else 0,
            )

            # 两次更新中的 A、B 都必须能在活动记忆或归档中找到。
            all_versions = [final_content] + archive_contents
            for phrase in (
                "原始记忆内容。",
                "并发更新 A 的内容。",
                "并发更新 B 的内容。",
            ):
                self.assertEqual(
                    sum(phrase in content for content in all_versions),
                    1,
                    f"版本内容丢失或重复：{phrase}",
                )


if __name__ == "__main__":
    unittest.main()
