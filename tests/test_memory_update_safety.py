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

from memory.memory_writer_v2 import (
    update_memory,
    archive_existing_memory,
)


class TestMemoryUpdateSafety(unittest.TestCase):
    def test_failed_new_version_write_preserves_original_memory(self):
        original_content = """---
id: mem_test123
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
            memory_file = Path(temp_dir) / "mem_test123.md"
            memory_file.write_text(original_content, encoding="utf-8")

            with patch(
                "memory.memory_writer_v2.find_memory_file",
                return_value=memory_file,
            ), patch(
                "memory.memory_writer_v2.archive_existing_memory"
            ) as archive, patch(
                "os.fsync",
                side_effect=OSError("模拟临时文件同步失败"),
            ):
                with self.assertRaises(OSError):
                    update_memory(
                        "mem_test123",
                        {
                            "type": "preferences",
                            "importance": 0.9,
                            "confidence": 0.95,
                            "memory": "更新后的记忆内容。",
                        },
                    )

                # 临时文件准备失败时，不应开始归档旧版本。
                archive.assert_not_called()

            self.assertEqual(
                memory_file.read_text(encoding="utf-8"),
                original_content,
                "更新失败后，原始记忆文件应保持完整。",
            )


    def test_failed_atomic_replace_preserves_original_memory(self):
        original_content = """---
id: mem_test456
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
            memory_file = Path(temp_dir) / "mem_test456.md"
            memory_file.write_text(original_content, encoding="utf-8")

            fake_archive = Path(temp_dir) / "archive" / "mem_test456.md"

            with patch(
                "memory.memory_writer_v2.find_memory_file",
                return_value=memory_file,
            ), patch(
                "memory.memory_writer_v2.archive_existing_memory",
                return_value=fake_archive,
            ) as archive, patch(
                "os.replace",
                side_effect=OSError("模拟原子替换失败"),
            ) as replace:
                with self.assertRaises(OSError):
                    update_memory(
                        "mem_test456",
                        {
                            "type": "preferences",
                            "importance": 0.9,
                            "confidence": 0.95,
                            "memory": "更新后的记忆内容。",
                        },
                    )

                archive.assert_called_once()
                replace.assert_called_once()

            self.assertEqual(
                memory_file.read_text(encoding="utf-8"),
                original_content,
                "原子替换失败后，正式记忆文件应保持完整。",
            )


    def test_failed_archive_preserves_original_memory_and_cleans_temp(self):
        original_content = """---
id: mem_test789
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
            memory_file = Path(temp_dir) / "mem_test789.md"
            memory_file.write_text(original_content, encoding="utf-8")

            with patch(
                "memory.memory_writer_v2.find_memory_file",
                return_value=memory_file,
            ), patch(
                "memory.memory_writer_v2.archive_existing_memory",
                side_effect=OSError("模拟归档失败"),
            ) as archive, patch(
                "os.replace",
            ) as replace:
                with self.assertRaises(OSError):
                    update_memory(
                        "mem_test789",
                        {
                            "type": "preferences",
                            "importance": 0.9,
                            "confidence": 0.95,
                            "memory": "更新后的记忆内容。",
                        },
                    )

                archive.assert_called_once()
                replace.assert_not_called()

            self.assertEqual(
                memory_file.read_text(encoding="utf-8"),
                original_content,
                "归档失败后，原始记忆文件必须保持完整。",
            )

            remaining_temp_files = list(
                Path(temp_dir).glob(".mem_test789.*.tmp")
            )
            self.assertEqual(
                remaining_temp_files,
                [],
                "归档失败后，不应残留临时文件。",
            )


    def test_existing_archive_is_never_overwritten(self):
        original_memory = """---
id: mem_archive_test
type: preferences
status: active
importance: 0.8
confidence: 0.9
created_at: 2026-10-01
updated_at: 2026-10-01
---

原始正式记忆。
"""

        existing_archive_content = "这是一份已经存在的历史归档，不能被覆盖。"

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            memory_file = root / "mem_archive_test.md"
            archive_dir = root / "archive"

            memory_file.write_text(original_memory, encoding="utf-8")

            version_id = "mem_archive_test__v20261009_120000_123456"
            existing_archive = archive_dir / f"{version_id}.md"
            archive_dir.mkdir()
            existing_archive.write_text(
                existing_archive_content,
                encoding="utf-8",
            )

            with patch(
                "memory.memory_writer_v2.ARCHIVE_ROOT",
                archive_dir,
            ), patch(
                "memory.memory_writer_v2.now_stamp",
                return_value="20261009_120000_123456",
            ), patch(
                "memory.memory_writer_v2.now_date",
                return_value="2026-10-09",
            ):
                with self.assertRaises(FileExistsError):
                    archive_existing_memory(memory_file, "mem_archive_test")

            self.assertEqual(
                existing_archive.read_text(encoding="utf-8"),
                existing_archive_content,
                "归档操作不得覆盖已有的历史记忆。",
            )

            self.assertEqual(
                memory_file.read_text(encoding="utf-8"),
                original_memory,
                "归档冲突时，正式记忆文件应保持完整。",
            )


    def test_failed_archive_write_does_not_leave_partial_archive(self):
        original_memory = """---
id: mem_archive_failure
type: preferences
status: active
importance: 0.8
confidence: 0.9
created_at: 2026-10-01
updated_at: 2026-10-01
---

原始正式记忆。
"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            memory_file = root / "mem_archive_failure.md"
            archive_dir = root / "archive"
            memory_file.write_text(original_memory, encoding="utf-8")

            version_id = "mem_archive_failure__v20261009_120000_123456"
            final_archive = archive_dir / f"{version_id}.md"
            original_named_temp_file = tempfile.NamedTemporaryFile

            class FailingWriter:
                def __init__(self, handle):
                    self.handle = handle
                    self.name = handle.name

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc_value, traceback):
                    return self.handle.__exit__(
                        exc_type, exc_value, traceback
                    )

                def write(self, content):
                    self.handle.write(content[:20])
                    self.handle.flush()
                    raise OSError("模拟写入部分归档内容后发生故障")

                def flush(self):
                    return self.handle.flush()

                def fileno(self):
                    return self.handle.fileno()

            def failing_named_temp_file(*args, **kwargs):
                handle = original_named_temp_file(*args, **kwargs)
                return FailingWriter(handle)

            with patch(
                "memory.memory_writer_v2.ARCHIVE_ROOT",
                archive_dir,
            ), patch(
                "memory.memory_writer_v2.now_stamp",
                return_value="20261009_120000_123456",
            ), patch(
                "memory.memory_writer_v2.now_date",
                return_value="2026-10-09",
            ), patch(
                "tempfile.NamedTemporaryFile",
                side_effect=failing_named_temp_file,
            ):
                with self.assertRaises(OSError):
                    archive_existing_memory(
                        memory_file,
                        "mem_archive_failure",
                    )

            self.assertFalse(
                final_archive.exists(),
                "写入中途失败后，不应留下不完整的最终归档文件。",
            )

            self.assertEqual(
                memory_file.read_text(encoding="utf-8"),
                original_memory,
                "归档失败后，原始正式记忆必须保持完整。",
            )

            remaining_temp_files = (
                list(archive_dir.glob("*.tmp"))
                if archive_dir.exists()
                else []
            )
            self.assertEqual(
                remaining_temp_files,
                [],
                "归档失败后，不应残留临时文件。",
            )


    def test_archive_cleanup_failure_emits_warning(self):
        original_memory = """---
id: mem_cleanup_test
type: preferences
status: active
importance: 0.8
confidence: 0.9
created_at: 2026-10-01
updated_at: 2026-10-01
---

原始正式记忆。
"""

        import io
        from contextlib import redirect_stdout

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            memory_file = root / "mem_cleanup_test.md"
            archive_dir = root / "archive"

            memory_file.write_text(
                original_memory,
                encoding="utf-8",
            )

            original_unlink = Path.unlink

            def fail_temp_cleanup(path, *args, **kwargs):
                if path.name.endswith(".tmp"):
                    raise OSError("模拟临时文件删除失败")

                return original_unlink(path, *args, **kwargs)

            output = io.StringIO()

            with patch(
                "memory.memory_writer_v2.ARCHIVE_ROOT",
                archive_dir,
            ), patch.object(
                Path,
                "unlink",
                autospec=True,
                side_effect=fail_temp_cleanup,
            ), redirect_stdout(output):
                archive_path = archive_existing_memory(
                    memory_file,
                    "mem_cleanup_test",
                )

            self.assertIsNotNone(
                archive_path,
                "临时文件清理失败不应使已成功的归档失效。",
            )

            self.assertTrue(
                archive_path.exists(),
                "成功发布的正式归档文件应当存在。",
            )

            self.assertEqual(
                memory_file.read_text(encoding="utf-8"),
                original_memory,
                "单独执行归档不应修改原始正式记忆。",
            )

            self.assertIn(
                "WARNING",
                output.getvalue(),
                "临时文件清理失败时应输出警告。",
            )

            self.assertIn(
                ".tmp",
                output.getvalue(),
                "警告应包含临时文件路径，便于后续处理。",
            )


    def test_update_cleanup_failure_emits_warning(self):
        original_content = """---
id: mem_update_cleanup_test
type: preferences
status: active
importance: 0.8
confidence: 0.9
created_at: 2026-10-01
updated_at: 2026-10-01
---

原始正式记忆。
"""

        import io
        from contextlib import redirect_stdout

        with tempfile.TemporaryDirectory() as temp_dir:
            memory_file = Path(temp_dir) / "mem_update_cleanup_test.md"
            memory_file.write_text(
                original_content,
                encoding="utf-8",
            )

            original_unlink = Path.unlink

            def fail_temp_cleanup(path, *args, **kwargs):
                if path.name.startswith(".mem_update_cleanup_test.") and path.name.endswith(".tmp"):
                    raise OSError("模拟更新临时文件删除失败")

                return original_unlink(path, *args, **kwargs)

            output = io.StringIO()

            with patch(
                "memory.memory_writer_v2.find_memory_file",
                return_value=memory_file,
            ), patch(
                "memory.memory_writer_v2.archive_existing_memory",
                return_value=Path(temp_dir) / "fake_archive.md",
            ), patch(
                "os.replace",
                side_effect=OSError("模拟正式文件替换失败"),
            ), patch.object(
                Path,
                "unlink",
                autospec=True,
                side_effect=fail_temp_cleanup,
            ), redirect_stdout(output):
                with self.assertRaisesRegex(
                    OSError,
                    "模拟正式文件替换失败",
                ):
                    update_memory(
                        "mem_update_cleanup_test",
                        {
                            "type": "preferences",
                            "importance": 0.9,
                            "confidence": 0.95,
                            "memory": "更新后的记忆内容。",
                        },
                    )

            self.assertEqual(
                memory_file.read_text(encoding="utf-8"),
                original_content,
                "正式替换失败后，原始记忆必须保持完整。",
            )

            self.assertIn(
                "WARNING",
                output.getvalue(),
                "临时文件清理失败时应输出警告。",
            )

            self.assertIn(
                ".tmp",
                output.getvalue(),
                "警告应包含临时文件路径。",
            )

if __name__ == "__main__":
    unittest.main()
