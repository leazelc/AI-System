import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout
from io import StringIO

BASE_DIR = Path(__file__).resolve().parents[1]
MEMORY_DIR = BASE_DIR / "memory"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if str(MEMORY_DIR) not in sys.path:
    sys.path.insert(0, str(MEMORY_DIR))

from memory.memory_validation import validate_decision


class TestMemoryWriterValidation(unittest.TestCase):
    def test_invalid_decision_is_rejected_before_any_write(self):
        invalid_decision = {
            "action": "NEW",
            "target_id": None,
            "type": "preferences",
            "importance": 0.8,
            "confidence": 1.5,
            "memory": "这条记忆不应该被写入。",
        }

        # 校验失败时，不允许进入候选记忆或正式记忆写入函数。
        with patch(
            "memory.memory_writer_v2.decide_update",
            return_value=invalid_decision,
        ), patch(
            "memory.memory_writer_v2.write_candidate"
        ) as write_candidate, patch(
            "memory.memory_writer_v2.write_new_memory"
        ) as write_new_memory:
            from memory.memory_writer_v2 import execute

            output = StringIO()
            with redirect_stdout(output):
                execute("模拟一条非法决策")

            self.assertIn("[REJECTED]", output.getvalue())
            write_candidate.assert_not_called()
            write_new_memory.assert_not_called()

    def test_valid_ignore_does_not_write_memory(self):
        ignore_decision = {
            "action": "IGNORE",
            "target_id": None,
            "type": None,
            "importance": 0.0,
            "confidence": 0.0,
            "memory": None,
        }

        with patch(
            "memory.memory_writer_v2.decide_update",
            return_value=ignore_decision,
        ), patch(
            "memory.memory_writer_v2.write_candidate"
        ) as write_candidate, patch(
            "memory.memory_writer_v2.write_new_memory"
        ) as write_new_memory, patch(
            "memory.memory_writer_v2.sync_indexes"
        ) as sync_indexes:
            from memory.memory_writer_v2 import execute

            output = StringIO()
            with redirect_stdout(output):
                execute("模拟一条无需保存的信息")

            self.assertIn("Nothing written.", output.getvalue())
            write_candidate.assert_not_called()
            write_new_memory.assert_not_called()
            sync_indexes.assert_not_called()


    def test_unknown_target_is_rejected_before_candidate_write(self):
        invalid_target_decision = {
            "action": "UPDATE",
            "target_id": "mem_does_not_exist",
            "type": "preferences",
            "importance": 0.8,
            "confidence": 0.95,
            "memory": "这条更新不应该被写入。",
        }

        with patch(
            "memory.memory_writer_v2.decide_update",
            return_value=invalid_target_decision,
        ), patch(
            "memory.memory_writer_v2.find_memory_file",
            return_value=None,
        ), patch(
            "memory.memory_writer_v2.write_candidate"
        ) as write_candidate, patch(
            "memory.memory_writer_v2.update_memory"
        ) as update_memory, patch(
            "memory.memory_writer_v2.sync_indexes"
        ) as sync_indexes:
            from memory.memory_writer_v2 import execute

            output = StringIO()
            with redirect_stdout(output):
                execute("模拟更新一条不存在的记忆")

            self.assertIn("[REJECTED]", output.getvalue())
            self.assertIn("does not exist", output.getvalue())
            write_candidate.assert_not_called()
            update_memory.assert_not_called()
            sync_indexes.assert_not_called()


    def test_sync_indexes_propagates_embedding_failure(self):
        from memory import memory_writer_v2

        output = StringIO()

        with patch.object(
            memory_writer_v2, "scan_core"
        ), patch.object(
            memory_writer_v2.keyword_index, "rebuild", return_value=13
        ), patch.object(
            memory_writer_v2, "sync_embeddings", return_value=False
        ), redirect_stdout(output):
            result = memory_writer_v2.sync_indexes()

        self.assertIs(
            result,
            False,
            "嵌入索引同步失败时，sync_indexes() 必须返回 False。",
        )
        self.assertIn(
            "synchronization was incomplete",
            output.getvalue().lower(),
        )
        self.assertNotIn(
            "Memory indexes synchronized.",
            output.getvalue(),
        )


    def test_index_sync_failure_is_not_reported_as_success(self):
        new_decision = {
            "action": "NEW",
            "target_id": None,
            "type": "preferences",
            "importance": 0.8,
            "confidence": 0.95,
            "memory": "测试索引同步失败时的状态报告。",
        }

        with patch(
            "memory.memory_writer_v2.decide_update",
            return_value=new_decision,
        ), patch(
            "memory.memory_writer_v2.write_candidate",
            return_value=Path("/tmp/test_candidate.md"),
        ), patch(
            "memory.memory_writer_v2.write_new_memory",
            return_value=Path("/tmp/test_memory.md"),
        ), patch(
            "memory.memory_writer_v2.archive_candidate"
        ), patch(
            "memory.memory_writer_v2.sync_indexes",
            return_value=False,
        ):
            from memory.memory_writer_v2 import execute

            output = StringIO()
            with redirect_stdout(output):
                execute("模拟正式记忆写入成功但索引同步失败")

        result = output.getvalue()
        self.assertIn("index synchronization failed", result.lower())
        self.assertNotIn("Writer operation completed.", result)


if __name__ == "__main__":
    unittest.main()
