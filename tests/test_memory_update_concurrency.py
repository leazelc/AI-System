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

from memory.memory_writer_v2 import update_memory


class TestMemoryUpdateConcurrency(unittest.TestCase):
    def test_concurrent_updates_to_same_memory_both_complete(self):
        original_content = """---
id: mem_concurrency_test
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
            memory_file = Path(temp_dir) / "mem_concurrency_test.md"
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

            with patch(
                "memory.memory_writer_v2.find_memory_file",
                return_value=memory_file,
            ), patch(
                "memory.memory_writer_v2.archive_existing_memory",
                return_value=Path(temp_dir) / "fake_archive.md",
            ):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_a = executor.submit(
                        update_memory, "mem_concurrency_test", update_a
                    )
                    future_b = executor.submit(
                        update_memory, "mem_concurrency_test", update_b
                    )

                    result_a = future_a.result(timeout=10)
                    result_b = future_b.result(timeout=10)

            final_content = memory_file.read_text(encoding="utf-8")

            self.assertEqual(result_a, memory_file)
            self.assertEqual(result_b, memory_file)
            self.assertTrue(
                "并发更新 A 的内容。" in final_content
                or "并发更新 B 的内容。" in final_content
            )

    def test_same_memory_updates_are_serialized(self):
        import threading
        from memory.memory_writer_v2 import _get_update_lock

        memory_id = "mem_lock_serialization_test"
        lock = _get_update_lock(memory_id)

        first_entered = threading.Event()
        second_attempting = threading.Event()
        release_first = threading.Event()
        second_acquired = threading.Event()

        def first_worker():
            with lock:
                first_entered.set()
                if not release_first.wait(timeout=5):
                    raise TimeoutError("等待释放锁超时")

        def second_worker():
            if not first_entered.wait(timeout=5):
                raise TimeoutError("第一个线程没有进入临界区")

            second_attempting.set()
            with _get_update_lock(memory_id):
                second_acquired.set()

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(first_worker)
            second = executor.submit(second_worker)

            try:
                self.assertTrue(first_entered.wait(timeout=2))
                self.assertTrue(second_attempting.wait(timeout=2))
                self.assertFalse(
                    second_acquired.is_set(),
                    "第一个线程持锁时，第二个线程不应进入临界区",
                )
            finally:
                release_first.set()

            first.result(timeout=5)
            second.result(timeout=5)

        self.assertTrue(second_acquired.is_set())

    def test_different_memories_can_update_concurrently(self):
        import threading
        from memory.memory_writer_v2 import _get_update_lock

        first_lock = _get_update_lock("mem_parallel_test_A")
        second_lock = _get_update_lock("mem_parallel_test_B")

        barrier = threading.Barrier(2)

        def worker(lock):
            with lock:
                barrier.wait(timeout=3)
                return True

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(worker, first_lock)
            future_b = executor.submit(worker, second_lock)

            self.assertTrue(future_a.result(timeout=5))
            self.assertTrue(future_b.result(timeout=5))

if __name__ == "__main__":
    unittest.main()
