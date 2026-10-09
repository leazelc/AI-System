import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from memory.memory_repair_backup import (
    create_repair_backup,
    verify_repair_backup,
)


class TestMemoryRepairBackup(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "sample.md"
        self.backup_root = self.root / "backups"
        self.original = (
            b"---\nid: test-001\nstatus: active\n---\nTest memory.\n"
        )
        self.source.write_bytes(self.original)

    def tearDown(self):
        self.temp.cleanup()

    def test_backup_preserves_source_and_bytes(self):
        result = create_repair_backup([self.source], self.backup_root)
        backup_file = (
            Path(result["backup_dir"])
            / result["files"][0]["backup_path"]
        )

        self.assertTrue(result["success"])
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertEqual(backup_file.read_bytes(), self.original)
        self.assertTrue(verify_repair_backup(result["backup_dir"])["valid"])
        self.assertFalse(result["repairs_executed"])

    def test_existing_backup_is_not_overwritten(self):
        first = create_repair_backup([self.source], self.backup_root)
        first_manifest = Path(first["manifest_path"])
        original_manifest = first_manifest.read_bytes()

        second = create_repair_backup([self.source], self.backup_root)

        self.assertNotEqual(first["backup_dir"], second["backup_dir"])
        self.assertEqual(first_manifest.read_bytes(), original_manifest)
        self.assertTrue(verify_repair_backup(first["backup_dir"])["valid"])
        self.assertTrue(verify_repair_backup(second["backup_dir"])["valid"])

    def test_tampered_backup_is_detected(self):
        result = create_repair_backup([self.source], self.backup_root)
        backup_file = (
            Path(result["backup_dir"])
            / result["files"][0]["backup_path"]
        )
        backup_file.write_bytes(b"tampered")

        self.assertFalse(verify_repair_backup(result["backup_dir"])["valid"])

    def test_failed_backup_cleans_partial_directory(self):
        with patch(
            "memory.memory_repair_backup.sha256_bytes",
            side_effect=RuntimeError("simulated failure"),
        ):
            with self.assertRaises(RuntimeError):
                create_repair_backup([self.source], self.backup_root)

        remaining = (
            list(self.backup_root.iterdir())
            if self.backup_root.exists()
            else []
        )
        self.assertEqual(remaining, [])
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            create_repair_backup([], self.backup_root)

    def test_duplicate_source_is_rejected(self):
        with self.assertRaises(ValueError):
            create_repair_backup(
                [self.source, self.source],
                self.backup_root,
            )


if __name__ == "__main__":
    unittest.main()
