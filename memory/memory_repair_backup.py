"""Create and verify pre-repair backups. Never execute repairs."""

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path.home() / "AI-System"
MEMORY_ROOT = PROJECT_ROOT / "memory"
DEFAULT_BACKUP_ROOT = PROJECT_ROOT / "repair_backups"


def sha256_bytes(data):
    """Return the SHA-256 hex digest of bytes."""
    return hashlib.sha256(data).hexdigest()


def create_repair_backup(paths, backup_root=None):
    """
    Back up files byte-for-byte into a unique directory.

    This function only creates backups and verifies their contents.
    It never modifies source files or executes repairs.
    """
    root = Path(backup_root) if backup_root else DEFAULT_BACKUP_ROOT
    sources = [Path(p).expanduser() for p in paths]

    if not sources:
        raise ValueError("拒绝创建空备份：没有提供源文件。")

    resolved_sources = []
    seen = set()

    for source in sources:
        if source.is_symlink():
            raise ValueError(f"拒绝备份符号链接：{source}")

        resolved = source.resolve(strict=True)

        if not resolved.is_file():
            raise ValueError(f"源路径不是普通文件：{source}")

        if resolved in seen:
            raise ValueError(f"源文件重复：{resolved}")

        seen.add(resolved)
        resolved_sources.append((source, resolved))

    root.mkdir(parents=True, exist_ok=True)

    backup_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    backup_dir = root / backup_id

    # mkdir without exist_ok prevents accidentally reusing a directory.
    backup_dir.mkdir()

    entries = []

    try:
        files_dir = backup_dir / "files"
        files_dir.mkdir()

        for number, (original, resolved) in enumerate(
            resolved_sources, start=1
        ):
            # Read the source once; hash and write the exact same bytes.
            original_bytes = resolved.read_bytes()
            original_hash = sha256_bytes(original_bytes)

            backup_name = f"{number:04d}_{resolved.name}"
            backup_path = files_dir / backup_name

            # Exclusive creation prevents overwriting any existing file.
            with backup_path.open("xb") as output:
                output.write(original_bytes)
                output.flush()
                os.fsync(output.fileno())

            backup_bytes = backup_path.read_bytes()
            backup_hash = sha256_bytes(backup_bytes)

            if backup_hash != original_hash:
                raise IOError(f"备份校验失败：{resolved}")

            if len(backup_bytes) != len(original_bytes):
                raise IOError(f"备份大小不一致：{resolved}")

            entries.append({
                "source_path": str(resolved),
                "backup_path": str(Path("files") / backup_name),
                "size_bytes": len(original_bytes),
                "sha256": original_hash,
                "verified": True,
            })

        manifest = {
            "backup_id": backup_id,
            "created_at": datetime.now().astimezone().isoformat(),
            "purpose": "pre_repair_backup",
            "repairs_executed": False,
            "files_count": len(entries),
            "files": entries,
        }

        manifest_path = backup_dir / "manifest.json"
        manifest_bytes = (
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
            ) + "\n"
        ).encode("utf-8")

        with manifest_path.open("xb") as output:
            output.write(manifest_bytes)
            output.flush()
            os.fsync(output.fileno())

        # Verify the published manifest can be read and parsed.
        with manifest_path.open("r", encoding="utf-8") as source:
            saved_manifest = json.load(source)

        if saved_manifest.get("files_count") != len(entries):
            raise IOError("备份清单校验失败。")

        return {
            "success": True,
            "backup_id": backup_id,
            "backup_dir": str(backup_dir),
            "manifest_path": str(manifest_path),
            "files_count": len(entries),
            "repairs_executed": False,
            "files": entries,
        }

    except Exception:
        # An incomplete backup must not look like a successful backup.
        shutil.rmtree(backup_dir, ignore_errors=True)
        raise


def verify_repair_backup(backup_dir):
    """Verify backup files against the saved manifest; read-only."""
    backup_dir = Path(backup_dir)
    manifest_path = backup_dir / "manifest.json"

    with manifest_path.open("r", encoding="utf-8") as source:
        manifest = json.load(source)

    results = []

    for entry in manifest.get("files", []):
        backup_path = backup_dir / entry["backup_path"]

        if not backup_path.is_file():
            results.append({
                "source_path": entry["source_path"],
                "valid": False,
                "reason": "backup_file_missing",
            })
            continue

        data = backup_path.read_bytes()
        actual_hash = sha256_bytes(data)

        valid = (
            actual_hash == entry["sha256"]
            and len(data) == entry["size_bytes"]
        )

        results.append({
            "source_path": entry["source_path"],
            "valid": valid,
            "sha256": actual_hash,
            "size_bytes": len(data),
            "reason": None if valid else "hash_or_size_mismatch",
        })

    return {
        "backup_id": manifest.get("backup_id"),
        "valid": (
            len(results) == manifest.get("files_count")
            and all(item["valid"] for item in results)
        ),
        "files_count": len(results),
        "files": results,
        "repairs_executed": False,
    }
