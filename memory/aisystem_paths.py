from pathlib import Path
import os

PROJECT_ROOT = Path(
    os.environ.get("AI_SYSTEM_HOME", str(Path.home() / "AI-System"))
).expanduser().resolve()

MEMORY_ROOT = Path(
    os.environ.get("AI_SYSTEM_MEMORY_ROOT", str(PROJECT_ROOT / "memory"))
).expanduser().resolve()

INDEX_ROOT = MEMORY_ROOT / "index"
CORE_DB_PATH = INDEX_ROOT / "memory.sqlite"
EMBEDDING_DB_PATH = INDEX_ROOT / "embeddings.sqlite"
ARCHIVE_ROOT = MEMORY_ROOT / "archive"
CANDIDATE_ROOT = MEMORY_ROOT / "candidates"
