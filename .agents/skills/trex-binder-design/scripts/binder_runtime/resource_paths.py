"""Resolve immutable resources shipped with the skill, independent of cwd."""
from __future__ import annotations
from pathlib import Path

PACKAGED_DATA_ROOT = Path(__file__).resolve().parent / "data"


def embedded_runtime_root() -> Path:
    """Directory placed on child Python paths to import the bundled runtime."""
    return Path(__file__).resolve().parent.parent


def publication_data_path(repository_root: Path | str, relative_path: Path | str) -> Path:
    """Resolve bundled data; the legacy repository argument is intentionally ignored."""
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"publication data path must be relative: {relative}")
    return PACKAGED_DATA_ROOT / relative
