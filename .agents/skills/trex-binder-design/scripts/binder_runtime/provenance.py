"""Asset manifest checks and legacy archive schema identifiers.

New execution provenance is recorded by trex_codex.runner from the bundle.
No source checkout or local model server is inspected here.
"""


from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Iterable


PROVENANCE_SCHEMA = "v7.3.3_run_provenance_v2"


LEGACY_PROVENANCE_SCHEMAS = frozenset(
    {
        "v7.3.3_run_provenance_v1",
    }
)


SUPPORTED_PROVENANCE_SCHEMAS = LEGACY_PROVENANCE_SCHEMAS | {PROVENANCE_SCHEMA}


MODEL_MANIFEST_SCHEMA = "v7.3.3_model_manifest_v1"


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_rows(rows: Iterable[tuple[str, int, str]]) -> str:
    digest = hashlib.sha256()
    for relative, size, content_sha256 in sorted(rows):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(content_sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_model_manifest(
    model_path: Path,
    manifest: dict[str, Any],
    *,
    verify_content: bool = False,
) -> str:
    """Validate a content manifest and optionally rehash every declared file.

    The default path is suitable for campaign startup: it validates manifest
    integrity plus every declared file name and byte size. ``verify_content``
    additionally compares every file's SHA256 and is intended for release
    audits because model trees can contain tens of gigabytes.
    """
    if manifest.get("schema_version") != MODEL_MANIFEST_SCHEMA:
        raise ValueError("unsupported model manifest schema")
    expected = manifest.get("files") or []
    if not expected or not manifest.get("content_sha256"):
        raise ValueError("model manifest is incomplete")
    rows: list[tuple[str, int, str]] = []
    seen: set[str] = set()
    for row in expected:
        relative = str(row.get("path", ""))
        candidate = Path(relative)
        if (
            not relative
            or candidate.is_absolute()
            or ".." in candidate.parts
            or relative in seen
        ):
            raise ValueError(f"unsafe or duplicate model path: {relative!r}")
        seen.add(relative)
        try:
            size = int(row["size"])
            expected_sha = str(row["sha256"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid model manifest row: {row!r}") from exc
        if (
            size < 0
            or len(expected_sha) != 64
            or any(char not in "0123456789abcdef" for char in expected_sha)
        ):
            raise ValueError(f"invalid model manifest row: {row!r}")
        rows.append((relative, size, expected_sha))
        path = model_path / relative
        if not path.is_file():
            raise FileNotFoundError(f"model file missing: {path}")
        if path.stat().st_size != size:
            raise ValueError(f"model file size changed: {path}")
        if verify_content and _sha256_file(path) != expected_sha:
            raise ValueError(f"model file content changed: {path}")
    if manifest.get("n_files") not in (None, len(rows)):
        raise ValueError("model manifest n_files does not match its file rows")
    total_bytes = sum(size for _, size, _ in rows)
    if manifest.get("total_bytes") not in (None, total_bytes):
        raise ValueError("model manifest total_bytes does not match its file rows")
    content_sha256 = _digest_rows(rows)
    if content_sha256 != manifest["content_sha256"]:
        raise ValueError("model manifest content digest does not match its file rows")
    return str(manifest["content_sha256"])


def model_digest_from_env() -> str | None:
    value = os.environ.get("TREX_MODEL_DIGEST", "").strip()
    return value or None

