"""Content-addressed blob storage for incoming documents."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

# Magic bytes we can identify without trusting the supplied content type.
_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"%PDF", ".pdf"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
    (b"BM", ".bmp"),
    (b"II*\x00", ".tif"),
    (b"MM\x00*", ".tif"),
)
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class StoredBlob:
    path: Path
    sha256: str
    byte_size: int
    filename: str


def sniff_suffix(content: bytes) -> str | None:
    """Detect the file type from its leading bytes."""
    header = content[:16]
    for signature, suffix in _SIGNATURES:
        if header.startswith(signature):
            return suffix
    if content[8:12] == b"WEBP":
        return ".webp"
    return None


def safe_filename(name: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", name.replace("\\", "/").rsplit("/", 1)[-1]).strip("._")
    return cleaned[:120] or "document"


def store_bytes(content: bytes, filename: str, blob_dir: Path) -> StoredBlob:
    """Write bytes to content-addressed storage.

    The extension is taken from the file's own magic bytes when they are
    recognised, so a PDF uploaded as `invoice.txt` is still rendered correctly.
    """
    if not content:
        raise ValueError("Refusing to store an empty document.")

    digest = hashlib.sha256(content).hexdigest()
    name = safe_filename(filename)
    sniffed = sniff_suffix(content)
    suffix = sniffed or Path(name).suffix.lower() or ".bin"
    if sniffed and Path(name).suffix.lower() != sniffed:
        name = f"{Path(name).stem}{sniffed}"

    blob_dir.mkdir(parents=True, exist_ok=True)
    path = blob_dir / f"{digest[:16]}{suffix}"
    if not path.exists():
        path.write_bytes(content)

    return StoredBlob(path=path, sha256=digest, byte_size=len(content), filename=name)
