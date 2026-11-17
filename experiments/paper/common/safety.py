"""Validate every experiment output before creating directories or files."""

from __future__ import annotations

import json
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROTECTED_ROOTS = (
    Path("/media/Elements/4DFlow"),
    Path("/media/fudan22/4DFlow"),
    Path("/SSDHome/share/4dFlow"),
    Path("/mnt/nas/nas3/openData/rawdata/4dFlow"),
)


class OutputGuard:
    def __init__(self, protected=()):
        resolved = sorted({p.resolve() for p in (*PROTECTED_ROOTS, *map(Path, protected))}, key=lambda p: len(p.parts))
        roots = []
        for path in resolved:
            if not any(path.is_relative_to(root) for root in roots):
                roots.append(path)
        self.protected = tuple(roots)

    def check(self, path):
        path = Path(path).expanduser().resolve()
        if any(path == p or path.is_relative_to(p) for p in self.protected):
            raise ValueError(f"Refusing output inside read-only dataset: {path}")
        if path == ROOT or self.protected and any(p.is_relative_to(path) for p in self.protected):
            raise ValueError(f"Refusing output at an ancestor of protected data: {path}")
        return path

    def directory(self, path):
        path = self.check(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def json(self, path, data):
        path = self.check(path)
        self.directory(path.parent)
        temporary = self.check(path.with_suffix(path.suffix + f".tmp-{uuid.uuid4().hex}"))
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
        return path


def sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
