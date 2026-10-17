"""Content-addressed object storage for images and overlays (§16, §25).

Objects are keyed by ``sha256`` of their bytes — upload filenames are never
used, logged, or stored (I-6). Postgres stores only the URI + hash. Bindings:
``mem://`` (tests), ``file://`` (dev/compose shared volume). An S3-compatible
binding is a deployment seam (one URI scheme) not implemented in V1 — the
factory fails closed on ``s3://`` instead of silently writing elsewhere.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from consilium.core.errors import InputRejected

__all__ = [
    "FileObjectStore",
    "MemoryObjectStore",
    "ObjectStore",
    "make_object_store",
    "sha256_hex",
]


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ObjectStore(Protocol):
    async def put(self, data: bytes, *, prefix: str, key: str | None = None) -> str:
        """Store ``data`` under ``<prefix>/<key>`` (default key: sha256 of the
        bytes — content addressing); return its URI. Idempotent."""
        ...

    async def load(self, uri: str) -> bytes: ...

    async def exists(self, uri: str) -> bool: ...
    async def delete(self, uri: str) -> bool:
        """Remove an object (right-to-erasure only). True if it existed."""
        ...

    def uri_for(self, digest: str, *, prefix: str) -> str: ...


@dataclass(slots=True)
class MemoryObjectStore:
    objects: dict[str, bytes] = field(default_factory=dict)

    def uri_for(self, digest: str, *, prefix: str) -> str:
        return f"mem://{prefix}/{digest}"

    async def put(self, data: bytes, *, prefix: str, key: str | None = None) -> str:
        uri = self.uri_for(key or sha256_hex(data), prefix=prefix)
        self.objects.setdefault(uri, data)
        return uri

    async def load(self, uri: str) -> bytes:
        try:
            return self.objects[uri]
        except KeyError:
            raise InputRejected("IMAGE_NOT_FOUND", detail="object missing") from None

    async def exists(self, uri: str) -> bool:
        return uri in self.objects

    async def delete(self, uri: str) -> bool:
        return self.objects.pop(uri, None) is not None


@dataclass(frozen=True, slots=True)
class FileObjectStore:
    root: Path

    def uri_for(self, digest: str, *, prefix: str) -> str:
        return f"file://{prefix}/{digest}"

    def _path(self, uri: str) -> Path:
        if not uri.startswith("file://"):
            raise InputRejected("IMAGE_NOT_FOUND", detail="foreign object uri scheme")
        rel = uri.removeprefix("file://")
        path = (self.root / rel).resolve()
        if self.root.resolve() not in path.parents:  # no traversal out of the store
            raise InputRejected("IMAGE_NOT_FOUND", detail="object uri outside store")
        return path

    async def put(self, data: bytes, *, prefix: str, key: str | None = None) -> str:
        uri = self.uri_for(key or sha256_hex(data), prefix=prefix)
        path = self._path(uri)

        def _write() -> None:
            if path.exists():
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)  # atomic publish

        await asyncio.to_thread(_write)
        return uri

    async def load(self, uri: str) -> bytes:
        path = self._path(uri)
        try:
            return await asyncio.to_thread(path.read_bytes)
        except FileNotFoundError:
            raise InputRejected("IMAGE_NOT_FOUND", detail="object missing") from None

    async def exists(self, uri: str) -> bool:
        return await asyncio.to_thread(self._path(uri).exists)

    async def delete(self, uri: str) -> bool:
        path = self._path(uri)

        def _unlink() -> bool:
            try:
                path.unlink()
            except FileNotFoundError:
                return False
            return True

        return await asyncio.to_thread(_unlink)


def make_object_store(uri: str) -> ObjectStore:
    if uri.startswith("mem://"):
        return MemoryObjectStore()
    if uri.startswith("file://"):
        return FileObjectStore(Path(uri.removeprefix("file://")))
    raise ValueError(
        "unsupported OBJECT_STORE_URI scheme (V1 supports file:// and mem://; "
        "S3 binding is a deployment seam — fail closed)"
    )
