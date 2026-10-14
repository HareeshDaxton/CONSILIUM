"""Column-level encryption helper for PHI-bearing columns (ARCHITECTURE.md §11).

AES-256-GCM with a random 96-bit nonce per value; ciphertext layout is
``nonce || ciphertext+tag``. The key comes from configuration
(``Settings.field_encryption_key``, env ``FIELD_ENCRYPTION_KEY``) as 32 raw
bytes or a base64/urlsafe-base64 encoding of 32 bytes. The helper NEVER logs
plaintext or keys, and only pattern names (never values) may travel in
provenance fields (I-6).

Used by the repositories for ``clinical_notes.raw_redacted_text_enc`` — the
only column holding redacted note text. ``cases.state_json`` must never
contain it (graph/state.py excludes the field from persisted state).
"""

from __future__ import annotations

import base64
import binascii
import os
from typing import Protocol

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_NONCE_BYTES = 12  # AES-GCM standard nonce size


class FieldEncryptor(Protocol):
    def encrypt(self, plaintext: str) -> bytes: ...
    def decrypt(self, ciphertext: bytes) -> str: ...


class AesGcmFieldEncryptor:
    """AES-256-GCM encryptor. Immutable; safe to share across sessions."""

    def __init__(self, key: bytes) -> None:
        if len(key) != 32:
            raise ValueError("field encryption key must be exactly 32 bytes (AES-256)")
        self._aead = AESGCM(key)

    def encrypt(self, plaintext: str) -> bytes:
        nonce = os.urandom(_NONCE_BYTES)
        return nonce + self._aead.encrypt(nonce, plaintext.encode("utf-8"), None)

    def decrypt(self, ciphertext: bytes) -> str:
        if len(ciphertext) < _NONCE_BYTES:
            raise ValueError("ciphertext too short")
        nonce, body = ciphertext[:_NONCE_BYTES], ciphertext[_NONCE_BYTES:]
        return self._aead.decrypt(nonce, body, None).decode("utf-8")


def parse_key(value: str) -> bytes:
    """Accept raw 32-char strings or (urlsafe-)base64 encodings of 32 bytes."""
    raw = value.encode("utf-8")
    if len(raw) == 32:
        return raw
    for decoder in (base64.urlsafe_b64decode, base64.b64decode):
        try:
            decoded = decoder(raw)
        except (binascii.Error, ValueError):
            continue
        if len(decoded) == 32:
            return decoded
    raise ValueError("FIELD_ENCRYPTION_KEY must be 32 bytes or a base64 encoding of 32 bytes")
