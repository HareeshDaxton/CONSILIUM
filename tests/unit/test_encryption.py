"""Column-encryption helper unit tests (offline, no DB)."""

import base64

import pytest

from consilium.persistence.encryption import AesGcmFieldEncryptor, parse_key


def test_round_trip() -> None:
    enc = AesGcmFieldEncryptor(b"k" * 32)
    blob = enc.encrypt("synthetic redacted note")
    assert b"synthetic" not in blob
    assert enc.decrypt(blob) == "synthetic redacted note"


def test_random_nonce_per_value() -> None:
    enc = AesGcmFieldEncryptor(b"k" * 32)
    assert enc.encrypt("same") != enc.encrypt("same")


def test_wrong_key_fails_closed() -> None:
    from cryptography.exceptions import InvalidTag

    blob = AesGcmFieldEncryptor(b"k" * 32).encrypt("secret")
    with pytest.raises(InvalidTag):
        AesGcmFieldEncryptor(b"x" * 32).decrypt(blob)


def test_key_must_be_32_bytes() -> None:
    with pytest.raises(ValueError, match="32 bytes"):
        AesGcmFieldEncryptor(b"short")


def test_parse_key_accepts_raw_and_base64() -> None:
    assert parse_key("k" * 32) == b"k" * 32
    b64 = base64.urlsafe_b64encode(b"z" * 32).decode()
    assert parse_key(b64) == b"z" * 32
    with pytest.raises(ValueError, match="32 bytes"):
        parse_key("not-a-key")
