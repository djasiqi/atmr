"""Couverture de ``security.crypto`` (AES-256, rotation de clés)."""

from __future__ import annotations

import base64
import os

import pytest

from security.crypto import (
    DEFAULT_KEY_LENGTH,
    EncryptionService,
    add_legacy_key,
    get_encryption_service,
    reset_encryption_service,
    rotate_to_new_key,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    reset_encryption_service()
    yield
    reset_encryption_service()


def test_derive_key_et_encrypt_erreur(monkeypatch):
    service = EncryptionService()
    derived = service._derive_key(b"password", b"salt" * 4)
    assert isinstance(derived, bytes)
    assert len(derived) == DEFAULT_KEY_LENGTH

    monkeypatch.setattr(
        "security.crypto.Cipher",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("cipher")),
    )
    with pytest.raises(RuntimeError, match="cipher"):
        service.encrypt_field("secret")


def test_decrypt_base64_invalide_et_trop_court(monkeypatch):
    service = EncryptionService()

    def _bad_b64(_data):
        raise ValueError("bad b64")

    monkeypatch.setattr("security.crypto.base64.b64decode", _bad_b64)
    with pytest.raises(ValueError, match="bad b64"):
        service.decrypt_field("AAAA")

    monkeypatch.undo()
    too_short = base64.b64encode(b"short").decode("utf-8")
    with pytest.raises(ValueError, match="trop court"):
        service.decrypt_field(too_short)


def test_decrypt_legacy_et_echec_toutes_cles():
    original = EncryptionService()
    ciphertext = original.encrypt_field("données")

    rotated = EncryptionService(
        master_key=os.urandom(32),
        legacy_keys=[original.master_key],
    )
    assert rotated.decrypt_field(ciphertext) == "données"

    other = EncryptionService(master_key=os.urandom(32))
    with pytest.raises(ValueError, match=r"(?i)padding|invalid"):
        other.decrypt_field(ciphertext)


def test_add_legacy_key_et_rotation():
    master = os.urandom(32)
    extra = os.urandom(32)
    service = EncryptionService(master_key=master)

    add_legacy_key(service, extra)
    assert extra in service.legacy_keys
    add_legacy_key(service, extra)
    add_legacy_key(service, master)
    assert service.legacy_keys.count(extra) == 1
    assert master not in service.legacy_keys

    ciphertext = service.encrypt_field("avant-rotation")
    old = rotate_to_new_key(service, extra)
    assert old == master
    assert service.master_key == extra
    assert master in service.legacy_keys
    assert service.decrypt_field(ciphertext) == "avant-rotation"

    third = os.urandom(32)
    rotate_to_new_key(service, third)
    rotate_to_new_key(service, extra)
    assert service.legacy_keys.count(extra) == 1


def test_get_encryption_service_cles_env(monkeypatch):
    master = os.urandom(32)
    legacy = os.urandom(32)
    monkeypatch.setenv("MASTER_ENCRYPTION_KEY", master.hex())
    monkeypatch.setenv(
        "LEGACY_ENCRYPTION_KEYS",
        f"{legacy.hex()}, not-a-hex, {master.hex()},   ",
    )
    service = get_encryption_service()
    assert service.master_key == master
    assert legacy in service.legacy_keys
    assert master not in service.legacy_keys

    reset_encryption_service()
    monkeypatch.setenv("MASTER_ENCRYPTION_KEY", "zzzz")
    monkeypatch.delenv("LEGACY_ENCRYPTION_KEYS", raising=False)
    invalid = get_encryption_service()
    assert len(invalid.master_key) == DEFAULT_KEY_LENGTH

    reset_encryption_service()
    monkeypatch.delenv("MASTER_ENCRYPTION_KEY", raising=False)
    generated = get_encryption_service()
    assert len(generated.master_key) == DEFAULT_KEY_LENGTH
