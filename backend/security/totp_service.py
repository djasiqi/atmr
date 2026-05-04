"""Service TOTP 2FA : generation, verification, recovery codes, chiffrement."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import secrets
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import redis as redis_mod

logger = logging.getLogger(__name__)

_ENCRYPTION_KEY = os.environ.get("TOTP_ENCRYPTION_KEY") or os.environ.get(
    "MASTER_ENCRYPTION_KEY", ""
)
_RECOVERY_CODE_COUNT = 10
_MAX_2FA_FAILURES = 10
_LOCKOUT_TTL_SECONDS = 1800
_CHALLENGE_JTI_TTL_SECONDS = 300


def _get_cipher_key() -> bytes:
    """Derive a 32-byte key from the configured hex key."""
    if not _ENCRYPTION_KEY:
        raise RuntimeError("TOTP_ENCRYPTION_KEY or MASTER_ENCRYPTION_KEY must be set")
    return bytes.fromhex(_ENCRYPTION_KEY[:64].ljust(64, "0"))


def encrypt_secret(plain: str) -> str:
    """Chiffre un secret TOTP avec AES-256-GCM. Retourne base64(nonce+ciphertext+tag)."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key = _get_cipher_key()
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, plain.encode("utf-8"), None)
    return base64.b64encode(nonce + ct).decode("ascii")


def decrypt_secret(encrypted: str) -> str:
    """Dechiffre un secret TOTP."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key = _get_cipher_key()
    raw = base64.b64decode(encrypted)
    nonce, ct = raw[:12], raw[12:]
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ct, None).decode("utf-8")


def generate_totp_secret(user_email: str, issuer: str = "ATMR") -> dict[str, str]:
    """Genere un secret TOTP + provisioning URI + QR code base64.

    Returns:
        {provisioning_uri, qr_code_base64, secret_display, secret_encrypted}
    """
    import pyotp  # type: ignore[import-untyped]
    import qrcode

    secret = pyotp.random_base32()
    totp = pyotp.TOTP(secret)
    uri = totp.provisioning_uri(name=user_email, issuer_name=issuer)

    img = qrcode.make(uri)
    buf = io.BytesIO()
    img.save(buf)
    qr_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "provisioning_uri": uri,
        "qr_code_base64": f"data:image/png;base64,{qr_b64}",
        "secret_display": secret,
        "secret_encrypted": encrypt_secret(secret),
    }


def verify_totp_code(encrypted_secret: str, code: str) -> bool:
    """Verifie un code TOTP a 6 chiffres (fenetre +/- 1)."""
    import pyotp  # type: ignore[import-untyped]

    secret = decrypt_secret(encrypted_secret)
    totp = pyotp.TOTP(secret)
    return totp.verify(code, valid_window=1)


def generate_recovery_codes(count: int = _RECOVERY_CODE_COUNT) -> tuple[list[str], str]:
    """Genere des recovery codes en clair + leurs hashes JSON.

    Returns:
        (plain_codes, hashes_json)
    """
    codes = [f"{secrets.randbelow(10**8):08d}" for _ in range(count)]
    hashes = [hashlib.sha256(c.encode()).hexdigest() for c in codes]
    return codes, json.dumps(hashes)


def verify_recovery_code(hashes_json: str, code: str) -> tuple[bool, str]:
    """Verifie un recovery code et le consomme.

    Returns:
        (is_valid, updated_hashes_json)
    """
    code_hash = hashlib.sha256(code.strip().encode()).hexdigest()
    hashes = json.loads(hashes_json) if hashes_json else []
    if code_hash in hashes:
        hashes.remove(code_hash)
        return True, json.dumps(hashes)
    return False, hashes_json


def _get_redis() -> redis_mod.Redis | None:
    """Return redis_client from ext, or None if unavailable."""
    try:
        from ext import redis_client

        return redis_client
    except Exception:
        return None


def check_2fa_lockout(user_id: int) -> bool:
    """Verifie si l'utilisateur est bloque pour trop de tentatives 2FA."""
    rc = _get_redis()
    if rc is None:
        return False
    try:
        raw = rc.get(f"2fa_failures:{user_id}")
        return int(raw) >= _MAX_2FA_FAILURES if raw is not None else False  # type: ignore[arg-type]
    except Exception:
        return False


def record_2fa_failure(user_id: int) -> int:
    """Incremente le compteur d'echecs 2FA. Bloque apres seuil pendant 30min."""
    rc = _get_redis()
    if rc is None:
        return 0
    try:
        key = f"2fa_failures:{user_id}"
        raw = rc.incr(key)
        count = int(raw)  # type: ignore[arg-type]
        if count == 1:
            rc.expire(key, _LOCKOUT_TTL_SECONDS)
        return count
    except Exception:
        return 0


def reset_2fa_failures(user_id: int) -> None:
    """Reset le compteur d'echecs apres un succes."""
    rc = _get_redis()
    if rc is None:
        return
    import contextlib

    with contextlib.suppress(Exception):
        rc.delete(f"2fa_failures:{user_id}")


def store_2fa_challenge_jti(jti: str) -> None:
    """Stocke un JTI de temp_token dans Redis (usage unique, 5min TTL)."""
    rc = _get_redis()
    if rc is None:
        logger.warning("Failed to store 2FA challenge JTI: Redis unavailable")
        return
    try:
        rc.set(f"2fa_challenge:{jti}", "1", ex=_CHALLENGE_JTI_TTL_SECONDS)
    except Exception:
        logger.warning("Failed to store 2FA challenge JTI in Redis")


def consume_2fa_challenge_jti(jti: str) -> bool:
    """Consomme un JTI (supprime de Redis). Retourne True si valide."""
    rc = _get_redis()
    if rc is None:
        return False
    try:
        raw = rc.delete(f"2fa_challenge:{jti}")
        return int(raw) > 0  # type: ignore[arg-type]
    except Exception:
        return False
