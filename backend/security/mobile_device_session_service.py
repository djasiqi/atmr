"""Service de sessions durables mobile (MobileDeviceSession).

Invariant : status=active ⇒ l'appareil peut obtenir de nouveaux credentials
indépendamment du temps écoulé, tant que la session n'est pas révoquée.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from ext import db
from models.mobile_device_session import (
    AuthRotationResult,
    MobileDeviceSession,
    MobileDeviceSessionStatus,
)

logger = logging.getLogger(__name__)

AUTH_CONTRACT_VERSION = "mobile-device-session-v1"
PREVIOUS_CREDENTIAL_GRACE_SECONDS = int(
    os.getenv("MOBILE_SESSION_PREVIOUS_CREDENTIAL_GRACE_SECONDS", "300")
)
DEFAULT_DEVICE_SESSION_LIMIT = int(os.getenv("MAX_MOBILE_DEVICE_SESSIONS_DRIVER", "5"))
ROTATION_RESULT_TTL_SECONDS = int(os.getenv("AUTH_ROTATION_RESULT_TTL_SECONDS", "600"))
SESSION_CACHE_TTL_SECONDS = int(os.getenv("MOBILE_SESSION_CACHE_TTL_SECONDS", "5"))
SESSION_NEGATIVE_CACHE_TTL_SECONDS = int(
    os.getenv("MOBILE_SESSION_NEGATIVE_CACHE_TTL_SECONDS", "30")
)


def auth_capabilities() -> dict[str, Any]:
    return {
        "auth_contract_version": AUTH_CONTRACT_VERSION,
        "capabilities": {
            "durable_device_session": True,
            "idempotent_rotation": True,
            "session_resume": True,
            "session_validation": True,
        },
    }


def _hash_secret(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def generate_opaque_secret() -> str:
    """Secret opaque ≥256 bits d'entropie."""
    return secrets.token_urlsafe(32)


def hash_credential(credential: str) -> str:
    return _hash_secret(credential)


def hash_revocation_secret(secret: str) -> str:
    return _hash_secret(secret)


def hash_idempotency_key(key: str) -> str:
    return _hash_secret(key)


def _now() -> datetime:
    return datetime.now(UTC)


def get_device_session_limit(role: str | None = None) -> int:
    if (role or "").lower() == "driver":
        return int(os.getenv("MAX_MOBILE_DEVICE_SESSIONS_DRIVER", str(DEFAULT_DEVICE_SESSION_LIMIT)))
    return int(os.getenv("MAX_MOBILE_DEVICE_SESSIONS", str(DEFAULT_DEVICE_SESSION_LIMIT)))


def list_active_sessions(user_id: int) -> list[MobileDeviceSession]:
    return (
        MobileDeviceSession.query.filter(
            MobileDeviceSession.user_id == user_id,
            MobileDeviceSession.status == MobileDeviceSessionStatus.active,
        )
        .order_by(MobileDeviceSession.last_seen_at.desc().nullslast())
        .all()
    )


def get_session_by_id(
    session_id: uuid.UUID | str, *, for_update: bool = False
) -> MobileDeviceSession | None:
    try:
        sid = session_id if isinstance(session_id, uuid.UUID) else uuid.UUID(str(session_id))
    except (ValueError, TypeError):
        return None
    query = MobileDeviceSession.query.filter_by(session_id=sid)
    if for_update:
        # Verrou ligne : évite les rotations concurrentes du même appareil (retries HTTP)
        query = query.with_for_update()
    return query.first()


def find_session_for_installation(
    user_id: int, device_installation_id: str
) -> MobileDeviceSession | None:
    return MobileDeviceSession.query.filter_by(
        user_id=user_id,
        device_installation_id=device_installation_id,
    ).first()


class DeviceSessionLimitReached(Exception):
    def __init__(self, sessions: list[MobileDeviceSession]):
        self.sessions = sessions
        super().__init__("device_session_limit_reached")


def find_active_session_for_installation(
    user_id: int, device_installation_id: str
) -> MobileDeviceSession | None:
    return MobileDeviceSession.query.filter_by(
        user_id=user_id,
        device_installation_id=device_installation_id,
        status=MobileDeviceSessionStatus.active,
    ).first()


def create_or_reuse_session(
    *,
    user_id: int,
    device_installation_id: str,
    device_name: str | None = None,
    driver_id: int | None = None,
    role: str | None = None,
    app_version: str | None = None,
    platform: str | None = None,
    context_id: str | None = None,
) -> tuple[MobileDeviceSession, str, str]:
    """Crée ou tourne les credentials d'une session active pour une installation.

    Sessions terminales : jamais réactivées — nouvelle ligne + nouveau session_id.
    Concurrence : IntegrityError sur index partiel → recharge gagnante + politique active.
    """
    from sqlalchemy.exc import IntegrityError

    for _attempt in range(2):
        recovery = generate_opaque_secret()
        revocation = generate_opaque_secret()
        now = _now()

        existing = find_active_session_for_installation(user_id, device_installation_id)
        if existing is not None:
            existing.previous_credential_hash = existing.credential_hash
            existing.previous_generation = existing.credential_generation or existing.generation
            existing.previous_credential_valid_until = now + timedelta(
                seconds=PREVIOUS_CREDENTIAL_GRACE_SECONDS
            )
            existing.credential_hash = hash_credential(recovery)
            existing.revocation_secret_hash = hash_revocation_secret(revocation)
            new_cred_gen = int(existing.credential_generation or existing.generation or 1) + 1
            existing.credential_generation = new_cred_gen
            existing.generation = new_cred_gen  # alias legacy
            # session_epoch inchangé (même session active)
            existing.device_name = device_name or existing.device_name
            existing.driver_id = driver_id if driver_id is not None else existing.driver_id
            existing.last_seen_at = now
            existing.last_refresh_at = now
            existing.last_app_version = app_version
            existing.last_platform = platform
            existing.last_context_id = context_id
            db.session.add(existing)
            db.session.flush()
            _invalidate_session_cache(existing.session_id)
            return existing, recovery, revocation

        active = list_active_sessions(user_id)
        limit = get_device_session_limit(role)
        if len(active) >= limit:
            raise DeviceSessionLimitReached(active)

        session = MobileDeviceSession(
            session_id=uuid.uuid4(),
            user_id=user_id,
            driver_id=driver_id,
            device_installation_id=device_installation_id,
            device_name=device_name,
            status=MobileDeviceSessionStatus.active,
            credential_hash=hash_credential(recovery),
            revocation_secret_hash=hash_revocation_secret(revocation),
            generation=1,
            session_epoch=1,
            credential_generation=1,
            refresh_generation=1,
            last_seen_at=now,
            last_refresh_at=now,
            last_app_version=app_version,
            last_platform=platform,
            last_context_id=context_id,
        )
        db.session.add(session)
        try:
            with db.session.begin_nested():
                db.session.flush()
            return session, recovery, revocation
        except IntegrityError:
            # Concurrent create : recharger la session gagnante
            continue

    # Dernier recours après 2 tentatives
    existing = find_active_session_for_installation(user_id, device_installation_id)
    if existing is not None:
        recovery = generate_opaque_secret()
        revocation = generate_opaque_secret()
        now = _now()
        existing.credential_hash = hash_credential(recovery)
        existing.revocation_secret_hash = hash_revocation_secret(revocation)
        new_cred_gen = int(existing.credential_generation or existing.generation or 1) + 1
        existing.credential_generation = new_cred_gen
        existing.generation = new_cred_gen
        existing.last_seen_at = now
        db.session.add(existing)
        db.session.flush()
        return existing, recovery, revocation
    raise DeviceSessionLimitReached([])


def verify_recovery_credential(
    session: MobileDeviceSession, credential: str
) -> bool:
    h = hash_credential(credential)
    if secrets.compare_digest(h, session.credential_hash):
        return True
    if (
        session.previous_credential_hash
        and session.previous_credential_valid_until
        and session.previous_credential_valid_until > _now()
        and secrets.compare_digest(h, session.previous_credential_hash)
    ):
        return True
    return False


def rotate_recovery_credential(
    session: MobileDeviceSession,
) -> str:
    """Tourne le recovery credential ; incrémente credential_generation (pas session_epoch)."""
    now = _now()
    new_secret = generate_opaque_secret()
    session.previous_credential_hash = session.credential_hash
    session.previous_generation = session.credential_generation or session.generation
    session.previous_credential_valid_until = now + timedelta(
        seconds=PREVIOUS_CREDENTIAL_GRACE_SECONDS
    )
    session.credential_hash = hash_credential(new_secret)
    new_gen = int(session.credential_generation or session.generation or 1) + 1
    session.credential_generation = new_gen
    session.generation = new_gen  # alias legacy
    session.last_refresh_at = now
    session.last_seen_at = now
    return new_secret


def bump_refresh_generation(session: MobileDeviceSession) -> int:
    """Incrémente refresh_generation sans toucher session_epoch ni recovery."""
    session.refresh_generation = int(getattr(session, "refresh_generation", 1) or 1) + 1
    session.last_refresh_at = _now()
    session.last_seen_at = session.last_refresh_at
    return int(session.refresh_generation)


def revoke_session(
    session: MobileDeviceSession,
    *,
    reason: str,
    revoked_by_user_id: int | None = None,
    status: MobileDeviceSessionStatus = MobileDeviceSessionStatus.revoked,
) -> None:
    session.status = status
    session.revoked_at = _now()
    session.revoked_reason = reason
    session.revoked_by_user_id = revoked_by_user_id
    # Rendre tokens/caches structurellement obsolètes
    session.session_epoch = int(getattr(session, "session_epoch", 1) or 1) + 1
    _cache_session_revoked(session.session_id)
    _invalidate_session_cache(session.session_id)


def revoke_all_user_sessions(
    user_id: int,
    *,
    reason: str,
    status: MobileDeviceSessionStatus = MobileDeviceSessionStatus.security_revoked,
    except_session_id: uuid.UUID | None = None,
) -> int:
    count = 0
    for sess in list_active_sessions(user_id):
        if except_session_id and sess.session_id == except_session_id:
            continue
        revoke_session(sess, reason=reason, status=status)
        count += 1
    return count


def revoke_user_security_sessions(
    user: Any,
    *,
    status: MobileDeviceSessionStatus = MobileDeviceSessionStatus.security_revoked,
    reason: str = "password_reset",
    increment_token_version: bool = True,
) -> int:
    """Révoque toutes les sessions actives après un événement de sécurité.

    Couvre reset/changement MDP, reset admin, révocation globale.
    """
    if increment_token_version and hasattr(user, "token_version"):
        user.token_version = int(getattr(user, "token_version", 0) or 0) + 1
    count = revoke_all_user_sessions(user.id, reason=reason, status=status)
    try:
        from security.refresh_token_service import revoke_all_user_tokens

        revoke_all_user_tokens(user.id, reason=reason)
    except Exception as exc:
        logger.warning("revoke_all_user_tokens après security revoke: %s", exc)
    return count


def disable_user_sessions(
    user: Any,
    *,
    status: MobileDeviceSessionStatus = MobileDeviceSessionStatus.account_disabled,
    reason: str = "account_disabled",
    increment_token_version: bool = True,
) -> int:
    """Révoque les sessions lors d'une désactivation de compte / profil."""
    return revoke_user_security_sessions(
        user,
        status=status,
        reason=reason,
        increment_token_version=increment_token_version,
    )


def verify_revocation_secret(session: MobileDeviceSession, secret: str) -> bool:
    return secrets.compare_digest(
        hash_revocation_secret(secret), session.revocation_secret_hash
    )


def consume_revocation_secret(session: MobileDeviceSession, secret: str) -> bool:
    """Legacy one-shot (sans operation_id). Préférer revoke_pending_idempotent."""
    if not verify_revocation_secret(session, secret):
        return False
    session.revocation_secret_hash = hash_revocation_secret(generate_opaque_secret())
    if session.is_active():
        revoke_session(session, reason="pending_revocation")
    return True


def revoke_pending_idempotent(
    session: MobileDeviceSession,
    secret: str,
    *,
    operation_id: str | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Révoque via revocation_secret avec rejeu idempotent (perte d'ACK).

    Returns:
        (payload_ok, error_code) — error_code None si succès.
    """
    if operation_id:
        existing = get_rotation_result(session.session_id, str(operation_id))
        if existing is not None:
            cached = load_rotation_response(existing)
            if cached is not None:
                secret_hash = cached.get("secret_hash")
                if secret_hash and secrets.compare_digest(
                    str(secret_hash), hash_revocation_secret(secret)
                ):
                    return (
                        {
                            "ok": True,
                            "already_revoked": True,
                            "error_code": cached.get("error_code"),
                        },
                        None,
                    )
                # Même operation_id mais preuve différente → rejeter
                return None, "invalid_revocation_secret"

    if not verify_revocation_secret(session, secret):
        return None, "invalid_revocation_secret"

    already = not session.is_active()
    if session.is_active():
        revoke_session(session, reason="pending_revocation")

    # One-shot après succès (rejeu via operation_id + receipt)
    consumed_hash = hash_revocation_secret(secret)
    session.revocation_secret_hash = hash_revocation_secret(generate_opaque_secret())

    payload = {
        "ok": True,
        "already_revoked": already,
        "secret_hash": consumed_hash,
    }
    if operation_id:
        store_rotation_result(
            session=session,
            idempotency_key=str(operation_id),
            request_generation=int(session.generation or 1),
            successor_generation=int(session.generation or 1),
            response_payload=payload,
            operation_type="logout_pending",
        )
    return {"ok": True, "already_revoked": already}, None


# --- Cache Redis session validation ---

def _cache_key(session_id: uuid.UUID | str) -> str:
    return f"auth:mobile_session:{session_id}"


def _get_redis():
    try:
        from shared.redis_client import get_redis_client

        return get_redis_client()
    except Exception:
        return None


def cache_session_snapshot(session: MobileDeviceSession) -> None:
    r = _get_redis()
    if not r:
        return
    try:
        import json

        payload = json.dumps(
            {
                "status": session.status.value,
                "generation": session.generation,
                "session_epoch": int(getattr(session, "session_epoch", 1) or 1),
                "credential_generation": int(
                    getattr(session, "credential_generation", session.generation) or 1
                ),
                "refresh_generation": int(
                    getattr(session, "refresh_generation", 1) or 1
                ),
                "user_id": session.user_id,
            }
        )
        r.setex(_cache_key(session.session_id), SESSION_CACHE_TTL_SECONDS, payload)
    except Exception as exc:
        logger.debug("cache session snapshot failed: %s", exc)


def _cache_session_revoked(session_id: uuid.UUID) -> None:
    """Cache négatif immédiat après révocation (SLO ≤ 5 s)."""
    r = _get_redis()
    if not r:
        return
    try:
        import json

        payload = json.dumps({"status": "revoked", "session_epoch": -1})
        r.setex(
            _cache_key(session_id), SESSION_NEGATIVE_CACHE_TTL_SECONDS, payload
        )
    except Exception as exc:
        logger.debug("cache session revoked failed: %s", exc)


def _invalidate_session_cache(session_id: uuid.UUID) -> None:
    r = _get_redis()
    if not r:
        return
    try:
        r.delete(_cache_key(session_id))
    except Exception:
        pass


def validate_mobile_session(
    *,
    session_id: str | None,
    session_generation: int | None = None,
    session_epoch: int | None = None,
    user_id: int | None = None,
    bypass_positive_cache: bool = False,
) -> tuple[str | None, bool]:
    """Valide une session mobile.

    Returns:
        (error_code | None, retryable)
        None error = OK

    bypass_positive_cache: ops sensibles (logout-all, MDP) → toujours PostgreSQL.
    """
    if not session_id:
        # Compat legacy : pas encore de session_id
        return None, False

    try:
        sid = uuid.UUID(str(session_id))
    except (ValueError, TypeError):
        return "session_revoked", False

    # epoch effectif : nouveau claim ou legacy session_generation
    effective_epoch = session_epoch
    if effective_epoch is None and session_generation is not None:
        effective_epoch = session_generation

    # Cache Redis (sauf ops sensibles)
    r = None if bypass_positive_cache else _get_redis()
    if r:
        try:
            import json

            raw = r.get(_cache_key(sid))
            if raw:
                data = json.loads(raw)
                if data.get("status") != "active":
                    return "session_revoked", False
                cached_epoch = data.get("session_epoch", data.get("generation"))
                if (
                    effective_epoch is not None
                    and cached_epoch is not None
                    and int(cached_epoch) != int(effective_epoch)
                ):
                    return "session_revoked", False
                if user_id is not None and int(data.get("user_id", -1)) != int(user_id):
                    return "session_revoked", False
                return None, False
        except Exception:
            pass

    try:
        session = get_session_by_id(sid)
    except Exception:
        return "session_validation_unavailable", True

    if session is None or not session.is_active():
        _cache_session_revoked(sid)
        return "session_revoked", False
    if user_id is not None and session.user_id != user_id:
        return "session_revoked", False
    db_epoch = int(getattr(session, "session_epoch", session.generation) or 1)
    if effective_epoch is not None and db_epoch != int(effective_epoch):
        return "session_revoked", False

    cache_session_snapshot(session)
    return None, False


# --- Idempotence rotation (AuthRotationResult) ---

def _get_encryption_key() -> tuple[bytes, str]:
    """Clé AEAD hors logique métier — réutilise APP_ENCRYPTION_KEY_B64."""
    from models.base import _load_encryption_key

    key = _load_encryption_key()
    key_id = (os.getenv("AUTH_ROTATION_ENCRYPTION_KEY_ID") or "v1").strip()
    return key, key_id


def encrypt_rotation_response(payload: dict[str, Any]) -> tuple[bytes, str]:
    import json
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key, key_id = _get_encryption_key()
    # AESGCM exige 16/24/32 ; tronquer/hasher si besoin
    if len(key) not in (16, 24, 32):
        key = hashlib.sha256(key).digest()
    aesgcm = AESGCM(key)
    nonce = secrets.token_bytes(12)
    plaintext = json.dumps(payload).encode("utf-8")
    ct = aesgcm.encrypt(nonce, plaintext, None)
    return nonce + ct, key_id


def decrypt_rotation_response(ciphertext: bytes, key_id: str) -> dict[str, Any]:
    import json
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key, _ = _get_encryption_key()
    if len(key) not in (16, 24, 32):
        key = hashlib.sha256(key).digest()
    aesgcm = AESGCM(key)
    nonce, ct = ciphertext[:12], ciphertext[12:]
    plaintext = aesgcm.decrypt(nonce, ct, None)
    return json.loads(plaintext.decode("utf-8"))


def store_rotation_result(
    *,
    session: MobileDeviceSession,
    idempotency_key: str,
    request_generation: int,
    successor_generation: int,
    response_payload: dict[str, Any],
    operation_type: str = "refresh",
) -> AuthRotationResult:
    ciphertext, key_id = encrypt_rotation_response(response_payload)
    row = AuthRotationResult(
        id=uuid.uuid4(),
        session_id=session.session_id,
        idempotency_key_hash=hash_idempotency_key(idempotency_key),
        request_generation=request_generation,
        successor_generation=successor_generation,
        response_ciphertext=ciphertext,
        encryption_key_id=key_id,
        operation_type=operation_type,
        expires_at=_now() + timedelta(seconds=ROTATION_RESULT_TTL_SECONDS),
    )
    db.session.add(row)
    return row


def get_rotation_result(
    session_id: uuid.UUID, idempotency_key: str
) -> AuthRotationResult | None:
    return AuthRotationResult.query.filter_by(
        session_id=session_id,
        idempotency_key_hash=hash_idempotency_key(idempotency_key),
    ).first()


def load_rotation_response(row: AuthRotationResult) -> dict[str, Any] | None:
    if row.expires_at and row.expires_at < _now():
        return None
    try:
        return decrypt_rotation_response(row.response_ciphertext, row.encryption_key_id)
    except Exception as exc:
        logger.warning("decrypt AuthRotationResult failed: %s", exc)
        return None
