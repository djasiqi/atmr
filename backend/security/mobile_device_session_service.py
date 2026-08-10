"""Service de sessions durables mobile (MobileDeviceSession).

Invariant : status=active ⇒ l'appareil peut obtenir de nouveaux credentials
indépendamment du temps écoulé, tant que la session n'est pas révoquée.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.exc import IntegrityError

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
SESSION_RESUME_RESULT_TTL_SECONDS = int(
    os.getenv("AUTH_SESSION_RESUME_RESULT_TTL_SECONDS", "86400")
)
SESSION_CACHE_TTL_SECONDS = int(os.getenv("MOBILE_SESSION_CACHE_TTL_SECONDS", "5"))
SESSION_NEGATIVE_CACHE_TTL_SECONDS = int(
    os.getenv("MOBILE_SESSION_NEGATIVE_CACHE_TTL_SECONDS", "30")
)
RESOLUTION_TOKEN_TTL_SECONDS = int(
    os.getenv("MOBILE_DEVICE_RESOLUTION_TOKEN_TTL_SECONDS", "300")
)
RESOLUTION_CLAIM_TTL_SECONDS = int(
    os.getenv("MOBILE_DEVICE_RESOLUTION_CLAIM_TTL_SECONDS", "60")
)
PROVISIONAL_SESSION_TTL_SECONDS = int(
    os.getenv("MOBILE_DEVICE_PROVISIONAL_TTL_SECONDS", "900")
)
# P1 : replace + provisional activés côté contrat (mobile gated sur capabilities)
DEVICE_SESSION_REPLACE_ENABLED = os.getenv(
    "MOBILE_DEVICE_SESSION_REPLACE_ENABLED", "true"
).lower() in {"1", "true", "yes", "on"}
PROVISIONAL_CONFIRMATION_ENABLED = os.getenv(
    "MOBILE_DEVICE_PROVISIONAL_CONFIRMATION_ENABLED", "true"
).lower() in {"1", "true", "yes", "on"}
ROTATION_META_KEY = "_rotation_meta"
ROTATION_IDEMPOTENCY_CONSTRAINT = "uq_auth_rotation_result_session_idempotency"
RESOLUTION_TOKEN_PREFIX = "auth:device_session_resolution:"
RESOLUTION_CLAIM_PREFIX = "auth:device_session_resolution_claim:"


def auth_capabilities() -> dict[str, Any]:
    return {
        "auth_contract_version": AUTH_CONTRACT_VERSION,
        "capabilities": {
            "durable_device_session": True,
            "idempotent_rotation": True,
            "session_resume": True,
            "session_validation": True,
            "device_session_management": True,
            "device_session_replace": bool(DEVICE_SESSION_REPLACE_ENABLED),
            "provisional_session_confirmation": bool(PROVISIONAL_CONFIRMATION_ENABLED),
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
        return int(
            os.getenv(
                "MAX_MOBILE_DEVICE_SESSIONS_DRIVER", str(DEFAULT_DEVICE_SESSION_LIMIT)
            )
        )
    return int(
        os.getenv("MAX_MOBILE_DEVICE_SESSIONS", str(DEFAULT_DEVICE_SESSION_LIMIT))
    )


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
        sid = (
            session_id
            if isinstance(session_id, uuid.UUID)
            else uuid.UUID(str(session_id))
        )
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


class DeviceSessionResolutionError(Exception):
    def __init__(self, code: str, message: str = ""):
        self.code = code
        self.message = message or code
        super().__init__(self.message)


@dataclass
class DeviceSessionMetadata:
    """Métadonnées appareil best-effort (headers client)."""

    device_name: str | None = None
    device_model: str | None = None
    device_manufacturer: str | None = None
    device_type: str | None = None
    platform: str | None = None
    os_version: str | None = None
    app_version: str | None = None
    app_build: str | None = None
    context_id: str | None = None


def apply_session_metadata(
    session: MobileDeviceSession,
    meta: DeviceSessionMetadata | None,
    *,
    touch_seen: bool = True,
) -> None:
    """Rafraîchit les métadonnées présentes (login / resume / refresh / confirm)."""
    if meta is None:
        return
    changed = False
    if meta.device_name:
        session.device_name = meta.device_name[:255]
        changed = True
    if meta.device_model:
        session.device_model = meta.device_model[:128]
        changed = True
    if meta.device_manufacturer:
        session.device_manufacturer = meta.device_manufacturer[:128]
        changed = True
    if meta.device_type:
        session.device_type = meta.device_type[:32]
        changed = True
    if meta.platform:
        session.last_platform = meta.platform[:32]
        changed = True
    if meta.os_version:
        session.last_os_version = meta.os_version[:64]
        changed = True
    if meta.app_version:
        session.last_app_version = meta.app_version[:64]
        changed = True
    if meta.app_build:
        session.last_app_build = meta.app_build[:64]
        changed = True
    if meta.context_id:
        session.last_context_id = meta.context_id[:128]
        changed = True
    now = _now()
    if touch_seen:
        session.last_seen_at = now
    if changed:
        session.metadata_updated_at = now


def mark_session_confirmed(session: MobileDeviceSession) -> bool:
    """Confirme une session provisional (idempotent). Retourne True si transition."""
    if session.confirmed_at is not None:
        session.provisional_expires_at = None
        return False
    session.confirmed_at = _now()
    session.provisional_expires_at = None
    return True


def _resolution_redis_key(token: str) -> str:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return f"{RESOLUTION_TOKEN_PREFIX}{digest}"


def _resolution_claim_key(token: str) -> str:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return f"{RESOLUTION_CLAIM_PREFIX}{digest}"


def issue_device_session_resolution_token(
    *,
    user_id: int,
    requested_device_installation_id: str,
    allowed_sessions: list[MobileDeviceSession],
    ttl_seconds: int | None = None,
) -> str | None:
    """Émet un challenge Redis (état issued) avec snapshot des sessions autorisées.

    Retourne None si replace désactivé ou Redis indisponible (fail-soft UX).
    """
    if not DEVICE_SESSION_REPLACE_ENABLED:
        return None

    import json

    r = _get_redis()
    if not r:
        return None
    token = secrets.token_urlsafe(32)
    ttl = int(ttl_seconds or RESOLUTION_TOKEN_TTL_SECONDS)
    operation_id = secrets.token_urlsafe(16)
    allowed_ids = [str(s.session_id) for s in allowed_sessions]
    payload = {
        "scope": "device_session_resolution",
        "state": "issued",
        "operation_id": operation_id,
        "user_id": int(user_id),
        "requested_device_installation_id": str(requested_device_installation_id),
        "allowed_session_ids": allowed_ids,
        "session_set_version": hashlib.sha256(
            "|".join(sorted(allowed_ids)).encode("utf-8")
        ).hexdigest()[:16],
    }
    try:
        r.setex(_resolution_redis_key(token), ttl, json.dumps(payload))
        return token
    except Exception as exc:
        logger.warning("issue_device_session_resolution_token failed: %s", exc)
        return None


def claim_device_session_resolution_token(
    *,
    token: str,
    requested_device_installation_id: str,
) -> dict[str, Any]:
    """Passe issued → claimed (atomique). Lève DeviceSessionResolutionError."""
    import json

    raw_token = (token or "").strip()
    if not raw_token:
        raise DeviceSessionResolutionError("resolution_token_required")
    if not DEVICE_SESSION_REPLACE_ENABLED:
        raise DeviceSessionResolutionError("resolution_unavailable")

    r = _get_redis()
    if not r:
        raise DeviceSessionResolutionError(
            "resolution_unavailable",
            "Service de résolution temporairement indisponible.",
        )

    key = _resolution_redis_key(raw_token)
    claim_key = _resolution_claim_key(raw_token)
    try:
        raw = r.get(key)
    except Exception as exc:
        logger.warning("resolution token redis get failed: %s", exc)
        raise DeviceSessionResolutionError("resolution_unavailable") from exc

    if not raw:
        raise DeviceSessionResolutionError(
            "resolution_token_expired",
            "Le délai pour remplacer un appareil a expiré. Reconnectez-vous.",
        )

    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        payload = json.loads(raw)
    except Exception as exc:
        raise DeviceSessionResolutionError("resolution_token_invalid") from exc

    if payload.get("scope") != "device_session_resolution":
        raise DeviceSessionResolutionError("resolution_token_invalid")

    state = payload.get("state") or "issued"
    if state == "consumed":
        raise DeviceSessionResolutionError(
            "resolution_token_expired",
            "Ce challenge a déjà été utilisé.",
        )

    expected_device = str(payload.get("requested_device_installation_id") or "")
    if not expected_device or expected_device != str(requested_device_installation_id):
        raise DeviceSessionResolutionError(
            "resolution_device_mismatch",
            "Cet appareil ne correspond pas au challenge de résolution.",
        )

    # Claim atomique : empêche double replace concurrent ; reclaimable sur rollback.
    try:
        claimed = r.set(
            claim_key,
            payload.get("operation_id") or "1",
            nx=True,
            ex=RESOLUTION_CLAIM_TTL_SECONDS,
        )
        if not claimed:
            raise DeviceSessionResolutionError(
                "resolution_token_in_use",
                "Un remplacement est déjà en cours. Réessayez.",
            )
        payload["state"] = "claimed"
        ttl = r.ttl(key)
        if ttl and int(ttl) > 0:
            r.setex(key, int(ttl), json.dumps(payload))
        else:
            r.setex(key, RESOLUTION_TOKEN_TTL_SECONDS, json.dumps(payload))
    except DeviceSessionResolutionError:
        raise
    except Exception as exc:
        logger.warning("resolution claim failed: %s", exc)
        raise DeviceSessionResolutionError("resolution_unavailable") from exc

    return payload


def release_device_session_resolution_claim(*, token: str) -> None:
    """Libère le claim après rollback PG pour permettre un retry."""
    r = _get_redis()
    if not r:
        return
    with contextlib.suppress(Exception):
        r.delete(_resolution_claim_key(token))
    # Remettre state=issued si le token existe encore
    import json

    key = _resolution_redis_key(token)
    try:
        raw = r.get(key)
        if not raw:
            return
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        payload = json.loads(raw)
        if payload.get("state") == "claimed":
            payload["state"] = "issued"
            ttl = r.ttl(key)
            if ttl and int(ttl) > 0:
                r.setex(key, int(ttl), json.dumps(payload))
    except Exception as exc:
        logger.debug("release resolution claim failed: %s", exc)


def consume_device_session_resolution_token(*, token: str) -> None:
    """Marque le challenge consumed après COMMIT PG réussi."""
    import json

    r = _get_redis()
    if not r:
        return
    key = _resolution_redis_key(token)
    claim_key = _resolution_claim_key(token)
    try:
        raw = r.get(key)
        if raw:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            payload = json.loads(raw)
            payload["state"] = "consumed"
            r.setex(key, 30, json.dumps(payload))
        r.delete(claim_key)
    except Exception as exc:
        logger.debug("consume resolution token failed: %s", exc)
        with contextlib.suppress(Exception):
            r.delete(key)
            r.delete(claim_key)


def find_active_session_for_installation(
    user_id: int, device_installation_id: str
) -> MobileDeviceSession | None:
    return MobileDeviceSession.query.filter_by(
        user_id=user_id,
        device_installation_id=device_installation_id,
        status=MobileDeviceSessionStatus.active,
    ).first()


def reap_expired_provisional_sessions(
    user_id: int,
    *,
    commit: bool = False,
) -> list[uuid.UUID]:
    """Révoque les sessions provisional expirées (SQL only). Retourne les session_ids.

    commit=False : flush uniquement — publication Redis à la charge de l'appelant
    après COMMIT (via publish_session_revoked).
    """
    now = _now()
    expired = (
        MobileDeviceSession.query.filter(
            MobileDeviceSession.user_id == user_id,
            MobileDeviceSession.status == MobileDeviceSessionStatus.active,
            MobileDeviceSession.confirmed_at.is_(None),
            MobileDeviceSession.provisional_expires_at.isnot(None),
            MobileDeviceSession.provisional_expires_at <= now,
        )
        .with_for_update()
        .all()
    )
    revoked_ids: list[uuid.UUID] = []
    for sess in expired:
        revoke_session_state(
            sess,
            reason="provisional_expired",
            status=MobileDeviceSessionStatus.revoked,
        )
        revoked_ids.append(sess.session_id)
        try:
            from security.refresh_token_service import revoke_tokens_for_session

            revoke_tokens_for_session(
                str(sess.session_id),
                reason="provisional_expired",
                commit=False,
            )
        except Exception as exc:
            logger.warning("reap provisional tokens failed: %s", exc)
    if revoked_ids:
        db.session.flush()
    if commit and revoked_ids:
        db.session.commit()
        for sid in revoked_ids:
            publish_session_revoked(sid)
    return revoked_ids


def _create_or_reuse_session_locked(
    *,
    user_id: int,
    device_installation_id: str,
    driver_id: int | None = None,
    role: str | None = None,
    meta: DeviceSessionMetadata | None = None,
) -> tuple[MobileDeviceSession, str, str]:
    """Primitive interne : suppose User FOR UPDATE déjà détenu."""
    for _attempt in range(2):
        recovery = generate_opaque_secret()
        revocation = generate_opaque_secret()
        now = _now()

        existing = find_active_session_for_installation(user_id, device_installation_id)
        if existing is not None:
            # Provisional expirée sur la même installation → révoquer puis recréer
            if existing.is_provisional_expired(now=now):
                revoke_session_state(
                    existing,
                    reason="provisional_expired_reuse",
                    status=MobileDeviceSessionStatus.revoked,
                )
                try:
                    from security.refresh_token_service import revoke_tokens_for_session

                    revoke_tokens_for_session(
                        str(existing.session_id),
                        reason="provisional_expired_reuse",
                        commit=False,
                    )
                except Exception as exc:
                    logger.warning("revoke tokens provisional reuse: %s", exc)
                db.session.flush()
                # Boucler pour créer une nouvelle ligne
                continue

            existing.previous_credential_hash = existing.credential_hash
            existing.previous_generation = (
                existing.credential_generation or existing.generation
            )
            existing.previous_credential_valid_until = now + timedelta(
                seconds=PREVIOUS_CREDENTIAL_GRACE_SECONDS
            )
            existing.credential_hash = hash_credential(recovery)
            existing.revocation_secret_hash = hash_revocation_secret(revocation)
            new_cred_gen = (
                int(existing.credential_generation or existing.generation or 1) + 1
            )
            existing.credential_generation = new_cred_gen
            existing.generation = new_cred_gen  # alias legacy
            existing.driver_id = (
                driver_id if driver_id is not None else existing.driver_id
            )
            existing.last_refresh_at = now
            apply_session_metadata(existing, meta, touch_seen=True)
            # Reuse confirmé : ne jamais repasser provisional
            if existing.confirmed_at is not None:
                existing.provisional_expires_at = None
            db.session.add(existing)
            db.session.flush()
            _invalidate_session_cache(existing.session_id)
            return existing, recovery, revocation

        # Reap sync avant count (INV-AUTH-DEVICE-01)
        reap_expired_provisional_sessions(user_id, commit=False)

        active = list_active_sessions(user_id)
        limit = get_device_session_limit(role)
        if len(active) >= limit:
            raise DeviceSessionLimitReached(active)

        provisional = bool(PROVISIONAL_CONFIRMATION_ENABLED)
        session = MobileDeviceSession(
            session_id=uuid.uuid4(),
            user_id=user_id,
            driver_id=driver_id,
            device_installation_id=device_installation_id,
            status=MobileDeviceSessionStatus.active,
            credential_hash=hash_credential(recovery),
            revocation_secret_hash=hash_revocation_secret(revocation),
            generation=1,
            session_epoch=1,
            credential_generation=1,
            refresh_generation=1,
            last_seen_at=now,
            last_refresh_at=now,
            confirmed_at=None if provisional else now,
            provisional_expires_at=(
                now + timedelta(seconds=PROVISIONAL_SESSION_TTL_SECONDS)
                if provisional
                else None
            ),
        )
        apply_session_metadata(session, meta, touch_seen=False)
        db.session.add(session)
        try:
            with db.session.begin_nested():
                db.session.flush()
            return session, recovery, revocation
        except IntegrityError:
            continue

    existing = find_active_session_for_installation(user_id, device_installation_id)
    if existing is not None:
        recovery = generate_opaque_secret()
        revocation = generate_opaque_secret()
        now = _now()
        existing.credential_hash = hash_credential(recovery)
        existing.revocation_secret_hash = hash_revocation_secret(revocation)
        new_cred_gen = (
            int(existing.credential_generation or existing.generation or 1) + 1
        )
        existing.credential_generation = new_cred_gen
        existing.generation = new_cred_gen
        existing.last_seen_at = now
        apply_session_metadata(existing, meta, touch_seen=True)
        db.session.add(existing)
        db.session.flush()
        return existing, recovery, revocation
    raise DeviceSessionLimitReached([])


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
    meta: DeviceSessionMetadata | None = None,
) -> tuple[MobileDeviceSession, str, str]:
    """Crée ou tourne les credentials d'une session active pour une installation.

    Prend User FOR UPDATE puis délègue à _create_or_reuse_session_locked.
    """
    from models import User

    User.query.filter_by(id=user_id).with_for_update().one()
    effective_meta = meta or DeviceSessionMetadata(
        device_name=device_name,
        platform=platform,
        app_version=app_version,
        context_id=context_id,
    )
    return _create_or_reuse_session_locked(
        user_id=user_id,
        device_installation_id=device_installation_id,
        driver_id=driver_id,
        role=role,
        meta=effective_meta,
    )


def replace_device_session(
    *,
    user_id: int,
    session_to_revoke: str,
    device_installation_id: str,
    allowed_session_ids: list[str] | None = None,
    driver_id: int | None = None,
    role: str | None = None,
    meta: DeviceSessionMetadata | None = None,
) -> tuple[MobileDeviceSession, str, str, uuid.UUID]:
    """Révoque une session puis crée/reprend sous verrou (SQL only, sans Redis).

    Retourne (new_session, recovery, revocation, revoked_session_id).
    Publication Redis + consume challenge : après COMMIT appelant.
    """
    from models import User

    User.query.filter_by(id=user_id).with_for_update().one()

    target = get_session_by_id(session_to_revoke)
    if target is None or target.user_id != user_id:
        raise DeviceSessionResolutionError(
            "session_not_found",
            "Session à révoquer introuvable.",
        )

    if allowed_session_ids is not None:
        allowed = {str(s) for s in allowed_session_ids}
        if str(target.session_id) not in allowed:
            raise DeviceSessionResolutionError(
                "session_not_in_challenge",
                "Cet appareil n'était pas dans la liste au moment du challenge.",
            )

    revoked_id = target.session_id
    if target.is_active():
        revoke_session_state(
            target,
            reason="Remplacement multi-appareils (resolution)",
            revoked_by_user_id=user_id,
            status=MobileDeviceSessionStatus.revoked,
        )
        try:
            from security.refresh_token_service import revoke_tokens_for_session

            revoke_tokens_for_session(
                str(target.session_id),
                reason="Remplacement multi-appareils",
                commit=False,
            )
        except Exception as exc:
            logger.warning("revoke_tokens_for_session during replace: %s", exc)

    new_session, recovery, revocation = _create_or_reuse_session_locked(
        user_id=user_id,
        device_installation_id=device_installation_id,
        driver_id=driver_id,
        role=role,
        meta=meta,
    )
    return new_session, recovery, revocation, revoked_id


def revoke_session_state(
    session: MobileDeviceSession,
    *,
    reason: str,
    revoked_by_user_id: int | None = None,
    status: MobileDeviceSessionStatus = MobileDeviceSessionStatus.revoked,
) -> None:
    """SQL / ORM uniquement — aucun effet Redis (appeler publish après COMMIT)."""
    session.status = status
    session.revoked_at = _now()
    session.revoked_reason = reason
    session.revoked_by_user_id = revoked_by_user_id
    session.session_epoch = int(getattr(session, "session_epoch", 1) or 1) + 1
    session.provisional_expires_at = None


def publish_session_revoked(session_id: uuid.UUID | str) -> None:
    """Après COMMIT uniquement : marqueur négatif Redis (pas d'invalidate contradictoire)."""
    try:
        sid = (
            session_id
            if isinstance(session_id, uuid.UUID)
            else uuid.UUID(str(session_id))
        )
    except (ValueError, TypeError):
        return
    _cache_session_revoked(sid)


def revoke_session(
    session: MobileDeviceSession,
    *,
    reason: str,
    revoked_by_user_id: int | None = None,
    status: MobileDeviceSessionStatus = MobileDeviceSessionStatus.revoked,
    publish_cache: bool = True,
) -> None:
    """Révocation session. publish_cache=True uniquement si COMMIT immédiat suit.

    Pour les flux transactionnels (replace), préférer revoke_session_state +
    publish_session_revoked après COMMIT.
    """
    revoke_session_state(
        session,
        reason=reason,
        revoked_by_user_id=revoked_by_user_id,
        status=status,
    )
    if publish_cache:
        # Ne PAS invalider après : cela effaçait le marqueur négatif (bug historique).
        publish_session_revoked(session.session_id)


def verify_recovery_credential(session: MobileDeviceSession, credential: str) -> bool:
    h = hash_credential(credential)
    if secrets.compare_digest(h, session.credential_hash):
        return True
    return bool(
        session.previous_credential_hash
        and session.previous_credential_valid_until
        and session.previous_credential_valid_until > _now()
        and secrets.compare_digest(h, session.previous_credential_hash)
    )


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
    mark_session_confirmed(session)
    return new_secret


def bump_refresh_generation(session: MobileDeviceSession) -> int:
    """Incrémente refresh_generation sans toucher session_epoch ni recovery."""
    session.refresh_generation = int(getattr(session, "refresh_generation", 1) or 1) + 1
    session.last_refresh_at = _now()
    session.last_seen_at = session.last_refresh_at
    mark_session_confirmed(session)
    return int(session.refresh_generation)


def revoke_all_user_sessions(
    user_id: int,
    *,
    reason: str,
    status: MobileDeviceSessionStatus = MobileDeviceSessionStatus.security_revoked,
    except_session_id: uuid.UUID | None = None,
) -> int:
    """Révoque les sessions actives (SQL only). publish_session_revoked après COMMIT."""
    count = 0
    for sess in list_active_sessions(user_id):
        if except_session_id and sess.session_id == except_session_id:
            continue
        revoke_session_state(sess, reason=reason, status=status)
        count += 1
    return count


def revoke_user_security_sessions(
    user: Any,
    *,
    status: MobileDeviceSessionStatus = MobileDeviceSessionStatus.security_revoked,
    reason: str = "password_reset",
    increment_token_version: bool = True,
    fail_closed: bool = False,
    commit_tokens: bool = True,
) -> int:
    """Révoque toutes les sessions actives après un événement de sécurité.

    Couvre reset/changement MDP, reset admin, révocation globale.

    fail_closed=True : propage l'échec refresh tokens (pas de swallow).
    commit_tokens=False : flush only — commit + publish_session_revoked à l'appelant.
    """
    if increment_token_version and hasattr(user, "token_version"):
        user.token_version = int(getattr(user, "token_version", 0) or 0) + 1
    revoked_ids = [s.session_id for s in list_active_sessions(user.id)]
    count = revoke_all_user_sessions(user.id, reason=reason, status=status)
    try:
        from security.refresh_token_service import revoke_all_user_tokens

        revoke_all_user_tokens(user.id, reason=reason, commit=commit_tokens)
    except Exception as exc:
        if fail_closed:
            raise
        logger.warning("revoke_all_user_tokens après security revoke: %s", exc)
    if commit_tokens:
        # Tokens déjà commités : publier le cache négatif maintenant.
        for sid in revoked_ids:
            publish_session_revoked(sid)
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
        revoke_session_state(session, reason="pending_revocation")
    return True


def revoke_pending_idempotent(
    session: MobileDeviceSession,
    secret: str,
    *,
    operation_id: str | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Révoque via revocation_secret avec rejeu idempotent (perte d'ACK).

    SQL only — publish_session_revoked après COMMIT appelant.

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
        revoke_session_state(session, reason="pending_revocation")

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
        from ext import redis_client

        return redis_client
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
        r.setex(_cache_key(session_id), SESSION_NEGATIVE_CACHE_TTL_SECONDS, payload)
    except Exception as exc:
        logger.debug("cache session revoked failed: %s", exc)


def _invalidate_session_cache(session_id: uuid.UUID) -> None:
    r = _get_redis()
    if not r:
        return
    with contextlib.suppress(Exception):
        r.delete(_cache_key(session_id))


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


class RotationIdempotencyStatus(str, Enum):
    MISS = "miss"
    REPLAY = "replay"
    EXPIRED = "expired"
    UNREADABLE = "unreadable"
    MISMATCH = "mismatch"


@dataclass(frozen=True)
class RotationIdempotencyResolution:
    status: RotationIdempotencyStatus
    payload: dict[str, Any] | None = None
    row: AuthRotationResult | None = None


@dataclass(frozen=True)
class RotationProof:
    """Preuve liée au receipt — authentifie un retry sans exiger le credential courant."""

    proof_hash: str
    device_installation_id: str
    operation_type: str
    request_generation: int | None = None


def rotation_result_ttl_seconds(operation_type: str) -> int:
    if operation_type == "session_resume":
        return SESSION_RESUME_RESULT_TTL_SECONDS
    return ROTATION_RESULT_TTL_SECONDS


def _normalize_aes_key(key: bytes) -> bytes:
    if len(key) not in (16, 24, 32):
        return hashlib.sha256(key).digest()
    return key


def _get_encryption_key_for_id(key_id: str | None = None) -> tuple[bytes, str]:
    """Résout une clé AEAD par encryption_key_id (keyring minimal).

    - Clé courante : APP_ENCRYPTION_KEY_B64 + AUTH_ROTATION_ENCRYPTION_KEY_ID (défaut v1)
    - Clés historiques : AUTH_ROTATION_ENCRYPTION_KEY_<ID>_B64 (ex. AUTH_ROTATION_ENCRYPTION_KEY_V1_B64)
    """
    from models.base import _load_encryption_key

    current_id = (os.getenv("AUTH_ROTATION_ENCRYPTION_KEY_ID") or "v1").strip()
    requested = (key_id or current_id).strip() or current_id

    if requested == current_id:
        return _normalize_aes_key(_load_encryption_key()), current_id

    env_name = f"AUTH_ROTATION_ENCRYPTION_KEY_{requested.upper()}_B64"
    b64 = (os.getenv(env_name) or "").strip()
    if not b64:
        raise KeyError(f"clé historique absente: {requested} ({env_name})")

    import base64
    import binascii

    padded = b64 + "=" * (-len(b64) % 4)
    try:
        key = base64.urlsafe_b64decode(padded.encode())
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError(f"{env_name} invalide (Base64)") from exc
    return _normalize_aes_key(key), requested


def _get_encryption_key() -> tuple[bytes, str]:
    """Clé AEAD courante (écriture)."""
    return _get_encryption_key_for_id(None)


def encrypt_rotation_response(payload: dict[str, Any]) -> tuple[bytes, str]:
    import json

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key, key_id = _get_encryption_key()
    aesgcm = AESGCM(key)
    nonce = secrets.token_bytes(12)
    plaintext = json.dumps(payload).encode("utf-8")
    ct = aesgcm.encrypt(nonce, plaintext, None)
    return nonce + ct, key_id


def decrypt_rotation_response(ciphertext: bytes, key_id: str) -> dict[str, Any]:
    import json

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key, _ = _get_encryption_key_for_id(key_id)
    aesgcm = AESGCM(key)
    nonce, ct = ciphertext[:12], ciphertext[12:]
    plaintext = aesgcm.decrypt(nonce, ct, None)
    return json.loads(plaintext.decode("utf-8"))


def strip_rotation_meta(payload: dict[str, Any]) -> dict[str, Any]:
    """Retire les métadonnées internes avant réponse HTTP."""
    return {k: v for k, v in payload.items() if k != ROTATION_META_KEY}


def build_rotation_storage_payload(
    response_payload: dict[str, Any],
    *,
    proof: RotationProof,
) -> dict[str, Any]:
    stored = dict(response_payload)
    stored[ROTATION_META_KEY] = {
        "proof_hash": proof.proof_hash,
        "device_installation_id": proof.device_installation_id,
        "operation_type": proof.operation_type,
        "request_generation": proof.request_generation,
    }
    return stored


def _proof_matches(
    meta: dict[str, Any] | None,
    proof: RotationProof | None,
    *,
    stored_payload: dict[str, Any] | None = None,
) -> bool:
    if proof is None:
        return True
    if not isinstance(meta, dict):
        # Anciens receipts sans meta : incompatibles avec replay authentifié
        return False
    if str(meta.get("device_installation_id") or "") != str(
        proof.device_installation_id
    ):
        return False
    if str(meta.get("operation_type") or "") != str(proof.operation_type):
        return False

    stored_proof = str(meta.get("proof_hash") or "")
    try:
        if stored_proof and secrets.compare_digest(stored_proof, proof.proof_hash):
            return True
    except (TypeError, ValueError):
        pass

    # Crash-safe : le client a pu déjà persister le secret successeur du reçu
    # et retente avec ce secret (hash ≠ proof d'origine).
    if stored_payload:
        successor: str | None = None
        if proof.operation_type == "session_resume":
            raw = stored_payload.get("recovery_credential")
            successor = raw if isinstance(raw, str) else None
        elif proof.operation_type == "refresh":
            raw = stored_payload.get("refresh_token")
            successor = raw if isinstance(raw, str) else None
        if successor:
            try:
                if secrets.compare_digest(hash_credential(successor), proof.proof_hash):
                    return True
            except (TypeError, ValueError):
                return False
    return False


def get_rotation_result(
    session_id: uuid.UUID, idempotency_key: str
) -> AuthRotationResult | None:
    return AuthRotationResult.query.filter_by(
        session_id=session_id,
        idempotency_key_hash=hash_idempotency_key(idempotency_key),
    ).first()


def resolve_rotation_idempotency(
    session_id: uuid.UUID,
    idempotency_key: str,
    *,
    proof: RotationProof | None = None,
) -> RotationIdempotencyResolution:
    """Résout un reçu d'idempotence sans ambiguïté expiration / decrypt / mismatch."""
    row = get_rotation_result(session_id, idempotency_key)
    if row is None:
        return RotationIdempotencyResolution(status=RotationIdempotencyStatus.MISS)

    if row.expires_at and row.expires_at < _now():
        return RotationIdempotencyResolution(
            status=RotationIdempotencyStatus.EXPIRED, row=row
        )

    try:
        stored = decrypt_rotation_response(
            row.response_ciphertext, row.encryption_key_id
        )
    except Exception as exc:
        logger.error(
            "AuthRotationResult UNREADABLE session_id=%s key_id=%s: %s",
            session_id,
            row.encryption_key_id,
            exc,
        )
        return RotationIdempotencyResolution(
            status=RotationIdempotencyStatus.UNREADABLE, row=row
        )

    meta = stored.get(ROTATION_META_KEY) if isinstance(stored, dict) else None
    public_payload = strip_rotation_meta(stored) if isinstance(stored, dict) else {}
    if proof is not None and not _proof_matches(
        meta if isinstance(meta, dict) else None,
        proof,
        stored_payload=public_payload if isinstance(public_payload, dict) else None,
    ):
        return RotationIdempotencyResolution(
            status=RotationIdempotencyStatus.MISMATCH, row=row
        )

    return RotationIdempotencyResolution(
        status=RotationIdempotencyStatus.REPLAY,
        payload=public_payload if isinstance(public_payload, dict) else {},
        row=row,
    )


def load_rotation_response(row: AuthRotationResult) -> dict[str, Any] | None:
    """Compat : rejoue si déchiffrable et non expiré (sans vérification de proof)."""
    if row.expires_at and row.expires_at < _now():
        return None
    try:
        stored = decrypt_rotation_response(
            row.response_ciphertext, row.encryption_key_id
        )
    except Exception as exc:
        logger.warning("decrypt AuthRotationResult failed: %s", exc)
        return None
    if not isinstance(stored, dict):
        return None
    return strip_rotation_meta(stored)


def is_rotation_idempotency_conflict(exc: BaseException) -> bool:
    """True uniquement pour uq_auth_rotation_result_session_idempotency (pas tout 23505)."""
    integrity: IntegrityError | None = None
    if isinstance(exc, IntegrityError):
        integrity = exc
    else:
        cause = getattr(exc, "__cause__", None)
        if isinstance(cause, IntegrityError):
            integrity = cause
    if integrity is None:
        return False

    orig = getattr(integrity, "orig", None)
    pgcode = getattr(orig, "pgcode", None) if orig is not None else None
    if pgcode is not None and str(pgcode) != "23505":
        return False

    diag = getattr(orig, "diag", None) if orig is not None else None
    constraint = getattr(diag, "constraint_name", None) if diag is not None else None
    if constraint:
        return str(constraint) == ROTATION_IDEMPOTENCY_CONSTRAINT

    msg = str(integrity).lower()
    return ROTATION_IDEMPOTENCY_CONSTRAINT in msg or (
        str(pgcode) == "23505" and "idempotency" in msg
    )


def store_rotation_result(
    *,
    session: MobileDeviceSession,
    idempotency_key: str,
    request_generation: int,
    successor_generation: int,
    response_payload: dict[str, Any],
    operation_type: str = "refresh",
    proof: RotationProof | None = None,
) -> AuthRotationResult | None:
    """Persiste le reçu. Retourne la ligne si insert gagnant, None si conflit (ON CONFLICT).

    En cas de conflit PostgreSQL (DO NOTHING), ne lève pas IntegrityError.
    """
    effective_proof = proof or RotationProof(
        proof_hash="",
        device_installation_id=str(session.device_installation_id or ""),
        operation_type=operation_type,
        request_generation=request_generation,
    )
    stored_payload = build_rotation_storage_payload(
        response_payload, proof=effective_proof
    )
    ciphertext, key_id = encrypt_rotation_response(stored_payload)
    row_id = uuid.uuid4()
    expires_at = _now() + timedelta(seconds=rotation_result_ttl_seconds(operation_type))
    key_hash = hash_idempotency_key(idempotency_key)

    bind = db.session.get_bind()
    dialect = getattr(getattr(bind, "dialect", None), "name", "") or ""

    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        stmt = (
            pg_insert(AuthRotationResult.__table__)
            .values(
                id=row_id,
                session_id=session.session_id,
                idempotency_key_hash=key_hash,
                request_generation=request_generation,
                successor_generation=successor_generation,
                response_ciphertext=ciphertext,
                encryption_key_id=key_id,
                operation_type=operation_type,
                expires_at=expires_at,
            )
            .on_conflict_do_nothing(constraint=ROTATION_IDEMPOTENCY_CONSTRAINT)
            .returning(AuthRotationResult.__table__.c.id)
        )
        result = db.session.execute(stmt)
        inserted_id = result.scalar_one_or_none()
        if inserted_id is None:
            return None
        return db.session.get(AuthRotationResult, inserted_id)

    row = AuthRotationResult(
        id=row_id,
        session_id=session.session_id,
        idempotency_key_hash=key_hash,
        request_generation=request_generation,
        successor_generation=successor_generation,
        response_ciphertext=ciphertext,
        encryption_key_id=key_id,
        operation_type=operation_type,
        expires_at=expires_at,
    )
    try:
        with db.session.begin_nested():
            db.session.add(row)
            db.session.flush()
        return row
    except IntegrityError as exc:
        if is_rotation_idempotency_conflict(exc):
            return None
        raise


def http_response_for_idempotency(
    resolution: RotationIdempotencyResolution,
) -> tuple[dict[str, Any], int] | None:
    """Mappe EXPIRED/UNREADABLE/MISMATCH/REPLAY vers une réponse HTTP. None si MISS."""
    if resolution.status == RotationIdempotencyStatus.MISS:
        return None
    if resolution.status == RotationIdempotencyStatus.REPLAY:
        payload = dict(resolution.payload or {})
        payload["error_code"] = "refresh_duplicate"
        return payload, 200
    if resolution.status == RotationIdempotencyStatus.EXPIRED:
        return {
            "error": "idempotency_result_expired",
            "error_code": "idempotency_result_expired",
            "retryable": False,
        }, 401
    if resolution.status == RotationIdempotencyStatus.UNREADABLE:
        return {
            "error": "rotation_result_unavailable",
            "error_code": "rotation_result_unavailable",
            "retryable": True,
        }, 503
    # MISMATCH
    return {
        "error": "idempotency_proof_mismatch",
        "error_code": "refresh_replay_detected",
        "retryable": False,
    }, 401
