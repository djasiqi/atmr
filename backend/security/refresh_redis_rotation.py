"""P0-1 / P0-2 — état Redis des refresh tokens (current / previous grace / compensation).

Modèle Redis :

* ``active_refresh_token:{hash}`` → user_id (CURRENT)
* ``refresh_previous:{hash}`` → JSON ``{user_id, successor_hash, session_id, valid_until}``
  avec TTL = grâce (défaut 300 s)

La grâce ne permet **pas** une nouvelle rotation depuis l'ancien token : elle permet
uniquement de le reconnaître comme prédécesseur et de restituer le successeur.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

PREVIOUS_TOKEN_PREFIX = "refresh_previous:"
DEFAULT_ROTATION_GRACE_SECONDS = 300


class RedisRefreshState(str, Enum):
    CURRENT = "CURRENT"
    PREVIOUS_WITHIN_GRACE = "PREVIOUS_WITHIN_GRACE"
    EXPIRED_PREVIOUS = "EXPIRED_PREVIOUS"
    UNKNOWN = "UNKNOWN"
    UNAVAILABLE = "UNAVAILABLE"


def rotation_grace_seconds() -> int:
    """Fenêtre de grâce Redis (secondes). Défaut 300. Convention age < grâce."""
    raw = (os.getenv("REFRESH_ROTATION_GRACE_SECONDS") or "").strip()
    if raw.isdigit():
        return max(int(raw), 1)
    return int(DEFAULT_ROTATION_GRACE_SECONDS)


@dataclass
class RedisIssuanceHandle:
    """Snapshot pour compensation si le commit SQL échoue après mutation Redis."""

    kind: str  # "publish" | "rotate"
    user_id: int
    new_token: str
    old_token: str | None = None
    old_was_active: bool = False
    old_active_ttl: int | None = None
    session_id: str | None = None
    previous_payload: str | None = None
    grace_seconds: int = 300


@dataclass
class RedisClassifyResult:
    state: RedisRefreshState
    successor_hash: str | None = None
    session_id: str | None = None
    valid_until: float | None = None


def _svc():
    from services.security.authentication import RefreshTokenService

    return RefreshTokenService()


def _hash(token: str) -> str:
    return _svc()._hash_token(token)


def publish_refresh_redis(
    user_id: int,
    token: str,
    *,
    ttl_seconds: int | None = None,
) -> RedisIssuanceHandle:
    """Publie un refresh CURRENT. Retourne un handle de compensation P0-1."""
    from security.refresh_token_service import (
        RefreshStoreUnavailableError,
        refresh_fail_closed_enabled,
        sync_refresh_token_to_redis,
    )

    try:
        sync_refresh_token_to_redis(user_id, token, ttl_seconds=ttl_seconds)
    except RefreshStoreUnavailableError:
        raise
    except Exception as exc:
        if refresh_fail_closed_enabled():
            raise RefreshStoreUnavailableError("redis_unavailable") from exc
        logger.warning("publish_refresh_redis fail-open user_id=%s: %s", user_id, exc)
    return RedisIssuanceHandle(
        kind="publish",
        user_id=user_id,
        new_token=token,
        grace_seconds=rotation_grace_seconds(),
    )


def rotate_refresh_redis(
    user_id: int,
    old_token: str,
    new_token: str,
    *,
    ttl_seconds: int,
    session_id: str | None = None,
    grace_seconds: int | None = None,
) -> RedisIssuanceHandle:
    """Transition atomique R0→R1 : current=R1, previous=R0 (grâce).

    Ne place **pas** R0 dans ``revoked_refresh_token`` pendant la grâce.
    """
    from security.auth_redis import raise_refresh_store_unavailable
    from security.refresh_token_service import (
        RefreshStoreUnavailableError,
        refresh_fail_closed_enabled,
    )

    grace = int(
        grace_seconds if grace_seconds is not None else rotation_grace_seconds()
    )
    svc = _svc()
    old_hash = _hash(old_token)
    new_hash = _hash(new_token)
    active_old = f"{svc.active_tokens_prefix}{old_hash}"
    previous_key = f"{PREVIOUS_TOKEN_PREFIX}{old_hash}"

    old_was_active = False
    old_ttl: int | None = None
    try:
        if svc.redis_client.exists(active_old):
            old_was_active = True
            try:
                ttl = int(svc.redis_client.ttl(active_old))
                old_ttl = ttl if ttl > 0 else None
            except Exception:
                old_ttl = None

        valid_until = time.time() + grace
        previous_payload = json.dumps(
            {
                "user_id": int(user_id),
                "successor_hash": new_hash,
                "session_id": str(session_id) if session_id else None,
                "valid_until": valid_until,
            }
        )

        # 1) Publier R1 via le même helper que le login (active_refresh_token:*)
        svc.store_token(user_id, new_token, ttl_seconds=int(ttl_seconds))
        # 2) Enregistrer R0 comme previous (grâce)
        svc.redis_client.setex(previous_key, int(grace), previous_payload)
        # 3) Retirer R0 des actifs SANS le marquer revoked
        svc.redis_client.delete(active_old)
        user_tokens_key = f"user_refresh_tokens:{user_id}"
        with contextlib.suppress(Exception):
            svc.redis_client.zrem(user_tokens_key, old_hash)
    except RefreshStoreUnavailableError:
        raise
    except Exception as exc:
        logger.error(
            "rotate_refresh_redis failed user_id=%s err=%s",
            user_id,
            type(exc).__name__,
        )
        if refresh_fail_closed_enabled():
            raise_refresh_store_unavailable(exc)
        logger.warning("rotate_refresh_redis fail-open user_id=%s: %s", user_id, exc)
        previous_payload = None

    handle = RedisIssuanceHandle(
        kind="rotate",
        user_id=user_id,
        new_token=new_token,
        old_token=old_token,
        old_was_active=old_was_active,
        old_active_ttl=old_ttl,
        session_id=session_id,
        previous_payload=previous_payload,
        grace_seconds=grace,
    )
    logger.info(
        "auth_redis_rotate_applied",
        extra={
            "event": "auth_redis_rotate_applied",
            "user_id": user_id,
            "session_id": session_id,
            "old_hash": old_hash[:8],
            "new_hash": new_hash[:8],
            "grace_seconds": grace,
        },
    )
    return handle


def compensate_redis_issuance(handle: RedisIssuanceHandle | None) -> None:
    """Annule une mutation Redis si le commit SQL a échoué (P0-1 hardening).

    En mode dual_write / auth_primary, applique aussi la compensation sur les
    deux stores bruts (legacy + auth) pour éviter un R1 orphelin sur le secondaire
    si le DualWrite a partiellement échoué.
    """
    if handle is None:
        return
    try:
        svc = _svc()
        _apply_compensation_ops(svc.redis_client, handle)
        _mirror_compensation_both_stores(handle)

        new_hash = _hash(handle.new_token)
        logger.error(
            "auth_redis_commit_compensated",
            extra={
                "event": "auth_redis_commit_compensated",
                "kind": handle.kind,
                "user_id": handle.user_id,
                "session_id": handle.session_id,
                "new_hash": new_hash[:8],
            },
        )
    except Exception:
        logger.exception(
            "auth_redis_compensation_failed kind=%s user_id=%s",
            handle.kind,
            handle.user_id,
        )


def _apply_compensation_ops(client: Any, handle: RedisIssuanceHandle) -> None:
    """Compensation idempotente sur un client Redis donné."""
    new_hash = _hash(handle.new_token)
    active_new = f"active_refresh_token:{new_hash}"
    user_tokens_key = f"user_refresh_tokens:{handle.user_id}"

    if handle.kind == "publish":
        client.delete(active_new)
        with contextlib.suppress(Exception):
            client.zrem(user_tokens_key, new_hash)
    elif handle.kind == "rotate" and handle.old_token:
        old_hash = _hash(handle.old_token)
        previous_key = f"{PREVIOUS_TOKEN_PREFIX}{old_hash}"
        active_old = f"active_refresh_token:{old_hash}"
        client.delete(active_new)
        client.delete(previous_key)
        with contextlib.suppress(Exception):
            client.zrem(user_tokens_key, new_hash)
        if handle.old_was_active:
            ttl = handle.old_active_ttl or max(handle.grace_seconds, 60)
            client.setex(active_old, int(ttl), str(handle.user_id))
            with contextlib.suppress(Exception):
                client.zadd(user_tokens_key, {old_hash: time.time()})


def _mirror_compensation_both_stores(handle: RedisIssuanceHandle) -> None:
    """Force la compensation sur legacy + auth dédié (modes migration)."""
    try:
        from security.auth_redis_migration import (
            AuthRedisMigrationMode,
            get_migration_mode,
        )

        mode = get_migration_mode()
        if mode not in {
            AuthRedisMigrationMode.DUAL_WRITE,
            AuthRedisMigrationMode.AUTH_PRIMARY,
        }:
            return
        from security.auth_redis import get_dedicated_auth_redis, get_legacy_redis

        for label, client in (
            ("legacy", get_legacy_redis()),
            ("auth", get_dedicated_auth_redis()),
        ):
            try:
                _apply_compensation_ops(client, handle)
            except Exception:
                logger.exception(
                    "auth_redis_compensation_peer_failed peer=%s kind=%s",
                    label,
                    handle.kind,
                )
    except Exception:
        logger.debug("mirror compensation skipped", exc_info=True)


def commit_db_after_redis(handle: RedisIssuanceHandle) -> None:
    """Commit SQL ; en échec → rollback + compensation Redis."""
    from ext import db

    try:
        db.session.commit()
    except Exception:
        db.session.rollback()
        compensate_redis_issuance(handle)
        raise


def classify_refresh_in_redis(
    token: str,
    *,
    user_id: int | None = None,
) -> RedisClassifyResult:
    """Classe un refresh côté Redis (P0-2)."""
    from security.refresh_token_service import (
        RefreshStoreUnavailableError,
        refresh_fail_closed_enabled,
    )

    try:
        svc = _svc()
        token_hash = _hash(token)
        active_key = f"{svc.active_tokens_prefix}{token_hash}"
        previous_key = f"{PREVIOUS_TOKEN_PREFIX}{token_hash}"

        try:
            svc.redis_client.ping()
        except Exception as ping_err:
            if refresh_fail_closed_enabled():
                raise RefreshStoreUnavailableError("redis_unavailable") from ping_err
            return RedisClassifyResult(state=RedisRefreshState.UNAVAILABLE)

        # Révoqué hard (hors grâce) ?
        if svc.redis_client.exists(f"{svc.revoked_tokens_prefix}{token_hash}"):
            # Peut coexister avec previous pendant transition legacy — previous gagne.
            pass

        stored = svc.redis_client.get(active_key)
        if stored is not None:
            if user_id is not None and int(stored) != int(user_id):
                return RedisClassifyResult(state=RedisRefreshState.UNKNOWN)
            return RedisClassifyResult(state=RedisRefreshState.CURRENT)

        prev_raw = svc.redis_client.get(previous_key)
        if prev_raw is not None:
            try:
                if isinstance(prev_raw, bytes):
                    prev_raw = prev_raw.decode("utf-8")
                data = json.loads(prev_raw)
            except (TypeError, ValueError, json.JSONDecodeError):
                return RedisClassifyResult(state=RedisRefreshState.UNKNOWN)
            valid_until = float(data.get("valid_until") or 0)
            # Convention : âge < grâce ⇔ valid_until > now (TTL Redis + timestamp).
            if valid_until > time.time():
                if user_id is not None and int(data.get("user_id") or -1) != int(
                    user_id
                ):
                    return RedisClassifyResult(state=RedisRefreshState.UNKNOWN)
                return RedisClassifyResult(
                    state=RedisRefreshState.PREVIOUS_WITHIN_GRACE,
                    successor_hash=str(data.get("successor_hash") or "") or None,
                    session_id=(
                        str(data["session_id"])
                        if data.get("session_id") is not None
                        else None
                    ),
                    valid_until=valid_until,
                )
            return RedisClassifyResult(state=RedisRefreshState.EXPIRED_PREVIOUS)

        # Clé previous absente mais était peut-être expirée (TTL écoulé) → UNKNOWN
        return RedisClassifyResult(state=RedisRefreshState.UNKNOWN)
    except RefreshStoreUnavailableError:
        return RedisClassifyResult(state=RedisRefreshState.UNAVAILABLE)
    except Exception as exc:
        from security.refresh_token_service import refresh_fail_closed_enabled

        if refresh_fail_closed_enabled():
            logger.error(
                "classify_refresh_in_redis fail-closed: %s", type(exc).__name__
            )
            return RedisClassifyResult(state=RedisRefreshState.UNAVAILABLE)
        logger.warning("classify_refresh_in_redis fail-open: %s", exc)
        return RedisClassifyResult(state=RedisRefreshState.UNAVAILABLE)


def delete_active_refresh_redis(token: str) -> None:
    """Retire un token des actifs sans le marquer revoked (compensation / cleanup)."""
    try:
        svc = _svc()
        token_hash = _hash(token)
        stored = svc.redis_client.get(f"{svc.active_tokens_prefix}{token_hash}")
        svc.redis_client.delete(f"{svc.active_tokens_prefix}{token_hash}")
        if stored is not None:
            with contextlib.suppress(Exception):
                svc.redis_client.zrem(f"user_refresh_tokens:{stored}", token_hash)
    except Exception:
        logger.warning("delete_active_refresh_redis failed", exc_info=True)


@dataclass
class OrphanRedisRepairResult:
    repaired: bool
    ambiguous: bool
    reason: str


def try_repair_orphan_redis_previous(
    *,
    predecessor_token: str,
    user_id: int,
    session_id: str | None,
    claimed_refresh_generation: int | None,
    db_refresh_generation: int | None,
    successor_hash: str | None,
) -> OrphanRedisRepairResult:
    """Répare un état Redis PREVIOUS dont la rotation n'a jamais été commitée en DB.

    Preuve non ambiguë requise :
    - R0 encore présent en DB (ligne non absente)
    - R1 (successor_hash) absent de la table refresh_token
    - aucun AuthRotationResult pour la session contenant ce successeur
    - generation DB encore égale à celle claimée par R0 (pas de bump)

    Si ambigu → ``ambiguous=True`` (fail-closed côté appelant).
    Si réparé → R0 remis CURRENT, previous + actif R1 orphelins supprimés.
    """
    from models import RefreshToken
    from models.mobile_device_session import AuthRotationResult
    from security.refresh_token_service import _hash_refresh_token

    if not successor_hash:
        return OrphanRedisRepairResult(
            repaired=False, ambiguous=True, reason="missing_successor_hash"
        )

    pred_hash = _hash_refresh_token(predecessor_token)
    r0_row = RefreshToken.query.filter_by(token_hash=pred_hash).first()
    if r0_row is None:
        return OrphanRedisRepairResult(
            repaired=False, ambiguous=True, reason="r0_absent_db"
        )

    r1_row = RefreshToken.query.filter_by(token_hash=str(successor_hash)).first()
    if r1_row is not None:
        return OrphanRedisRepairResult(
            repaired=False, ambiguous=True, reason="r1_present_db"
        )

    if (
        claimed_refresh_generation is not None
        and db_refresh_generation is not None
        and int(claimed_refresh_generation) != int(db_refresh_generation)
    ):
        return OrphanRedisRepairResult(
            repaired=False,
            ambiguous=True,
            reason="generation_already_bumped",
        )

    if session_id:
        try:
            sid = __import__("uuid").UUID(str(session_id))
        except (ValueError, TypeError):
            return OrphanRedisRepairResult(
                repaired=False, ambiguous=True, reason="bad_session_id"
            )
        # Receipts récents : si un payload déchiffrable porte ce successor → pas orphelin.
        from security.mobile_device_session_service import (
            decrypt_rotation_response,
            strip_rotation_meta,
        )

        rows = (
            AuthRotationResult.query.filter_by(session_id=sid, operation_type="refresh")
            .order_by(AuthRotationResult.created_at.desc())
            .limit(20)
            .all()
        )
        for row in rows:
            try:
                stored = decrypt_rotation_response(
                    row.response_ciphertext, row.encryption_key_id
                )
            except Exception:
                continue
            if not isinstance(stored, dict):
                continue
            public = strip_rotation_meta(stored)
            succ = public.get("refresh_token")
            if isinstance(succ, str) and succ and _hash(succ) == str(successor_hash):
                return OrphanRedisRepairResult(
                    repaired=False,
                    ambiguous=True,
                    reason="receipt_has_successor",
                )

    # Réparation Redis
    try:
        svc = _svc()
        previous_key = f"{PREVIOUS_TOKEN_PREFIX}{pred_hash}"
        orphan_active = f"{svc.active_tokens_prefix}{successor_hash}"
        active_r0 = f"{svc.active_tokens_prefix}{pred_hash}"
        svc.redis_client.delete(previous_key)
        svc.redis_client.delete(orphan_active)
        with contextlib.suppress(Exception):
            svc.redis_client.zrem(f"user_refresh_tokens:{user_id}", successor_hash)
        # Remettre R0 CURRENT (TTL large ; aligné sur refresh restant si possible)
        ttl = 90 * 24 * 3600
        with contextlib.suppress(Exception):
            from datetime import UTC, datetime

            if r0_row.expires_at is not None:
                exp = r0_row.expires_at
                if exp.tzinfo is None:
                    exp = exp.replace(tzinfo=UTC)
                ttl = max(int((exp - datetime.now(UTC)).total_seconds()), 60)
        svc.redis_client.setex(active_r0, ttl, str(user_id))
        with contextlib.suppress(Exception):
            svc.redis_client.zadd(
                f"user_refresh_tokens:{user_id}", {pred_hash: time.time()}
            )
        logger.error(
            "auth_redis_orphan_repaired",
            extra={
                "event": "auth_redis_orphan_repaired",
                "user_id": user_id,
                "session_id": session_id,
                "pred_hash": pred_hash[:8],
                "successor_hash": str(successor_hash)[:8],
                "reason": "redis_before_db_commit_crash",
            },
        )
        return OrphanRedisRepairResult(
            repaired=True, ambiguous=False, reason="repaired"
        )
    except Exception as exc:
        logger.exception("auth_redis_orphan_repair_failed: %s", type(exc).__name__)
        return OrphanRedisRepairResult(
            repaired=False, ambiguous=True, reason="repair_failed"
        )
