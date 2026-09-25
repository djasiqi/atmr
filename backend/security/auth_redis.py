"""Client Redis dédié au store refresh (P0-6).

Autorité unique pour CURRENT / PREVIOUS / revoked refresh.
Ne pas utiliser ``ext.redis_client`` (cache / Celery / sockets) pour ces clés.

``AUTH_REDIS_URL`` prime ; fallback temporaire ``REDIS_URL`` pour compat tests/legacy
(log warning). En production, ``AUTH_REDIS_URL`` doit pointer vers ``redis-auth``.

Rollout (``AUTH_REDIS_MIGRATION_MODE``) : voir ``auth_redis_migration.py``.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from typing import Any, TypeVar

import redis

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_client: Any | None = None
_client_url: str | None = None
_legacy_client: Any | None = None
_legacy_url: str | None = None
_dedicated_client: Any | None = None
_dedicated_url: str | None = None

T = TypeVar("T")


def _flask_env_name() -> str:
    flask_env = (os.getenv("FLASK_CONFIG") or os.getenv("FLASK_ENV") or "").strip().lower()
    try:
        from flask import current_app, has_app_context

        if has_app_context():
            cfg_name = (current_app.config.get("ENV") or "").strip().lower()
            if cfg_name:
                flask_env = cfg_name
            if current_app.config.get("TESTING"):
                flask_env = "testing"
    except Exception:
        pass
    return flask_env


def _config_or_env(key: str) -> str:
    try:
        from flask import current_app, has_app_context

        if has_app_context():
            val = (current_app.config.get(key) or "").strip()
            if val:
                return val
    except Exception:
        pass
    return (os.getenv(key) or "").strip()


def resolve_legacy_redis_url() -> str:
    """URL du Redis général (cache / legacy refresh pendant migration)."""
    return _config_or_env("REDIS_URL") or "redis://127.0.0.1:6379/0"


def resolve_dedicated_auth_redis_url() -> str:
    """URL redis-auth dédiée (sans fallback). Vide si non configurée."""
    return _config_or_env("AUTH_REDIS_URL")


def resolve_auth_redis_url() -> str:
    """Résout l'URL du store auth refresh (client unique / mode off).

    Production : ``AUTH_REDIS_URL`` obligatoire (pas de fallback silencieux vers
    ``REDIS_URL`` / allkeys-lru). Dev/test : fallback autorisé, ou via
    ``AUTH_REDIS_ALLOW_LEGACY_FALLBACK=1``.
    """
    dedicated = resolve_dedicated_auth_redis_url()
    general = resolve_legacy_redis_url()
    flask_env = _flask_env_name()

    if dedicated:
        return dedicated

    allow_legacy = (os.getenv("AUTH_REDIS_ALLOW_LEGACY_FALLBACK") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    is_prod = flask_env in ("production", "prod")
    if is_prod and not allow_legacy:
        raise RuntimeError(
            "AUTH_REDIS_URL est requis en production (P0-6). "
            + "Sans cette URL le store refresh retomberait sur REDIS_URL "
            + "(allkeys-lru). Définir AUTH_REDIS_URL ou, temporairement, "
            + "AUTH_REDIS_ALLOW_LEGACY_FALLBACK=1."
        )

    logger.warning("AUTH_REDIS_URL absent — fallback REDIS_URL (P0-6 non isolé)")
    return general or "redis://127.0.0.1:6379/0"


def _from_url(url: str) -> redis.Redis:
    return redis.from_url(url, decode_responses=True)


def get_legacy_redis(*, force_new: bool = False) -> redis.Redis:
    """Client Redis général (REDIS_URL)."""
    global _legacy_client, _legacy_url
    url = resolve_legacy_redis_url()
    if force_new:
        return _from_url(url)
    with _lock:
        if _legacy_client is not None and _legacy_url == url:
            return _legacy_client
        _legacy_client = _from_url(url)
        _legacy_url = url
        return _legacy_client


def get_dedicated_auth_redis(*, force_new: bool = False) -> redis.Redis:
    """Client redis-auth uniquement. Lève si AUTH_REDIS_URL absent."""
    global _dedicated_client, _dedicated_url
    url = resolve_dedicated_auth_redis_url()
    if not url:
        raise RuntimeError(
            "AUTH_REDIS_URL requis pour le client dédié (mode migration dual_write/"
            "auth_primary/auth_only)."
        )
    if force_new:
        return _from_url(url)
    with _lock:
        if _dedicated_client is not None and _dedicated_url == url:
            return _dedicated_client
        _dedicated_client = _from_url(url)
        _dedicated_url = url
        return _dedicated_client


def get_auth_redis(*, force_new: bool = False) -> Any:
    """Retourne le client utilisé par RefreshTokenService (éventuellement dual-write).

    ``force_new=True`` crée une connexion fraîche (tests restart / isolation).
    """
    from security.auth_redis_migration import (
        AuthRedisMigrationMode,
        DualWriteRedisClient,
        get_migration_mode,
        migration_requires_dedicated_auth,
    )

    mode = get_migration_mode()

    if migration_requires_dedicated_auth():
        dedicated_url = resolve_dedicated_auth_redis_url()
        legacy_url = resolve_legacy_redis_url()
        if not dedicated_url:
            raise RuntimeError(
                f"AUTH_REDIS_URL requis quand AUTH_REDIS_MIGRATION_MODE={mode.value}"
            )
        if dedicated_url == legacy_url:
            raise RuntimeError(
                "AUTH_REDIS_URL et REDIS_URL doivent être distincts en mode migration "
                f"({mode.value}) — sinon dual-write / parité impossibles."
            )

        dedicated = get_dedicated_auth_redis(force_new=force_new)
        legacy = get_legacy_redis(force_new=force_new)

        if mode == AuthRedisMigrationMode.AUTH_ONLY:
            return dedicated

        if mode == AuthRedisMigrationMode.DUAL_WRITE:
            # READ legacy ; WRITE legacy (primary) + auth (secondary)
            return DualWriteRedisClient(
                read_client=legacy,
                write_primary=legacy,
                write_secondary=dedicated,
                mode=mode,
                secondary_label="redis-auth",
            )

        # auth_primary : READ auth ; WRITE auth (primary) + legacy (secondary rollback)
        return DualWriteRedisClient(
            read_client=dedicated,
            write_primary=dedicated,
            write_secondary=legacy,
            mode=mode,
            secondary_label="redis-legacy",
        )

    # Mode off/legacy : client unique (comportement historique)
    global _client, _client_url
    url = resolve_auth_redis_url()
    if force_new:
        return _from_url(url)

    with _lock:
        if _client is not None and _client_url == url:
            return _client
        _client = _from_url(url)
        _client_url = url
        return _client


def reset_auth_redis_client() -> None:
    """Invalide les singletons (tests / reconfig)."""
    global _client, _client_url, _legacy_client, _legacy_url, _dedicated_client, _dedicated_url
    with _lock:
        for client in (_client, _legacy_client, _dedicated_client):
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
        _client = None
        _client_url = None
        _legacy_client = None
        _legacy_url = None
        _dedicated_client = None
        _dedicated_url = None


def is_redis_oom_error(exc: BaseException) -> bool:
    """True si Redis a refusé une écriture pour maxmemory / OOM."""
    if exc.__class__.__name__ in ("OutOfMemoryError", "ResponseError"):
        msg = str(exc).upper()
        if "OOM" in msg or "MAXMEMORY" in msg or "USED MEMORY" in msg:
            return True
    msg = str(exc).upper()
    return (
        "OOM" in msg
        or "MAXMEMORY" in msg
        or "COMMAND NOT ALLOWED WHEN USED MEMORY" in msg
    )


def raise_refresh_store_unavailable(exc: BaseException) -> None:
    """Traduit une erreur Redis auth en ``RefreshStoreUnavailableError`` (+ métriques)."""
    from security.refresh_token_service import RefreshStoreUnavailableError

    try:
        from security.security_metrics import (
            redis_auth_rejected_writes_total,
            redis_auth_store_unavailable_total,
        )

        if is_redis_oom_error(exc):
            redis_auth_rejected_writes_total.inc()
        redis_auth_store_unavailable_total.inc()
    except Exception:
        pass

    reason = "redis_oom" if is_redis_oom_error(exc) else "redis_unavailable"
    logger.error(
        "auth_redis_store_unavailable reason=%s err=%s",
        reason,
        type(exc).__name__,
    )
    raise RefreshStoreUnavailableError(reason) from exc


def auth_redis_write(fn: Callable[[], T]) -> T:
    """Exécute une écriture Redis auth ; OOM / erreur → fail-closed store unavailable."""
    from security.refresh_token_service import RefreshStoreUnavailableError

    try:
        return fn()
    except RefreshStoreUnavailableError:
        raise
    except Exception as exc:
        raise_refresh_store_unavailable(exc)
        raise  # unreachable — pour le type-checker
