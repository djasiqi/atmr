"""P0-6 rollout — modes de migration Redis auth (dual-write / auth-primary).

Modes (``AUTH_REDIS_MIGRATION_MODE``) — contrat explicite :

* ``legacy`` — READ = legacy (``REDIS_URL``) ; WRITE = legacy
  (jamais ``resolve_auth_redis_url()`` / redis-auth)
* ``dual_write`` — READ = legacy ; WRITE = legacy + redis-auth
* ``auth_primary`` — READ = redis-auth ; WRITE = redis-auth + legacy
* ``auth_only`` — READ/WRITE = redis-auth
* ``off`` — compat tests/dev uniquement ; **interdit en production** dès que
  ``AUTH_REDIS_URL`` est défini et distinct de ``REDIS_URL``
  (évite l'ancienne fenêtre silencieuse auth-only involontaire)

Invariant Phase F : aucun ``auth missing → lire legacy`` par requête.
"""

from __future__ import annotations

import contextlib
import logging
import os
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

ACTIVE_PREFIX = "active_refresh_token:"
PREVIOUS_PREFIX = "refresh_previous:"
REVOKED_PREFIX = "revoked_refresh_token:"
USER_ZSET_PREFIX = "user_refresh_tokens:"

REFRESH_KEY_PATTERNS = (
    f"{ACTIVE_PREFIX}*",
    f"{PREVIOUS_PREFIX}*",
    f"{REVOKED_PREFIX}*",
    f"{USER_ZSET_PREFIX}*",
)


class AuthRedisMigrationMode(str, Enum):
    OFF = "off"
    LEGACY = "legacy"
    DUAL_WRITE = "dual_write"
    AUTH_PRIMARY = "auth_primary"
    AUTH_ONLY = "auth_only"


def get_migration_mode() -> AuthRedisMigrationMode:
    # Env prime (cutover ops sans redéploiement de config figée).
    raw = (os.getenv("AUTH_REDIS_MIGRATION_MODE") or "").strip()
    if not raw:
        try:
            from flask import current_app, has_app_context

            if has_app_context():
                raw = (
                    current_app.config.get("AUTH_REDIS_MIGRATION_MODE") or ""
                ).strip()
        except Exception:
            pass
    normalized = raw.lower().replace("-", "_")
    if not normalized or normalized in {"off", "none", "disabled"}:
        return AuthRedisMigrationMode.OFF
    if normalized == "legacy":
        return AuthRedisMigrationMode.LEGACY
    if normalized in {"dual_write", "dualwrite"}:
        return AuthRedisMigrationMode.DUAL_WRITE
    if normalized in {"auth_primary", "authprimary", "primary"}:
        return AuthRedisMigrationMode.AUTH_PRIMARY
    if normalized in {"auth_only", "authonly", "auth"}:
        return AuthRedisMigrationMode.AUTH_ONLY
    raise RuntimeError(
        f"AUTH_REDIS_MIGRATION_MODE invalide: {raw!r}. "
        "Valeurs: off|legacy|dual_write|auth_primary|auth_only"
    )


def migration_requires_dedicated_auth() -> bool:
    return get_migration_mode() in {
        AuthRedisMigrationMode.DUAL_WRITE,
        AuthRedisMigrationMode.AUTH_PRIMARY,
        AuthRedisMigrationMode.AUTH_ONLY,
    }


def assert_prod_migration_mode_not_ambiguous() -> None:
    """Refuse ``off`` + ``AUTH_REDIS_URL`` distinct en production.

    Cette combinaison faisait écrire silencieusement uniquement vers redis-auth
    via ``resolve_auth_redis_url()`` — cause des CURRENT auth-only observés.
    """
    from security.auth_redis import (
        _flask_env_name,
        resolve_dedicated_auth_redis_url,
        resolve_legacy_redis_url,
    )

    flask_env = _flask_env_name()
    is_prod = flask_env in ("production", "prod")
    try:
        from flask import current_app, has_app_context

        if has_app_context() and bool(current_app.config.get("TESTING")):
            return
    except Exception:
        pass
    if not is_prod:
        return

    mode = get_migration_mode()
    dedicated = resolve_dedicated_auth_redis_url()
    legacy = resolve_legacy_redis_url()
    if (
        mode == AuthRedisMigrationMode.OFF
        and dedicated
        and dedicated != legacy
    ):
        raise RuntimeError(
            "AUTH_REDIS_MIGRATION_MODE=off est interdit en production lorsque "
            "AUTH_REDIS_URL est défini et distinct de REDIS_URL. "
            "Choisir explicitement: legacy|dual_write|auth_primary|auth_only "
            "(évite l'écriture silencieuse auth-only)."
        )


def read_authority_is_auth() -> bool:
    return get_migration_mode() in {
        AuthRedisMigrationMode.AUTH_PRIMARY,
        AuthRedisMigrationMode.AUTH_ONLY,
    }


def dual_write_enabled() -> bool:
    return get_migration_mode() in {
        AuthRedisMigrationMode.DUAL_WRITE,
        AuthRedisMigrationMode.AUTH_PRIMARY,
    }


def _inc_dual_write_error() -> None:
    try:
        from security.security_metrics import auth_redis_dual_write_errors_total

        auth_redis_dual_write_errors_total.inc()
    except Exception:
        pass


def _inc_migration_metric(name: str) -> None:
    try:
        from security import security_metrics as sm

        gauge = getattr(sm, name, None)
        if gauge is not None:
            gauge.inc()
    except Exception:
        pass


class DualWriteRedisClient:
    """Proxy Redis : lectures selon autorité ; écritures duales selon le mode.

    Les méthodes non listées sont déléguées au client d'autorité de lecture
    (best-effort pour compat redis-py).
    """

    def __init__(
        self,
        *,
        read_client: Any,
        write_primary: Any,
        write_secondary: Any | None,
        mode: AuthRedisMigrationMode,
        secondary_label: str,
    ) -> None:
        self._read = read_client
        self._primary = write_primary
        self._secondary = write_secondary
        self._mode = mode
        self._secondary_label = secondary_label

    # --- lectures (autorité unique, pas de fallback) ---
    def get(self, name: str) -> Any:
        return self._read.get(name)

    def exists(self, *names: str) -> Any:
        return self._read.exists(*names)

    def ttl(self, name: str) -> Any:
        return self._read.ttl(name)

    def pttl(self, name: str) -> Any:
        return self._read.pttl(name)

    def ping(self) -> Any:
        return self._read.ping()

    def type(self, name: str) -> Any:
        return self._read.type(name)

    def zrange(self, name: str, start: int, end: int, **kwargs: Any) -> Any:
        return self._read.zrange(name, start, end, **kwargs)

    def zcard(self, name: str) -> Any:
        return self._read.zcard(name)

    def zscore(self, name: str, value: Any) -> Any:
        return self._read.zscore(name, value)

    def info(self, section: str | None = None, **kwargs: Any) -> Any:
        if section is None:
            return self._read.info(**kwargs)
        return self._read.info(section=section, **kwargs)

    def scan(
        self,
        cursor: int = 0,
        match: str | None = None,
        count: int | None = None,
        **kwargs: Any,
    ) -> Any:
        return self._read.scan(cursor=cursor, match=match, count=count, **kwargs)

    def scan_iter(
        self, match: str | None = None, count: int | None = None, **kwargs: Any
    ):
        return self._read.scan_iter(match=match, count=count, **kwargs)

    # --- écritures ---
    def _write(self, method: str, *args: Any, **kwargs: Any) -> Any:
        primary_fn = getattr(self._primary, method)
        result = primary_fn(*args, **kwargs)
        if self._secondary is None:
            return result
        try:
            secondary_fn = getattr(self._secondary, method)
            secondary_fn(*args, **kwargs)
        except Exception as exc:
            _inc_dual_write_error()
            logger.error(
                "auth_redis_dual_write_error target=%s method=%s err=%s mode=%s",
                self._secondary_label,
                method,
                type(exc).__name__,
                self._mode.value,
            )
            # dual_write : legacy est primaire — ne pas faire échouer l'utilisateur
            # auth_primary : auth est primaire — legacy secondaire best-effort
            if self._mode == AuthRedisMigrationMode.DUAL_WRITE:
                return result
            # auth_primary : primaire déjà OK ; secondaire optionnel
            return result
        return result

    def setex(self, name: str, time: Any, value: Any) -> Any:
        return self._write("setex", name, time, value)

    def psetex(self, name: str, time_ms: Any, value: Any) -> Any:
        return self._write("psetex", name, time_ms, value)

    def set(self, name: str, value: Any, **kwargs: Any) -> Any:
        return self._write("set", name, value, **kwargs)

    def delete(self, *names: str) -> Any:
        return self._write("delete", *names)

    def expire(self, name: str, time: Any) -> Any:
        return self._write("expire", name, time)

    def pexpire(self, name: str, time: Any) -> Any:
        return self._write("pexpire", name, time)

    def zadd(self, name: str, mapping: dict, **kwargs: Any) -> Any:
        return self._write("zadd", name, mapping, **kwargs)

    def zrem(self, name: str, *values: Any) -> Any:
        return self._write("zrem", name, *values)

    def close(self) -> None:
        for client in (self._read, self._primary, self._secondary):
            if client is None:
                continue
            with contextlib.suppress(Exception):
                client.close()

    def __getattr__(self, item: str) -> Any:
        # Délégation lecture pour méthodes redis-py non listées
        return getattr(self._read, item)
