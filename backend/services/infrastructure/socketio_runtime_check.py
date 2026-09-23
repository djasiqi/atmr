"""Vérifications runtime Socket.IO / Redis (multi-workers Gunicorn)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("app")


def _worker_affinity_guaranteed() -> bool:
    """Vrai seulement si l'opérateur affirme une affinité HTTP vers le même worker.

    Gunicorn ne la fournit pas. Le défaut est donc faux. Cette variable ne change
    pas le routage : elle qualifie uniquement le diagnostic.
    """
    raw = (os.getenv("SOCKETIO_WORKER_AFFINITY_GUARANTEED") or "").strip().lower()
    return raw in {"1", "true", "yes"}


@dataclass(frozen=True)
class SocketIoRuntimeDiagnostics:
    redis_url_configured: bool
    message_queue_enabled: bool
    gunicorn_workers: int
    async_mode: str
    redis_ping_ok: bool | None
    redis_ping_error: str | None
    worker_affinity_guaranteed: bool
    engineio_polling_safe: bool
    # Alias historique : même valeur que engineio_polling_safe.
    # Ne signifie pas « Redis rend le long-polling sûr ».
    multi_worker_safe: bool
    warnings: tuple[str, ...]

    def to_log_extra(self) -> dict[str, Any]:
        return {
            "socketio_redis_configured": self.redis_url_configured,
            "socketio_message_queue": self.message_queue_enabled,
            "gunicorn_workers": self.gunicorn_workers,
            "socketio_async_mode": self.async_mode,
            "redis_ping_ok": self.redis_ping_ok,
            "socketio_worker_affinity_guaranteed": self.worker_affinity_guaranteed,
            "socketio_engineio_polling_safe": self.engineio_polling_safe,
            "socketio_multi_worker_safe": self.multi_worker_safe,
        }


def _parse_gunicorn_workers() -> int:
    raw = (os.getenv("GUNICORN_WORKERS") or "1").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def _ping_redis(redis_url: str) -> tuple[bool | None, str | None]:
    if not redis_url or redis_url.startswith("memory://"):
        return None, None
    try:
        import redis

        client = redis.Redis.from_url(
            redis_url,
            socket_connect_timeout=3,
            socket_timeout=3,
        )
        client.ping()
        return True, None
    except Exception as exc:
        return False, str(exc)


def collect_socketio_runtime_diagnostics(
    *,
    message_queue: str | None,
    redis_url: str | None = None,
) -> SocketIoRuntimeDiagnostics:
    """Sépare la file d'émissions Redis de la propriété du sid Engine.IO.

    ``message_queue_enabled`` décrit la coordination cross-process des émissions.
    ``engineio_polling_safe`` est vrai seulement si chaque sid reste dans le même
    process. Redis ne copie pas ``self.sockets`` : avec plusieurs workers Gunicorn
    et sans affinité de worker, le long-polling n'est pas sûr.
    """
    url = (redis_url if redis_url is not None else os.getenv("REDIS_URL", "")).strip()
    redis_configured = bool(url) and not url.startswith("memory://")
    mq_enabled = bool(message_queue and str(message_queue).strip())
    workers = _parse_gunicorn_workers()
    async_mode = (os.getenv("SOCKETIO_ASYNC_MODE") or "gevent").strip().lower()
    ping_ok, ping_err = _ping_redis(url) if redis_configured else (None, None)
    affinity = _worker_affinity_guaranteed()
    # Un seul process, ou une affinité explicite vers ce process. Redis n'entre pas ici.
    engineio_polling_safe = workers <= 1 or affinity

    warnings: list[str] = []
    if not engineio_polling_safe:
        warnings.append(
            "GUNICORN_WORKERS>1 sans affinité vers le même worker : long-polling "
            "Engine.IO non sûr (sid dans self.sockets du process). Un handshake 200 "
            "puis une requête sid sur un autre worker produit « Invalid session » / "
            "HTTP 400. message_queue Redis ne partage pas ces sid."
        )
    if workers > 1 and not mq_enabled:
        warnings.append(
            "GUNICORN_WORKERS>1 sans message_queue Redis : les émissions Socket.IO "
            "ne sont pas coordonnées entre workers."
        )
    if redis_configured and ping_ok is False:
        warnings.append(f"REDIS_URL configurée mais ping échoue : {ping_err}")
    if workers > 1 and mq_enabled and ping_ok is False:
        warnings.append(
            "message_queue activée mais Redis injoignable : les émissions "
            "cross-process ne seront pas délivrées. Cela ne partage pas non plus "
            "les sid Engine.IO."
        )

    return SocketIoRuntimeDiagnostics(
        redis_url_configured=redis_configured,
        message_queue_enabled=mq_enabled,
        gunicorn_workers=workers,
        async_mode=async_mode,
        redis_ping_ok=ping_ok,
        redis_ping_error=ping_err,
        worker_affinity_guaranteed=affinity,
        engineio_polling_safe=engineio_polling_safe,
        multi_worker_safe=engineio_polling_safe,
        warnings=tuple(warnings),
    )


def log_socketio_runtime_diagnostics(
    app_logger: logging.Logger,
    *,
    message_queue: str | None,
    redis_url: str | None = None,
) -> SocketIoRuntimeDiagnostics:
    """Log structuré au boot — à appeler après socketio.init_app()."""
    diag = collect_socketio_runtime_diagnostics(
        message_queue=message_queue,
        redis_url=redis_url,
    )
    level = logging.WARNING if diag.warnings else logging.INFO
    app_logger.log(
        level,
        "[Socket.IO] Runtime: workers=%s async_mode=%s message_queue=%s redis_ping=%s "
        "worker_affinity=%s engineio_polling_safe=%s",
        diag.gunicorn_workers,
        diag.async_mode,
        "enabled" if diag.message_queue_enabled else "disabled",
        diag.redis_ping_ok if diag.redis_ping_ok is not None else "n/a",
        diag.worker_affinity_guaranteed,
        diag.engineio_polling_safe,
        extra=diag.to_log_extra(),
    )
    for warning in diag.warnings:
        app_logger.warning("[Socket.IO] %s", warning)
    if diag.message_queue_enabled and diag.redis_ping_ok:
        print(
            f"✅ [Socket.IO] Message queue Redis active "
            f"(workers={diag.gunicorn_workers}) — coordination des émissions, "
            "pas la propriété des sid Engine.IO",
            flush=True,
        )
    if diag.gunicorn_workers == 1:
        print(
            "✅ [Socket.IO] Mode single-worker : long-polling Engine.IO "
            "dans le même process",
            flush=True,
        )
    elif not diag.engineio_polling_safe:
        print(
            "⚠️ [Socket.IO] Long-polling Engine.IO non sûr : plusieurs "
            "workers sans affinité",
            flush=True,
        )
    return diag
