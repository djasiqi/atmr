"""Vérifications runtime Socket.IO / Redis (multi-workers Gunicorn)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("app")


@dataclass(frozen=True)
class SocketIoRuntimeDiagnostics:
    redis_url_configured: bool
    message_queue_enabled: bool
    gunicorn_workers: int
    async_mode: str
    redis_ping_ok: bool | None
    redis_ping_error: str | None
    multi_worker_safe: bool
    warnings: tuple[str, ...]

    def to_log_extra(self) -> dict[str, Any]:
        return {
            "socketio_redis_configured": self.redis_url_configured,
            "socketio_message_queue": self.message_queue_enabled,
            "gunicorn_workers": self.gunicorn_workers,
            "socketio_async_mode": self.async_mode,
            "redis_ping_ok": self.redis_ping_ok,
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
    except Exception as exc:  # noqa: BLE001 — diagnostic only
        return False, str(exc)


def collect_socketio_runtime_diagnostics(
    *,
    message_queue: str | None,
    redis_url: str | None = None,
) -> SocketIoRuntimeDiagnostics:
    """Évalue si la stack Socket.IO est sûre pour N workers Gunicorn."""
    url = (redis_url if redis_url is not None else os.getenv("REDIS_URL", "")).strip()
    redis_configured = bool(url) and not url.startswith("memory://")
    mq_enabled = bool(message_queue and str(message_queue).strip())
    workers = _parse_gunicorn_workers()
    async_mode = (os.getenv("SOCKETIO_ASYNC_MODE") or "gevent").strip().lower()
    ping_ok, ping_err = _ping_redis(url) if redis_configured else (None, None)

    warnings: list[str] = []
    if workers > 1 and not mq_enabled:
        warnings.append(
            "GUNICORN_WORKERS>1 sans message_queue Redis : risque « Invalid session » "
            "(handshake sur un worker, requête suivante sur un autre)."
        )
    if redis_configured and ping_ok is False:
        warnings.append(f"REDIS_URL configurée mais ping échoue : {ping_err}")
    if workers > 1 and mq_enabled and ping_ok is False:
        warnings.append(
            "message_queue activée mais Redis injoignable : les workers ne partageront pas les SID."
        )

    multi_worker_safe = workers <= 1 or (mq_enabled and ping_ok is not False)

    return SocketIoRuntimeDiagnostics(
        redis_url_configured=redis_configured,
        message_queue_enabled=mq_enabled,
        gunicorn_workers=workers,
        async_mode=async_mode,
        redis_ping_ok=ping_ok,
        redis_ping_error=ping_err,
        multi_worker_safe=multi_worker_safe,
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
        "[Socket.IO] Runtime: workers=%s async_mode=%s message_queue=%s redis_ping=%s multi_worker_safe=%s",
        diag.gunicorn_workers,
        diag.async_mode,
        "enabled" if diag.message_queue_enabled else "disabled",
        diag.redis_ping_ok if diag.redis_ping_ok is not None else "n/a",
        diag.multi_worker_safe,
        extra=diag.to_log_extra(),
    )
    for warning in diag.warnings:
        app_logger.warning("[Socket.IO] %s", warning)
    if diag.message_queue_enabled and diag.redis_ping_ok:
        print(
            f"✅ [Socket.IO] Message queue Redis active (workers={diag.gunicorn_workers})",
            flush=True,
        )
    elif diag.gunicorn_workers == 1:
        print(
            "✅ [Socket.IO] Mode single-worker (message_queue optionnelle)", flush=True
        )
    return diag
