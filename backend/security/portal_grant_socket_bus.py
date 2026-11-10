"""Coupure des sockets de tous les workers quand un droit est révoqué.

Le registre des sockets est local à chaque processus. La révocation est
publiée sur Redis après le commit ; chaque worker déconnecte les sockets
qu'il détient. Pub/Sub n'est pas une file : si l'abonnement tombe, le
worker cesse les réceptions professionnelles jusqu'au retour du canal.
Aucun jeton, message ou dossier n'est publié.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import suppress
from typing import Any

logger = logging.getLogger(__name__)

REVOCATION_CHANNEL = "portal_grant:revocation"
LISTENER_CLIENT_NAME = "portal-grant-revocation"
_RETRY_SECONDS = 1.0
_LISTENER_STARTED = False
_LISTENER_LOCK = threading.Lock()
_HEALTH_LOCK = threading.Lock()
_HEALTHY = True
_ON_REVOCATION: Any = None
_ON_CHANNEL_LOST: Any = None
_ON_CHANNEL_RECOVERED: Any = None


def revocation_channel_healthy() -> bool:
    """Le canal est sain, ou Redis n'est pas configuré (un seul processus)."""
    if _redis() is None:
        return True
    with _HEALTH_LOCK:
        return _HEALTHY


def publish_portal_revocation_payload(payload: dict[str, Any]) -> None:
    """Publie une révocation déjà validée. Un échec ferme le canal local."""
    user_id = payload.get("user_id")
    if not isinstance(user_id, int):
        return
    client = _redis()
    if client is None:
        return
    try:
        client.publish(REVOCATION_CHANNEL, json.dumps(payload))
        logger.info(
            "portal_grant_revocation_published user_id=%s portal=%s",
            user_id,
            payload.get("portal"),
        )
    except Exception:
        logger.exception(
            "portal_grant_revocation_publish_failed user_id=%s", user_id
        )
        _set_unhealthy("publish_failed")


def publish_mobile_session_socket_cut(session_id: str) -> None:
    """Demande à chaque worker de fermer les sockets de cette session mobile."""
    if not session_id:
        return
    client = _redis()
    if client is None:
        return
    try:
        client.publish(
            REVOCATION_CHANNEL,
            json.dumps(
                {
                    "kind": "mobile_session",
                    "mobile_session_id": str(session_id),
                }
            ),
        )
    except Exception:
        logger.exception(
            "mobile_session_socket_cut_publish_failed session=%s", session_id
        )


def publish_portal_revocation(grant: Any) -> None:
    """Publie tout de suite. La révocation métier passe par l'après-commit."""
    user_id = getattr(grant, "user_id", None)
    if not isinstance(user_id, int):
        return
    publish_portal_revocation_payload(
        {
            "user_id": user_id,
            "portal": str(getattr(grant, "portal", "") or ""),
            "company_id": getattr(grant, "company_id", None),
            "institution_id": getattr(grant, "institution_id", None),
        }
    )


def publish_mission_read_cutoff(user_id: int, conversation_id: int) -> None:
    """Demande à chaque worker de retirer ce chauffeur de la salle de mission."""
    publish_portal_revocation_payload(
        {
            "kind": "mission_read_cutoff",
            "user_id": int(user_id),
            "conversation_id": int(conversation_id),
            "portal": "driver",
        }
    )


def start_portal_revocation_listener(
    on_revocation: Any,
    *,
    on_channel_lost: Any = None,
    on_channel_recovered: Any = None,
) -> None:
    """Démarre un fil d'écoute. Sans effet s'il tourne déjà dans ce processus."""
    global _LISTENER_STARTED, _ON_REVOCATION, _ON_CHANNEL_LOST, _ON_CHANNEL_RECOVERED
    with _LISTENER_LOCK:
        _ON_REVOCATION = on_revocation
        if on_channel_lost is not None:
            _ON_CHANNEL_LOST = on_channel_lost
        if on_channel_recovered is not None:
            _ON_CHANNEL_RECOVERED = on_channel_recovered
        if _LISTENER_STARTED or _redis() is None:
            return
        thread = threading.Thread(
            target=_listen,
            name="portal-grant-revocation",
            daemon=True,
        )
        thread.start()
        _LISTENER_STARTED = True


def _redis() -> Any:
    from ext import redis_client

    return redis_client


def _set_unhealthy(reason: str) -> bool:
    global _HEALTHY
    with _HEALTH_LOCK:
        was_healthy = _HEALTHY
        _HEALTHY = False
    if not was_healthy:
        return False
    logger.warning(
        "portal_grant_channel_unhealthy reason=%s at=%.6f",
        reason,
        time.time(),
    )
    callback = _ON_CHANNEL_LOST
    if callback is not None:
        try:
            callback()
        except Exception:
            logger.exception("portal_grant_channel_lost_callback_failed")
    return True


def _note_channel_alive() -> None:
    global _HEALTHY
    with _HEALTH_LOCK:
        was_healthy = _HEALTHY
        _HEALTHY = True
    if was_healthy:
        return
    logger.warning("portal_grant_channel_recovered at=%.6f", time.time())
    callback = _ON_CHANNEL_RECOVERED
    if callback is None:
        return
    try:
        callback()
    except Exception:
        logger.exception("portal_grant_channel_recovered_callback_failed")


def _ping_redis(url: str) -> None:
    import redis

    probe = redis.Redis.from_url(
        url,
        socket_timeout=2,
        socket_connect_timeout=2,
    )
    try:
        probe.ping()
    finally:
        probe.close()


def _apply_message(message: Any) -> None:
    if not isinstance(message, dict) or message.get("type") != "message":
        return
    raw = message.get("data")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        return
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return
    if not isinstance(payload, dict):
        return
    callback = _ON_REVOCATION
    if callback is None:
        return
    try:
        callback(payload)
    except Exception:
        logger.exception(
            "portal_grant_revocation_apply_failed user_id=%s",
            payload.get("user_id"),
        )


def _listen() -> None:
    import redis

    url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    while True:
        client = None
        try:
            client = redis.Redis.from_url(
                url,
                socket_timeout=5,
                socket_connect_timeout=2,
                client_name=LISTENER_CLIENT_NAME,
            )
            pubsub = client.pubsub(ignore_subscribe_messages=True)
            pubsub.subscribe(REVOCATION_CHANNEL)
            _note_channel_alive()
            while True:
                message = pubsub.get_message(timeout=1.0)
                if message is None:
                    _ping_redis(url)
                    _note_channel_alive()
                    continue
                _apply_message(message)
                _note_channel_alive()
        except Exception:
            transition = _set_unhealthy("listener_failed")
            if transition:
                logger.exception("portal_grant_revocation_listener_failed")
            else:
                logger.warning("portal_grant_revocation_listener_retry")
            time.sleep(_RETRY_SECONDS)
        finally:
            if client is not None:
                with suppress(Exception):
                    client.close()
