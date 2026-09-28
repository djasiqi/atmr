"""Dédup push d'assignation : le claim n'est pas une preuve d'envoi.

IN_FLIGHT est un bail court, pris par le worker juste avant l'appel fournisseur.
SENT n'est posé qu'après une acceptation provider. Un crash avant l'envoi libère
le bail (ou le laisse expirer) pour qu'un retry ou un nouvel événement puisse partir.

Fenêtre DELIVERY-06 : le provider a accepté, puis le processus meurt avant
l'écriture de SENT. Ce n'est pas exactement-une-fois. Le retry peut renvoyer.
Sémantique assumée : at-least-once. Le même ``notification_id`` (``event_id``
du payload, réutilisé par Celery) trace les deux acceptations.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_DEDUP_NS = os.getenv("DRIVER_PUSH_DEDUP_REDIS_NS", "atmr:driver_push:dedup")
_SENT_TTL_SEC = int(os.getenv("DRIVER_PUSH_DEDUP_TTL_SEC", "45"))
_IN_FLIGHT_TTL_SEC = int(os.getenv("DRIVER_PUSH_IN_FLIGHT_TTL_SEC", "20"))
_ENABLED = os.getenv("DRIVER_PUSH_DEDUP_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)


def _redis():
    from ext import redis_client

    return redis_client


def _sent_key(driver_id: int, booking_id: int) -> str:
    return f"{_DEDUP_NS}:sent:{driver_id}:{booking_id}"


def _inflight_key(driver_id: int, booking_id: int) -> str:
    return f"{_DEDUP_NS}:inflight:{driver_id}:{booking_id}"


def _usable(driver_id: int, booking_id: int) -> bool:
    return _ENABLED and driver_id > 0 and booking_id > 0


def driver_booking_push_already_sent(driver_id: int, booking_id: int) -> bool:
    """Vrai seulement si un provider a déjà accepté cet envoi."""
    if not _usable(driver_id, booking_id):
        return False
    rc = _redis()
    if not rc:
        return False
    try:
        return bool(rc.get(_sent_key(driver_id, booking_id)))
    except Exception as exc:
        logger.debug("driver_booking_push_already_sent fail-open: %s", exc)
        return False


def begin_driver_booking_push(driver_id: int, booking_id: int) -> str:
    """Démarre une tentative worker.

    Retours :
    - ``open`` : Redis absent ou dédup désactivé (fail-open, pas de marqueur)
    - ``sent`` : déjà accepté par un provider, ne pas renvoyer
    - ``claimed`` : bail IN_FLIGHT obtenu
    - ``busy`` : une autre tentative tient le bail, retry plus tard
    """
    if not _usable(driver_id, booking_id):
        return "open"
    rc = _redis()
    if not rc:
        return "open"
    try:
        if rc.get(_sent_key(driver_id, booking_id)):
            return "sent"
        ttl = max(5, min(60, _IN_FLIGHT_TTL_SEC))
        ok = rc.set(_inflight_key(driver_id, booking_id), "1", nx=True, ex=ttl)
        return "claimed" if ok else "busy"
    except Exception as exc:
        logger.debug("begin_driver_booking_push fail-open: %s", exc)
        return "open"


def mark_driver_booking_push_sent(driver_id: int, booking_id: int) -> None:
    """Pose SENT après acceptation provider et libère le bail."""
    if not _usable(driver_id, booking_id):
        return
    rc = _redis()
    if not rc:
        return
    try:
        ttl = max(30, min(120, _SENT_TTL_SEC))
        rc.set(_sent_key(driver_id, booking_id), "1", ex=ttl)
        rc.delete(_inflight_key(driver_id, booking_id))
    except Exception as exc:
        logger.debug("mark_driver_booking_push_sent: %s", exc)


def release_driver_booking_push(driver_id: int, booking_id: int) -> None:
    """Libère IN_FLIGHT sans poser SENT. L'événement reste retryable."""
    if not _usable(driver_id, booking_id):
        return
    rc = _redis()
    if not rc:
        return
    try:
        rc.delete(_inflight_key(driver_id, booking_id))
    except Exception as exc:
        logger.debug("release_driver_booking_push: %s", exc)


def claim_driver_booking_push(
    driver_id: int,
    booking_id: int,
    *,
    ttl_sec: int | None = None,
) -> bool:
    """Compatibilité : True si la tentative peut continuer (claimed ou open).

    Ne pose pas SENT. Un second appel concurrent reçoit False (busy ou sent).
    """
    del ttl_sec
    state = begin_driver_booking_push(driver_id, booking_id)
    return state in {"claimed", "open"}
