# backend/services/notifications/notification_targets.py
"""Routage centralisé des notifications booking_updated (exclude_actor, policy canonique).

P0: Le chauffeur qui change le statut ne doit JAMAIS recevoir de push pour sa propre action.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

DEBUG_NOTIF_ROUTING = os.environ.get("DEBUG_NOTIF_ROUTING", "").lower() in (
    "1",
    "true",
    "yes",
)


@dataclass(frozen=True)
class NotificationTargets:
    """Cibles de notification pour booking_updated."""

    notify_driver_socket: bool
    notify_driver_push: bool
    notify_company_socket: bool
    notify_company_push: bool
    exclude_driver_id: int | None  # Si driver_id == exclude_driver_id, skip push/socket driver
    exclude_company_id: int | None  # Si company_id == exclude_company_id, skip push company (actor)


def compute_notification_targets(
    *,
    driver_id: int,
    company_id: int,
    actor_role: str | None,
    actor_id: int | None,
    status: str | None,
    source: str | None = None,
) -> NotificationTargets:
    """Calcule les cibles de notification selon la politique exclude_actor.

    Policy:
    - actor_role == "driver" et actor_id == driver_id:
        -> company only (socket + push), driver NEVER push, driver socket optionnel (non pour éviter confusion)
    - actor_role == "company":
        -> driver (socket + push), company socket (pas push actor si applicable)
    - actor_role inconnu:
        -> fallback safe: company socket+push, driver socket+push SEULEMENT si status
           n'est pas un "driver progress" (en_route, in_progress, completed, return_completed)
    """
    status_lower = (status or "").lower()
    driver_progress_statuses = {"en_route", "in_progress", "completed", "return_completed"}

    is_driver_actor = (
        actor_role == "driver"
        and actor_id is not None
        and int(actor_id) == int(driver_id)
    )
    is_company_actor = actor_role == "company" and actor_id is not None

    if is_driver_actor:
        # Chauffeur a changé le statut -> notifier UNIQUEMENT l'entreprise
        targets = NotificationTargets(
            notify_driver_socket=False,
            notify_driver_push=False,
            notify_company_socket=True,
            notify_company_push=True,
            exclude_driver_id=driver_id,
            exclude_company_id=None,
        )
    elif is_company_actor:
        # Entreprise a modifié -> notifier le chauffeur
        targets = NotificationTargets(
            notify_driver_socket=True,
            notify_driver_push=True,
            notify_company_socket=True,
            notify_company_push=False,  # Pas de push à l'entreprise acteur
            exclude_driver_id=None,
            exclude_company_id=int(actor_id) if actor_id else None,
        )
    else:
        # actor_role inconnu: fallback avec source si présent
        # source=driver_api => actor implicite = driver, company only
        if source == "driver_api":
            targets = NotificationTargets(
                notify_driver_socket=False,
                notify_driver_push=False,
                notify_company_socket=True,
                notify_company_push=True,
                exclude_driver_id=driver_id,
                exclude_company_id=None,
            )
        elif source == "company_api":
            targets = NotificationTargets(
                notify_driver_socket=True,
                notify_driver_push=True,
                notify_company_socket=True,
                notify_company_push=False,
                exclude_driver_id=None,
                exclude_company_id=None,
            )
        else:
            # source absent ou system: fallback conservateur
            is_driver_progress = status_lower in driver_progress_statuses
            actor_is_other_driver = (
                actor_role == "driver"
                and actor_id is not None
                and int(actor_id) != int(driver_id)
            )
            skip_driver_push = is_driver_progress and not actor_is_other_driver
            targets = NotificationTargets(
                notify_driver_socket=True,
                notify_driver_push=not skip_driver_push,
                notify_company_socket=True,
                notify_company_push=True,
                exclude_driver_id=driver_id if skip_driver_push else None,
                exclude_company_id=None,
            )

    if DEBUG_NOTIF_ROUTING:
        _log_routing_debug(
            driver_id=driver_id,
            company_id=company_id,
            actor_role=actor_role,
            actor_id=actor_id,
            status=status,
            targets=targets,
        )

    return targets


def _log_routing_debug(
    *,
    driver_id: int,
    company_id: int,
    actor_role: str | None,
    actor_id: int | None,
    status: str | None,
    targets: NotificationTargets,
) -> None:
    """Log structuré JSON pour debug du routage (DEBUG_NOTIF_ROUTING=1)."""
    try:
        from ext import app_logger

        payload = {
            "event_type": "booking_updated",
            "driver_id": driver_id,
            "company_id": company_id,
            "actor_role": actor_role,
            "actor_id": actor_id,
            "status": status,
            "recipients": {
                "driver_socket": [driver_id] if targets.notify_driver_socket else [],
                "driver_push": [driver_id] if targets.notify_driver_push else [],
                "company_socket": [company_id] if targets.notify_company_socket else [],
                "company_push": [company_id] if targets.notify_company_push else [],
            },
            "exclude_actor_applied": targets.exclude_driver_id is not None
            or targets.exclude_company_id is not None,
        }
        app_logger.info(
            "[DEBUG_NOTIF_ROUTING] %s",
            json.dumps(payload, default=str),
        )
    except Exception:
        pass
