# backend/services/notifications/notification_targets.py
"""Routage centralisé des notifications.

- NotificationTargets / compute_notification_targets : legacy, booking_updated only.
- BookingNotificationContext / FullNotificationTargets / compute_all_notification_targets :
  contrat unifie couvrant tous les events x tous les acteurs.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

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
    exclude_driver_id: (
        int | None
    )  # Si driver_id == exclude_driver_id, skip push/socket driver
    exclude_company_id: (
        int | None
    )  # Si company_id == exclude_company_id, skip push company (actor)


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
    driver_progress_statuses = {
        "en_route",
        "in_progress",
        "completed",
        "return_completed",
    }

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
    elif source == "driver_api":
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


# ---------------------------------------------------------------------------
# Contrat unifié V2 : contexte + matrice multi-acteurs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BookingNotificationContext:
    """Contexte résolu d'un booking pour le routage de notifications."""

    booking_id: int
    owner_company_id: int
    executing_company_id: int | None
    driver_id: int | None
    institution_id: int | None
    request_id: int | None
    request_public_id: str | None
    is_institution_sourced: bool
    is_subcontracted: bool


def resolve_booking_notification_context(
    booking_id: int,
) -> BookingNotificationContext | None:
    """Résout le contexte complet d'un booking pour le routage de notifications.

    Returns:
        BookingNotificationContext ou None si booking introuvable.
    """
    try:
        from ext import db
        from models import Booking
        from services.events.institution_events import get_request_info_from_booking

        booking = db.session.get(Booking, int(booking_id))
        if not booking:
            return None

        owner_company_id = int(getattr(booking, "company_id", 0) or 0)
        exec_id_raw = getattr(booking, "executing_company_id", None)
        executing_company_id = int(exec_id_raw) if exec_id_raw else None
        driver_id = getattr(booking, "driver_id", None)
        if driver_id is not None:
            driver_id = int(driver_id)

        req_info = get_request_info_from_booking(int(booking_id))
        institution_id = req_info.get("institution_id") if req_info else None
        request_id = req_info.get("request_id") if req_info else None
        request_public_id = req_info.get("public_id") if req_info else None

        is_subcontracted = (
            executing_company_id is not None
            and executing_company_id != owner_company_id
        )

        return BookingNotificationContext(
            booking_id=int(booking_id),
            owner_company_id=owner_company_id,
            executing_company_id=executing_company_id,
            driver_id=driver_id,
            institution_id=institution_id,
            request_id=request_id,
            request_public_id=request_public_id,
            is_institution_sourced=institution_id is not None,
            is_subcontracted=is_subcontracted,
        )
    except Exception:
        logger.exception(
            "[notification_targets] Failed to resolve context for booking %s",
            booking_id,
        )
        return None


@dataclass(frozen=True)
class FullNotificationTargets:
    """Cibles de notification pour tous les events x tous les acteurs."""

    # Driver
    notify_driver_socket: bool = False
    notify_driver_push: bool = False
    exclude_driver_id: int | None = None

    # Owner company (donneur d'ordre)
    notify_owner_socket: bool = False
    notify_owner_push: bool = False
    owner_company_id: int | None = None

    # Executing company (partenaire, si sous-traitée)
    notify_executing_socket: bool = False
    notify_executing_push: bool = False
    notify_executing_persist: bool = False
    executing_company_id: int | None = None

    # Institution (Socket.IO + InstitutionNotification, jamais push)
    notify_institution_socket: bool = False
    notify_institution_persist: bool = False
    institution_id: int | None = None


# Statuts pour lesquels l'institution est notifiée (skip in_progress)
_INSTITUTION_NOTIFY_STATUSES = {"en_route", "completed", "cancelled", "canceled"}
# Statuts pour lesquels l'executing company reçoit un push (pas en_route/in_progress)
_EXECUTING_PUSH_STATUSES = {"completed", "cancelled", "canceled", "return_completed"}


def compute_all_notification_targets(
    event_type: str,
    ctx: BookingNotificationContext,
    actor_role: str | None = None,
    actor_id: int | None = None,
    status: str | None = None,
) -> FullNotificationTargets:
    """Matrice de routage centralisee : qui recoit quoi pour chaque event.

    Si actor_role est None, il est traite comme "system" (admin, cron, script, etc.)
    pour garantir que tous les acteurs concernes soient notifies.
    """
    effective_actor_role = actor_role or "system"
    status_lower = (status or "").lower()

    if event_type == "booking_assigned":
        return _targets_booking_assigned(ctx)
    if event_type == "booking_reassigned":
        return _targets_booking_reassigned(ctx)
    if event_type == "booking_updated":
        return _targets_booking_updated(
            ctx, effective_actor_role, actor_id, status_lower
        )
    if event_type == "booking_cancelled":
        return _targets_booking_cancelled(ctx, effective_actor_role)

    # Fallback conservateur : owner + driver
    return FullNotificationTargets(
        notify_driver_socket=ctx.driver_id is not None,
        notify_driver_push=ctx.driver_id is not None,
        notify_owner_socket=True,
        notify_owner_push=True,
        owner_company_id=ctx.owner_company_id,
        executing_company_id=ctx.executing_company_id,
        institution_id=ctx.institution_id,
    )


def _targets_booking_assigned(
    ctx: BookingNotificationContext,
) -> FullNotificationTargets:
    return FullNotificationTargets(
        notify_driver_socket=ctx.driver_id is not None,
        notify_driver_push=ctx.driver_id is not None,
        notify_owner_socket=True,
        notify_owner_push=False,
        owner_company_id=ctx.owner_company_id,
        notify_institution_socket=ctx.is_institution_sourced,
        notify_institution_persist=ctx.is_institution_sourced,
        institution_id=ctx.institution_id if ctx.is_institution_sourced else None,
        notify_executing_socket=ctx.is_subcontracted,
        notify_executing_push=False,
        notify_executing_persist=False,
        executing_company_id=ctx.executing_company_id if ctx.is_subcontracted else None,
    )


def _targets_booking_reassigned(
    ctx: BookingNotificationContext,
) -> FullNotificationTargets:
    return FullNotificationTargets(
        notify_driver_socket=True,
        notify_driver_push=True,
        notify_owner_socket=True,
        notify_owner_push=False,
        owner_company_id=ctx.owner_company_id,
        # Institution : NON (règle explicite)
        notify_institution_socket=False,
        notify_institution_persist=False,
        institution_id=None,
        notify_executing_socket=ctx.is_subcontracted,
        notify_executing_push=False,
        notify_executing_persist=False,
        executing_company_id=ctx.executing_company_id if ctx.is_subcontracted else None,
    )


def _targets_booking_updated(
    ctx: BookingNotificationContext,
    actor_role: str | None,
    actor_id: int | None,
    status_lower: str,
) -> FullNotificationTargets:
    is_driver_actor = (
        actor_role == "driver"
        and actor_id is not None
        and ctx.driver_id is not None
        and int(actor_id) == int(ctx.driver_id)
    )
    is_company_actor = actor_role == "company"

    # Driver/Owner : réutilise la logique exclude_actor existante
    notify_driver_socket = not is_driver_actor
    notify_driver_push = not is_driver_actor
    notify_owner_socket = True
    notify_owner_push = not is_company_actor

    # Institution : en_route + completed (skip in_progress)
    inst_notify = (
        ctx.is_institution_sourced and status_lower in _INSTITUTION_NOTIFY_STATUSES
    )

    # Executing : toujours socket, push seulement completed/cancelled
    exec_push = ctx.is_subcontracted and status_lower in _EXECUTING_PUSH_STATUSES
    exec_persist = ctx.is_subcontracted and status_lower in _EXECUTING_PUSH_STATUSES

    return FullNotificationTargets(
        notify_driver_socket=notify_driver_socket,
        notify_driver_push=notify_driver_push,
        exclude_driver_id=ctx.driver_id if is_driver_actor else None,
        notify_owner_socket=notify_owner_socket,
        notify_owner_push=notify_owner_push,
        owner_company_id=ctx.owner_company_id,
        notify_institution_socket=inst_notify,
        notify_institution_persist=inst_notify,
        institution_id=ctx.institution_id if inst_notify else None,
        notify_executing_socket=ctx.is_subcontracted,
        notify_executing_push=exec_push,
        notify_executing_persist=exec_persist,
        executing_company_id=ctx.executing_company_id if ctx.is_subcontracted else None,
    )


def _targets_booking_cancelled(
    ctx: BookingNotificationContext,
    actor_role: str | None,
) -> FullNotificationTargets:
    is_company_actor = actor_role == "company"
    is_driver_actor = actor_role == "driver"
    is_institution_actor = actor_role == "institution"

    return FullNotificationTargets(
        # Driver : notifié sauf s'il est l'acteur
        notify_driver_socket=ctx.driver_id is not None and not is_driver_actor,
        notify_driver_push=ctx.driver_id is not None and not is_driver_actor,
        exclude_driver_id=ctx.driver_id if is_driver_actor else None,
        # Owner : notifié sauf s'il est l'acteur
        notify_owner_socket=not is_company_actor,
        notify_owner_push=not is_company_actor,
        owner_company_id=ctx.owner_company_id,
        # Institution : notifiée sauf si elle est l'acteur
        notify_institution_socket=ctx.is_institution_sourced
        and not is_institution_actor,
        notify_institution_persist=ctx.is_institution_sourced
        and not is_institution_actor,
        institution_id=(
            ctx.institution_id
            if ctx.is_institution_sourced and not is_institution_actor
            else None
        ),
        # Executing : toujours notifié si sous-traitée
        notify_executing_socket=ctx.is_subcontracted,
        notify_executing_push=ctx.is_subcontracted,
        notify_executing_persist=ctx.is_subcontracted,
        executing_company_id=ctx.executing_company_id if ctx.is_subcontracted else None,
    )
