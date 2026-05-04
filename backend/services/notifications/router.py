# backend/services/notifications/router.py
"""NotificationRouter : orchestre Socket.IO + Push selon la matrice de routage."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from services.notifications.dedup_throttle import check_dedup_and_throttle
from services.notifications.domain_event_canonical import DomainEventCanonical
from services.notifications.router_config import (
    ROUTER_CONFIG,
    EventTypeConfig,
    is_driver_progress_status,
    is_driver_status_progress_type,
)

logger = logging.getLogger(__name__)


@dataclass
class SocketEmit:
    """Une émission Socket.IO à effectuer."""

    role: str  # driver | company
    role_id: int
    event: str
    payload: dict[str, Any]


@dataclass
class PushRequest:
    """Une demande de push à effectuer (après filtrage router)."""

    role: str
    role_id: int
    title: str
    body: str
    data: dict[str, Any]
    dedupe_key: str
    collapse_key: str
    severity: str


@dataclass
class RouterResult:
    """Résultat du routage : quoi émettre en socket, quoi envoyer en push."""

    socket_emits: list[SocketEmit] = field(default_factory=list)
    push_requests: list[PushRequest] = field(default_factory=list)
    skip_reasons: list[tuple[str, int, str | None]] = field(default_factory=list)
    # (recipient_role, recipient_id, reason)


def _is_actor_recipient(
    config: EventTypeConfig,
    actor_role: str | None,
    actor_id: int | None,
    recipient_role: str,
    recipient_id: int,
) -> bool:
    """True si le destinataire est l'acteur -> exclude_actor => skip push."""
    if not config.exclude_actor or not actor_role or actor_id is None:
        return False
    return (
        recipient_role == "driver"
        and actor_role == "driver"
        and actor_id == recipient_id
    ) or (
        recipient_role == "company"
        and actor_role == "company"
        and actor_id == recipient_id
    )


def route(
    event: DomainEventCanonical,
    *,
    presence_driver: bool | None = None,
    presence_company: bool | None = None,
) -> RouterResult:
    """Décide les socket_emits et push_requests pour un DomainEventCanonical.

    presence_* : True = connecté/actif, False = inactif, None = inconnu.
    Si if_inactive et presence inconnue -> on envoie la push.
    """
    result = RouterResult()
    config = ROUTER_CONFIG.get(event.type)
    if not config:
        config = EventTypeConfig(
            event_type=event.type,
            recipients="both",
            push_policy="if_inactive",
            exclude_actor=True,
        )

    # Déterminer les destinataires selon config.recipients
    driver_id = event.driver_id
    company_id = event.company_id
    actor_role = event.actor_role
    actor_id = event.actor_id

    # P0: pour DRIVER_EN_ROUTE / ONBOARD / COMPLETED, jamais de push chauffeur
    if is_driver_status_progress_type(event.type):
        driver_id_for_push = None
    else:
        driver_id_for_push = driver_id

    # Socket emits (toujours selon la logique métier appelante; le router peut
    # n'être utilisé que pour la décision push). Ici on remplit socket_emits
    # pour cohérence avec la spec "socket d'abord".
    if driver_id and config.recipients in ("driver", "both"):
        result.socket_emits.append(
            SocketEmit(
                role="driver",
                role_id=driver_id,
                event=event.type.lower(),
                payload=event.to_payload(),
            )
        )
    if company_id and config.recipients in ("company", "both"):
        result.socket_emits.append(
            SocketEmit(
                role="company",
                role_id=company_id,
                event=event.type.lower(),
                payload=event.to_payload(),
            )
        )

    # Push requests
    throttle = config.throttle
    throttle_scope = f"booking_{event.booking_id or 0}" if throttle else None
    throttle_window = throttle.window_s if throttle else 0
    throttle_max = throttle.max_per_window if throttle else 0
    dedupe_key = event.dedupe_key()

    def consider_push(recipient_role: str, recipient_id: int) -> None:
        if recipient_role == "driver" and driver_id_for_push is None:
            result.skip_reasons.append(
                ("driver", recipient_id, "driver_status_progress_never_push")
            )
            return
        if recipient_id <= 0:
            return
        if _is_actor_recipient(
            config, actor_role, actor_id, recipient_role, recipient_id
        ):
            result.skip_reasons.append((recipient_role, recipient_id, "exclude_actor"))
            return
        if config.push_policy == "never":
            result.skip_reasons.append((recipient_role, recipient_id, "policy_never"))
            return
        if config.push_policy == "if_inactive":
            pres = presence_driver if recipient_role == "driver" else presence_company
            if pres is True:
                result.skip_reasons.append(
                    (recipient_role, recipient_id, "if_inactive_recipient_active")
                )
                return
        skip, reason = check_dedup_and_throttle(
            recipient_role,
            recipient_id,
            dedupe_key,
            throttle_scope,
            throttle_window,
            throttle_max,
        )
        if skip:
            result.skip_reasons.append((recipient_role, recipient_id, reason))
            return
        result.push_requests.append(
            PushRequest(
                role=recipient_role,
                role_id=recipient_id,
                title=event.title,
                body=event.body,
                data=event.to_payload(),
                dedupe_key=dedupe_key,
                collapse_key=event.collapse_key(),
                severity=event.severity,
            )
        )

    if config.recipients in ("driver", "both") and driver_id_for_push:
        consider_push("driver", driver_id_for_push)
    if config.recipients in ("company", "both") and company_id:
        consider_push("company", company_id)

    return result


def should_skip_push_for_driver(
    event_type: str,
    status: str | None,
    actor_role: str | None,
    actor_id: int | None,
    driver_id: int | None,
) -> bool:
    """Helper P0 : True si on ne doit jamais envoyer de push au chauffeur pour cet événement.

    Utilisable depuis handle_booking_updated / fanout sans passer par le router complet.
    """
    if not driver_id:
        return True
    # Statut "driver progress" => l'acteur est le chauffeur => pas de push chauffeur
    return (
        is_driver_progress_status(status)
        or is_driver_status_progress_type(event_type)
        or (actor_role == "driver" and actor_id == driver_id)
    )
