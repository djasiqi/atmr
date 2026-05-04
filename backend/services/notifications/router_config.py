# backend/services/notifications/router_config.py
"""Configuration du routage par type d'événement (matrice notifications)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Policies pour la push
PushPolicy = Literal["never", "if_inactive", "always"]
RecipientRole = Literal["driver", "company", "both"]


@dataclass(frozen=True)
class ThrottleRule:
    """Règle de throttle par scope."""

    window_s: int
    max_per_window: int
    scope: str  # "booking_id+recipient" | "booking_id+company" | etc.


@dataclass(frozen=True)
class EventTypeConfig:
    """Config par type d'événement pour le NotificationRouter."""

    event_type: str
    recipients: RecipientRole  # driver | company | both
    push_policy: PushPolicy  # never | if_inactive | always
    exclude_actor: bool = True
    throttle: ThrottleRule | None = None
    collapse_scope: str = "booking_id+category"  # pour collapse_key


# Types "driver status progress" : JAMAIS de push au chauffeur (il est l'acteur)
DRIVER_STATUS_PROGRESS_TYPES = frozenset(
    {"DRIVER_EN_ROUTE", "DRIVER_ONBOARD", "DRIVER_COMPLETED", "DRIVER_RETURN_COMPLETED"}
)

# Statuts booking qui indiquent une action chauffeur -> pas de push chauffeur
DRIVER_PROGRESS_STATUS_VALUES = frozenset(
    {"en_route", "in_progress", "completed", "return_completed"}
)

ROUTER_CONFIG: dict[str, EventTypeConfig] = {
    # ----- Entreprise -> Chauffeur -----
    "BOOKING_ASSIGNED": EventTypeConfig(
        event_type="BOOKING_ASSIGNED",
        recipients="driver",
        push_policy="if_inactive",
        exclude_actor=True,
        throttle=None,
    ),
    "BOOKING_UPDATED": EventTypeConfig(
        event_type="BOOKING_UPDATED",
        recipients="both",
        push_policy="if_inactive",
        exclude_actor=True,
        throttle=ThrottleRule(
            window_s=60, max_per_window=1, scope="booking_id+recipient"
        ),
    ),
    "BOOKING_CANCELED": EventTypeConfig(
        event_type="BOOKING_CANCELED",
        recipients="driver",
        push_policy="always",
        exclude_actor=True,
    ),
    "BOOKING_UNASSIGNED": EventTypeConfig(
        event_type="BOOKING_UNASSIGNED",
        recipients="driver",
        push_policy="always",
        exclude_actor=True,
    ),
    "BOOKING_REASSIGNED": EventTypeConfig(
        event_type="BOOKING_REASSIGNED",
        recipients="driver",
        push_policy="always",
        exclude_actor=True,
    ),
    "BOOKING_MESSAGE": EventTypeConfig(
        event_type="BOOKING_MESSAGE",
        recipients="driver",
        push_policy="if_inactive",
        exclude_actor=True,
    ),
    # ----- Chauffeur -> Entreprise (P0: jamais push au chauffeur) -----
    "DRIVER_EN_ROUTE": EventTypeConfig(
        event_type="DRIVER_EN_ROUTE",
        recipients="company",
        push_policy="if_inactive",
        exclude_actor=True,
        throttle=ThrottleRule(
            window_s=30, max_per_window=1, scope="booking_id+company"
        ),
    ),
    "DRIVER_ONBOARD": EventTypeConfig(
        event_type="DRIVER_ONBOARD",
        recipients="company",
        push_policy="if_inactive",
        exclude_actor=True,
        throttle=ThrottleRule(
            window_s=30, max_per_window=1, scope="booking_id+company"
        ),
    ),
    "DRIVER_COMPLETED": EventTypeConfig(
        event_type="DRIVER_COMPLETED",
        recipients="company",
        push_policy="if_inactive",
        exclude_actor=True,
    ),
    "DRIVER_RETURN_COMPLETED": EventTypeConfig(
        event_type="DRIVER_RETURN_COMPLETED",
        recipients="company",
        push_policy="if_inactive",
        exclude_actor=True,
    ),
    # ----- Alertes -----
    "DELAY_DETECTED": EventTypeConfig(
        event_type="DELAY_DETECTED",
        recipients="both",
        push_policy="always",
        exclude_actor=False,  # alerte double chauffeur + entreprise
    ),
    "ASSIGNED_STILL_NOT_STARTED": EventTypeConfig(
        event_type="ASSIGNED_STILL_NOT_STARTED",
        recipients="both",
        push_policy="always",
        exclude_actor=True,
    ),
}

# Mapping des types "métier" existants (fanout/socket) vers les types canoniques
BOOKING_UPDATED_STATUS_TO_CANONICAL_TYPE: dict[str, str] = {
    "en_route": "DRIVER_EN_ROUTE",
    "in_progress": "DRIVER_ONBOARD",
    "completed": "DRIVER_COMPLETED",
    "return_completed": "DRIVER_RETURN_COMPLETED",
}


def get_canonical_type_for_booking_updated(
    status: str | None, actor_role: str | None
) -> str:
    """Retourne le type canonique pour un booking_updated selon statut et acteur."""
    if not status:
        return "BOOKING_UPDATED"
    s = (status or "").lower()
    if actor_role == "driver" and s in BOOKING_UPDATED_STATUS_TO_CANONICAL_TYPE:
        return BOOKING_UPDATED_STATUS_TO_CANONICAL_TYPE[s]
    return "BOOKING_UPDATED"


def is_driver_status_progress_type(event_type: str) -> bool:
    """True si l'événement est un "driver status progress" (jamais push driver)."""
    return event_type in DRIVER_STATUS_PROGRESS_TYPES


def is_driver_progress_status(status: str | None) -> bool:
    """True si le statut booking indique une action chauffeur (en_route, onboard, completed)."""
    return (status or "").lower() in DRIVER_PROGRESS_STATUS_VALUES
