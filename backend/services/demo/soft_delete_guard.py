from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import event
from sqlalchemy.orm import Session

from models import Booking, Client, Company, Driver, Institution, Invoice, User, Vehicle
from models.enums import BookingStatus, InvoiceStatus

logger = logging.getLogger(__name__)

_LISTENER_REGISTERED = False
_DEMO_DOMAINS = ("@demo.lirie.ch", "@demo.local")


def _is_demo_email(value: str | None) -> bool:
    email = str(value or "").strip().lower()
    if not email or "@" not in email:
        return False
    return email.startswith("demo-") or email.endswith(_DEMO_DOMAINS)


def _company_is_demo(company: Company | None) -> bool:
    if not company:
        return False
    if _is_demo_email(getattr(company, "contact_email", None)):
        return True
    owner = getattr(company, "user", None)
    return _is_demo_email(getattr(owner, "email", None))


def _institution_is_demo(institution: Institution | None) -> bool:
    """Vérifie si une institution est une institution démo (seed ou accès démo)."""
    if not institution:
        return False
    if _is_demo_email(getattr(institution, "contact_email", None)):
        return True
    users = getattr(institution, "users", None)
    if users:
        for u in users:
            if _is_demo_email(getattr(u, "email", None)):
                return True
    return False


def _user_is_demo(user: User | None) -> bool:
    if not user:
        return False
    return _is_demo_email(getattr(user, "email", None))


def _client_is_demo(client: Client) -> bool:
    return _user_is_demo(getattr(client, "user", None)) or _is_demo_email(
        getattr(client, "contact_email", None)
    )


def _driver_is_demo(driver: Driver) -> bool:
    return _user_is_demo(getattr(driver, "user", None)) or _company_is_demo(
        getattr(driver, "company", None)
    )


def _vehicle_is_demo(vehicle: Vehicle) -> bool:
    return _company_is_demo(getattr(vehicle, "company", None))


def _booking_is_demo(booking: Booking) -> bool:
    if _company_is_demo(getattr(booking, "company", None)):
        return True
    if _user_is_demo(getattr(booking, "user", None)):
        return True
    client = getattr(booking, "client", None)
    return bool(client and _client_is_demo(client))


def _invoice_is_demo(invoice: Invoice) -> bool:
    if _company_is_demo(getattr(invoice, "company", None)):
        return True
    client = getattr(invoice, "client", None)
    return bool(client and _client_is_demo(client))


def _convert_demo_delete_to_soft_delete(session: Session, obj: Any) -> bool:
    if isinstance(obj, Client) and _client_is_demo(obj):
        obj.is_active = False
        session.add(obj)
        return True

    if isinstance(obj, Driver) and _driver_is_demo(obj):
        obj.is_active = False
        obj.is_available = False
        session.add(obj)
        return True

    if isinstance(obj, Vehicle) and _vehicle_is_demo(obj):
        if hasattr(obj, "is_active"):
            obj.is_active = False
            session.add(obj)
            return True
        return False

    if isinstance(obj, Booking) and _booking_is_demo(obj):
        obj.status = BookingStatus.CANCELED
        session.add(obj)
        return True

    if isinstance(obj, Invoice) and _invoice_is_demo(obj):
        obj.status = InvoiceStatus.CANCELLED
        session.add(obj)
        return True

    return False


def register_demo_soft_delete_guard() -> None:
    """Empêche les suppressions physiques des enregistrements démo clés.

    On convertit les `session.delete()` en suppression logique pour:
    clients, chauffeurs, véhicules, réservations, factures.
    """
    global _LISTENER_REGISTERED
    if _LISTENER_REGISTERED:
        return

    @event.listens_for(Session, "before_flush")
    def _demo_before_flush(
        session: Session, _flush_context: Any, _instances: Any
    ) -> None:
        converted = 0
        for obj in list(session.deleted):
            if _convert_demo_delete_to_soft_delete(session, obj):
                converted += 1
        if converted:
            logger.info(
                "[demo_soft_delete_guard] suppressions converties en soft delete: %s",
                converted,
            )

    _LISTENER_REGISTERED = True


def institution_is_demo(institution: Institution | None) -> bool:
    """Public: vérifie si une institution est démo (pour filtrer les notifications)."""
    return _institution_is_demo(institution)


def company_is_demo(company: Company | None) -> bool:
    """Public: vérifie si une entreprise est démo (pour filtrer les notifications)."""
    return _company_is_demo(company)


def filter_companies_for_institution(
    companies: list[Company],
    institution: Institution | None,
) -> list[Company]:
    """Filtre les transporteurs selon le type d'institution (démo ↔ réel).

    - Institution démo : uniquement entreprises démo (@demo.lirie.ch, etc.).
    - Institution réelle : exclut les entreprises démo (LIRIE Demo, comptes test).
    """
    if institution_is_demo(institution):
        return [c for c in companies if company_is_demo(c)]
    return [c for c in companies if not company_is_demo(c)]
