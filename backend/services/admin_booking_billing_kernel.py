"""Noyau partagé : classification source, montant observable, qualification billing.

Utilisé par la liste admin / export CSV et par le moteur facturation plateforme.
Les versions permettent de tracer les changements de règles métier.
"""

from __future__ import annotations

import logging
from typing import Any

from models import Booking
from models.enums import UserRole
from services.admin_booking_investigation import (
    compute_needs_investigation_booking,
    evaluate_incomplete,
)

logger = logging.getLogger(__name__)

CLASSIFICATION_VERSION = 1
QUALIFICATION_VERSION = 1

# Seuils score de fiabilité (0–100, ajusté) — documentés pour la transparence produit
RELIABILITY_GOOD_MIN = 70.0
RELIABILITY_MEDIUM_MIN = 40.0


def is_synthetic_demo_email(email: str | None) -> bool:
    if not email:
        return False
    lowered = email.strip().lower()
    return (
        lowered.endswith("@demo.local")
        or lowered.endswith("@demo.lirie.ch")
        or lowered.startswith("demo-")
        or lowered.endswith("@internal.atmr.local")
    )


def is_synthetic_demo_booking(booking: Booking) -> bool:
    """Hors périmètre pilotage : comptes / emails de démo (aligné esprit admin)."""
    try:
        from ext import db
        from models import User

        u = None
        if getattr(booking, "user_id", None):
            u = db.session.get(User, booking.user_id)
        if u and is_synthetic_demo_email(getattr(u, "email", None)):
            return True
        cli = booking.client
        if (
            cli
            and getattr(cli, "user", None)
            and is_synthetic_demo_email(getattr(cli.user, "email", None))
        ):
            return True
    except Exception:
        logger.debug("is_synthetic_demo_booking: erreur soft", exc_info=True)
    return False


def classify_booking_source(booking: Booking) -> str:
    """Taxonomie fine : institution_request | company_manual | client_direct | admin_created | unknown_source."""
    try:
        tl = booking._get_institution_timeline()
        if tl and tl.get("created_by_name"):
            return "institution_request"
    except Exception:
        pass
    cli = booking.client
    if cli and cli.user:
        role = getattr(cli.user, "role", None)
        rv = getattr(role, "value", role)
        if rv == UserRole.ADMIN.value:
            return "admin_created"
        if rv == UserRole.COMPANY.value:
            return "company_manual"
        if rv == UserRole.CLIENT.value:
            return "client_direct"
        if rv == UserRole.INSTITUTION.value:
            return "institution_request"
        if rv == UserRole.DRIVER.value:
            return "company_manual"
    return "unknown_source"


def observed_transport_amount(booking: Booking) -> float | None:
    """Montant transport observable pour le pilotage (une seule règle pour toute l'admin).

    Priorité :
    1. ``amount`` (Float métier) si strictement > 0
    2. Sinon ``price_amount`` (Numeric) si présent et > 0
    3. Sinon None — la réservation compte en volume mais pas en somme montant
    """
    try:
        a = float(booking.amount) if booking.amount is not None else 0.0
    except (TypeError, ValueError):
        a = 0.0
    if a > 0:
        return a
    if getattr(booking, "price_amount", None) is not None:
        try:
            pa = float(booking.price_amount)
        except (TypeError, ValueError):
            pa = 0.0
        if pa > 0:
            return pa
    return None


def booking_is_executed(booking: Booking) -> bool:
    """Aligné sur la lecture « course exécutée » côté pilotage : terminée (trajet effectué)."""
    st = booking.status
    key = st.value if hasattr(st, "value") else str(st).upper()
    return key in ("COMPLETED", "RETURN_COMPLETED")


def qualify_booking(
    booking: Booking,
    *,
    has_transfer: bool,
    has_pending_transfer: bool,
) -> dict[str, Any]:
    """Retourne state (eligible|ambiguous|needs_review|excluded), reasons[], families[]."""
    if is_synthetic_demo_booking(booking):
        return {
            "state": "excluded",
            "reasons": ["out_of_scope"],
            "families": ["perimeter"],
        }

    src = classify_booking_source(booking)
    amt = observed_transport_amount(booking)

    if evaluate_incomplete(booking):
        return {
            "state": "needs_review",
            "reasons": ["incomplete_data"],
            "families": ["investigation"],
        }

    if amt is None:
        return {
            "state": "needs_review",
            "reasons": ["missing_amount"],
            "families": ["montant"],
        }

    if src == "unknown_source":
        return {
            "state": "needs_review",
            "reasons": ["unknown_source"],
            "families": ["source"],
        }

    if has_pending_transfer:
        return {
            "state": "needs_review",
            "reasons": ["investigation_flag"],
            "families": ["transfert"],
        }

    if compute_needs_investigation_booking(
        booking, has_pending_transfer=has_pending_transfer
    ):
        return {
            "state": "needs_review",
            "reasons": ["investigation_flag"],
            "families": ["investigation"],
        }

    # Transfert accepté + non institution : ambiguïté opérationnelle (pas exclusion auto)
    if has_transfer and src != "institution_request":
        return {
            "state": "ambiguous",
            "reasons": ["transfer_complex"],
            "families": ["transfert"],
        }

    return {"state": "eligible", "reasons": [], "families": []}


def reliability_bucket_and_percent(
    *,
    eligible: int,
    needs_review: int,
    ambiguous: int,
    excluded: int,
    total: int,
) -> tuple[str, float | None]:
    """Indicateur Bon / Moyen / Faible + pourcentage ajusté (simple, documenté)."""
    if total <= 0:
        return "unknown", None
    _ = excluded
    # Pénalité : needs_review plus lourd qu'ambiguous
    adjusted = (
        100.0 * eligible / total - 18.0 * needs_review / total - 6.0 * ambiguous / total
    )
    adjusted = max(0.0, min(100.0, adjusted))
    if adjusted >= RELIABILITY_GOOD_MIN:
        label = "good"
    elif adjusted >= RELIABILITY_MEDIUM_MIN:
        label = "medium"
    else:
        label = "low"
    return label, round(adjusted, 1)


def build_pilotage_payload_for_booking(
    booking: Booking,
    *,
    has_transfer: bool,
    has_pending_transfer: bool,
) -> dict[str, Any]:
    """Payload JSON stable pour liste admin et exports."""
    src = classify_booking_source(booking)
    amt = observed_transport_amount(booking)
    q = qualify_booking(
        booking,
        has_transfer=has_transfer,
        has_pending_transfer=has_pending_transfer,
    )
    ui_group = "manual_direct"
    if src == "institution_request":
        ui_group = "institution"
    elif src == "unknown_source":
        ui_group = "unknown"

    return {
        "classification_version": CLASSIFICATION_VERSION,
        "qualification_version": QUALIFICATION_VERSION,
        "source_code": src,
        "ui_source_group": ui_group,
        "observed_transport_amount": amt,
        "qualification": {
            "state": q["state"],
            "reasons": q["reasons"],
            "families": q["families"],
        },
    }
