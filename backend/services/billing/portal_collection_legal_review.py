"""Revue juridique et éligibilité de transmission humaine (6G-A).

Aucune transmission externe. L'approbation référence le hash exact du dossier.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from ext import db
from models.portal_collection_legal_review import (
    LEGAL_REVIEW_PROTOCOL_VERSION,
    REVIEW_APPROVED,
    REVIEW_PENDING,
    REVIEW_REJECTED,
    PortalCollectionLegalReview,
)
from models.portal_receivable import (
    DISPUTE_OPEN,
    RECEIVABLE_CANCELLED,
    RECEIVABLE_DISPUTED,
    RECEIVABLE_PAID,
    PortalReceivable,
    PortalReceivableDispute,
)
from models.portal_receivable_collection_action import (
    PortalReceivableCollectionTransmission,
)
from services.billing.portal_receivable import PortalReceivableError
from services.billing.portal_receivable_pursuit import (
    PURSUIT_READY,
    resolve_portal_pursuit_readiness,
)

TRANSMISSION_AUTHORIZED = "transmission_authorized"
TRANSMISSION_NOT_AUTHORIZED = "transmission_not_authorized"

REASON_LEGAL_REVIEW_MISSING = "legal_review_missing"
REASON_LEGAL_REVIEW_STALE = "legal_review_stale"
REASON_LEGAL_REVIEW_REJECTED = "legal_review_rejected"
REASON_BALANCE_CHANGED = "balance_changed_since_draft"
REASON_CREDITOR_NOT_CONFIRMED = "creditor_confirmation_required"
REASON_PURSUIT_NOT_READY = "pursuit_not_ready"


@dataclass(frozen=True, slots=True)
class TransmissionEligibility:
    state: str
    reasons: tuple[str, ...]
    details: dict[str, Any]

    @property
    def is_authorized(self) -> bool:
        return self.state == TRANSMISSION_AUTHORIZED


def _money(value: object) -> Decimal:
    return Decimal(str(value or "0"))


def create_legal_review_pending(
    *,
    transmission: PortalReceivableCollectionTransmission,
    requested_by_user_id: int | None = None,
) -> PortalCollectionLegalReview:
    """Ouvre une revue pending liée au hash actuel du draft."""
    del requested_by_user_id  # réservé audit futur (qui a ouvert la revue)
    existing = (
        PortalCollectionLegalReview.query.filter_by(
            transmission_id=int(transmission.id),
            dossier_hash=str(transmission.export_hash),
            review_version=LEGAL_REVIEW_PROTOCOL_VERSION,
        )
        .one_or_none()
    )
    if existing is not None:
        return existing
    row = PortalCollectionLegalReview(
        receivable_id=int(transmission.receivable_id),
        transmission_id=int(transmission.id),
        creditor_company_id=int(transmission.creditor_company_id),
        dossier_hash=str(transmission.export_hash),
        review_status=REVIEW_PENDING,
        review_version=LEGAL_REVIEW_PROTOCOL_VERSION,
        notes=None,
        reviewed_by_user_id=None,
        reviewed_at=None,
    )
    db.session.add(row)
    db.session.flush()
    return row


def decide_legal_review(
    *,
    review: PortalCollectionLegalReview,
    transmission: PortalReceivableCollectionTransmission,
    reviewed_by_user_id: int,
    approve: bool,
    notes: str | None = None,
) -> PortalCollectionLegalReview:
    """Approuve ou rejette. Refuse si le hash du draft a changé."""
    if str(review.dossier_hash) != str(transmission.export_hash):
        raise PortalReceivableError(
            "Le hash du dossier a changé — nouvelle revue requise.",
            code=REASON_LEGAL_REVIEW_STALE,
        )
    if review.review_status in (REVIEW_APPROVED, REVIEW_REJECTED):
        raise PortalReceivableError(
            "Cette revue est déjà décidée.",
            code="legal_review_already_decided",
        )
    review.review_status = REVIEW_APPROVED if approve else REVIEW_REJECTED
    review.reviewed_by_user_id = int(reviewed_by_user_id)
    review.reviewed_at = datetime.now(UTC)
    review.notes = (notes or "").strip() or None
    db.session.flush()
    return review


def latest_matching_approval(
    transmission: PortalReceivableCollectionTransmission,
) -> PortalCollectionLegalReview | None:
    return (
        PortalCollectionLegalReview.query.filter_by(
            transmission_id=int(transmission.id),
            dossier_hash=str(transmission.export_hash),
            review_status=REVIEW_APPROVED,
            review_version=LEGAL_REVIEW_PROTOCOL_VERSION,
        )
        .order_by(PortalCollectionLegalReview.id.desc())
        .first()
    )


def _add_reason(reasons: list[str], code: str) -> None:
    if code not in reasons:
        reasons.append(code)


def resolve_transmission_eligibility(
    transmission_id: int,
) -> TransmissionEligibility:
    """Éligibilité à une transmission humaine future (6G-B) — pas d'envoi."""
    transmission = db.session.get(
        PortalReceivableCollectionTransmission, int(transmission_id)
    )
    if transmission is None:
        return TransmissionEligibility(
            state=TRANSMISSION_NOT_AUTHORIZED,
            reasons=("transmission_not_found",),
            details={},
        )
    receivable = db.session.get(PortalReceivable, int(transmission.receivable_id))
    if receivable is None:
        return TransmissionEligibility(
            state=TRANSMISSION_NOT_AUTHORIZED,
            reasons=("receivable_not_found",),
            details={},
        )

    reasons: list[str] = []
    readiness = resolve_portal_pursuit_readiness(int(receivable.id))
    if readiness.state != PURSUIT_READY:
        _add_reason(reasons, REASON_PURSUIT_NOT_READY)
        for r in readiness.reasons:
            _add_reason(reasons, r)

    if not bool(transmission.creditor_confirmed):
        _add_reason(reasons, REASON_CREDITOR_NOT_CONFIRMED)

    if receivable.status == RECEIVABLE_CANCELLED or receivable.cancelled_at is not None:
        _add_reason(reasons, "cancelled")
    open_d = (
        PortalReceivableDispute.query.filter_by(
            receivable_id=int(receivable.id), status=DISPUTE_OPEN
        ).first()
    )
    if (
        receivable.status == RECEIVABLE_DISPUTED
        or receivable.disputed_at is not None
        or open_d is not None
    ):
        _add_reason(reasons, "disputed")
    if receivable.status == RECEIVABLE_PAID or _money(receivable.balance_due) <= 0:
        _add_reason(reasons, "paid")

    # Paiement partiel après le draft : le montant snapshot n'est plus courant.
    if _money(receivable.balance_due) != _money(transmission.balance_snapshot):
        _add_reason(reasons, REASON_BALANCE_CHANGED)

    approval = latest_matching_approval(transmission)
    if approval is None:
        rejected = (
            PortalCollectionLegalReview.query.filter_by(
                transmission_id=int(transmission.id),
                dossier_hash=str(transmission.export_hash),
                review_status=REVIEW_REJECTED,
            )
            .first()
        )
        if rejected is not None:
            _add_reason(reasons, REASON_LEGAL_REVIEW_REJECTED)
        else:
            _add_reason(reasons, REASON_LEGAL_REVIEW_MISSING)

    details = {
        "transmission_id": transmission.id,
        "dossier_hash": transmission.export_hash,
        "balance_snapshot": float(transmission.balance_snapshot),
        "balance_current": float(receivable.balance_due),
        "legal_review_id": approval.id if approval else None,
        "legal_review_status": approval.review_status if approval else None,
        "pursuit_readiness": readiness.state,
        "disclaimer": (
            "Éligibilité théorique uniquement. Aucune transmission externe "
            "n'est effectuée (EasyGov / office / recouvrement)."
        ),
    }
    if reasons:
        return TransmissionEligibility(
            state=TRANSMISSION_NOT_AUTHORIZED,
            reasons=tuple(reasons),
            details=details,
        )
    return TransmissionEligibility(
        state=TRANSMISSION_AUTHORIZED, reasons=(), details=details
    )


def serialize_legal_review(row: PortalCollectionLegalReview) -> dict[str, Any]:
    return {
        "id": row.id,
        "receivable_id": row.receivable_id,
        "transmission_id": row.transmission_id,
        "creditor_company_id": row.creditor_company_id,
        "dossier_hash": row.dossier_hash,
        "review_status": row.review_status,
        "review_version": row.review_version,
        "notes": row.notes,
        "reviewed_by_user_id": row.reviewed_by_user_id,
        "reviewed_at": row.reviewed_at.isoformat() if row.reviewed_at else None,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def serialize_transmission_eligibility(
    result: TransmissionEligibility,
) -> dict[str, Any]:
    return {
        "state": result.state,
        "reasons": list(result.reasons),
        "details": result.details,
    }
