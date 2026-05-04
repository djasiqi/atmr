"""Service pour résoudre les bons de transport valides (P3).

Objectif: détecter si un booking est couvert par un bon de transport validé
et proposer automatiquement le payeur correspondant (clinique/assurance/tiers).

Priorité dans la résolution de payeur : bon validé > séjour actif > payeur par défaut > patient.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from models import BillingParty, Booking, TransportVoucher
from models.enums import BillingSource, TransportVoucherStatus, TransportVoucherType

logger = logging.getLogger(__name__)


def find_valid_voucher_for_booking(
    *,
    booking: Booking,
) -> TransportVoucher | None:
    """Trouve un bon de transport validé qui couvre une date de booking (P3).

    Règle: un booking est "couvert par un bon" si :
    - Le bon est lié directement au booking (`booking_id`), OU
    - Le bon est lié au client et sa période de validité (`valid_from` <= booking_date <= `valid_to`)
      contient la date du booking.

    Uniquement les bons avec `status=validated` sont considérés.

    Args:
        booking: Booking à vérifier

    Returns:
        TransportVoucher validé si trouvé, None sinon
    """
    if not booking:
        return None

    client_id = getattr(booking, "client_id", None)
    if not client_id:
        return None

    # Date de référence: pickup_at (priorité) ou scheduled_time (fallback)
    booking_date: datetime | None = None
    if hasattr(booking, "pickup_at") and booking.pickup_at:
        booking_date = booking.pickup_at
        if booking_date and booking_date.tzinfo is None:
            booking_date = booking_date.replace(tzinfo=UTC)
    elif booking.scheduled_time:
        booking_date = booking.scheduled_time
        if booking_date and booking_date.tzinfo is None:
            booking_date = booking_date.replace(tzinfo=UTC)

    if not booking_date:
        return None

    # 1. Chercher un bon directement lié au booking
    direct_voucher = TransportVoucher.query.filter_by(
        booking_id=booking.id,
        status=TransportVoucherStatus.VALIDATED.value,
    ).first()
    if direct_voucher:
        return direct_voucher

    # 2. Chercher un bon lié au client avec période de validité qui couvre la date
    vouchers = (
        TransportVoucher.query.filter_by(
            client_id=client_id,
            status=TransportVoucherStatus.VALIDATED.value,
        )
        .filter(
            (TransportVoucher.valid_from.is_(None))
            | (TransportVoucher.valid_from <= booking_date)
        )
        .filter(
            (TransportVoucher.valid_to.is_(None))
            | (TransportVoucher.valid_to >= booking_date)
        )
        .order_by(TransportVoucher.created_at.desc())
        .limit(1)
        .all()
    )

    return vouchers[0] if vouchers else None


def resolve_payer_from_voucher(
    *,
    voucher: TransportVoucher,
    company_id: int,
) -> dict[str, Any] | None:
    """Résout le payeur (billing_party_id) basé sur un bon de transport validé (P3).

    Si un bon validé existe, le payeur indiqué dans le bon devient la référence :
    - Si `billing_party_id` est défini → utiliser ce payeur
    - Si `type=clinic` → résoudre via mapping clinique (comme pour séjour)
    - Sinon → utiliser le billing_party_id du bon

    Args:
        voucher: TransportVoucher validé trouvé
        company_id: ID de l'entreprise (transporteur)

    Returns:
        Dict avec `billing_party_id`, `billed_to_type`, `billed_to_company_id`, `billing_source`, `billing_source_ref`
        si résolu, None sinon
    """
    if not voucher:
        return None

    # Si le bon a déjà un billing_party_id défini, l'utiliser
    if voucher.billing_party_id:
        billing_party = BillingParty.query.filter_by(
            id=voucher.billing_party_id, company_id=company_id
        ).first()
        if billing_party:
            result: dict[str, Any] = {
                "billing_party_id": billing_party.id,
                "billed_to_type": billing_party.type.value
                if hasattr(billing_party.type, "value")
                else str(billing_party.type),
                "billed_to_company_id": None,
                "billing_source": BillingSource.TRANSPORT_VOUCHER.value,
                "billing_source_ref": f"voucher#{voucher.id}",
            }
            # Si c'est une clinique, ajouter billed_to_company_id
            if (
                billing_party.type.value == "clinic"
                if hasattr(billing_party.type, "value")
                else str(billing_party.type) == "clinic"
            ):
                # Essayer de trouver la company_id depuis le billing_party
                # (nécessite un mapping ou un champ dans BillingParty)
                # Pour l'instant, on laisse None si pas disponible
                pass
            return result

    # Si type=clinic mais pas de billing_party_id, résoudre via mapping
    voucher_type_value = (
        voucher.type.value if hasattr(voucher.type, "value") else str(voucher.type)
    )
    if voucher_type_value == TransportVoucherType.CLINIC.value:
        # Chercher si le bon référence une clinique via external_ref ou autre
        # Pour l'instant, on retourne None si pas de billing_party_id
        logger.warning(
            (
                "[TransportVoucherResolver] Bon trouvé (voucher_id=%s, type=clinic) "
                "mais pas de billing_party_id configuré"
            ),
            voucher.id,
        )
        return None

    # Autres types (insurance, other) nécessitent un billing_party_id
    logger.warning(
        (
            "[TransportVoucherResolver] Bon trouvé (voucher_id=%s, type=%s) "
            "mais pas de billing_party_id configuré"
        ),
        voucher.id,
        voucher.type.value if hasattr(voucher.type, "value") else str(voucher.type),
    )
    return None


def detect_billing_conflict_with_voucher(
    *,
    booking: Booking,
    voucher: TransportVoucher,
    company_id: int,
) -> tuple[bool, str | None]:
    """Détecte un conflit entre le payeur actuel du booking et le bon (P3).

    Conflit = bon validé indique payeur A mais booking explicitement facturé à payeur B
    (sauf si override documenté).

    Args:
        booking: Booking à vérifier
        voucher: TransportVoucher validé trouvé
        company_id: ID de l'entreprise (transporteur)

    Returns:
        Tuple (has_conflict: bool, reason: str | None)
    """
    if not voucher or not booking:
        return False, None

    # Résoudre le payeur attendu depuis le bon
    expected_payer = resolve_payer_from_voucher(voucher=voucher, company_id=company_id)
    if not expected_payer:
        return False, None  # Pas de conflit si le bon ne résout pas de payeur

    expected_billing_party_id = expected_payer.get("billing_party_id")
    current_billing_party_id = getattr(booking, "billing_party_id", None)
    current_billed_to_type = (
        getattr(booking, "billed_to_type", None) or "patient"
    ).lower()

    # Pas de conflit si le payeur correspond
    if current_billing_party_id == expected_billing_party_id:
        return False, None

    # Conflit si le booking a un payeur différent
    reason = (
        f"Conflit bon/payeur: bon validé (voucher_id={voucher.id}) "
        f"indique billing_party_id={expected_billing_party_id} "
        f"mais booking a billing_party_id={current_billing_party_id} "
        f"ou billed_to_type={current_billed_to_type}."
    )
    return True, reason
