"""Service pour résoudre les séjours actifs d'un client (P2.2).

Objectif: détecter si un booking est "lié à un séjour" et proposer automatiquement
le payeur clinique basé sur les séjours actifs.

Gère aussi les transitions temporelles :
- Curatelle qui commence/finit → payeur par défaut change
- Hospitalisation qui se termine → retour au payeur par défaut (curatelle ou patient)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from models import BillingParty, Booking, ClientBillingParty, ClientStay, Company
from services.billing.billing_party_linker import resolve_billing_party_for_clinic
from services.billing.transport_voucher_resolver import (
    find_valid_voucher_for_booking,
    resolve_payer_from_voucher,
)

logger = logging.getLogger(__name__)


def find_active_stay_for_booking(
    *,
    booking: Booking,
) -> ClientStay | None:
    """Trouve le séjour actif d'un client pour une date de booking donnée (P2.2).

    Règle: une course (booking) est "liée à un séjour" si sa `pickup_at` (ou `scheduled_time`)
    tombe dans l'intervalle `start_date <= booking_date <= end_date` (ou `end_date IS NULL`).

    Args:
        booking: Booking à vérifier

    Returns:
        ClientStay actif si trouvé, None sinon
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

    # Chercher un séjour actif qui contient cette date
    stays = (
        ClientStay.query.filter_by(
            client_id=client_id,
            status="active",
        )
        .filter(ClientStay.start_date <= booking_date)
        .filter(
            (ClientStay.end_date.is_(None)) | (ClientStay.end_date >= booking_date)
        )
        .order_by(ClientStay.start_date.desc())
        .limit(1)
        .all()
    )

    return stays[0] if stays else None


def resolve_payer_from_stay(
    *,
    stay: ClientStay,
    company_id: int,
) -> dict[str, Any] | None:
    """Résout le payeur (billed_to_company_id, billing_party_id) basé sur un séjour (P2.2).

    Si un séjour existe, la clinique du séjour devient la référence pour :
    - proposer `billed_to_type="clinic"`
    - proposer `billed_to_company_id = company_id` (clinique)
    - proposer `billing_party_id` (destinataire) selon configuration (ClinicBillingPartyMapping)

    Args:
        stay: ClientStay actif trouvé
        company_id: ID de l'entreprise (transporteur)

    Returns:
        Dict avec `billed_to_company_id`, `billing_party_id`, `billed_to_type` si résolu,
        None si pas de mapping configuré
    """
    if not stay or not stay.company_id:
        return None

    clinic_company_id = stay.company_id

    # Résoudre le BillingParty via mapping
    billing_party = resolve_billing_party_for_clinic(
        company_id=company_id,
        clinic_company_id=clinic_company_id,
    )

    result: dict[str, Any] = {
        "billed_to_type": "clinic",
        "billed_to_company_id": clinic_company_id,
        "billing_party_id": billing_party.id if billing_party else None,
    }

    if not billing_party:
        logger.warning(
            (
                "[ClientStayResolver] Séjour trouvé (stay_id=%s, clinic_company_id=%s) "
                "mais pas de mapping billing_party configuré pour company_id=%s"
            ),
            stay.id,
            clinic_company_id,
            company_id,
        )

    return result


def detect_billing_conflict_with_stay(
    *,
    booking: Booking,
    stay: ClientStay,
    company_id: int,
) -> tuple[bool, str | None]:
    """Détecte un conflit entre le payeur actuel du booking et le séjour (P2.2).

    Conflit = séjour clinique A mais booking explicitement facturé patient/tiers
    (sauf si override documenté).

    Args:
        booking: Booking à vérifier
        stay: ClientStay actif trouvé
        company_id: ID de l'entreprise (transporteur)

    Returns:
        Tuple (has_conflict: bool, reason: str | None)
    """
    if not stay or not booking:
        return False, None

    current_billed_to_type = (getattr(booking, "billed_to_type", None) or "patient").lower()
    current_billed_to_company_id = getattr(booking, "billed_to_company_id", None)
    current_billing_party_id = getattr(booking, "billing_party_id", None)

    # Si le booking est déjà facturé à la clinique du séjour → pas de conflit
    if current_billed_to_company_id == stay.company_id:
        return False, None

    # Si le booking a un billing_party_id qui correspond au mapping de la clinique → pas de conflit
    if current_billing_party_id:
        billing_party = resolve_billing_party_for_clinic(
            company_id=company_id,
            clinic_company_id=stay.company_id,
        )
        if billing_party and billing_party.id == current_billing_party_id:
            return False, None

    # Conflit si :
    # - séjour actif existe (clinique A)
    # - mais booking est facturé patient OU à une autre clinique/tiers
    if current_billed_to_type == "patient":
        reason = (
            f"Conflit séjour/payeur: séjour actif à la clinique {stay.company_id} "
            + f"(stay_id={stay.id}) mais booking facturé au patient."
        )
        return True, reason

    if (
        current_billed_to_type in ("clinic", "insurance")
        and current_billed_to_company_id
        and current_billed_to_company_id != stay.company_id
    ):
        reason = (
            f"Conflit séjour/payeur: séjour actif à la clinique {stay.company_id} "
            + f"(stay_id={stay.id}) mais booking facturé à la clinique/tiers {current_billed_to_company_id}."
        )
        return True, reason

    return False, None


def resolve_default_billing_party_for_client(
    *,
    client_id: int,
    company_id: int,
) -> BillingParty | None:
    """Résout le payeur par défaut d'un client (curatelle, parents, etc.).

    Utilise la table `ClientBillingParty` avec `is_default=True` pour trouver
    le payeur par défaut configuré pour ce client.

    Args:
        client_id: ID du client
        company_id: ID de l'entreprise (transporteur)

    Returns:
        BillingParty par défaut si trouvé, None sinon (facturation directe au patient)
    """
    link = (
        ClientBillingParty.query.filter_by(client_id=client_id, is_default=True)
        .join(BillingParty)
        .filter(BillingParty.company_id == company_id, BillingParty.is_active.is_(True))
        .first()
    )
    if link and link.billing_party:
        return link.billing_party
    return None


def resolve_billing_party_for_booking(
    *,
    booking: Booking,
    company_id: int,
) -> dict[str, Any] | None:
    """Résout le payeur (billing_party_id) pour un booking selon la date (P2.2 + P3 + transitions).

    Logique de résolution (par ordre de priorité) :
    1. **Bon de transport validé** (P3) → facturation selon bon (priorité maximale)
    2. **Séjour actif** (P2) → facturation clinique
    3. **Payeur par défaut** (curatelle, parents, etc.) via `ClientBillingParty.is_default`
    4. **Fallback** : patient direct (retourne None, le PDF utilisera client_id)

    Cette fonction gère automatiquement les transitions :
    - Bon validé → facture selon bon
    - Client hospitalisé → facture à la clinique
    - Client sort de l'hôpital → retour au payeur par défaut (curatelle ou patient)
    - Curatelle qui commence → facture au curateur (payeur par défaut)
    - Curatelle qui se termine → facture au patient

    Args:
        booking: Booking à analyser
        company_id: ID de l'entreprise (transporteur)

    Returns:
        Dict avec `billing_party_id`, `billed_to_type`, `billed_to_company_id`, `billing_source`, `billing_source_ref`
        si résolu, None si facturation directe au patient
    """
    client_id = getattr(booking, "client_id", None)
    if not booking or not client_id:
        return None

    # ✅ P3: 1. Vérifier si un bon de transport validé existe pour cette date (priorité maximale)
    voucher = find_valid_voucher_for_booking(booking=booking)
    if voucher:
        payer_from_voucher = resolve_payer_from_voucher(
            voucher=voucher,
            company_id=company_id,
        )
        if payer_from_voucher and payer_from_voucher.get("billing_party_id"):
            logger.info(
                (
                    "[ClientStayResolver] Booking %s: payeur résolu via bon de transport validé "
                    "(voucher_id=%s, billing_party_id=%s, source=%s)"
                ),
                booking.id,
                voucher.id,
                payer_from_voucher["billing_party_id"],
                payer_from_voucher.get("billing_source", "transport_voucher"),
            )
            return payer_from_voucher

    # 2. Pas de bon validé → vérifier si un séjour actif existe pour cette date
    stay = find_active_stay_for_booking(booking=booking)
    if stay:
        payer_from_stay = resolve_payer_from_stay(
            stay=stay,
            company_id=company_id,
        )
        if payer_from_stay and payer_from_stay.get("billing_party_id"):
            # Ajouter billing_source pour traçabilité
            payer_from_stay["billing_source"] = "client_stay"
            payer_from_stay["billing_source_ref"] = f"stay#{stay.id}"
            logger.info(
                (
                    "[ClientStayResolver] Booking %s: payeur résolu via séjour actif "
                    "(stay_id=%s, billing_party_id=%s)"
                ),
                booking.id,
                stay.id,
                payer_from_stay["billing_party_id"],
            )
            return payer_from_stay

    # 3. Pas de séjour actif → utiliser le payeur par défaut (curatelle, etc.)
    default_bp = resolve_default_billing_party_for_client(
        client_id=int(client_id),
        company_id=company_id,
    )
    if default_bp:
        logger.info(
            (
                "[ClientStayResolver] Booking %s: payeur résolu via payeur par défaut "
                "(billing_party_id=%s, type=%s)"
            ),
            booking.id,
            default_bp.id,
            default_bp.type.value if hasattr(default_bp.type, "value") else str(default_bp.type),
        )
        return {
            "billing_party_id": default_bp.id,
            "billed_to_type": "patient",  # Le payeur est tiers mais le bénéficiaire reste le patient
            "billed_to_company_id": None,
            "billing_source": "default_client",
            "billing_source_ref": f"billing_party#{default_bp.id}",
        }

    # 4. Aucun payeur spécifique → facturation directe au patient
    logger.debug(
        "[ClientStayResolver] Booking %s: aucun payeur spécifique → facturation directe au patient",
        booking.id,
    )
    return None


def find_active_stay_for_client(
    *,
    client_id: int,
    reference_date: datetime | None = None,
) -> ClientStay | None:
    """Trouve le séjour actif d'un client pour une date donnée (ou maintenant).

    Args:
        client_id: ID du client
        reference_date: Date de référence (par défaut: maintenant UTC)

    Returns:
        ClientStay actif si trouvé, None sinon
    """
    if not client_id:
        return None

    if reference_date is None:
        reference_date = datetime.now(UTC)
    elif reference_date.tzinfo is None:
        reference_date = reference_date.replace(tzinfo=UTC)

    # Chercher un séjour actif qui contient cette date
    stays = (
        ClientStay.query.filter_by(
            client_id=client_id,
            status="active",
        )
        .filter(ClientStay.start_date <= reference_date)
        .filter(
            (ClientStay.end_date.is_(None)) | (ClientStay.end_date >= reference_date)
        )
        .order_by(ClientStay.start_date.desc())
        .limit(1)
        .all()
    )

    return stays[0] if stays else None


def get_clinic_rate_for_booking(
    *,
    booking: Booking,
    clinic_company_id: int,
) -> Decimal | None:
    """Récupère le tarif clinique pour un booking.

    ⚠️ LIMITATION ACTUELLE :
    Le tarif clinique provient actuellement de Company.preferential_rate de la clinique.
    Ce champ peut être conceptuellement ambigu :
    - client.preferential_rate = tarif patient (souvent 40 CHF)
    - Company.preferential_rate = tarif clinique (devrait être 40 CHF dans votre cas)
    
    Si patient=45 et clinique=40, alors Company.preferential_rate doit valoir 40.
    
    🔮 AMÉLIORATION FUTURE :
    Pour une séparation conceptuelle claire, envisager d'ajouter :
    - clinic_rate dans ClinicBillingPartyMapping (tarif spécifique au contrat transporteur-clinique)
    - ou clinic_rate dans CompanyBillingProfile (tarif par clinique)
    
    Args:
        booking: Booking concerné (pour logging)
        clinic_company_id: ID de la clinique (Company) payeur

    Returns:
        Decimal du tarif clinique (Company.preferential_rate) ou None si introuvable
    """
    if not clinic_company_id:
        logger.warning(
            "[get_clinic_rate_for_booking] clinic_company_id manquant pour booking %s",
            booking.id if booking else None,
        )
        return None

    clinic = Company.query.filter_by(id=clinic_company_id).first()
    if not clinic:
        logger.warning(
            "[get_clinic_rate_for_booking] Clinique introuvable: clinic_company_id=%s (booking %s)",
            clinic_company_id,
            booking.id if booking else None,
        )
        return None

    if clinic.preferential_rate is None:
        logger.warning(
            "[get_clinic_rate_for_booking] Tarif préférentiel non configuré pour clinique %s (booking %s). "
            "Source actuelle: Company.preferential_rate. "
            "Vérifier que ce champ contient bien le tarif clinique (ex: 40 CHF) et non le tarif patient.",
            clinic_company_id,
            booking.id if booking else None,
        )
        return None

    # ✅ Log pour traçabilité (DEBUG temporaire)
    logger.debug(
        "[get_clinic_rate_for_booking] Tarif clinique trouvé: clinic_company_id=%s, "
        "clinic_name=%s, preferential_rate=%.2f CHF, source=Company.preferential_rate (booking %s)",
        clinic_company_id,
        clinic.name,
        clinic.preferential_rate,
        booking.id if booking else None,
    )

    return clinic.preferential_rate


def get_clinic_address_for_stay(stay: ClientStay) -> dict[str, Any] | None:
    """Récupère l'adresse complète de la clinique pour un séjour.

    Args:
        stay: ClientStay avec company_id pointant vers la clinique

    Returns:
        Dict avec 'address' (string), 'lat', 'lon', 'preferential_rate', ou None si clinique introuvable
    """
    if not stay or not stay.company_id:
        return None

    clinic = Company.query.filter_by(id=stay.company_id).first()
    if not clinic:
        return None

    # Construire l'adresse complète
    address_parts = []
    if clinic.address:
        address_parts.append(clinic.address)
    elif clinic.domicile_address_line1:
        address_parts.append(clinic.domicile_address_line1)
        if clinic.domicile_address_line2:
            address_parts.append(clinic.domicile_address_line2)
    if clinic.domicile_zip:
        address_parts.append(clinic.domicile_zip)
    if clinic.domicile_city:
        address_parts.append(clinic.domicile_city)

    address = ", ".join(address_parts) if address_parts else None

    return {
        "address": address,
        "lat": float(clinic.latitude) if clinic.latitude else None,
        "lon": float(clinic.longitude) if clinic.longitude else None,
        "preferential_rate": (
            float(clinic.preferential_rate) if clinic.preferential_rate is not None else None
        ),
        "clinic_id": clinic.id,
        "clinic_name": clinic.name,
    }
