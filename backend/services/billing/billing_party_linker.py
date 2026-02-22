"""Helpers pour lier une facture à un destinataire (BillingParty).

Objectif:
- permettre une migration progressive depuis `Invoice.bill_to_client_id` (legacy)
  vers `Invoice.billing_party_id` (source de vérité).
"""

from __future__ import annotations

import logging

from ext import db
from models import (
    BillingParty,
    BillingPartyType,
    Client,
    ClinicBillingPartyMapping,
    Company,
)

logger = logging.getLogger(__name__)


def _guess_billing_party_type_from_client(client: Client) -> BillingPartyType:
    """Heuristique safe pour typer un payeur issu d'un Client legacy."""
    name = (getattr(client, "institution_name", None) or "").strip().lower()
    if "opad" in name:
        return BillingPartyType.OPAD
    if "ems" in name:
        return BillingPartyType.EMS
    if "hôpital" in name or "hopital" in name or "hospital" in name:
        return BillingPartyType.HOSPITAL
    if bool(getattr(client, "is_institution", False)):
        return BillingPartyType.CLINIC
    return BillingPartyType.OTHER


def _best_effort_address_from_client(client: Client) -> str:
    # Priorité: billing_address (déjà utilisé côté PDF) → domicile → user.address → placeholder
    addr = ""
    try:
        addr = getattr(client, "billing_address_secure", None) or ""
    except Exception:
        addr = getattr(client, "billing_address", "") or ""
    if (addr or "").strip():
        return addr.strip()

    domicile = getattr(client, "domicile_address", None) or ""
    if str(domicile).strip():
        postal = (getattr(client, "domicile_zip", "") or "").strip()
        city = (getattr(client, "domicile_city", "") or "").strip()
        if postal and city:
            return f"{domicile}\n{postal} {city}"
        return str(domicile).strip()

    user = getattr(client, "user", None)
    if user is not None and (getattr(user, "address", None) or "").strip():
        return str(user.address).strip()

    # Important: BillingParty non-patient exige une adresse non vide.
    return "Adresse non renseignée"


def _best_effort_contacts_from_client(client: Client) -> tuple[str | None, str | None]:
    email = (getattr(client, "contact_email", None) or "").strip() or None
    phone = ""
    try:
        phone = getattr(client, "contact_phone_secure", None) or ""
    except Exception:
        phone = getattr(client, "contact_phone", "") or ""
    phone = phone.strip()
    if not phone:
        user = getattr(client, "user", None)
        phone = (getattr(user, "phone", None) or "").strip() if user is not None else ""
    return email, (phone or None)


def get_or_create_billing_party_for_legacy_bill_to_client(
    *,
    company_id: int,
    bill_to_client_id: int,
) -> BillingParty | None:
    """Crée (si nécessaire) un BillingParty correspondant à un `bill_to_client_id` legacy.

    On utilise `external_ref` pour éviter les doublons.
    """
    client = Client.query.filter_by(id=bill_to_client_id, company_id=company_id).first()
    if not client:
        logger.warning(
            "[billing_party_linker] Client payeur introuvable: company_id=%s client_id=%s",
            company_id,
            bill_to_client_id,
        )
        return None

    external_ref = f"legacy_client:{bill_to_client_id}"
    existing = BillingParty.query.filter_by(
        company_id=company_id, external_ref=external_ref
    ).first()
    if existing:
        return existing

    display_name = (
        (getattr(client, "institution_name", None) or "").strip()
        or (
            f"{getattr(client.user, 'first_name', '') or ''} {getattr(client.user, 'last_name', '') or ''}".strip()
            if getattr(client, "user", None)
            else ""
        )
        or (getattr(getattr(client, "user", None), "username", None) or "").strip()
        or f"Payeur #{bill_to_client_id}"
    )

    billing_party = BillingParty()
    billing_party.company_id = company_id
    billing_party.type = _guess_billing_party_type_from_client(client)
    billing_party.display_name = display_name
    billing_party.billing_address = _best_effort_address_from_client(client)
    email, phone = _best_effort_contacts_from_client(client)
    billing_party.contact_email = email
    billing_party.contact_phone = phone
    billing_party.external_ref = external_ref

    db.session.add(billing_party)
    db.session.flush()
    return billing_party


def get_or_create_billing_party_for_clinic_company(
    *,
    company_id: int,
    clinic_company_id: int,
) -> BillingParty | None:
    """Crée (si nécessaire) un BillingParty correspondant à une clinique (Company).

    Note:
        On n'active pas de mapping automatique ici: la sélection du destinataire
        reste une décision de configuration. Cette fonction sert surtout à
        pré-créer un destinataire "clinique" standard (fallback).
    """
    clinic = Company.query.filter_by(id=clinic_company_id).first()
    if not clinic:
        logger.warning(
            "[billing_party_linker] Clinique introuvable: clinic_company_id=%s",
            clinic_company_id,
        )
        return None

    external_ref = f"clinic_company:{clinic_company_id}"
    existing = BillingParty.query.filter_by(
        company_id=company_id, external_ref=external_ref
    ).first()
    if existing:
        return existing

    # Adresse best-effort depuis Company
    address = (getattr(clinic, "address", None) or "").strip()
    if not address:
        # Essayer l'adresse de domiciliation structurée (si renseignée)
        line1 = (getattr(clinic, "domicile_address_line1", None) or "").strip()
        line2 = (getattr(clinic, "domicile_address_line2", None) or "").strip()
        postal = (getattr(clinic, "domicile_zip", None) or "").strip()
        city = (getattr(clinic, "domicile_city", None) or "").strip()
        parts = [p for p in [line1, line2] if p]
        if postal and city:
            parts.append(f"{postal} {city}")
        address = "\n".join(parts).strip()
    if not address:
        address = "Adresse non renseignée"

    billing_party = BillingParty()
    billing_party.company_id = company_id
    billing_party.type = BillingPartyType.CLINIC
    billing_party.display_name = (getattr(clinic, "name", None) or "").strip() or (
        f"Clinique #{clinic_company_id}"
    )
    billing_party.billing_address = address
    billing_party.contact_email = (getattr(clinic, "contact_email", None) or "").strip() or None
    billing_party.contact_phone = (getattr(clinic, "contact_phone", None) or "").strip() or None
    billing_party.external_ref = external_ref

    db.session.add(billing_party)
    db.session.flush()
    return billing_party


def resolve_billing_party_for_clinic(
    *,
    company_id: int,
    clinic_company_id: int,
) -> BillingParty | None:
    """Résout le destinataire BillingParty configuré pour une clinique.

    1. Cherche un mapping ClinicBillingPartyMapping actif (flux classique).
    2. Fallback: auto-crée un BillingParty pour les cliniques institution
       (bookings créés via le flux institution n'ont pas de mapping préconfiguré).
    """
    mapping = ClinicBillingPartyMapping.query.filter_by(
        company_id=company_id, clinic_company_id=clinic_company_id, is_active=True
    ).first()
    if mapping:
        return BillingParty.query.filter_by(id=mapping.billing_party_id).first()

    # Fallback pour les bookings institution :
    # clinic_company_id == company_id signifie que le client institution
    # a company_id = l'entreprise de transport (pas une vraie clinique Company).
    # Dans ce cas, chercher le BillingParty institution existant (external_ref=institution:X).
    if clinic_company_id == company_id:
        logger.info(
            "[billing_party_linker] clinic_company_id=%s == company_id (institution case). "
            "Looking for existing institution BillingParty.",
            clinic_company_id,
        )
        # Chercher un BP institution pour cette company
        institution_bp = BillingParty.query.filter(
            BillingParty.company_id == company_id,
            BillingParty.external_ref.like("institution:%"),
            BillingParty.type == BillingPartyType.CLINIC,
        ).first()
        if institution_bp:
            logger.info(
                "[billing_party_linker] Found institution BP id=%s: %s",
                institution_bp.id,
                institution_bp.display_name,
            )
            return institution_bp

    # Fallback classique: auto-créer le billing party depuis la Company clinique
    logger.info(
        "[billing_party_linker] No mapping found for company_id=%s, clinic_company_id=%s. "
        "Trying auto-create fallback.",
        company_id,
        clinic_company_id,
    )
    return get_or_create_billing_party_for_clinic_company(
        company_id=company_id,
        clinic_company_id=clinic_company_id,
    )

