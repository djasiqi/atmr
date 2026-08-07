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


def _direct_patient_external_ref(client_id: int) -> str:
    return f"patient_client:{int(client_id)}"


def _display_name_for_direct_patient(client: Client) -> str:
    """Nom affiché pour un destinataire PATIENT portefeuille."""
    institution = (getattr(client, "institution_name", None) or "").strip()
    if institution and bool(getattr(client, "is_institution", False)):
        return institution

    user = getattr(client, "user", None)
    if user is not None:
        first = (getattr(user, "first_name", None) or "").strip()
        last = (getattr(user, "last_name", None) or "").strip()
        full = f"{first} {last}".strip()
        if full:
            return full
        username = (getattr(user, "username", None) or "").strip()
        if username:
            return username

    first = (getattr(client, "first_name", None) or "").strip()
    last = (getattr(client, "last_name", None) or "").strip()
    full = f"{first} {last}".strip()
    if full:
        return full

    return f"Patient #{getattr(client, 'id', '?')}"


def _domicile_address_for_direct_patient(client: Client) -> str | None:
    """Adresse domicile best-effort ; vide autorisé pour BillingParty PATIENT."""
    domicile = (getattr(client, "domicile_address", None) or "").strip()
    postal = (getattr(client, "domicile_zip", None) or "").strip()
    city = (getattr(client, "domicile_city", None) or "").strip()
    parts: list[str] = []
    if domicile:
        parts.append(domicile)
    postal_city = " ".join(p for p in [postal, city] if p).strip()
    if postal_city:
        parts.append(postal_city)
    if parts:
        return "\n".join(parts)

    # Fallback billing / user — sans placeholder « Adresse non renseignée »
    try:
        billing = (getattr(client, "billing_address_secure", None) or "").strip()
    except Exception:
        billing = (getattr(client, "billing_address", None) or "").strip()
    if billing:
        return billing

    user = getattr(client, "user", None)
    if user is not None:
        user_addr = (getattr(user, "address", None) or "").strip()
        if user_addr:
            return user_addr
    return None


def get_or_create_billing_party_for_direct_patient(
    *,
    company_id: int,
    client: Client,
) -> BillingParty:
    """Crée ou met à jour le destinataire technique PATIENT pour un client portefeuille.

    Ce BillingParty n'est **pas** un tiers payeur : aucune ligne ClientBillingParty
    n'est créée. L'UI peut donc continuer d'afficher « Aucun tiers payeur configuré ».

    Idempotence via ``external_ref = patient_client:{client.id}`` (scopé company_id).
    """
    if client is None or getattr(client, "id", None) is None:
        raise ValueError("client requis pour get_or_create_billing_party_for_direct_patient")

    client_id = int(client.id)
    if int(getattr(client, "company_id", 0) or 0) != int(company_id):
        raise ValueError(
            "client.company_id ne correspond pas à company_id "
            f"(client={client_id}, company={company_id})"
        )

    external_ref = _direct_patient_external_ref(client_id)
    display_name = _display_name_for_direct_patient(client)
    address = _domicile_address_for_direct_patient(client)
    email, phone = _best_effort_contacts_from_client(client)

    existing = BillingParty.query.filter_by(
        company_id=company_id, external_ref=external_ref
    ).first()
    if existing:
        changed = False
        if display_name and existing.display_name != display_name:
            existing.display_name = display_name
            changed = True
        if address and existing.billing_address != address:
            existing.billing_address = address
            changed = True
        if email and existing.contact_email != email:
            existing.contact_email = email
            changed = True
        if phone and existing.contact_phone != phone:
            existing.contact_phone = phone
            changed = True
        if existing.type != BillingPartyType.PATIENT:
            existing.type = BillingPartyType.PATIENT
            changed = True
        if not existing.is_active:
            existing.is_active = True
            changed = True
        if changed:
            db.session.flush()
        return existing

    billing_party = BillingParty()
    billing_party.company_id = company_id
    billing_party.type = BillingPartyType.PATIENT
    billing_party.display_name = display_name
    billing_party.billing_address = address
    billing_party.contact_email = email
    billing_party.contact_phone = phone
    billing_party.external_ref = external_ref
    billing_party.is_active = True

    db.session.add(billing_party)
    db.session.flush()

    logger.info(
        "[billing_party_linker] Created PATIENT BillingParty id=%s "
        "(external_ref=%s, company_id=%s, client_id=%s)",
        billing_party.id,
        external_ref,
        company_id,
        client_id,
    )
    return billing_party


_ESTABLISHMENT_BP_TYPES = frozenset(
    {
        BillingPartyType.CLINIC,
        BillingPartyType.EMS,
        BillingPartyType.HOSPITAL,
    }
)


def is_establishment_billing_party(billing_party: BillingParty | None) -> bool:
    """True si le BP est un établissement (clinique / EMS / hôpital)."""
    if billing_party is None:
        return False
    bp_type = getattr(billing_party, "type", None)
    if bp_type in _ESTABLISHMENT_BP_TYPES:
        return True
    raw = getattr(bp_type, "value", None) or str(bp_type or "")
    return str(raw).lower().strip() in {"clinic", "ems", "hospital"}


def get_or_create_billing_party_for_institution_patient(
    *,
    company_id: int,
    institution_patient,
) -> BillingParty:
    """BillingParty PATIENT pour un InstitutionPatient (external_ref patient:{id})."""
    from models.institution_patient import InstitutionPatient

    if not isinstance(institution_patient, InstitutionPatient):
        raise ValueError(
            "institution_patient requis pour get_or_create_billing_party_for_institution_patient"
        )

    patient_id = int(institution_patient.id)
    external_ref = f"patient:{patient_id}"
    existing = BillingParty.query.filter_by(
        company_id=company_id, external_ref=external_ref
    ).first()

    first = (getattr(institution_patient, "first_name", None) or "").strip()
    last = (getattr(institution_patient, "last_name", None) or "").strip()
    display_name = f"{first} {last}".strip() or f"Patient #{patient_id}"

    addr_parts: list[str] = []
    street = (getattr(institution_patient, "address", None) or "").strip()
    if street:
        addr_parts.append(street)
    postal = (getattr(institution_patient, "postal_code", None) or "").strip()
    city = (getattr(institution_patient, "city", None) or "").strip()
    if postal or city:
        addr_parts.append(f"{postal} {city}".strip())
    address = "\n".join(addr_parts) or None
    phone = (getattr(institution_patient, "phone", None) or "").strip() or None

    if existing:
        if existing.type != BillingPartyType.PATIENT:
            existing.type = BillingPartyType.PATIENT
        if display_name and existing.display_name != display_name:
            existing.display_name = display_name
        if address and existing.billing_address != address:
            existing.billing_address = address
        if phone and not (existing.contact_phone or "").strip():
            existing.contact_phone = phone
        return existing

    billing_party = BillingParty()
    billing_party.company_id = company_id
    billing_party.type = BillingPartyType.PATIENT
    billing_party.display_name = display_name
    billing_party.billing_address = address
    billing_party.contact_phone = phone
    billing_party.external_ref = external_ref
    billing_party.is_active = True
    db.session.add(billing_party)
    db.session.flush()
    logger.info(
        "[billing_party_linker] Created PATIENT BillingParty id=%s "
        "external_ref=%s company_id=%s",
        billing_party.id,
        external_ref,
        company_id,
    )
    return billing_party


def resolve_billing_party_for_portfolio_patient(
    *,
    company_id: int,
    client: Client,
) -> BillingParty:
    """Résout le destinataire V2 pour une course portefeuille ``billed_to_type=patient``.

    Ordre :
    1. Tiers payeur actif non-établissement (``ClientBillingParty``)
    2. Sinon BillingParty PATIENT technique (pas de lien ClientBillingParty)
    """
    from services.billing.client_stay_resolver import (
        resolve_default_billing_party_for_client,
    )

    third_party = resolve_default_billing_party_for_client(
        client_id=int(client.id),
        company_id=int(company_id),
    )
    # Ne pas réutiliser un BP clinique/EMS/hôpital via ClientBillingParty
    if third_party is not None and not is_establishment_billing_party(third_party):
        return third_party
    return get_or_create_billing_party_for_direct_patient(
        company_id=int(company_id),
        client=client,
    )


def ensure_patient_destination_billing_party(booking) -> BillingParty | None:
    """Quand ``billed_to_type=patient``, aligne ``billing_party_id`` sur un BP patient.

    - Conserve un BP non-établissement (patient, curatelle, famille…).
    - Remplace un BP clinique/EMS/hôpital (ou absent) par un BP PATIENT
      (InstitutionPatient ou portefeuille).
    - Force ``billed_to_company_id = NULL``.
    """
    btype = str(getattr(booking, "billed_to_type", None) or "").lower().strip()
    if btype != "patient":
        return None

    company_id = getattr(booking, "company_id", None)
    if company_id is None:
        return None

    booking.billed_to_company_id = None

    current_bp: BillingParty | None = None
    bp_id = getattr(booking, "billing_party_id", None)
    if bp_id is not None:
        current_bp = db.session.get(BillingParty, int(bp_id))

    if current_bp is not None and not is_establishment_billing_party(current_bp):
        return current_bp

    ip_id = getattr(booking, "institution_patient_id", None)
    if ip_id is not None:
        from models.institution_patient import InstitutionPatient

        patient = db.session.get(InstitutionPatient, int(ip_id))
        if patient is not None:
            bp = get_or_create_billing_party_for_institution_patient(
                company_id=int(company_id),
                institution_patient=patient,
            )
            booking.billing_party_id = int(bp.id)
            return bp

    client = getattr(booking, "client", None)
    if client is None and getattr(booking, "client_id", None):
        client = db.session.get(Client, int(booking.client_id))
    if client is not None:
        bp = resolve_billing_party_for_portfolio_patient(
            company_id=int(company_id),
            client=client,
        )
        booking.billing_party_id = int(bp.id)
        return bp

    logger.warning(
        "[billing_party_linker] Impossible d'assurer un BP patient "
        "(booking_id=%s company_id=%s)",
        getattr(booking, "id", None),
        company_id,
    )
    return None


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
    billing_party.contact_email = (
        getattr(clinic, "contact_email", None) or ""
    ).strip() or None
    billing_party.contact_phone = (
        getattr(clinic, "contact_phone", None) or ""
    ).strip() or None
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
