"""Résolution du BillingParty pour les bookings issus d'une institution.

Garantit que billing_party_id est résolu au moment de la conversion
request → booking, éliminant les cas "facturé au mauvais".

Logique par billing_intent:
- institution → BillingParty CLINIC/EMS/HOSPITAL (depuis Institution)
- patient → BillingParty PATIENT (snapshot patient, plus de fallback Client fragile)
- curator/spc/other → BillingParty tiers (depuis billing_details)

Source de vérité: BillingParty (pas billing_details, pas Institution).
billing_details est un snapshot informatif copié dans metadata_json.

Chaque résolution produit un billing_resolution_status et billing_resolution_source
dans booking.metadata_json pour observabilité et portail transporteur.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sqlalchemy import func

from ext import db
from models import BillingParty, BillingPartyType, ClinicBillingPartyMapping, Client, Company

if TYPE_CHECKING:
    from models import Booking, Institution, TransportRequest
    from models.institution_patient import InstitutionPatient

logger = logging.getLogger(__name__)


# ── Constantes ──────────────────────────────────────────────────────────

# Mapping institution_type → BillingPartyType
_INSTITUTION_TYPE_TO_BP_TYPE = {
    "clinic": BillingPartyType.CLINIC,
    "ems": BillingPartyType.EMS,
    "hospital": BillingPartyType.HOSPITAL,
    "imad": BillingPartyType.OTHER,
    "curatelle": BillingPartyType.OTHER,
}

# Mapping billing_intent → BillingPartyType (pour tiers)
_INTENT_TO_BP_TYPE = {
    "curator": BillingPartyType.CURATORSHIP,
    "spc": BillingPartyType.OPAD,
    "insurance": BillingPartyType.INSURANCE,
    "other": BillingPartyType.OTHER,
}

# Statuts de résolution
STATUS_SUCCESS = "success"
STATUS_FAILED_MISSING_ADDRESS = "failed_missing_institution_address"
STATUS_FAILED_MISSING_PAYER = "failed_missing_payer_info"
STATUS_FAILED_MISSING_PATIENT = "failed_missing_patient_info"
STATUS_FAILED_ERROR = "failed_error"

# Sources de résolution
SOURCE_MAPPING = "clinic_billing_mapping"
SOURCE_EXTERNAL_REF = "external_ref_lookup"
SOURCE_CREATED_INSTITUTION = "created_from_institution"
SOURCE_CREATED_PATIENT = "created_from_patient_snapshot"
SOURCE_CREATED_THIRD_PARTY = "created_from_billing_details"
SOURCE_EXISTING_THIRD_PARTY = "existing_third_party"
SOURCE_EXISTING_PATIENT = "existing_patient_ref"


# ── Fonction principale ─────────────────────────────────────────────────


def resolve_billing_party_for_institution_booking(
    *,
    booking: Booking,
    transport_request: TransportRequest,
    company_id: int,
    billing_intent_override: str | None = None,
) -> dict[str, Any]:
    """Résout et attache le billing_party_id au booking selon billing_intent.

    Couvre les 3 cas : institution, patient, tiers.
    Produit toujours un billing_resolution_status dans metadata_json.

    Args:
        booking: Le booking fraîchement créé
        transport_request: La demande de transport source
        company_id: ID de l'entreprise de transport qui accepte
        billing_intent_override: Intent effectif (ex. override destination par leg)

    Returns:
        Dict avec les champs mis à jour (pour audit/metrics)
    """
    billing_intent = (
        billing_intent_override
        if billing_intent_override
        else (transport_request.billing_intent or "patient")
    )
    billing_intent = str(billing_intent).lower()
    institution = transport_request.institution
    patient = transport_request.patient

    result: dict[str, Any] = {"billing_intent": billing_intent}
    resolution_status = STATUS_SUCCESS
    resolution_source = ""

    # ── Intent: Institution ──────────────────────────────────────────
    if billing_intent == "institution" and institution:
        bp, source = _resolve_or_create_institution_bp(
            company_id=company_id,
            institution=institution,
        )
        if bp:
            # Conserver la clinique payeuse résolue en amont (AcceptOffer),
            # sinon la déduire depuis le client institution ou le mapping BP.
            clinic_company_id = _resolve_clinic_company_id_for_institution(
                booking=booking,
                company_id=company_id,
                billing_party_id=bp.id,
                institution=institution,
            )
            booking.billing_party_id = bp.id
            booking.billed_to_type = "clinic"
            if clinic_company_id is not None:
                booking.billed_to_company_id = clinic_company_id
            result["billing_party_id"] = bp.id
            result["billing_party_name"] = bp.display_name
            result["billed_to_company_id"] = booking.billed_to_company_id
            resolution_source = source
            logger.info(
                "[BillingResolver] intent=institution → bp_id=%s (%s) source=%s booking=%s clinic_company_id=%s",
                bp.id,
                bp.display_name,
                source,
                booking.id,
                booking.billed_to_company_id,
            )
        else:
            resolution_status = STATUS_FAILED_MISSING_ADDRESS
            logger.warning(
                "[BillingResolver] FAILED intent=institution: institution=%s has no address. booking=%s company=%s",
                institution.id,
                booking.id,
                company_id,
            )

    # ── Intent: Patient ──────────────────────────────────────────────
    elif billing_intent == "patient":
        bp, source = _resolve_or_create_patient_bp(
            company_id=company_id,
            transport_request=transport_request,
            patient=patient,
        )
        if bp:
            booking.billing_party_id = bp.id
            booking.billed_to_type = "patient"
            result["billing_party_id"] = bp.id
            result["billing_party_name"] = bp.display_name
            resolution_source = source
            logger.info(
                "[BillingResolver] intent=patient → bp_id=%s source=%s booking=%s",
                bp.id,
                source,
                booking.id,
            )
        else:
            # Patient sans infos suffisantes — non-bloquant (le fallback Client
            # dans le PDF reste actif), mais on flag pour le transporteur
            resolution_status = STATUS_FAILED_MISSING_PATIENT
            logger.warning(
                "[BillingResolver] FAILED intent=patient: insufficient patient data. booking=%s",
                booking.id,
            )

    # ── Intent: Tiers (curator/spc/insurance/other) ─────────────────
    elif billing_intent in ("curator", "spc", "other", "insurance"):
        # Auto-enrichir billing_details depuis les infos curateur du patient
        _enrich_billing_details_from_patient_guardian(transport_request, patient)

        bp, source, fail_reason = _resolve_or_create_third_party_bp(
            company_id=company_id,
            transport_request=transport_request,
            billing_intent=billing_intent,
        )
        if bp:
            booking.billing_party_id = bp.id
            booking.billed_to_type = billing_intent
            booking.billed_to_company_id = company_id
            result["billing_party_id"] = bp.id
            result["billing_party_name"] = bp.display_name
            resolution_source = source
            logger.info(
                "[BillingResolver] intent=%s → bp_id=%s (%s) source=%s booking=%s",
                billing_intent,
                bp.id,
                bp.display_name,
                source,
                booking.id,
            )
        else:
            resolution_status = fail_reason or STATUS_FAILED_MISSING_PAYER
            logger.warning(
                "[BillingResolver] FAILED intent=%s: %s. booking=%s",
                billing_intent,
                resolution_status,
                booking.id,
            )

    # ── Metadata (loggé car Booking n'a pas de colonne metadata_json) ──
    meta: dict[str, Any] = {}

    # Billing resolution tracking
    meta["billing_resolution_status"] = resolution_status
    meta["billing_resolution_source"] = resolution_source
    meta["billing_resolution_intent"] = billing_intent

    # Snapshot billing_details (informatif, pour traçabilité)
    billing_details = transport_request.billing_details
    if billing_details:
        meta["institution_billing_details"] = billing_details

    if institution:
        meta["institution_name"] = institution.name
        meta["institution_id"] = institution.id

    # Logger les metadata au lieu de les stocker (Booking n'a pas metadata_json)
    logger.info(
        "[BillingResolver] booking=%s billing_meta=%s",
        booking.id,
        meta,
    )
    result["billing_resolution_status"] = resolution_status
    result["billing_resolution_source"] = resolution_source

    # ── Metrics ──────────────────────────────────────────────────────
    _track_billing_resolution(
        intent=billing_intent,
        status=resolution_status,
        source=resolution_source,
        company_id=company_id,
        booking_id=booking.id,
    )

    return result


# ── Mapping guardianship_type → BillingPartyType ─────────────────────────

_GUARDIANSHIP_TYPE_TO_BP_TYPE = {
    "curatorship": BillingPartyType.CURATORSHIP,
    "opad": BillingPartyType.OPAD,
    "lawyer": BillingPartyType.LAWYER,
    "family": BillingPartyType.FAMILY,
    "other": BillingPartyType.OTHER,
}

# Mapping guardianship_type → billing_intent
_GUARDIANSHIP_TYPE_TO_INTENT = {
    "curatorship": "curator",
    "opad": "spc",
    "lawyer": "curator",
    "family": "curator",
    "other": "other",
}


def _enrich_billing_details_from_patient_guardian(
    transport_request: TransportRequest,
    patient: InstitutionPatient | None,
) -> None:
    """Auto-remplit billing_details depuis les infos curateur du patient.

    Si le patient a has_guardianship=True et des infos curateur renseignées,
    on les utilise comme fallback pour billing_details (payer_name,
    payer_address, payer_email) si non déjà fournis par l'institution.

    Cela permet à l'institution de ne renseigner le curateur qu'une seule
    fois dans la fiche patient, et la facturation s'en sert automatiquement.
    """
    if not patient or not patient.has_guardianship:
        return

    details = transport_request.billing_details
    if details is None:
        details = {}
        transport_request.billing_details = details

    # payer_name: guardian_name + guardian_organization
    if not (details.get("payer_name") or "").strip():
        name_parts = []
        if patient.guardian_name:
            name_parts.append(patient.guardian_name.strip())
        if patient.guardian_organization:
            name_parts.append(f"({patient.guardian_organization.strip()})")
        if name_parts:
            details["payer_name"] = " ".join(name_parts)

    # payer_address: guardian_address
    if not (details.get("payer_address") or "").strip() and patient.guardian_address:
        details["payer_address"] = patient.guardian_address.strip()

    # payer_email: guardian_email
    if not (details.get("payer_email") or "").strip() and patient.guardian_email:
        details["payer_email"] = patient.guardian_email.strip()

    # Enrichir le BillingPartyType si guardianship_type renseigné
    if patient.guardianship_type and not details.get("guardianship_type"):
        details["guardianship_type"] = patient.guardianship_type

    logger.debug(
        "[BillingResolver] Enriched billing_details from patient guardian"
        " (patient=%s, guardianship_type=%s)",
        patient.id,
        patient.guardianship_type,
    )


# ── Résolution Institution ───────────────────────────────────────────────


def _resolve_or_create_institution_bp(
    *,
    company_id: int,
    institution: Institution,
) -> tuple[BillingParty | None, str]:
    """Résout ou crée un BillingParty pour une institution.

    Logique (3 niveaux):
    1. ClinicBillingPartyMapping configuré par le transporteur
    2. BillingParty existant via external_ref
    3. Création depuis données Institution

    Returns:
        (BillingParty | None, source_label)
    """
    external_ref = f"institution:{institution.id}"

    # 1. Mapping configuré par le transporteur (prioritaire)
    mappings = ClinicBillingPartyMapping.query.filter_by(
        company_id=company_id,
        is_active=True,
    ).all()

    for m in mappings:
        bp = BillingParty.query.get(m.billing_party_id)
        if bp and bp.external_ref == external_ref:
            return bp, SOURCE_MAPPING

    # 2. Lookup par external_ref
    existing = BillingParty.query.filter_by(
        company_id=company_id,
        external_ref=external_ref,
    ).first()
    if existing:
        return existing, SOURCE_EXTERNAL_REF

    # 3. Créer depuis Institution
    billing_address = (institution.billing_address or "").strip()
    if not billing_address:
        billing_address = (institution.address or "").strip()

    # Si aucune adresse du tout → on ne peut pas créer de BP exploitable
    if not billing_address:
        return None, ""

    bp_type = _INSTITUTION_TYPE_TO_BP_TYPE.get(
        (institution.institution_type or "").lower(),
        BillingPartyType.CLINIC,
    )

    bp = BillingParty()
    bp.company_id = company_id
    bp.type = bp_type
    bp.display_name = institution.name or "Institution"
    bp.billing_address = billing_address
    bp.contact_email = (
        (institution.billing_email or "").strip()
        or (institution.contact_email or "").strip()
        or None
    )
    bp.contact_phone = (institution.contact_phone or "").strip() or None
    bp.external_ref = external_ref
    bp.is_active = True

    db.session.add(bp)
    db.session.flush()

    logger.info(
        "[BillingResolver] Created institution BillingParty id=%s (institution=%s, type=%s, company=%s)",
        bp.id,
        institution.id,
        bp_type.value,
        company_id,
    )

    return bp, SOURCE_CREATED_INSTITUTION


def _resolve_clinic_company_id_for_institution(
    *,
    booking: Booking,
    company_id: int,
    billing_party_id: int,
    institution: Institution | None = None,
) -> int | None:
    """Déduit la clinique payeuse pour un booking institution.

    Priorité:
    1) ``billed_to_company_id`` déjà défini (sauf erreur historique = ID transporteur)
    2) Client institution lié (``default_billed_to_company_id``)
    3) Entreprise clinique homonyme (``Company.name`` ≈ ``institution_name``)
    4) Mapping explicite (company_id, billing_party_id) -> clinic_company_id
    """
    existing = getattr(booking, "billed_to_company_id", None)
    if existing is not None and int(existing) != int(company_id):
        return int(existing)

    client = getattr(booking, "client", None)
    if client is None and getattr(booking, "client_id", None):
        client = Client.query.get(int(booking.client_id))

    if client and getattr(client, "is_institution", False):
        client_default = getattr(client, "default_billed_to_company_id", None)
        if client_default is not None:
            return int(client_default)
        inst_name = (getattr(client, "institution_name", None) or "").strip()
        if inst_name:
            co = (
                Company.query.filter(
                    func.lower(Company.name) == func.lower(inst_name)
                )
                .order_by(Company.id.asc())
                .first()
            )
            if co is not None:
                return int(co.id)

    if institution is not None:
        inst_name = (getattr(institution, "name", None) or "").strip()
        if inst_name:
            co = (
                Company.query.filter(
                    func.lower(Company.name) == func.lower(inst_name)
                )
                .order_by(Company.id.asc())
                .first()
            )
            if co is not None:
                return int(co.id)

    mapping = (
        ClinicBillingPartyMapping.query.filter_by(
            company_id=company_id,
            billing_party_id=billing_party_id,
            is_active=True,
        )
        .order_by(ClinicBillingPartyMapping.id.desc())
        .first()
    )
    if mapping and mapping.clinic_company_id is not None:
        return int(mapping.clinic_company_id)

    return None


# ── Résolution Patient ───────────────────────────────────────────────────


def _resolve_or_create_patient_bp(
    *,
    company_id: int,
    transport_request: TransportRequest,
    patient: InstitutionPatient | None,
) -> tuple[BillingParty | None, str]:
    """Résout ou crée un BillingParty PATIENT depuis le snapshot patient.

    Logique:
    1. Chercher un BP existant via external_ref "patient:{patient_id}"
    2. Construire un snapshot (nom + adresse) depuis InstitutionPatient
       ou depuis billing_details ou depuis transport_request pickup
    3. Créer un BillingParty PATIENT

    Returns:
        (BillingParty | None, source_label)
    """
    # ── Construire le snapshot patient ──
    patient_name = ""
    patient_address = ""

    if patient:
        patient_name = f"{patient.first_name or ''} {patient.last_name or ''}".strip()

        # Adresse : domicile patient > pickup (si c'est le domicile)
        addr_parts = []
        if patient.address:
            addr_parts.append(patient.address.strip())
        postal_city = ""
        if patient.postal_code:
            postal_city += patient.postal_code.strip()
        if patient.city:
            postal_city += (
                f" {patient.city.strip()}" if postal_city else patient.city.strip()
            )
        if postal_city:
            addr_parts.append(postal_city)
        patient_address = "\n".join(addr_parts)

    # Fallback depuis billing_details
    if not patient_name:
        details = transport_request.billing_details or {}
        patient_name = (details.get("patient_name") or "").strip()
    if not patient_address:
        details = transport_request.billing_details or {}
        patient_address = (details.get("patient_billing_address") or "").strip()

    # Fallback depuis customer_name (déjà résolu dans accept_offer)
    if not patient_name:
        patient_name = (
            getattr(transport_request, "patient_first_name", "") or ""
        ).strip()
        ln = (getattr(transport_request, "patient_last_name", "") or "").strip()
        if ln:
            patient_name = f"{patient_name} {ln}".strip()

    if not patient_name:
        return None, ""

    # ── Chercher un BP existant ──
    if patient:
        external_ref = f"patient:{patient.id}"
        existing = BillingParty.query.filter_by(
            company_id=company_id,
            external_ref=external_ref,
        ).first()
        if existing:
            # Mettre à jour si adresse a changé
            if patient_address and existing.billing_address != patient_address:
                existing.billing_address = patient_address
            return existing, SOURCE_EXISTING_PATIENT
    else:
        external_ref = None

    # ── Créer un BillingParty PATIENT ──
    bp = BillingParty()
    bp.company_id = company_id
    bp.type = BillingPartyType.PATIENT
    bp.display_name = patient_name
    # BillingParty PATIENT accepte billing_address vide (validation modèle)
    bp.billing_address = patient_address or None
    bp.contact_phone = (patient.phone if patient else None) or None
    bp.external_ref = external_ref
    bp.is_active = True

    db.session.add(bp)
    db.session.flush()

    logger.info(
        "[BillingResolver] Created patient BillingParty id=%s (patient_ref=%s, company=%s)",
        bp.id,
        external_ref or "no_ref",
        company_id,
    )

    return bp, SOURCE_CREATED_PATIENT


# ── Résolution Tiers ─────────────────────────────────────────────────────


def _resolve_or_create_third_party_bp(
    *,
    company_id: int,
    transport_request: TransportRequest,
    billing_intent: str,
) -> tuple[BillingParty | None, str, str]:
    """Résout ou crée un BillingParty tiers depuis billing_details.

    Validation: payer_name est obligatoire, payer_address recommandé.

    Returns:
        (BillingParty | None, source_label, fail_reason)
    """
    details = transport_request.billing_details or {}
    payer_name = (details.get("payer_name") or "").strip()
    payer_address = (details.get("payer_address") or "").strip()

    # Validation: payer_name obligatoire pour tiers
    if not payer_name:
        return None, "", STATUS_FAILED_MISSING_PAYER

    # Si guardianship_type est présent dans billing_details, utiliser un type
    # plus précis (ex: OPAD, LAWYER) au lieu du générique CURATORSHIP
    guardianship_type = details.get("guardianship_type")
    if guardianship_type and guardianship_type in _GUARDIANSHIP_TYPE_TO_BP_TYPE:
        bp_type = _GUARDIANSHIP_TYPE_TO_BP_TYPE[guardianship_type]
    else:
        bp_type = _INTENT_TO_BP_TYPE.get(billing_intent, BillingPartyType.OTHER)

    # Chercher un existant (par nom + type + company)
    existing = BillingParty.query.filter_by(
        company_id=company_id,
        display_name=payer_name,
        type=bp_type,
    ).first()
    if existing:
        # Mettre à jour adresse si on en a une meilleure
        if payer_address and existing.billing_address != payer_address:
            existing.billing_address = payer_address
        return existing, SOURCE_EXISTING_THIRD_PARTY, ""

    # Créer
    bp = BillingParty()
    bp.company_id = company_id
    bp.type = bp_type
    bp.display_name = payer_name
    bp.billing_address = payer_address or "Adresse non renseignée"
    bp.contact_email = (details.get("payer_email") or "").strip() or None
    bp.is_active = True

    db.session.add(bp)
    db.session.flush()

    logger.info(
        "[BillingResolver] Created third-party BillingParty id=%s (%s, type=%s, company=%s)",
        bp.id,
        payer_name,
        bp_type.value,
        company_id,
    )

    return bp, SOURCE_CREATED_THIRD_PARTY, ""


# ── Sync Institution → BillingParty ─────────────────────────────────────


def sync_institution_to_billing_parties(institution: Institution) -> int:
    """Synchronise les données Institution vers les BillingParty liés.

    Appelé après PUT /institutions/me pour maintenir la cohérence.

    Returns:
        Nombre de BillingParty mis à jour
    """
    external_ref = f"institution:{institution.id}"
    bps = BillingParty.query.filter_by(external_ref=external_ref).all()

    if not bps:
        return 0

    updated = 0
    billing_address = (institution.billing_address or "").strip()
    if not billing_address:
        billing_address = (institution.address or "").strip()
    if not billing_address:
        billing_address = "Adresse non renseignée"

    billing_email = (
        (institution.billing_email or "").strip()
        or (institution.contact_email or "").strip()
        or None
    )

    for bp in bps:
        changed = False
        if bp.display_name != institution.name:
            bp.display_name = institution.name
            changed = True
        if bp.billing_address != billing_address:
            bp.billing_address = billing_address
            changed = True
        if bp.contact_email != billing_email:
            bp.contact_email = billing_email
            changed = True
        phone = (institution.contact_phone or "").strip() or None
        if bp.contact_phone != phone:
            bp.contact_phone = phone
            changed = True
        if changed:
            updated += 1

    if updated:
        logger.info(
            "[BillingResolver] Synced %d BillingParty(s) for institution %s",
            updated,
            institution.id,
        )

    return updated


# ── Metrics ──────────────────────────────────────────────────────────────


def _track_billing_resolution(
    *,
    intent: str,
    status: str,
    source: str,
    company_id: int,
    booking_id: int | None,
) -> None:
    """Tracke un événement de résolution facturation (logs structurés + metrics).

    Utilise le même pattern que institution_metrics (log-based).
    """
    is_success = status == STATUS_SUCCESS
    emoji = "✅" if is_success else "⚠️"

    logger.info(
        "%s [Metric:BillingResolution] intent=%s status=%s source=%s company_id=%s booking_id=%s",
        emoji,
        intent,
        status,
        source or "none",
        company_id,
        booking_id,
    )
