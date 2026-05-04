# services/patient_sync/patient_identity_service.py
# pyright: reportCallIssue=false
"""Service de synchronisation du Patient Master Index.

Responsabilités :
- Trouver ou créer une PatientIdentity à partir d'un AVS
- Créer / gérer les PatientIdentityLinks
- Calculer les deltas (before / after) et émettre des PatientSyncEvents (outbox)
- Générer des suggestions de lien pour les patients sans AVS
- Anti-cascade via contextvars (compatible Flask + Celery)
"""

from __future__ import annotations

import contextvars
import hashlib
import logging
from datetime import UTC, date, datetime
from typing import Any

from ext import db
from models.institution import Institution
from models.institution_patient import InstitutionPatient
from models.patient_identity import (
    PatientAuditLog,
    PatientIdentity,
    PatientIdentityLink,
    PatientLinkSuggestion,
    PatientSyncEvent,
)
from shared.avs_utils import hash_avs, last4_avs, validate_avs

logger = logging.getLogger(__name__)

# ── Anti-cascade ──
_sync_origin_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "sync_origin", default=None
)


def with_sync_origin(fn, *args, **kwargs):
    """Execute fn avec sync_origin set, reset garanti même en cas d'exception."""
    token = _sync_origin_var.set("patient_sync")
    try:
        return fn(*args, **kwargs)
    finally:
        _sync_origin_var.reset(token)


# ── Champs propagés vers d'autres InstitutionPatient ──
SYNCABLE_FIELDS: list[str] = [
    "dob",
    "address",
    "city",
    "postal_code",
    "phone",
    "insurance_name",
    "insurance_number",
    "has_guardianship",
    "guardianship_type",
    "guardian_name",
    "guardian_organization",
    "guardian_phone",
    "guardian_email",
    "guardian_address",
]

# Champs qui acceptent le vidage forcé via force_clear_fields
FORCE_CLEARABLE_FIELDS: frozenset[str] = frozenset(
    {
        "insurance_name",
        "insurance_number",
        "guardian_phone",
        "guardian_email",
        "guardian_address",
        "phone",
        "notes",
    }
)


def ensure_identity_and_link(
    patient: InstitutionPatient,
    user_id: int | None = None,
) -> PatientIdentity | None:
    """Trouve ou crée une PatientIdentity pour ce patient (basé sur son AVS).

    Crée automatiquement le lien si absent.
    Retourne None si le patient n'a pas d'AVS.
    """
    if not patient.avs_number:
        return None

    avs_h = hash_avs(patient.avs_number)
    avs_l4 = last4_avs(patient.avs_number)
    avs_st = validate_avs(patient.avs_number)

    identity = PatientIdentity.query.filter_by(avs_hash=avs_h).first()

    if not identity:
        identity = PatientIdentity(
            avs_hash=avs_h,
            avs_last4=avs_l4,
            avs_status=avs_st,
            canonical_first_name=patient.first_name,
            canonical_last_name=patient.last_name,
            canonical_dob=patient.dob,
            version=1,
            confidence_level="high" if avs_st == "valid" else "medium",
            source_institution_id=patient.institution_id,
            source_patient_id=patient.id,
        )
        if avs_st == "valid" and user_id:
            identity.avs_verified_at = datetime.now(UTC)
            identity.avs_verified_by_user_id = user_id
        db.session.add(identity)
        db.session.flush()
        logger.info(
            "[PatientIdentity] Créée: id=%s, avs_last4=***%s",
            identity.id,
            avs_l4,
        )

    # Créer le lien institution_patient si absent
    existing_link = PatientIdentityLink.query.filter_by(
        patient_identity_id=identity.id,
        entity_type="institution_patient",
        entity_id=patient.id,
    ).first()

    if not existing_link:
        link = PatientIdentityLink(
            patient_identity_id=identity.id,
            entity_type="institution_patient",
            entity_id=patient.id,
            link_method="avs_exact",
            is_active=True,
            linked_by_user_id=user_id,
        )
        db.session.add(link)
        db.session.flush()

        db.session.add(
            PatientAuditLog(
                actor_user_id=user_id,
                action="LINK_CONFIRMED",
                entity_type="institution_patient",
                entity_id=patient.id,
                metadata_json={
                    "identity_id": identity.id,
                    "link_method": "avs_exact",
                    "avs_last4": avs_l4,
                },
            )
        )

    # Chercher les Client (transporteur) avec le même AVS et les lier
    _auto_link_clients_by_avs(identity, patient.avs_number, user_id)

    return identity


def compute_changed_fields(
    patient: InstitutionPatient,
    old_values: dict[str, Any],
    force_clear_fields: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Calcule le delta avant/après pour les champs synchronisables.

    Ne propage jamais de valeurs vides sauf si le champ est dans
    force_clear_fields ET dans FORCE_CLEARABLE_FIELDS.
    """
    force_clear = set(force_clear_fields or []) & FORCE_CLEARABLE_FIELDS
    changed: dict[str, dict[str, Any]] = {}
    for field in SYNCABLE_FIELDS:
        new_val = getattr(patient, field, None)
        old_val = old_values.get(field)
        if _values_differ(old_val, new_val):
            new_serialized = _serialize_val(new_val)
            if _is_empty(new_serialized) and field not in force_clear:
                continue
            changed[field] = {
                "before": _serialize_val(old_val),
                "after": new_serialized,
            }
    return changed


def compute_creation_delta(
    patient: InstitutionPatient,
) -> dict[str, dict[str, Any]]:
    """Calcule le delta pour une création (before=null, after=valeur).

    N'inclut que les champs non-null et non-vides.
    """
    changed: dict[str, dict[str, Any]] = {}
    for field in SYNCABLE_FIELDS:
        val = getattr(patient, field, None)
        serialized = _serialize_val(val)
        if not _is_empty(serialized):
            changed[field] = {"before": None, "after": serialized}
    return changed


def create_sync_event(
    identity: PatientIdentity,
    patient: InstitutionPatient,
    changed_fields: dict[str, dict[str, Any]],
    user_id: int | None = None,
) -> PatientSyncEvent | None:
    """Crée un événement de sync dans l'outbox (si pas de doublon idempotent)."""
    if not changed_fields:
        return None

    field_keys = ":".join(sorted(changed_fields.keys()))
    idem_raw = (
        f"{identity.id}:{identity.version}:"
        f"institution_patient:{patient.id}:{field_keys}"
    )
    idem_key = hashlib.sha256(idem_raw.encode()).hexdigest()

    existing = PatientSyncEvent.query.filter_by(idempotency_key=idem_key).first()
    if existing:
        logger.debug("[PatientSync] Event idempotent déjà existant: %s", idem_key[:12])
        return None

    event = PatientSyncEvent(
        patient_identity_id=identity.id,
        source_entity_type="institution_patient",
        source_entity_id=patient.id,
        changed_fields=changed_fields,
        idempotency_key=idem_key,
        event_version=identity.version,
        status="pending",
    )
    db.session.add(event)
    identity.version += 1

    db.session.add(
        PatientAuditLog(
            actor_user_id=user_id,
            action="SYNC_TRIGGERED",
            entity_type="institution_patient",
            entity_id=patient.id,
            metadata_json={
                "identity_id": identity.id,
                "changed_fields": list(changed_fields.keys()),
                "event_version": identity.version - 1,
            },
        )
    )

    db.session.flush()
    logger.info(
        "[PatientSync] Event créé: identity=%s, patient=%s, fields=%s",
        identity.id,
        patient.id,
        list(changed_fields.keys()),
    )
    return event


def trigger_sync_if_needed(
    patient: InstitutionPatient,
    old_values: dict[str, Any],
    user_id: int | None = None,
    force_clear_fields: list[str] | None = None,
) -> PatientSyncEvent | None:
    """Point d'entrée principal : vérifie si un sync est nécessaire et crée l'event.

    Appelé depuis PUT /institutions/patients/<id> après commit.
    Ne fait rien si le patient n'a pas d'AVS ou si l'institution n'est pas curatelle.
    """
    if _sync_origin_var.get(None):
        return None

    institution = Institution.query.get(patient.institution_id)
    if not institution or (institution.institution_type or "").lower() != "curatelle":
        return None

    if not patient.avs_number:
        return None

    identity = ensure_identity_and_link(patient, user_id)
    if not identity:
        return None

    changed = compute_changed_fields(patient, old_values, force_clear_fields)
    if not changed:
        return None

    return create_sync_event(identity, patient, changed, user_id)


def trigger_sync_on_create(
    patient: InstitutionPatient,
    user_id: int | None = None,
) -> dict[str, Any] | None:
    """Déclenche le sync à la création d'un patient.

    - Si AVS fourni : lien automatique + outbox PATIENT_SYNC_FIELDS
    - Si pas d'AVS : génère des suggestions de lien (confirmation humaine)

    Retourne un dict stable {"status", "suggestions_count", "identity_id"}.
    """
    if _sync_origin_var.get(None):
        return None

    institution = Institution.query.get(patient.institution_id)
    if not institution or (institution.institution_type or "").lower() != "curatelle":
        return None

    if patient.avs_number:
        identity = ensure_identity_and_link(patient, user_id)
        if not identity:
            return {"status": "none", "suggestions_count": 0, "identity_id": None}

        db.session.add(
            PatientAuditLog(
                actor_user_id=user_id,
                action="IDENTITY_LINK_CREATED",
                entity_type="institution_patient",
                entity_id=patient.id,
                metadata_json={
                    "identity_id": identity.id,
                    "link_method": "avs_exact",
                    "trigger": "on_create",
                },
            )
        )

        changed = compute_creation_delta(patient)
        if changed:
            create_sync_event(identity, patient, changed, user_id)

        return {
            "status": "linked",
            "suggestions_count": 0,
            "identity_id": identity.id,
        }

    suggestions = generate_link_suggestions(patient, user_id)
    if suggestions:
        return {
            "status": "suggestions",
            "suggestions_count": len(suggestions),
            "identity_id": None,
        }

    return {"status": "none", "suggestions_count": 0, "identity_id": None}


def generate_link_suggestions(
    patient: InstitutionPatient,
    user_id: int | None = None,
) -> list[PatientLinkSuggestion]:
    """Génère des suggestions de lien pour un patient sans AVS.

    Utilise find_potential_matches() puis crée des PatientLinkSuggestion.
    Auto-lien uniquement si candidat unique + aucun conflit strict.
    """
    from services.patient_sync.patient_matching_service import find_potential_matches

    if not patient.first_name or not patient.last_name:
        return []

    matches = find_potential_matches(
        patient_id=patient.id,
        first_name=patient.first_name,
        last_name=patient.last_name,
        dob=patient.dob,
        city=patient.city,
        phone=patient.phone,
    )

    if not matches:
        return []

    if _can_auto_link(matches, patient):
        match = matches[0]
        identity_id = match.get("identity_id")
        if identity_id:
            identity = PatientIdentity.query.get(identity_id)
            if identity:
                _create_link_and_sync(
                    identity, patient, "name_dob_auto_unique", user_id
                )
                return []

    created: list[PatientLinkSuggestion] = []
    for match in matches:
        target_type = (
            "institution_patient"
            if match["type"] in ("identity", "cross_patient")
            else "client"
        )
        target_id = match.get("patient_id") or match.get("identity_id", 0)
        target_identity_id = match.get("identity_id")

        existing = PatientLinkSuggestion.query.filter_by(
            source_patient_id=patient.id,
            target_entity_type=target_type,
            target_entity_id=target_id,
            status="pending",
        ).first()

        if existing:
            continue

        suggestion = PatientLinkSuggestion(
            source_patient_id=patient.id,
            target_identity_id=target_identity_id,
            target_entity_type=target_type,
            target_entity_id=target_id,
            match_score=match.get("match_score", 0),
            match_signals=_extract_signals(match),
            status="pending",
        )
        db.session.add(suggestion)
        created.append(suggestion)

    if created:
        db.session.flush()

    return created


def _can_auto_link(
    matches: list[dict[str, Any]],
    patient: InstitutionPatient,
) -> bool:
    """Vérifie si un auto-lien est safe (candidat unique + aucun conflit strict)."""
    if len(matches) != 1:
        return False

    match = matches[0]
    if match["type"] != "identity":
        return False

    identity_id = match.get("identity_id")
    if not identity_id:
        return False

    identity = PatientIdentity.query.get(identity_id)
    if not identity:
        return False

    active_link = PatientIdentityLink.query.filter_by(
        patient_identity_id=identity.id,
        is_active=True,
        entity_type="institution_patient",
        entity_id=patient.id,
    ).first()
    if active_link:
        return False

    any_active = PatientIdentityLink.query.filter_by(
        entity_type="institution_patient",
        entity_id=patient.id,
        is_active=True,
    ).first()
    if any_active:
        return False

    if identity.avs_status == "valid" and identity.avs_hash and patient.avs_number:
        patient_hash = hash_avs(patient.avs_number)
        if patient_hash != identity.avs_hash:
            return False

    if patient.dob:
        day = patient.dob.day
        month = patient.dob.month
        if day == 1 and month == 1:
            return False

    return bool(patient.first_name and patient.last_name)


def _create_link_and_sync(
    identity: PatientIdentity,
    patient: InstitutionPatient,
    link_method: str,
    user_id: int | None,
) -> None:
    """Crée un lien + outbox event pour un auto-lien confirmé."""
    existing_link = PatientIdentityLink.query.filter_by(
        patient_identity_id=identity.id,
        entity_type="institution_patient",
        entity_id=patient.id,
    ).first()

    if not existing_link:
        link = PatientIdentityLink(
            patient_identity_id=identity.id,
            entity_type="institution_patient",
            entity_id=patient.id,
            link_method=link_method,
            is_active=True,
            linked_by_user_id=user_id,
        )
        db.session.add(link)
        db.session.flush()

    db.session.add(
        PatientAuditLog(
            actor_user_id=user_id,
            action="IDENTITY_LINK_CREATED",
            entity_type="institution_patient",
            entity_id=patient.id,
            metadata_json={
                "identity_id": identity.id,
                "link_method": link_method,
                "trigger": "on_create_auto_unique",
            },
        )
    )

    changed = compute_creation_delta(patient)
    if changed:
        create_sync_event(identity, patient, changed, user_id)


def _extract_signals(match: dict[str, Any]) -> dict[str, bool]:
    """Extrait les signaux de matching en format dict stable."""
    signals_list = match.get("signals", [])
    return dict.fromkeys(signals_list, True)


def apply_sync_to_institution_patient(
    target_patient_id: int,
    changed_fields: dict[str, dict[str, Any]],
    source_identity_id: int | None = None,
) -> None:
    """Applique les champs modifiés à un InstitutionPatient cible.

    Garde-fou DOB : propagée seulement si source = curatelle + AVS confirmé.
    Désérialise les dates ISO avant setattr().
    """
    patient = InstitutionPatient.query.get(target_patient_id)
    if not patient:
        return

    dob_allowed = False
    if source_identity_id:
        identity = PatientIdentity.query.get(source_identity_id)
        if (
            identity
            and identity.avs_status == "valid"
            and identity.source_institution_id
        ):
            src_inst = Institution.query.get(identity.source_institution_id)
            if src_inst and (src_inst.institution_type or "").lower() == "curatelle":
                dob_allowed = True

    flags = patient.data_source_flags or {}
    for field, delta in changed_fields.items():
        if field == "dob" and not dob_allowed:
            continue
        if hasattr(patient, field):
            value = delta["after"]
            if field == "dob" and isinstance(value, str):
                value = date.fromisoformat(value)
            setattr(patient, field, value)
            flags[field] = "sync_curatelle"
    patient.data_source_flags = flags
    db.session.flush()


def apply_sync_to_client(
    target_client_id: int,
    changed_fields: dict[str, dict[str, Any]],
) -> None:
    """Applique les champs modifiés à un Client (transporteur).

    Mapping InstitutionPatient -> Client :
    - address -> domicile_address
    - city -> domicile_city
    - postal_code -> domicile_zip
    - phone -> contact_phone
    - has_guardianship + guardian_* -> BillingParty curateur (OPAD/curatorship)
    - insurance_name -> (info dans BillingParty si applicable)
    """
    from models.client import Client

    client = Client.query.get(target_client_id)
    if not client:
        return

    if "address" in changed_fields:
        client.domicile_address = changed_fields["address"]["after"]
    if "city" in changed_fields:
        client.domicile_city = changed_fields["city"]["after"]
    if "postal_code" in changed_fields:
        client.domicile_zip = changed_fields["postal_code"]["after"]
    if "phone" in changed_fields:
        client.contact_phone = changed_fields["phone"]["after"]

    _sync_guardianship_to_client(client, changed_fields)

    db.session.flush()


def _sync_guardianship_to_client(
    client,
    changed_fields: dict[str, dict[str, Any]],
) -> None:
    """Propage les infos de curatelle vers un BillingParty lie au client."""
    from models.billing_party import BillingParty, ClientBillingParty
    from models.enums import BillingPartyType

    has_guardianship = changed_fields.get("has_guardianship", {}).get("after")
    if has_guardianship is None and not any(
        k.startswith("guardian") for k in changed_fields
    ):
        return

    if has_guardianship is False:
        return

    if not has_guardianship:
        has_guardianship = True

    guardian_name = changed_fields.get("guardian_name", {}).get("after", "")
    guardian_org = changed_fields.get("guardian_organization", {}).get("after", "")
    guardian_phone = changed_fields.get("guardian_phone", {}).get("after")
    guardian_email = changed_fields.get("guardian_email", {}).get("after")
    guardian_address = changed_fields.get("guardian_address", {}).get("after")
    guardianship_type = changed_fields.get("guardianship_type", {}).get("after", "")

    bp_type = BillingPartyType.OPAD
    if guardianship_type and guardianship_type.lower() not in ("opad", "opad / spad"):
        bp_type = BillingPartyType.CURATORSHIP

    display_name = guardian_org or guardian_name or "Curatelle"
    if not client.company_id:
        return

    existing_link = (
        ClientBillingParty.query.join(BillingParty)
        .filter(
            ClientBillingParty.client_id == client.id,
            BillingParty.type.in_(
                [
                    BillingPartyType.OPAD,
                    BillingPartyType.CURATORSHIP,
                ]
            ),
        )
        .first()
    )

    if existing_link:
        bp = existing_link.billing_party
        bp.display_name = display_name
        bp.type = bp_type
        if guardian_address:
            bp.billing_address = guardian_address
        if guardian_phone:
            bp.contact_phone = guardian_phone
        if guardian_email:
            bp.contact_email = guardian_email
        if guardian_name:
            existing_link.contact_name = guardian_name
        if guardian_phone:
            existing_link.contact_phone = guardian_phone
        if guardian_email:
            existing_link.contact_email = guardian_email
    else:
        bp = BillingParty(
            company_id=client.company_id,
            type=bp_type,
            display_name=display_name,
            billing_address=guardian_address or "N/A",
            contact_phone=guardian_phone,
            contact_email=guardian_email,
        )
        db.session.add(bp)
        db.session.flush()

        link = ClientBillingParty(
            client_id=client.id,
            billing_party_id=bp.id,
            role="curatelle",
            contact_name=guardian_name,
            contact_phone=guardian_phone,
            contact_email=guardian_email,
            is_default=True,
        )
        db.session.add(link)
        db.session.flush()

    logger.info(
        "[PatientSync] Curatelle synced to client %s: %s (%s)",
        client.id,
        display_name,
        bp_type.value,
    )


def _auto_link_clients_by_avs(
    identity: PatientIdentity,
    avs_number: str,
    user_id: int | None = None,
) -> None:
    """Recherche les Client (transporteur) avec le même AVS et crée les liens.

    Permet au worker Celery de propager les données vers les clients existants
    quand un curateur crée/met à jour un patient.
    """
    from models.client import Client

    clients = Client.query.filter(
        Client.avs_number == avs_number,
        Client.is_active.is_(True),
    ).all()

    for client in clients:
        existing_link = PatientIdentityLink.query.filter_by(
            patient_identity_id=identity.id,
            entity_type="client",
            entity_id=client.id,
        ).first()

        if existing_link:
            continue

        link = PatientIdentityLink(
            patient_identity_id=identity.id,
            entity_type="client",
            entity_id=client.id,
            link_method="avs_exact",
            is_active=True,
            linked_by_user_id=user_id,
        )
        db.session.add(link)
        db.session.flush()

        db.session.add(
            PatientAuditLog(
                actor_user_id=user_id,
                action="IDENTITY_LINK_CREATED",
                entity_type="client",
                entity_id=client.id,
                metadata_json={
                    "identity_id": identity.id,
                    "link_method": "avs_exact",
                    "trigger": "auto_link_by_avs",
                },
            )
        )
        logger.info(
            "[PatientSync] Auto-linked client %s to identity %s (avs match)",
            client.id,
            identity.id,
        )


# ── Helpers internes ──


def _values_differ(old: Any, new: Any) -> bool:
    """Compare deux valeurs en gérant None, bool, str."""
    if old is None and new is None:
        return False
    if old is None or new is None:
        return True
    return str(old) != str(new)


def _serialize_val(val: Any) -> Any:
    """Sérialise une valeur pour le JSON du delta."""
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, date):
        return val.isoformat()
    return str(val)


def _is_empty(val: Any) -> bool:
    """Vérifie si une valeur est vide (None, chaîne vide, False pour bool)."""
    if val is None:
        return True
    return bool(isinstance(val, str) and val.strip() == "")
