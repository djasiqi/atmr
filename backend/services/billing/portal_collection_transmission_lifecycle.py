"""Transmission externe humaine avec preuves (étape 6G-B).

Aucun connecteur EasyGov / office / prestataire.
EXPORT_PREPARED ≠ TRANSMITTED ≠ ACKNOWLEDGED.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ext import db
from models.portal_collection_transmission_evidence import (
    CHANNEL_COLLECTION_PROVIDER_MANUAL,
    CHANNEL_EMAIL,
    CHANNEL_LOCAL_ARTIFACT,
    CHANNEL_MANUAL_OFFICE,
    CHANNEL_REGISTERED_MAIL,
    EVIDENCE_ACKNOWLEDGMENT,
    EVIDENCE_EVENT_ACKNOWLEDGED,
    EVIDENCE_EVENT_EXPORT,
    EVIDENCE_EVENT_TRANSMITTED,
    EVIDENCE_EXPORT_ARTIFACT,
    EVIDENCE_POSTAL_TRACKING,
    EVIDENCE_PROVIDER_MESSAGE_ID,
    EVIDENCE_RECEIPT,
    KIND_PRIVATE_COLLECTION,
    KIND_PURSUIT,
    SUPPORTED_CHANNELS,
    PortalCollectionTransmissionEvidence,
)

# Réexport pour les tests / routes.
__all_channels__ = (
    CHANNEL_MANUAL_OFFICE,
    CHANNEL_REGISTERED_MAIL,
    CHANNEL_EMAIL,
    CHANNEL_COLLECTION_PROVIDER_MANUAL,
)
from models.portal_receivable_collection_action import (
    ACTION_ACKNOWLEDGED,
    ACTION_EXPORT_PREPARED,
    ACTION_TRANSMISSION_AUTHORIZED,
    ACTION_TRANSMITTED,
    STATUS_CANCELLED,
    STATUS_DRAFT,
    TRANSMISSION_PRIVATE_COLLECTION,
    TRANSMISSION_PURSUIT_DRAFT,
    PortalReceivableCollectionAction,
    PortalReceivableCollectionTransmission,
)
from services.billing.portal_collection_legal_review import (
    latest_matching_approval,
    resolve_transmission_eligibility,
)
from services.billing.portal_receivable import PortalReceivableError

EXPORT_PROTOCOL_VERSION = "lirie-collection-export-v1"

# Statuts dérivés (resolver) — jamais synonymes.
STATUS_LIFECYCLE_DRAFT = "draft"
STATUS_LIFECYCLE_APPROVED = "approved"
STATUS_LIFECYCLE_EXPORT_PREPARED = "export_prepared"
STATUS_LIFECYCLE_TRANSMITTED = "transmitted"
STATUS_LIFECYCLE_ACKNOWLEDGED = "acknowledged"
STATUS_LIFECYCLE_CANCELLED = "cancelled"
STATUS_LIFECYCLE_STALE = "stale"

UI_LABEL_EXPORT = "Dossier préparé"
UI_LABEL_TRANSMITTED = "Transmission enregistrée"
UI_LABEL_ACKNOWLEDGED = "Réception confirmée"
UI_LABEL_DRAFT = "Brouillon"
UI_LABEL_APPROVED = "Revue approuvée"
UI_LABEL_CANCELLED = "Annulé"
UI_LABEL_STALE = "Approbation obsolète"

# Canaux explicitement refusés (non implémentés).
FORBIDDEN_CHANNELS = frozenset(
    {"easygov", "office_api", "collection_provider_api", "intrum", "creditreform"}
)

REASON_EVIDENCE_REQUIRED = "evidence_required"
REASON_CHANNEL_UNSUPPORTED = "channel_unsupported"
REASON_HASH_MISMATCH = "dossier_hash_mismatch"
REASON_NOT_ELIGIBLE = "transmission_not_authorized"
REASON_ALREADY_TRANSMITTED = "already_transmitted"
REASON_NOT_TRANSMITTED = "not_transmitted"
REASON_RECIPIENT_REQUIRED = "recipient_confirmation_required"
REASON_JURISDICTION_REQUIRED = "pursuit_jurisdiction_required"
REASON_ACK_EVIDENCE_REQUIRED = "acknowledgment_evidence_required"
REASON_CANCEL_AFTER_TRANSMIT = "cannot_cancel_after_transmitted"
REASON_EXTERNAL_CONNECTOR = "external_connector_not_implemented"


@dataclass(frozen=True, slots=True)
class TransmissionLifecycleStatus:
    state: str
    ui_label: str
    reasons: tuple[str, ...]
    details: dict[str, Any]


def _hash_payload(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _transmission_kind(transmission: PortalReceivableCollectionTransmission) -> str:
    if transmission.transmission_type == TRANSMISSION_PURSUIT_DRAFT:
        return KIND_PURSUIT
    if transmission.transmission_type == TRANSMISSION_PRIVATE_COLLECTION:
        return KIND_PRIVATE_COLLECTION
    raise PortalReceivableError(
        "Type de transmission inconnu.",
        code="transmission_type_invalid",
    )


def _latest_evidence(
    transmission_id: int, event_kind: str
) -> PortalCollectionTransmissionEvidence | None:
    return (
        PortalCollectionTransmissionEvidence.query.filter_by(
            transmission_id=int(transmission_id), event_kind=event_kind
        )
        .order_by(PortalCollectionTransmissionEvidence.id.desc())
        .first()
    )


def resolve_collection_transmission_status(
    transmission_id: int,
) -> TransmissionLifecycleStatus:
    """Statut courant dérivé — jamais un booléen « mark_as_transmitted »."""
    transmission = db.session.get(
        PortalReceivableCollectionTransmission, int(transmission_id)
    )
    if transmission is None:
        return TransmissionLifecycleStatus(
            state="not_found",
            ui_label="Introuvable",
            reasons=("transmission_not_found",),
            details={},
        )

    details: dict[str, Any] = {
        "transmission_id": transmission.id,
        "dossier_hash": transmission.export_hash,
        "row_status": transmission.status,
        "export_version": transmission.export_version,
    }

    if transmission.status == STATUS_CANCELLED:
        return TransmissionLifecycleStatus(
            state=STATUS_LIFECYCLE_CANCELLED,
            ui_label=UI_LABEL_CANCELLED,
            reasons=(),
            details=details,
        )

    ack = _latest_evidence(int(transmission.id), EVIDENCE_EVENT_ACKNOWLEDGED)
    if ack is not None:
        details["acknowledged_at"] = ack.occurred_at.isoformat()
        details["acknowledgment_reference"] = ack.acknowledgment_reference
        return TransmissionLifecycleStatus(
            state=STATUS_LIFECYCLE_ACKNOWLEDGED,
            ui_label=UI_LABEL_ACKNOWLEDGED,
            reasons=(),
            details=details,
        )

    tx_ev = _latest_evidence(int(transmission.id), EVIDENCE_EVENT_TRANSMITTED)
    if tx_ev is not None:
        details["transmitted_at"] = tx_ev.occurred_at.isoformat()
        details["channel"] = tx_ev.channel
        details["external_reference"] = tx_ev.external_reference
        details["evidence_hash"] = tx_ev.evidence_hash
        return TransmissionLifecycleStatus(
            state=STATUS_LIFECYCLE_TRANSMITTED,
            ui_label=UI_LABEL_TRANSMITTED,
            reasons=(),
            details=details,
        )

    eligibility = resolve_transmission_eligibility(int(transmission.id))
    approval = latest_matching_approval(transmission)
    if approval is not None and not eligibility.is_authorized:
        return TransmissionLifecycleStatus(
            state=STATUS_LIFECYCLE_STALE,
            ui_label=UI_LABEL_STALE,
            reasons=eligibility.reasons,
            details={**details, "eligibility": eligibility.details},
        )

    export_ev = _latest_evidence(int(transmission.id), EVIDENCE_EVENT_EXPORT)
    if export_ev is not None or transmission.export_prepared_at is not None:
        return TransmissionLifecycleStatus(
            state=STATUS_LIFECYCLE_EXPORT_PREPARED,
            ui_label=UI_LABEL_EXPORT,
            reasons=(),
            details=details,
        )

    if approval is not None and eligibility.is_authorized:
        return TransmissionLifecycleStatus(
            state=STATUS_LIFECYCLE_APPROVED,
            ui_label=UI_LABEL_APPROVED,
            reasons=(),
            details=details,
        )

    return TransmissionLifecycleStatus(
        state=STATUS_LIFECYCLE_DRAFT,
        ui_label=UI_LABEL_DRAFT,
        reasons=(),
        details=details,
    )


def build_recipient_summary(
    transmission: PortalReceivableCollectionTransmission,
) -> dict[str, Any]:
    """Résumé à confirmer avant transmission (§15)."""
    import json as _json

    debtor = _json.loads(transmission.debtor_snapshot)
    creditor = _json.loads(transmission.creditor_snapshot)
    return {
        "debtor_name": debtor.get("name"),
        "debtor_domicile": debtor.get("domicile_address")
        or debtor.get("billing_address"),
        "office_or_jurisdiction": transmission.pursuit_jurisdiction,
        "recipient": transmission.recipient_label,
        "creditor_name": creditor.get("legal_name") or creditor.get("display_name"),
        "amount_chf": float(transmission.balance_snapshot),
        "claim_reason": transmission.claim_reason_snapshot,
        "invoice_reference": transmission.invoice_reference_snapshot,
        "dossier_hash": transmission.export_hash,
        "disclaimer": (
            "Confirmer explicitement ce résumé avant toute transmission externe. "
            "Aucune juridiction n'est préremplie."
        ),
    }


def confirm_recipient_and_jurisdiction(
    *,
    transmission: PortalReceivableCollectionTransmission,
    confirmed_by_user_id: int,
    pursuit_jurisdiction: str | None,
    recipient_label: str,
    recipient_contact: str | None = None,
    summary_confirmed: bool,
) -> PortalReceivableCollectionTransmission:
    """Enregistre destinataire / juridiction + confirmation humaine du résumé."""
    if transmission.status == STATUS_CANCELLED:
        raise PortalReceivableError(
            "Transmission annulée.",
            code="transmission_cancelled",
        )
    if _latest_evidence(int(transmission.id), EVIDENCE_EVENT_TRANSMITTED):
        raise PortalReceivableError(
            "Destinataire figé après transmission.",
            code=REASON_ALREADY_TRANSMITTED,
        )
    if not summary_confirmed:
        raise PortalReceivableError(
            "Confirmation explicite du résumé destinataire obligatoire.",
            code=REASON_RECIPIENT_REQUIRED,
        )
    label = (recipient_label or "").strip()
    if not label:
        raise PortalReceivableError(
            "Destinataire / office obligatoire.",
            code=REASON_RECIPIENT_REQUIRED,
        )

    kind = _transmission_kind(transmission)
    if kind == KIND_PURSUIT:
        jurisdiction = (pursuit_jurisdiction or "").strip()
        if not jurisdiction:
            raise PortalReceivableError(
                "Juridiction / office de poursuite à confirmer explicitement "
                "(aucun défaut Genève).",
                code=REASON_JURISDICTION_REQUIRED,
            )
        transmission.pursuit_jurisdiction = jurisdiction
    else:
        contact = (recipient_contact or "").strip()
        if not contact:
            raise PortalReceivableError(
                "Contact du mandataire de recouvrement obligatoire.",
                code=REASON_RECIPIENT_REQUIRED,
            )
        transmission.recipient_contact = contact
        if pursuit_jurisdiction:
            transmission.pursuit_jurisdiction = pursuit_jurisdiction.strip() or None

    transmission.recipient_label = label
    transmission.recipient_summary_confirmed = True
    transmission.recipient_confirmed_at = datetime.now(UTC)
    transmission.recipient_confirmed_by_user_id = int(confirmed_by_user_id)

    action = PortalReceivableCollectionAction(
        receivable_id=int(transmission.receivable_id),
        creditor_company_id=int(transmission.creditor_company_id),
        transmission_id=int(transmission.id),
        action_type=ACTION_TRANSMISSION_AUTHORIZED,
        occurred_at=datetime.now(UTC),
        requested_by_user_id=int(confirmed_by_user_id),
        payload_snapshot=json.dumps(
            {
                "recipient_label": transmission.recipient_label,
                "recipient_contact": transmission.recipient_contact,
                "pursuit_jurisdiction": transmission.pursuit_jurisdiction,
                "dossier_hash": transmission.export_hash,
                "summary": build_recipient_summary(transmission),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    )
    db.session.add(action)
    db.session.flush()
    return transmission


def prepare_export_artifact(
    *,
    transmission: PortalReceivableCollectionTransmission,
    prepared_by_user_id: int,
) -> PortalCollectionTransmissionEvidence:
    """Marque EXPORT_PREPARED — ne signifie jamais TRANSMITTED."""
    if transmission.status == STATUS_CANCELLED:
        raise PortalReceivableError(
            "Transmission annulée.",
            code="transmission_cancelled",
        )

    now = datetime.now(UTC)
    transmission.export_version = EXPORT_PROTOCOL_VERSION
    transmission.export_prepared_at = now

    payload = {
        "export_hash": transmission.export_hash,
        "export_version": EXPORT_PROTOCOL_VERSION,
        "generated_at": now.isoformat(),
        "status_label": UI_LABEL_EXPORT,
        "transmitted": False,
    }
    evidence = PortalCollectionTransmissionEvidence(
        transmission_id=int(transmission.id),
        receivable_id=int(transmission.receivable_id),
        creditor_company_id=int(transmission.creditor_company_id),
        dossier_hash=str(transmission.export_hash),
        transmission_kind=_transmission_kind(transmission),
        channel=CHANNEL_LOCAL_ARTIFACT,
        event_kind=EVIDENCE_EVENT_EXPORT,
        occurred_at=now,
        recorded_by_user_id=int(prepared_by_user_id),
        recipient=None,
        external_reference=None,
        evidence_type=EVIDENCE_EXPORT_ARTIFACT,
        evidence_payload=json.dumps(payload, ensure_ascii=False, sort_keys=True),
        evidence_hash=_hash_payload(
            json.dumps(payload, ensure_ascii=False, sort_keys=True)
        ),
        acknowledgment_reference=None,
    )
    db.session.add(evidence)

    action = PortalReceivableCollectionAction(
        receivable_id=int(transmission.receivable_id),
        creditor_company_id=int(transmission.creditor_company_id),
        transmission_id=int(transmission.id),
        action_type=ACTION_EXPORT_PREPARED,
        occurred_at=now,
        requested_by_user_id=int(prepared_by_user_id),
        payload_snapshot=json.dumps(
            {
                "export_hash": transmission.export_hash,
                "export_version": EXPORT_PROTOCOL_VERSION,
                "transmitted": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    )
    db.session.add(action)
    db.session.flush()
    return evidence


def _validate_channel_evidence(
    *,
    channel: str,
    recipient: str | None,
    external_reference: str | None,
    evidence_type: str | None,
    evidence_fields: dict[str, Any],
    transmitted_at: datetime | None,
) -> tuple[str, str, dict[str, Any]]:
    """Retourne (evidence_type, external_reference, normalized_fields)."""
    if channel in FORBIDDEN_CHANNELS:
        raise PortalReceivableError(
            "Connecteur externe non implémenté.",
            code=REASON_EXTERNAL_CONNECTOR,
        )
    if channel not in SUPPORTED_CHANNELS:
        raise PortalReceivableError(
            "Canal non supporté.",
            code=REASON_CHANNEL_UNSUPPORTED,
        )
    if transmitted_at is None:
        raise PortalReceivableError(
            "transmitted_at obligatoire.",
            code=REASON_EVIDENCE_REQUIRED,
        )
    recipient_clean = (recipient or "").strip()
    if not recipient_clean:
        raise PortalReceivableError(
            "Destinataire obligatoire.",
            code=REASON_EVIDENCE_REQUIRED,
        )

    fields = dict(evidence_fields or {})
    ext_ref = (external_reference or "").strip() or None
    _ = evidence_type  # type dérivé du canal, pas du client

    if channel == CHANNEL_EMAIL:
        msg_id = (
            str(fields.get("provider_message_id") or "").strip()
            or (ext_ref or "")
        )
        sent_at = fields.get("sent_at") or transmitted_at.isoformat()
        if not msg_id:
            raise PortalReceivableError(
                "provider_message_id obligatoire pour le canal email.",
                code=REASON_EVIDENCE_REQUIRED,
            )
        fields = {
            "provider_message_id": msg_id,
            "sent_at": sent_at,
            "recipient": recipient_clean,
        }
        return EVIDENCE_PROVIDER_MESSAGE_ID, msg_id, fields

    if channel == CHANNEL_MANUAL_OFFICE:
        receipt = (
            str(fields.get("receipt") or "").strip()
            or str(fields.get("proof_of_deposit") or "").strip()
            or (ext_ref or "")
        )
        if not receipt:
            raise PortalReceivableError(
                "Récépissé / référence office obligatoire.",
                code=REASON_EVIDENCE_REQUIRED,
            )
        fields = {
            "receipt": receipt,
            "transmitted_at": transmitted_at.isoformat(),
            "recipient": recipient_clean,
        }
        return EVIDENCE_RECEIPT, receipt, fields

    if channel == CHANNEL_REGISTERED_MAIL:
        tracking = (
            str(fields.get("postal_tracking_reference") or "").strip()
            or (ext_ref or "")
        )
        proof = str(fields.get("proof_of_deposit") or "").strip()
        if not tracking or not proof:
            raise PortalReceivableError(
                "postal_tracking_reference et proof_of_deposit obligatoires.",
                code=REASON_EVIDENCE_REQUIRED,
            )
        fields = {
            "postal_tracking_reference": tracking,
            "proof_of_deposit": proof,
            "transmitted_at": transmitted_at.isoformat(),
            "recipient": recipient_clean,
        }
        return EVIDENCE_POSTAL_TRACKING, tracking, fields

    if channel == CHANNEL_COLLECTION_PROVIDER_MANUAL:
        contact = str(fields.get("recipient_contact") or "").strip()
        proof = (
            str(fields.get("handoff_evidence") or "").strip()
            or str(fields.get("receipt") or "").strip()
            or (ext_ref or "")
        )
        if not contact or not proof:
            raise PortalReceivableError(
                "Contact mandataire et preuve de remise obligatoires.",
                code=REASON_EVIDENCE_REQUIRED,
            )
        fields = {
            "provider": recipient_clean,
            "recipient_contact": contact,
            "handoff_evidence": proof,
            "transmitted_at": transmitted_at.isoformat(),
        }
        return EVIDENCE_RECEIPT, proof, fields

    raise PortalReceivableError(
        "Canal non supporté.",
        code=REASON_CHANNEL_UNSUPPORTED,
    )


def record_external_transmission(
    *,
    transmission: PortalReceivableCollectionTransmission,
    recorded_by_user_id: int,
    channel: str,
    recipient: str | None,
    external_reference: str | None,
    evidence_type: str | None,
    evidence_fields: dict[str, Any] | None,
    transmitted_at: datetime,
    expected_dossier_hash: str | None = None,
) -> PortalCollectionTransmissionEvidence:
    """Enregistre une transmission humaine réelle avec preuve.

    Aucun appel API externe. Refuse sans preuve / sans éligibilité.
    """
    if transmission.status != STATUS_DRAFT:
        raise PortalReceivableError(
            "Seuls les drafts actifs peuvent être marqués transmis.",
            code="transmission_not_draft",
        )
    if _latest_evidence(int(transmission.id), EVIDENCE_EVENT_TRANSMITTED):
        raise PortalReceivableError(
            "Transmission déjà enregistrée.",
            code=REASON_ALREADY_TRANSMITTED,
        )

    expected = (expected_dossier_hash or "").strip()
    if expected and expected != str(transmission.export_hash):
        raise PortalReceivableError(
            "Le hash fourni ne correspond pas au dossier approuvé.",
            code=REASON_HASH_MISMATCH,
        )

    # Revalidation juste avant l'action (§4).
    eligibility = resolve_transmission_eligibility(int(transmission.id))
    if not eligibility.is_authorized:
        raise PortalReceivableError(
            "Transmission refusée — dossier non éligible : "
            + ", ".join(eligibility.reasons),
            code=REASON_NOT_ELIGIBLE,
        )

    approval = latest_matching_approval(transmission)
    if approval is None or str(approval.dossier_hash) != str(transmission.export_hash):
        raise PortalReceivableError(
            "Revue juridique absente ou hash divergent.",
            code=REASON_HASH_MISMATCH,
        )

    if not bool(transmission.recipient_summary_confirmed):
        raise PortalReceivableError(
            "Confirmation du résumé destinataire obligatoire avant transmission.",
            code=REASON_RECIPIENT_REQUIRED,
        )
    kind = _transmission_kind(transmission)
    if kind == KIND_PURSUIT and not (transmission.pursuit_jurisdiction or "").strip():
        raise PortalReceivableError(
            "Juridiction de poursuite non confirmée.",
            code=REASON_JURISDICTION_REQUIRED,
        )
    if kind == KIND_PRIVATE_COLLECTION and not (
        transmission.recipient_contact or ""
    ).strip():
        raise PortalReceivableError(
            "Contact mandataire non confirmé.",
            code=REASON_RECIPIENT_REQUIRED,
        )

    recipient_final = (recipient or transmission.recipient_label or "").strip()
    ev_type, ext_ref, fields = _validate_channel_evidence(
        channel=channel,
        recipient=recipient_final,
        external_reference=external_reference,
        evidence_type=evidence_type,
        evidence_fields=evidence_fields or {},
        transmitted_at=transmitted_at,
    )

    # Lier éventuellement l'export préparé.
    fields["dossier_hash"] = transmission.export_hash
    fields["export_hash"] = transmission.export_hash
    fields["export_version"] = transmission.export_version
    payload_json = json.dumps(fields, ensure_ascii=False, sort_keys=True)

    evidence = PortalCollectionTransmissionEvidence(
        transmission_id=int(transmission.id),
        receivable_id=int(transmission.receivable_id),
        creditor_company_id=int(transmission.creditor_company_id),
        dossier_hash=str(transmission.export_hash),
        transmission_kind=kind,
        channel=channel,
        event_kind=EVIDENCE_EVENT_TRANSMITTED,
        occurred_at=transmitted_at,
        recorded_by_user_id=int(recorded_by_user_id),
        recipient=recipient_final,
        external_reference=ext_ref,
        evidence_type=ev_type,
        evidence_payload=payload_json,
        evidence_hash=_hash_payload(payload_json),
        acknowledgment_reference=None,
    )
    db.session.add(evidence)

    action = PortalReceivableCollectionAction(
        receivable_id=int(transmission.receivable_id),
        creditor_company_id=int(transmission.creditor_company_id),
        transmission_id=int(transmission.id),
        action_type=ACTION_TRANSMITTED,
        occurred_at=datetime.now(UTC),
        requested_by_user_id=int(recorded_by_user_id),
        payload_snapshot=json.dumps(
            {
                "channel": channel,
                "recipient": recipient_final,
                "external_reference": ext_ref,
                "dossier_hash": transmission.export_hash,
                "evidence_hash": evidence.evidence_hash,
                "ui_label": UI_LABEL_TRANSMITTED,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    )
    db.session.add(action)
    db.session.flush()
    return evidence


def record_acknowledgment(
    *,
    transmission: PortalReceivableCollectionTransmission,
    recorded_by_user_id: int,
    acknowledged_at: datetime,
    acknowledgment_reference: str,
    acknowledgment_evidence: str,
) -> PortalCollectionTransmissionEvidence:
    """ACKNOWLEDGED exige une preuve de réception distincte de TRANSMITTED."""
    if _latest_evidence(int(transmission.id), EVIDENCE_EVENT_TRANSMITTED) is None:
        raise PortalReceivableError(
            "Aucune transmission enregistrée — ACKNOWLEDGED impossible.",
            code=REASON_NOT_TRANSMITTED,
        )
    if _latest_evidence(int(transmission.id), EVIDENCE_EVENT_ACKNOWLEDGED):
        raise PortalReceivableError(
            "Réception déjà enregistrée.",
            code="already_acknowledged",
        )
    ref = (acknowledgment_reference or "").strip()
    proof = (acknowledgment_evidence or "").strip()
    if not ref or not proof:
        raise PortalReceivableError(
            "Référence et preuve de réception externes obligatoires.",
            code=REASON_ACK_EVIDENCE_REQUIRED,
        )

    tx_ev = _latest_evidence(int(transmission.id), EVIDENCE_EVENT_TRANSMITTED)
    assert tx_ev is not None
    payload = {
        "acknowledgment_reference": ref,
        "acknowledgment_evidence": proof,
        "acknowledged_at": acknowledged_at.isoformat(),
        "linked_transmission_evidence_id": tx_ev.id,
        "dossier_hash": transmission.export_hash,
    }
    payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    evidence = PortalCollectionTransmissionEvidence(
        transmission_id=int(transmission.id),
        receivable_id=int(transmission.receivable_id),
        creditor_company_id=int(transmission.creditor_company_id),
        dossier_hash=str(transmission.export_hash),
        transmission_kind=_transmission_kind(transmission),
        channel=tx_ev.channel,
        event_kind=EVIDENCE_EVENT_ACKNOWLEDGED,
        occurred_at=acknowledged_at,
        recorded_by_user_id=int(recorded_by_user_id),
        recipient=tx_ev.recipient,
        external_reference=ref,
        evidence_type=EVIDENCE_ACKNOWLEDGMENT,
        evidence_payload=payload_json,
        evidence_hash=_hash_payload(payload_json),
        acknowledgment_reference=ref,
    )
    db.session.add(evidence)

    action = PortalReceivableCollectionAction(
        receivable_id=int(transmission.receivable_id),
        creditor_company_id=int(transmission.creditor_company_id),
        transmission_id=int(transmission.id),
        action_type=ACTION_ACKNOWLEDGED,
        occurred_at=datetime.now(UTC),
        requested_by_user_id=int(recorded_by_user_id),
        payload_snapshot=json.dumps(
            {
                "acknowledgment_reference": ref,
                "dossier_hash": transmission.export_hash,
                "ui_label": UI_LABEL_ACKNOWLEDGED,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    )
    db.session.add(action)
    db.session.flush()
    return evidence


def assert_cancel_allowed(
    transmission: PortalReceivableCollectionTransmission,
) -> None:
    """Annulation avant transmission uniquement — pas de retrait externe."""
    if _latest_evidence(int(transmission.id), EVIDENCE_EVENT_TRANSMITTED):
        raise PortalReceivableError(
            "Impossible d'annuler une transmission déjà enregistrée "
            "(retrait externe non implémenté).",
            code=REASON_CANCEL_AFTER_TRANSMIT,
        )


def serialize_evidence(row: PortalCollectionTransmissionEvidence) -> dict[str, Any]:
    return {
        "id": row.id,
        "transmission_id": row.transmission_id,
        "receivable_id": row.receivable_id,
        "creditor_company_id": row.creditor_company_id,
        "dossier_hash": row.dossier_hash,
        "transmission_kind": row.transmission_kind,
        "channel": row.channel,
        "event_kind": row.event_kind,
        "occurred_at": row.occurred_at.isoformat() if row.occurred_at else None,
        "recorded_by_user_id": row.recorded_by_user_id,
        "recipient": row.recipient,
        "external_reference": row.external_reference,
        "evidence_type": row.evidence_type,
        "evidence_hash": row.evidence_hash,
        "acknowledgment_reference": row.acknowledgment_reference,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def serialize_lifecycle_status(
    result: TransmissionLifecycleStatus,
) -> dict[str, Any]:
    return {
        "state": result.state,
        "ui_label": result.ui_label,
        "reasons": list(result.reasons),
        "details": result.details,
    }
