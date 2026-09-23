"""Poursuite / recouvrement privé PORTAL — drafts explicites, sans transmission.

Couche plus stricte que la collection readiness 6E.
Aucune intégration EasyGov / office / société de recouvrement.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from ext import db
from models.client_booking_contract_event import (
    EVENT_BOOKING_CANCELLED,
    EVENT_BOOKING_CREATED,
    EVENT_BOOKING_MODIFIED,
    ClientBookingContractEvent,
)
from models.client_terms_acceptance import ClientTermsAcceptance
from models.company import Company
from models.portal_receivable import (
    DISPUTE_OPEN,
    RECEIVABLE_CANCELLED,
    RECEIVABLE_DISPUTED,
    RECEIVABLE_PAID,
    PortalReceivable,
    PortalReceivableDispute,
)
from models.portal_receivable_collection_action import (
    ACTION_PRIVATE_COLLECTION_DRAFT_PREPARED,
    ACTION_PURSUIT_DRAFT_PREPARED,
    ACTION_TRANSMISSION_CANCELLED,
    STATUS_CANCELLED,
    STATUS_DRAFT,
    TRANSMISSION_PRIVATE_COLLECTION,
    TRANSMISSION_PURSUIT_DRAFT,
    PortalReceivableCollectionAction,
    PortalReceivableCollectionTransmission,
)
from models.portal_receivable_dunning import (
    CHANNEL_EMAIL,
    CHANNEL_LETTER_DRAFT,
    DELIVERY_SENT,
    DUNNING_COLLECTION_PREPARED,
    DUNNING_FORMAL_NOTICE,
    PortalReceivableDunningEvent,
)
from models.user import User
from services.billing.portal_payment_hold import (
    business_calendar_date,
    current_business_date,
)
from services.billing.portal_pursuit_form_mapping import (
    pursuit_form_mapping_report,
)
from services.billing.portal_receivable import PortalReceivableError
from services.billing.portal_receivable_dunning import (
    READY as COLLECTION_READY,
)
from services.billing.portal_receivable_dunning import (
    REASON_CANCELLED,
    REASON_DISPUTED,
    REASON_FORMAL_NOTICE_MISSING,
    REASON_INVOICE_REFERENCE_MISSING,
    REASON_NOT_OVERDUE,
    REASON_PAID,
    build_claim_title,
    build_collection_dossier,
    resolve_portal_collection_readiness,
)

PURSUIT_READY = "pursuit_ready"
PURSUIT_NOT_READY = "pursuit_not_ready"

REASON_CREDITOR_LEGAL_NAME_MISSING = "creditor_legal_name_missing"
REASON_CREDITOR_ADDRESS_MISSING = "creditor_address_missing"
REASON_CREDITOR_IDENTITY_INCOMPLETE = "creditor_identity_incomplete"
REASON_CLAIM_REASON_MISSING = "claim_reason_missing"
REASON_CURRENCY_NOT_CHF = "currency_not_chf"
REASON_COLLECTION_NOT_READY = "collection_not_ready"
REASON_DEBTOR_NAME_MISSING = "debtor_name_missing"
REASON_DEBTOR_DOMICILE_MISSING = "debtor_domicile_missing"
REASON_DEBTOR_DOMICILE_UNVERIFIED = "debtor_domicile_unverified_or_unknown"
REASON_CONFIRMATION_REQUIRED = "creditor_confirmation_required"
REASON_OFFICIAL_FORM_MAPPING_INCOMPLETE = "official_form_mapping_incomplete"

DOMICILE_SEMANTICS = "domicile"
BILLING_SEMANTICS = "billing_address"

# Champs exclus de l'export de recouvrement (minimisation).
_HEALTH_SENSITIVE_KEYS = frozenset(
    {
        "wheelchair_need",
        "wheelchair_need_snapshot",
        "notes_medical",
        "doctor",
        "doctor_name",
        "medical_facility",
        "medical_facility_name",
        "establishment",
        "passenger_medical",
    }
)

# Rôle technique existant : propriétaire COMPANY.
# Gap documenté : pas de rôle « représentant juridique » distinct.
AUTHORIZED_CREDITOR_ROLES = ("COMPANY",)
AUTHORIZED_CREDITOR_ROLES_GAP = (
    "Aucun rôle company distinct « représentant juridique » : "
    "seul le compte UserRole.COMPANY lié à company.user_id peut agir. "
    "Pas de délégation juridique inventée."
)
REASON_COLLECTION_PREPARED_MISSING = "collection_prepared_missing"


def _nonempty(value: object | None) -> bool:
    return bool(str(value or "").strip())


def _creditor_postal_address(company: Company) -> str | None:
    line1 = getattr(company, "domicile_address_line1", None)
    zip_c = getattr(company, "domicile_zip", None)
    city = getattr(company, "domicile_city", None)
    if _nonempty(line1) and (_nonempty(zip_c) or _nonempty(city)):
        parts = [str(line1).strip()]
        line2 = getattr(company, "domicile_address_line2", None)
        if _nonempty(line2):
            parts.append(str(line2).strip())
        loc = " ".join(
            p for p in [str(zip_c or "").strip(), str(city or "").strip()] if p
        )
        if loc:
            parts.append(loc)
        return ", ".join(parts)
    addr = getattr(company, "address", None)
    return str(addr).strip() if _nonempty(addr) else None


@dataclass(frozen=True, slots=True)
class FieldAvailability:
    field: str
    status: str  # AVAILABLE | PARTIAL | MISSING
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class PursuitReadiness:
    state: str
    reasons: tuple[str, ...]
    audit: dict[str, Any]

    @property
    def is_ready(self) -> bool:
        return self.state == PURSUIT_READY


def _money(value: object) -> Decimal:
    return Decimal(str(value or "0"))


def _hash_payload(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _format_date(value: datetime | None) -> str:
    d = business_calendar_date(value)
    if d is None:
        return "—"
    return d.strftime("%d.%m.%Y")


def debtor_address_semantics() -> str:
    """Sémantique des adresses de facturation (non domicile LP)."""
    return BILLING_SEMANTICS


def debtor_domicile_semantics(receivable: PortalReceivable) -> str | None:
    """Sémantique du snapshot domicile figé — jamais billing implicite."""
    sem = getattr(receivable, "debtor_domicile_semantics", None)
    if _nonempty(sem):
        return str(sem).strip()
    if _nonempty(getattr(receivable, "debtor_domicile_address_snapshot", None)):
        return DOMICILE_SEMANTICS
    return None


def creditor_legal_identity_complete(company: Company | None) -> bool:
    """Identité légale créancier : legal_name + adresse postale officielle."""
    if company is None:
        return False
    return _nonempty(getattr(company, "legal_name", None)) and _nonempty(
        _creditor_postal_address(company)
    )


def audit_creditor_identity(company: Company | None) -> dict[str, FieldAvailability]:
    if company is None:
        return {
            "commercial_display_name": FieldAvailability(
                "commercial_display_name", "MISSING"
            ),
            "legal_company_name": FieldAvailability("legal_company_name", "MISSING"),
            "legal_form": FieldAvailability("legal_form", "MISSING"),
            "uid_ide": FieldAvailability("uid_ide", "MISSING"),
            "registered_postal_address": FieldAvailability(
                "registered_postal_address", "MISSING"
            ),
            "billing_email": FieldAvailability("billing_email", "MISSING"),
            "legal_identity_complete": FieldAvailability(
                "legal_identity_complete", "MISSING", "NO"
            ),
        }
    name_ok = _nonempty(company.name)
    legal_ok = _nonempty(getattr(company, "legal_name", None))
    postal = _creditor_postal_address(company)
    postal_status = "AVAILABLE" if _nonempty(postal) else "MISSING"
    complete = creditor_legal_identity_complete(company)
    return {
        "commercial_display_name": FieldAvailability(
            "commercial_display_name",
            "AVAILABLE" if name_ok else "MISSING",
            str(company.name) if name_ok else None,
        ),
        "legal_company_name": FieldAvailability(
            "legal_company_name",
            "AVAILABLE" if legal_ok else "MISSING",
            str(company.legal_name) if legal_ok else None,
        ),
        "legal_form": FieldAvailability(
            "legal_form",
            "AVAILABLE" if _nonempty(getattr(company, "legal_form", None)) else "MISSING",
            getattr(company, "legal_form", None),
        ),
        "uid_ide": FieldAvailability(
            "uid_ide",
            "AVAILABLE" if _nonempty(getattr(company, "uid_ide", None)) else "MISSING",
            getattr(company, "uid_ide", None),
        ),
        "registered_postal_address": FieldAvailability(
            "registered_postal_address",
            postal_status,
            postal,
        ),
        "billing_email": FieldAvailability(
            "billing_email",
            "AVAILABLE"
            if _nonempty(
                getattr(company, "billing_email", None)
                or getattr(company, "contact_email", None)
            )
            else "MISSING",
        ),
        "legal_identity_complete": FieldAvailability(
            "legal_identity_complete",
            "AVAILABLE" if complete else "MISSING",
            "YES" if complete else "NO",
        ),
    }


def audit_debtor_identity(receivable: PortalReceivable) -> dict[str, FieldAvailability]:
    user = db.session.get(User, int(receivable.debtor_user_id))
    first = str(getattr(user, "first_name", "") or "").strip() if user else ""
    last = str(getattr(user, "last_name", "") or "").strip() if user else ""
    full = str(receivable.debtor_name_snapshot or "").strip()
    if first and last:
        name_status = "AVAILABLE"
    elif full:
        name_status = "PARTIAL"
    else:
        name_status = "MISSING"
    billing = receivable.debtor_billing_address_snapshot
    domicile = getattr(receivable, "debtor_domicile_address_snapshot", None)
    domicile_sem = debtor_domicile_semantics(receivable)
    return {
        "first_name": FieldAvailability(
            "first_name", "AVAILABLE" if first else "MISSING", first or None
        ),
        "last_name": FieldAvailability(
            "last_name", "AVAILABLE" if last else "MISSING", last or None
        ),
        "full_name": FieldAvailability("full_name", name_status, full or None),
        "billing_address": FieldAvailability(
            "billing_address",
            "AVAILABLE" if _nonempty(billing) else "MISSING",
            str(billing).strip() if _nonempty(billing) else None,
        ),
        "domicile_address": FieldAvailability(
            "domicile_address",
            "AVAILABLE" if _nonempty(domicile) else "MISSING",
            str(domicile).strip() if _nonempty(domicile) else None,
        ),
        "domicile_semantics": FieldAvailability(
            "domicile_semantics",
            "AVAILABLE" if domicile_sem == DOMICILE_SEMANTICS else "MISSING",
            domicile_sem,
        ),
        "address_semantics": FieldAvailability(
            "address_semantics",
            "PARTIAL",
            BILLING_SEMANTICS,
        ),
    }



def build_claim_reason(receivable: PortalReceivable) -> str:
    """Motif factuel — pas une qualification juridique."""
    issued = _format_date(receivable.issued_at)
    title = build_claim_title(receivable)
    return (
        f"Facture {receivable.external_invoice_number} du {issued}, "
        f"{title[0].lower() + title[1:] if title else 'prestations de transport'}."
    )


def resolve_portal_enforcement_evidence(
    receivable_id: int,
) -> dict[str, Any]:
    """Classification factuelle des preuves — sans conclusion de mainlevée."""
    receivable = db.session.get(PortalReceivable, int(receivable_id))
    if receivable is None:
        return {"error": "receivable_not_found"}

    booking_ids = [int(line.booking_id) for line in (receivable.lines or [])]
    contract_types: set[str] = set()
    if booking_ids:
        events = (
            ClientBookingContractEvent.query.filter(
                ClientBookingContractEvent.booking_id.in_(booking_ids)
            )
            .order_by(ClientBookingContractEvent.id.asc())
            .all()
        )
        contract_types = {str(e.event_type) for e in events}

    terms = (
        ClientTermsAcceptance.query.filter_by(user_id=int(receivable.debtor_user_id))
        .limit(1)
        .first()
    )
    dunning = PortalReceivableDunningEvent.query.filter_by(
        receivable_id=int(receivable.id)
    ).all()
    formal_email = any(
        e.event_type == DUNNING_FORMAL_NOTICE
        and e.channel == CHANNEL_EMAIL
        and e.delivery_status == DELIVERY_SENT
        for e in dunning
    )
    formal_letter_draft = any(
        e.event_type == DUNNING_FORMAL_NOTICE and e.channel == CHANNEL_LETTER_DRAFT
        for e in dunning
    )

    categories = []
    user = db.session.get(User, int(receivable.debtor_user_id))
    if user is not None and getattr(user, "phone_verified_at", None):
        categories.append("IDENTITY_VERIFIED")
    if terms is not None:
        categories.append("TERMS_ACCEPTED")
    if EVENT_BOOKING_CREATED in contract_types:
        categories.append("ORDER_CONFIRMED")
    if booking_ids:
        categories.append("SERVICE_LINKED")
    categories.append("INVOICE_ISSUED")
    if receivable.payments:
        categories.append("PAYMENTS_TRACKED")
    if any(e.event_type == DUNNING_FORMAL_NOTICE for e in dunning):
        categories.append("FORMAL_NOTICE_SENT")

    return {
        "booking_contractual_evidence": (
            "available" if EVENT_BOOKING_CREATED in contract_types else "absent"
        ),
        "booking_modified_evidence": (
            "available" if EVENT_BOOKING_MODIFIED in contract_types else "absent"
        ),
        "booking_cancelled_evidence": (
            "available" if EVENT_BOOKING_CANCELLED in contract_types else "absent"
        ),
        "terms_acceptances": "available" if terms is not None else "absent",
        "invoice": "available",
        "dunning": "available" if dunning else "absent",
        "qualified_manual_signature": "absent",
        "formal_debt_acknowledgment": "absent",
        "factual_categories": categories,
        "formal_notice_email_proof": (
            "EMAIL_SENT_WITH_PROVIDER_ID"
            if formal_email
            else "MISSING"
        ),
        "formal_notice_postal_proof": (
            "MISSING"
            if not formal_letter_draft
            else "LETTER_DRAFT_ONLY_NOT_POSTAL_PROOF"
        ),
        "mainlevee_classification": "NOT_AUTOMATICALLY_DETERMINED",
        "disclaimer": (
            "Classification factuelle uniquement. "
            "Le tribunal reste compétent pour apprécier le titre."
        ),
    }


def _add_reason(reasons: list[str], code: str) -> None:
    if code not in reasons:
        reasons.append(code)


def resolve_portal_pursuit_readiness(
    receivable_id: int,
    *,
    as_of: date | datetime | None = None,
) -> PursuitReadiness:
    receivable = db.session.get(PortalReceivable, int(receivable_id))
    if receivable is None:
        return PursuitReadiness(
            state=PURSUIT_NOT_READY,
            reasons=("receivable_not_found",),
            audit={},
        )

    as_of_date = current_business_date(as_of)
    reasons: list[str] = []
    collection = resolve_portal_collection_readiness(
        int(receivable.id), as_of=as_of_date
    )
    if collection.state != COLLECTION_READY:
        _add_reason(reasons, REASON_COLLECTION_NOT_READY)
        for r in collection.reasons:
            _add_reason(reasons, r)

    if receivable.status == RECEIVABLE_CANCELLED or receivable.cancelled_at is not None:
        _add_reason(reasons, REASON_CANCELLED)
    open_dispute = (
        PortalReceivableDispute.query.filter_by(
            receivable_id=int(receivable.id), status=DISPUTE_OPEN
        ).first()
    )
    if (
        receivable.status == RECEIVABLE_DISPUTED
        or receivable.disputed_at is not None
        or open_dispute is not None
    ):
        _add_reason(reasons, REASON_DISPUTED)
    if receivable.status == RECEIVABLE_PAID or _money(receivable.balance_due) <= 0:
        _add_reason(reasons, REASON_PAID)
    due = business_calendar_date(receivable.due_date)
    if due is None or due >= as_of_date:
        _add_reason(reasons, REASON_NOT_OVERDUE)

    if str(receivable.currency or "").upper() != "CHF":
        _add_reason(reasons, REASON_CURRENCY_NOT_CHF)

    company = db.session.get(Company, int(receivable.creditor_company_id))
    creditor_audit = audit_creditor_identity(company)
    if creditor_audit["legal_company_name"].status == "MISSING":
        _add_reason(reasons, REASON_CREDITOR_LEGAL_NAME_MISSING)
    if creditor_audit["registered_postal_address"].status == "MISSING":
        _add_reason(reasons, REASON_CREDITOR_ADDRESS_MISSING)
    if not creditor_legal_identity_complete(company):
        _add_reason(reasons, REASON_CREDITOR_IDENTITY_INCOMPLETE)

    debtor_audit = audit_debtor_identity(receivable)
    if debtor_audit["full_name"].status == "MISSING":
        _add_reason(reasons, REASON_DEBTOR_NAME_MISSING)
    domicile_ok = (
        debtor_audit["domicile_address"].status == "AVAILABLE"
        and debtor_audit["domicile_semantics"].status == "AVAILABLE"
    )
    if not domicile_ok:
        if debtor_audit["domicile_address"].status == "MISSING":
            _add_reason(reasons, REASON_DEBTOR_DOMICILE_MISSING)
        else:
            _add_reason(reasons, REASON_DEBTOR_DOMICILE_UNVERIFIED)
        # billing seul ne suffit jamais
        if (
            debtor_audit["billing_address"].status == "AVAILABLE"
            and debtor_audit["domicile_address"].status == "MISSING"
        ):
            _add_reason(reasons, REASON_DEBTOR_DOMICILE_UNVERIFIED)

    if not _nonempty(receivable.external_invoice_number):
        _add_reason(reasons, REASON_INVOICE_REFERENCE_MISSING)

    claim_reason = build_claim_reason(receivable)
    if not _nonempty(claim_reason):
        _add_reason(reasons, REASON_CLAIM_REASON_MISSING)

    done_types = {
        str(e.event_type)
        for e in PortalReceivableDunningEvent.query.filter_by(
            receivable_id=int(receivable.id)
        ).all()
    }
    if DUNNING_FORMAL_NOTICE not in done_types:
        _add_reason(reasons, REASON_FORMAL_NOTICE_MISSING)
    if DUNNING_COLLECTION_PREPARED not in done_types:
        _add_reason(reasons, REASON_COLLECTION_PREPARED_MISSING)

    evidence = resolve_portal_enforcement_evidence(int(receivable.id))
    audit = {
        "debtor": {
            k: {"status": v.status, "detail": v.detail} for k, v in debtor_audit.items()
        },
        "creditor": {
            k: {"status": v.status, "detail": v.detail}
            for k, v in creditor_audit.items()
        },
        "claim": {
            "currency": {
                "status": (
                    "AVAILABLE"
                    if str(receivable.currency or "").upper() == "CHF"
                    else "MISSING"
                ),
                "detail": receivable.currency,
            },
            "principal_source": "PortalReceivable.balance_due",
            "principal": float(receivable.balance_due),
            "invoice_reference": {
                "status": (
                    "AVAILABLE"
                    if _nonempty(receivable.external_invoice_number)
                    else "MISSING"
                ),
                "detail": receivable.external_invoice_number,
            },
            "title_reason": {
                "status": "AVAILABLE" if _nonempty(claim_reason) else "MISSING",
                "detail": claim_reason,
            },
            "interest": "NOT_IMPLEMENTED",
            "fees": "NOT_IMPLEMENTED",
        },
        "debtor_address_semantics": debtor_address_semantics(),
        "debtor_domicile_semantics": debtor_domicile_semantics(receivable),
        "debtor_domicile_source": "Client.domicile_address + domicile_zip + domicile_city",
        "collection_readiness": {
            "state": collection.state,
            "reasons": list(collection.reasons),
        },
        "enforcement_evidence": evidence,
        "official_form_field_mapping": pursuit_form_mapping_report()["overall"],
        "official_form_mapping": pursuit_form_mapping_report(),
        "easygov_integration": "NOT_IMPLEMENTED",
        "authorized_creditor_roles": list(AUTHORIZED_CREDITOR_ROLES),
        "authorized_creditor_roles_gap": AUTHORIZED_CREDITOR_ROLES_GAP,
    }

    if reasons:
        return PursuitReadiness(
            state=PURSUIT_NOT_READY, reasons=tuple(reasons), audit=audit
        )
    return PursuitReadiness(state=PURSUIT_READY, reasons=(), audit=audit)


def _strip_sensitive(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            k: _strip_sensitive(v)
            for k, v in obj.items()
            if k not in _HEALTH_SENSITIVE_KEYS
            and not any(s in k.lower() for s in ("medical", "doctor", "wheelchair"))
        }
    if isinstance(obj, list):
        return [_strip_sensitive(x) for x in obj]
    return obj


def build_minimized_export(
    receivable: PortalReceivable,
    *,
    transmission_type: str,
    claim_reason: str,
) -> dict[str, Any]:
    """Export financier/contractuel minimal — pas de données de santé."""
    company = db.session.get(Company, int(receivable.creditor_company_id))
    dossier = build_collection_dossier(receivable)
    contract_events_min = []
    for ev in dossier.get("contract_events") or []:
        contract_events_min.append(
            {
                "id": ev.get("id"),
                "booking_id": ev.get("booking_id"),
                "event_type": ev.get("event_type"),
                "occurred_at": ev.get("occurred_at"),
            }
        )
    # Recharger preuves contractuelles sans snapshots médicaux.
    booking_ids = [int(line.booking_id) for line in (receivable.lines or [])]
    evidence_rows = []
    if booking_ids:
        rows = (
            ClientBookingContractEvent.query.filter(
                ClientBookingContractEvent.booking_id.in_(booking_ids)
            )
            .order_by(ClientBookingContractEvent.id.asc())
            .all()
        )
        for row in rows:
            evidence_rows.append(
                {
                    "id": row.id,
                    "booking_id": row.booking_id,
                    "event_type": row.event_type,
                    "occurred_at": (
                        row.occurred_at.isoformat() if row.occurred_at else None
                    ),
                    "debtor_name_snapshot": row.debtor_name_snapshot,
                    "debtor_billing_address_snapshot": (
                        row.debtor_billing_address_snapshot
                    ),
                    "pickup_snapshot": None,  # non nécessaire au recouvrement
                    "dropoff_snapshot": None,
                }
            )

    terms = (
        ClientTermsAcceptance.query.filter_by(user_id=int(receivable.debtor_user_id))
        .order_by(ClientTermsAcceptance.id.asc())
        .all()
    )
    terms_payload = [
        {
            "id": t.id,
            "document_type": getattr(t, "document_type", None),
            "terms_version": getattr(t, "terms_version", None),
            "accepted_at": t.accepted_at.isoformat()
            if getattr(t, "accepted_at", None)
            else None,
        }
        for t in terms
    ]

    formal = (
        PortalReceivableDunningEvent.query.filter_by(
            receivable_id=int(receivable.id), event_type=DUNNING_FORMAL_NOTICE
        )
        .order_by(PortalReceivableDunningEvent.id.desc())
        .first()
    )
    prepared = (
        PortalReceivableDunningEvent.query.filter_by(
            receivable_id=int(receivable.id), event_type=DUNNING_COLLECTION_PREPARED
        )
        .order_by(PortalReceivableDunningEvent.id.desc())
        .first()
    )

    export = {
        "document_kind": "lirie_collection_draft_export",
        "status_label": "DRAFT — NON TRANSMIS",
        "disclaimer": (
            "Ce fichier est un dossier de travail préparé pour le créancier. "
            "Ce n'est pas un formulaire officiel de réquisition ni une preuve "
            "de transmission à un office ou une société de recouvrement."
        ),
        "transmission_type": transmission_type,
        "receivable_id": receivable.id,
        "creditor": {
            "company_id": receivable.creditor_company_id,
            "display_name": receivable.creditor_name_snapshot,
            "legal_form": getattr(company, "legal_form", None) if company else None,
            "uid_ide": getattr(company, "uid_ide", None) if company else None,
            "postal_address": (
                _creditor_postal_address(company) if company else None
            ),
            "billing_email": (
                (
                    getattr(company, "billing_email", None)
                    or getattr(company, "contact_email", None)
                )
                if company
                else None
            ),
        },
        "debtor": {
            "user_id": receivable.debtor_user_id,
            "name": receivable.debtor_name_snapshot,
            "billing_address": receivable.debtor_billing_address_snapshot,
            "domicile_address": getattr(
                receivable, "debtor_domicile_address_snapshot", None
            ),
            "domicile_semantics": debtor_domicile_semantics(receivable),
            "billing_semantics": debtor_address_semantics(),
            "email": receivable.debtor_email_snapshot,
        },
        "claim": {
            "currency": "CHF",
            "principal": float(receivable.balance_due),
            "principal_source": "PortalReceivable.balance_due",
            "total_initial": float(receivable.total_amount),
            "amount_paid": float(receivable.amount_paid),
            "invoice_reference": receivable.external_invoice_number,
            "issued_at": (
                receivable.issued_at.isoformat() if receivable.issued_at else None
            ),
            "due_date": (
                receivable.due_date.isoformat() if receivable.due_date else None
            ),
            "reason": claim_reason,
            "interest": "NOT_IMPLEMENTED",
            "fees": "NOT_IMPLEMENTED",
        },
        "lines": [
            {
                "booking_id": line.booking_id,
                "invoiced_amount": float(line.invoiced_amount),
                "description": line.description,
            }
            for line in (receivable.lines or [])
        ],
        "payments": [
            {
                "amount": float(p.amount),
                "paid_at": p.paid_at.isoformat() if p.paid_at else None,
                "method": p.method,
                "reference": p.reference,
            }
            for p in (receivable.payments or [])
        ],
        "contract_events": evidence_rows or contract_events_min,
        "terms_acceptances": terms_payload,
        "dunning_history": [
            {
                "id": e.id,
                "event_type": e.event_type,
                "occurred_at": e.occurred_at.isoformat() if e.occurred_at else None,
                "channel": e.channel,
                "delivery_status": e.delivery_status,
                "balance_due_snapshot": float(e.balance_due_snapshot),
            }
            for e in PortalReceivableDunningEvent.query.filter_by(
                receivable_id=int(receivable.id)
            )
            .order_by(PortalReceivableDunningEvent.id.asc())
            .all()
        ],
        "dispute_history": [
            {
                "id": d.id,
                "status": d.status,
                "reason": d.reason,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            }
            for d in (receivable.disputes or [])
        ],
        "formal_notice_event_id": formal.id if formal else None,
        "collection_prepared_event_id": prepared.id if prepared else None,
        "pursuit_jurisdiction": None,
    }
    return _strip_sensitive(export)


def prepare_collection_transmission(
    *,
    receivable: PortalReceivable,
    transmission_type: str,
    requested_by_user_id: int,
    creditor_confirmed: bool,
    as_of: date | datetime | None = None,
) -> PortalReceivableCollectionTransmission:
    """Crée un draft immuable. Aucun appel externe."""
    if transmission_type not in (
        TRANSMISSION_PRIVATE_COLLECTION,
        TRANSMISSION_PURSUIT_DRAFT,
    ):
        raise PortalReceivableError(
            "Type de transmission invalide.",
            code="transmission_type_invalid",
        )
    if not creditor_confirmed:
        raise PortalReceivableError(
            "Confirmation explicite du créancier obligatoire.",
            code=REASON_CONFIRMATION_REQUIRED,
        )

    readiness = resolve_portal_pursuit_readiness(int(receivable.id), as_of=as_of)
    if not readiness.is_ready:
        raise PortalReceivableError(
            "Dossier insuffisant pour préparer une transmission : "
            + ", ".join(readiness.reasons),
            code="pursuit_not_ready",
        )

    claim_reason = build_claim_reason(receivable)
    export = build_minimized_export(
        receivable,
        transmission_type=transmission_type,
        claim_reason=claim_reason,
    )
    # Sanity : aucune donnée santé
    blob = json.dumps(export, ensure_ascii=False).lower()
    for banned in ("wheelchair", "notes_medical", "doctor", "medical_facility"):
        if banned in blob:
            raise PortalReceivableError(
                "Export non conforme à la minimisation des données.",
                code="export_contains_sensitive_data",
            )

    company = db.session.get(Company, int(receivable.creditor_company_id))
    creditor_snap = {
        "company_id": receivable.creditor_company_id,
        "display_name": receivable.creditor_name_snapshot,
        "legal_name": getattr(company, "legal_name", None) if company else None,
        "legal_form": getattr(company, "legal_form", None) if company else None,
        "uid_ide": getattr(company, "uid_ide", None) if company else None,
        "postal_address": _creditor_postal_address(company) if company else None,
    }
    debtor_snap = {
        "user_id": receivable.debtor_user_id,
        "name": receivable.debtor_name_snapshot,
        "billing_address": receivable.debtor_billing_address_snapshot,
        "domicile_address": getattr(
            receivable, "debtor_domicile_address_snapshot", None
        ),
        "domicile_semantics": debtor_domicile_semantics(receivable),
        "email": receivable.debtor_email_snapshot,
    }
    payments_snap = [
        {
            "amount": float(p.amount),
            "paid_at": p.paid_at.isoformat() if p.paid_at else None,
            "method": p.method,
            "reference": p.reference,
        }
        for p in (receivable.payments or [])
    ]
    prepared = (
        PortalReceivableDunningEvent.query.filter_by(
            receivable_id=int(receivable.id), event_type=DUNNING_COLLECTION_PREPARED
        )
        .order_by(PortalReceivableDunningEvent.id.desc())
        .first()
    )
    export_json = json.dumps(export, ensure_ascii=False, sort_keys=True)
    row = PortalReceivableCollectionTransmission(
        receivable_id=int(receivable.id),
        creditor_company_id=int(receivable.creditor_company_id),
        transmission_type=transmission_type,
        status=STATUS_DRAFT,
        collection_prepared_event_id=prepared.id if prepared else None,
        creditor_snapshot=json.dumps(creditor_snap, ensure_ascii=False, sort_keys=True),
        debtor_snapshot=json.dumps(debtor_snap, ensure_ascii=False, sort_keys=True),
        claim_principal_snapshot=_money(receivable.balance_due),
        payments_snapshot=json.dumps(payments_snap, ensure_ascii=False, sort_keys=True),
        balance_snapshot=_money(receivable.balance_due),
        currency_snapshot="CHF",
        invoice_reference_snapshot=str(receivable.external_invoice_number),
        claim_reason_snapshot=claim_reason,
        due_date_snapshot=receivable.due_date,
        export_payload=export_json,
        export_hash=_hash_payload(export_json),
        pursuit_jurisdiction=None,
        creditor_confirmed=True,
        requested_at=datetime.now(UTC),
        requested_by_user_id=int(requested_by_user_id),
    )
    db.session.add(row)
    db.session.flush()

    action_type = (
        ACTION_PURSUIT_DRAFT_PREPARED
        if transmission_type == TRANSMISSION_PURSUIT_DRAFT
        else ACTION_PRIVATE_COLLECTION_DRAFT_PREPARED
    )
    action = PortalReceivableCollectionAction(
        receivable_id=int(receivable.id),
        creditor_company_id=int(receivable.creditor_company_id),
        transmission_id=int(row.id),
        action_type=action_type,
        occurred_at=datetime.now(UTC),
        requested_by_user_id=int(requested_by_user_id),
        payload_snapshot=json.dumps(
            {
                "transmission_id": row.id,
                "transmission_type": transmission_type,
                "balance_snapshot": float(row.balance_snapshot),
                "export_hash": row.export_hash,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    )
    db.session.add(action)
    db.session.flush()
    return row


def cancel_collection_transmission(
    *,
    transmission: PortalReceivableCollectionTransmission,
    requested_by_user_id: int,
) -> PortalReceivableCollectionTransmission:
    if transmission.status == STATUS_CANCELLED:
        return transmission
    transmission.status = STATUS_CANCELLED
    action = PortalReceivableCollectionAction(
        receivable_id=int(transmission.receivable_id),
        creditor_company_id=int(transmission.creditor_company_id),
        transmission_id=int(transmission.id),
        action_type=ACTION_TRANSMISSION_CANCELLED,
        occurred_at=datetime.now(UTC),
        requested_by_user_id=int(requested_by_user_id),
        payload_snapshot=None,
    )
    db.session.add(action)
    db.session.flush()
    return transmission


def serialize_transmission(
    row: PortalReceivableCollectionTransmission,
) -> dict[str, Any]:
    return {
        "id": row.id,
        "receivable_id": row.receivable_id,
        "creditor_company_id": row.creditor_company_id,
        "transmission_type": row.transmission_type,
        "status": row.status,
        "collection_prepared_event_id": row.collection_prepared_event_id,
        "balance_snapshot": float(row.balance_snapshot),
        "claim_principal_snapshot": float(row.claim_principal_snapshot),
        "currency_snapshot": row.currency_snapshot,
        "invoice_reference_snapshot": row.invoice_reference_snapshot,
        "claim_reason_snapshot": row.claim_reason_snapshot,
        "export_hash": row.export_hash,
        "pursuit_jurisdiction": row.pursuit_jurisdiction,
        "creditor_confirmed": bool(row.creditor_confirmed),
        "requested_at": row.requested_at.isoformat() if row.requested_at else None,
        "requested_by_user_id": row.requested_by_user_id,
        "disclaimer": (
            "Draft de travail uniquement — aucune transmission externe effectuée."
        ),
    }


def serialize_pursuit_readiness(result: PursuitReadiness) -> dict[str, Any]:
    return {
        "state": result.state,
        "reasons": list(result.reasons),
        "audit": result.audit,
    }
