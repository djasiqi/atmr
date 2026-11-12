"""Dossier de facture patient. Aucune facture émise, aucune signature réelle.

L'acceptation enregistrée n'est pas juridiquement liante. L'attestation
de transport ne devient pas un consentement. Une case « patient payeur »
ne crée pas la dette.
"""

from __future__ import annotations

import hashlib
from decimal import Decimal
from typing import Any

from sqlalchemy import Integer, bindparam, select, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.exc import IntegrityError

from application.settlement.production_gates import (
    amount_due,
    authorize_settlement_history_read,
    debtor_label,
    decision_version,
    fingerprint,
    refuse_booking,
    reject_client_overrides,
    require_prep,
    status_of,
)
from domain.end_of_trip_settlement.patient_invoice_procedure import DOCUMENT_VERSION
from domain.end_of_trip_settlement.production_prep_barrier import PrepRefusal
from ext import db
from models import Company
from models.enums import BookingStatus
from models.settlement_production import PatientBillingDossier, PatientBillingEvent

_BANNER = "PRÉPARATION — FACTURE NON ÉMISE"
_KINDS = {
    "prior_information": ("prior_presented", "preparing"),
    "acceptance": ("acceptance_unapproved", "administrative"),
    "refusal": ("refused", "administrative"),
    "representative": ("representative", "administrative"),
    "unable": ("administrative", "administrative"),
    "not_understood": ("administrative", "administrative"),
}


def open_dossier(
    booking: Any, *, driver_id: int, company_id: int, actor_id: int | None
) -> dict[str, Any]:
    require_prep()
    refuse_booking(booking, driver_id=driver_id, company_id=company_id)
    existing = _lock(int(booking.id))
    if existing is not None:
        view = _view(existing)
        view["replay"] = True
        return view
    row = PatientBillingDossier(
        booking_id=int(booking.id),
        company_id=int(company_id),
        driver_id=int(driver_id),
        debtor_label=debtor_label(booking),
        amount_chf=amount_due(booking),
        currency="CHF",
        contractual_basis="missing",
        document_version=DOCUMENT_VERSION,
        document_approved=False,
        legally_binding=False,
        real_signature_collected=False,
        invoice_issued=False,
        attestation_used_as_consent=False,
        production_activation=False,
        state="preparing",
        decision_version=decision_version(booking),
        recipe_contract_fixture=False,
        representative_powers_verified=False,
        representative_decision="absent",
    )
    db.session.add(row)
    try:
        db.session.flush()
    except IntegrityError:
        db.session.rollback()
        winner = _lock(int(booking.id))
        if winner is None:
            raise
        view = _view(winner)
        view["replay"] = True
        return view
    _event(
        row,
        key=f"open-{row.id}",
        action="open",
        before="preparing",
        after="preparing",
        actor_role="driver",
        actor_id=actor_id,
        payload={"request_fingerprint": "open"},
    )
    db.session.commit()
    return _view(row)


def record_event(
    booking: Any,
    payload: dict,
    *,
    driver_id: int,
    company_id: int,
    actor_id: int | None,
) -> dict[str, Any]:
    require_prep()
    refuse_booking(booking, driver_id=driver_id, company_id=company_id)
    due = amount_due(booking)
    reject_client_overrides(payload, due=due, company_id=company_id)
    action = str(payload.get("action") or "")
    if action == "mark_patient_payer":
        raise PrepRefusal("case_cochee_insuffisante", 400)
    if action == "use_attestation_as_consent":
        raise PrepRefusal("attestation_distincte", 409)
    if action == "issue_invoice":
        raise PrepRefusal("emission_desactivee", 409)
    if action == "approve_representative":
        raise PrepRefusal("representation_non_approuvee", 403)
    if payload.get("powers_verified") is True:
        raise PrepRefusal("representation_non_approuvee", 403)
    if action == "declare_recipe_fixture":
        return _recipe_fixture(
            booking,
            payload,
            driver_id=driver_id,
            company_id=company_id,
            actor_id=actor_id,
        )
    if action == "prepare_draft":
        return prepare_invoice_draft(
            booking,
            payload,
            driver_id=driver_id,
            company_id=company_id,
            actor_id=actor_id,
        )
    if action not in _KINDS:
        raise PrepRefusal("action_inconnue", 400)
    key = str(payload.get("idempotency_key") or "").strip()
    if not key:
        raise PrepRefusal("cle_absente", 400)
    if payload.get("signature_image"):
        raise PrepRefusal("signature_reelle_interdite", 400)
    row = _lock(int(booking.id))
    if row is None:
        raise PrepRefusal("dossier_absent", 404)
    replay = _replay(row, key, payload)
    if replay is not None:
        return replay
    if decision_version(booking) != row.decision_version:
        raise PrepRefusal("decision_obsolete", 409)
    basis, state = _KINDS[action]
    before = row.state
    row.contractual_basis = basis
    row.state = state
    if action == "representative":
        row.representative_name = (
            str(payload.get("representative_name") or "")[:160] or None
        )
        row.representative_kind = (
            str(payload.get("representative_kind") or "")[:40] or None
        )
        row.representative_proof_expected = str(
            payload.get("representative_proof_expected") or "justificatif de pouvoir"
        )[:160]
        row.representative_powers_verified = False
        row.representative_decision = "pending_review"
    presented = str(payload.get("presented_text") or DOCUMENT_VERSION)
    _event(
        row,
        key=key,
        action=action,
        before=before,
        after=state,
        actor_role="driver",
        actor_id=actor_id,
        payload={
            "request_fingerprint": fingerprint({**payload, "kind": action}),
            "kind": action,
            "document_version": DOCUMENT_VERSION,
            "document_approved": False,
            "legally_binding": False,
            "real_signature_collected": False,
            "signature_image": None,
            "replaces_attestation": False,
            "creates_debt": False,
            "presented_sha256": hashlib.sha256(presented.encode()).hexdigest(),
            "signatory_role": payload.get("signatory_role") or "patient",
        },
    )
    db.session.commit()
    return _view(row)


def read_dossier(
    booking: Any, *, driver_id: int, company_id: int
) -> dict[str, Any] | None:
    """Consulte un dossier patient déjà enregistré sans ouvrir une écriture."""
    require_prep()
    authorize_settlement_history_read(
        booking, driver_id=driver_id, company_id=company_id
    )
    row = db.session.scalar(
        select(PatientBillingDossier).where(
            PatientBillingDossier.booking_id == int(booking.id)
        )
    )
    if row is None:
        return None
    return _view(row)


def _recipe_fixture(
    booking: Any,
    payload: dict,
    *,
    driver_id: int,
    company_id: int,
    actor_id: int | None,
) -> dict[str, Any]:
    key = str(payload.get("idempotency_key") or "").strip()
    if not key:
        raise PrepRefusal("cle_absente", 400)
    if driver_id <= 0 or company_id <= 0:
        raise PrepRefusal("chauffeur_non_affecte", 403)
    if payload.get("recipe_fixture") != "fictif-identifie":
        raise PrepRefusal("fondement_insuffisant", 409)
    row = _lock(int(booking.id))
    if row is None:
        raise PrepRefusal("dossier_absent", 404)
    replay = _replay(row, key, payload)
    if replay is not None:
        return replay
    before = row.state
    row.recipe_contract_fixture = True
    row.legally_binding = False
    row.document_approved = False
    _event(
        row,
        key=key,
        action="declare_recipe_fixture",
        before=before,
        after=row.state,
        actor_role="driver",
        actor_id=actor_id,
        payload={
            "request_fingerprint": fingerprint(
                {**payload, "kind": "declare_recipe_fixture"}
            ),
            "recipe_fixture": "fictif-identifie",
            "legally_binding": False,
            "creates_debt": False,
        },
    )
    db.session.commit()
    return _view(row)


def prepare_invoice_draft(
    booking: Any,
    payload: dict,
    *,
    driver_id: int,
    company_id: int,
    actor_id: int | None,
) -> dict[str, Any]:
    """Note le dossier. Ne crée plus de spécimen et ne pose aucun verrou."""
    require_prep()
    refuse_booking(booking, driver_id=driver_id, company_id=company_id)
    due = amount_due(booking)
    reject_client_overrides(payload, due=due, company_id=company_id)
    key = str(payload.get("idempotency_key") or "").strip()
    if not key:
        raise PrepRefusal("cle_absente", 400)
    row = _lock(int(booking.id))
    if row is None:
        raise PrepRefusal("dossier_absent", 404)
    replay = _replay(row, key, payload)
    if replay is not None:
        return replay
    if not row.recipe_contract_fixture:
        raise PrepRefusal("fondement_insuffisant", 409)
    if row.draft_invoice_id is not None:
        view = _view(row)
        view["replay"] = True
        return view
    if decision_version(booking) != row.decision_version:
        raise PrepRefusal("decision_obsolete", 409)
    from application.invoices.confirmed_payment_coverage import (
        classify_coverage,
        confirmed_amounts_by_booking,
    )

    paid = confirmed_amounts_by_booking([int(booking.id)]).get(
        int(booking.id), Decimal("0.00")
    )
    coverage = classify_coverage(due, paid)
    if coverage in {"full", "surplus"}:
        raise PrepRefusal("deja_regle", 409)
    if coverage == "ambiguous":
        raise PrepRefusal("couverture_ambigue", 409)
    from models.booking import Booking

    locked_booking = db.session.scalar(
        select(Booking).where(Booking.id == int(booking.id)).with_for_update()
    )
    if locked_booking is not None and getattr(locked_booking, "invoice_line_id", None):
        raise PrepRefusal("deja_facture", 409)
    _event(
        row,
        key=key,
        action="prepare_draft",
        before=row.state,
        after=row.state,
        actor_role="driver",
        actor_id=actor_id,
        payload={
            "request_fingerprint": fingerprint({**payload, "kind": "prepare_draft"}),
            "draft_invoice_id": None,
            "specimen_created": False,
            "invoice_line_locked": False,
            "issued": False,
        },
    )
    db.session.commit()
    view = _view(row)
    view["draft_created"] = False
    view["specimen_created"] = False
    return view


def _meta(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _specimen_reason(invoice: Any, line: Any, booking: Any) -> str:
    """Décide sans deviner. Seul un accord complet autorise la libération."""
    meta = _meta(invoice["meta"])
    number_ok = str(invoice["invoice_number"]).startswith("SPECIMEN-")
    specimen_ok = meta.get("specimen") is True
    payable_ok = meta.get("payable") is False
    draft_ok = str(invoice["status"]) == "draft"
    if not draft_ok and number_ok and specimen_ok:
        return "facture_emise"
    link_ok = (
        booking is not None
        and line is not None
        and booking["invoice_line_id"] is not None
        and int(booking["invoice_line_id"]) == int(line["id"])
        and line["reservation_id"] is not None
        and int(line["reservation_id"]) == int(booking["id"])
        and int(line["invoice_id"]) == int(invoice["id"])
        and int(booking["company_id"]) == int(invoice["company_id"])
    )
    if number_ok and specimen_ok and payable_ok and draft_ok and link_ok:
        return "release"
    if (
        booking is not None
        and booking["invoice_line_id"] is None
        and meta.get("lock_released") is True
    ):
        return "deja_libere"
    return "ambigu"


def release_specimen_locks(company_id: int) -> dict[str, Any]:
    """Retire le lien d'un spécimen concordant. N'efface ni facture ni dossier.

    N'est pas appelée par la génération d'une facture.
    """
    company_id = int(company_id)
    invoices = (
        db.session.execute(
            text(
                """
                SELECT id, company_id, invoice_number, status::text AS status, meta
                FROM invoices
                WHERE company_id = :company_id
                  AND (
                    invoice_number LIKE 'SPECIMEN-%'
                    OR COALESCE(meta->'specimen', 'false'::jsonb) = 'true'::jsonb
                  )
                ORDER BY id
                FOR UPDATE
                """
            ),
            {"company_id": company_id},
        )
        .mappings()
        .all()
    )
    if not invoices:
        db.session.commit()
        return {
            "company_id": company_id,
            "released": [],
            "ignored": [],
            "wrote": False,
        }
    invoice_ids = [int(row["id"]) for row in invoices]
    lines = (
        db.session.execute(
            text(
                """
                SELECT id, invoice_id, reservation_id
                FROM invoice_lines
                WHERE invoice_id = ANY(:invoice_ids)
                ORDER BY id
                FOR UPDATE
                """
            ).bindparams(bindparam("invoice_ids", type_=ARRAY(Integer))),
            {"invoice_ids": invoice_ids},
        )
        .mappings()
        .all()
    )
    line_ids = [int(row["id"]) for row in lines]
    reservation_ids = [
        int(row["reservation_id"]) for row in lines if row["reservation_id"] is not None
    ]
    bookings = []
    if line_ids or reservation_ids:
        bookings = (
            db.session.execute(
                text(
                    """
                    SELECT id, company_id, invoice_line_id, price_amount
                    FROM booking
                    WHERE company_id = :company_id
                      AND (
                        invoice_line_id = ANY(:line_ids)
                        OR id = ANY(:reservation_ids)
                      )
                    ORDER BY id
                    FOR UPDATE
                    """
                ).bindparams(
                    bindparam("line_ids", type_=ARRAY(Integer)),
                    bindparam("reservation_ids", type_=ARRAY(Integer)),
                ),
                {
                    "company_id": company_id,
                    "line_ids": line_ids or [0],
                    "reservation_ids": reservation_ids or [0],
                },
            )
            .mappings()
            .all()
        )
    by_line = {
        int(row["invoice_line_id"]): row
        for row in bookings
        if row["invoice_line_id"] is not None
    }
    by_id = {int(row["id"]): row for row in bookings}
    lines_by_invoice: dict[int, list[Any]] = {}
    for line in lines:
        lines_by_invoice.setdefault(int(line["invoice_id"]), []).append(line)
    released: list[dict[str, int]] = []
    ignored: list[dict[str, Any]] = []
    for invoice in invoices:
        owned = lines_by_invoice.get(int(invoice["id"]), [])
        if not owned:
            ignored.append({"invoice_id": int(invoice["id"]), "reason": "ambigu"})
            continue
        released_one = False
        for line in owned:
            pointed = by_line.get(int(line["id"]))
            reserved = (
                by_id.get(int(line["reservation_id"]))
                if line["reservation_id"] is not None
                else None
            )
            if (
                pointed is not None
                and reserved is not None
                and int(pointed["id"]) != int(reserved["id"])
            ):
                ignored.append(
                    {
                        "invoice_id": int(invoice["id"]),
                        "booking_id": int(pointed["id"]),
                        "reason": "ambigu",
                    }
                )
                continue
            booking = pointed or reserved
            reason = _specimen_reason(invoice, line, booking)
            if reason != "release":
                ignored.append(
                    {
                        "invoice_id": int(invoice["id"]),
                        "booking_id": None if booking is None else int(booking["id"]),
                        "reason": reason,
                    }
                )
                continue
            updated = db.session.execute(
                text(
                    """
                    UPDATE booking
                    SET invoice_line_id = NULL
                    WHERE id = :booking_id
                      AND company_id = :company_id
                      AND invoice_line_id = :line_id
                    """
                ),
                {
                    "booking_id": int(booking["id"]),
                    "company_id": company_id,
                    "line_id": int(line["id"]),
                },
            )
            if updated.rowcount != 1:
                ignored.append(
                    {
                        "invoice_id": int(invoice["id"]),
                        "booking_id": int(booking["id"]),
                        "reason": "lien_change",
                    }
                )
                continue
            released.append(
                {
                    "booking_id": int(booking["id"]),
                    "invoice_id": int(invoice["id"]),
                    "line_id": int(line["id"]),
                }
            )
            released_one = True
        if released_one:
            marked = db.session.execute(
                text(
                    """
                    UPDATE invoices
                    SET meta = COALESCE(meta, '{}'::jsonb)
                      || '{"blocking": false, "lock_released": true}'::jsonb
                    WHERE id = :invoice_id
                      AND company_id = :company_id
                      AND status::text = 'draft'
                      AND invoice_number LIKE 'SPECIMEN-%'
                      AND COALESCE(meta->'specimen', 'false'::jsonb) = 'true'::jsonb
                      AND COALESCE(meta->'payable', 'true'::jsonb) = 'false'::jsonb
                    """
                ),
                {"invoice_id": int(invoice["id"]), "company_id": company_id},
            )
            if marked.rowcount != 1:
                db.session.rollback()
                raise PrepRefusal("specimen_non_marque", 409)
    db.session.commit()
    return {
        "company_id": company_id,
        "released": released,
        "ignored": ignored,
        "wrote": bool(released),
    }


def refuse_issue(booking: Any, *, company_id: int) -> None:
    """L'émission reste fermée, même si le dossier existe."""
    require_prep()
    row = _lock(int(booking.id))
    if row is None or int(row.company_id) != int(company_id):
        raise PrepRefusal("dossier_absent", 404)
    raise PrepRefusal("emission_desactivee", 409)


_REVIEW_RESULTS = frozenset(
    {"pending_review", "documented_scope", "refused", "insufficient_proof"}
)


def review_representative(
    dossier_id: int,
    payload: dict,
    *,
    company_id: int,
    actor_id: int,
) -> dict[str, Any]:
    """Note administrative. N'approuve pas un pouvoir et ne crée pas l'acceptation."""
    require_prep()
    if payload.get("powers_verified") is True:
        raise PrepRefusal("representation_non_approuvee", 403)
    if payload.get("contractual_basis") or payload.get("creates_acceptance"):
        raise PrepRefusal("acceptation_non_fabriquee", 403)
    result = str(payload.get("result") or "")
    if result not in _REVIEW_RESULTS:
        raise PrepRefusal("revue_inconnue", 400)
    key = str(payload.get("idempotency_key") or "").strip()
    if not key:
        raise PrepRefusal("cle_absente", 400)
    row = db.session.get(PatientBillingDossier, int(dossier_id))
    if row is None or int(row.company_id) != int(company_id):
        raise PrepRefusal("dossier_absent", 404)
    body = {**payload, "action": "review_representative", "disposition": result}
    replay = _replay(row, key, body)
    if replay is not None:
        return replay
    before = row.state
    _event(
        row,
        key=key,
        action="review_representative",
        before=before,
        after=before,
        actor_role="company",
        actor_id=actor_id,
        payload={
            "request_fingerprint": fingerprint(
                {**body, "kind": "review_representative"}
            ),
            "result": result,
            "document_reference": str(payload.get("document_reference") or "")[:160],
            "scope_note": str(payload.get("scope_note") or "")[:240],
            "relation": str(payload.get("relation") or "")[:80],
            "powers_verified": False,
            "creates_debt": False,
            "legally_binding": False,
        },
    )
    db.session.commit()
    return _view(row)


def list_company(company_id: int) -> dict[str, Any]:
    require_prep()
    rows = db.session.scalars(
        select(PatientBillingDossier)
        .where(PatientBillingDossier.company_id == int(company_id))
        .order_by(PatientBillingDossier.id)
    ).all()
    return {
        "banner": _BANNER,
        "invoice_issuance": False,
        "real_signature_collection": False,
        "qr_bill": False,
        "cases": [_view(row) for row in rows],
    }


_SPECIMEN_IBAN = "CH4431999123000889012"
_QR_BANNER = "SPECIMEN — NON EMIS — NE PAS PAYER"


def specimen_qr(row: PatientBillingDossier) -> dict[str, Any]:
    """Structure QR de recette. Aucune image scannable, aucune référence payable."""
    return {
        "payable": False,
        "reference_type": "NON",
        "reference": None,
        "iban_role": "specimen_six",
        "iban": _SPECIMEN_IBAN,
        "amount_chf": f"{Decimal(row.amount_chf):.2f}",
        "currency": "CHF",
        "banner": _QR_BANNER,
        "spc_payload": None,
    }


def specimen_pdf(row_view: dict[str, Any]) -> str:
    from application.settlement.cash_ledger import specimen_pdf as _pdf

    lines = [
        "SPECIMEN — NON EMIS — NE PAS PAYER",
        f"Creancier : {row_view.get('company_name')}",
        f"Debiteur : {row_view['debtor_label']}",
        f"Transport : {row_view['booking_id']}",
        f"Montant : {row_view['amount_chf']} CHF",
        "TVA : non configuree, taux non presume",
        f"Fondement : {row_view['contractual_basis']}",
        "QR absent",
        "QR non payable",
        "Signature reelle absente",
    ]
    return _pdf(lines)


def _lock(booking_id: int) -> PatientBillingDossier | None:
    return db.session.scalar(
        select(PatientBillingDossier)
        .where(PatientBillingDossier.booking_id == booking_id)
        .with_for_update()
    )


def _replay(
    row: PatientBillingDossier, key: str, payload: dict
) -> dict[str, Any] | None:
    event = db.session.scalar(
        select(PatientBillingEvent).where(
            PatientBillingEvent.dossier_id == row.id,
            PatientBillingEvent.idempotency_key == key,
        )
    )
    if event is None:
        return None
    if event.payload.get("request_fingerprint") != fingerprint(
        {**payload, "kind": payload.get("action")}
    ):
        raise PrepRefusal("conflit_idempotence", 409)
    view = _view(row)
    view["replay"] = True
    return view


def _event(
    row: PatientBillingDossier,
    *,
    key: str,
    action: str,
    before: str,
    after: str,
    actor_role: str,
    actor_id: int | None,
    payload: dict,
) -> None:
    db.session.add(
        PatientBillingEvent(
            dossier_id=row.id,
            idempotency_key=key,
            action=action,
            from_state=before,
            to_state=after,
            actor_role=actor_role,
            actor_id=actor_id,
            payload=payload,
        )
    )


def _representative_review(row: PatientBillingDossier, events: list[Any]) -> str:
    review = "absent" if row.representative_decision == "absent" else "pending_review"
    for event in events:
        if event.action == "review_representative":
            review = str(event.payload.get("result") or review)
    return review


def _representative_reviewer(events: list[Any]) -> int | None:
    reviewer = None
    for event in events:
        if event.action == "review_representative":
            reviewer = event.actor_id
    return reviewer


def _followup_payload(events: list[Any]) -> dict[str, Any] | None:
    found = None
    for event in events:
        if (
            event.action in {"followup", "reconcile"}
            and isinstance(event.payload, dict)
            and "balance_chf" in event.payload
        ):
            found = event.payload
    return found


def _view(row: PatientBillingDossier) -> dict[str, Any]:
    from models.booking import Booking

    company = db.session.get(Company, int(row.company_id))
    booking = db.session.get(Booking, int(row.booking_id))
    events = db.session.scalars(
        select(PatientBillingEvent)
        .where(PatientBillingEvent.dossier_id == row.id)
        .order_by(PatientBillingEvent.id)
    ).all()
    follow = _followup_payload(events)
    from application.settlement.leg_settlement import leg_public

    legs = {} if booking is None else leg_public(booking)
    return {
        "id": row.id,
        "booking_id": row.booking_id,
        "client_id": None
        if booking is None or booking.client_id is None
        else int(booking.client_id),
        "company_id": row.company_id,
        "company_name": company.name if company is not None else None,
        "driver_id": row.driver_id,
        "debtor_label": row.debtor_label,
        "amount_chf": f"{Decimal(row.amount_chf):.2f}",
        "currency": "CHF",
        "contractual_basis": row.contractual_basis,
        "document_version": row.document_version,
        "document_approved": False,
        "legally_binding": False,
        "real_signature_collected": False,
        "invoice_issued": False,
        "attestation_used_as_consent": False,
        "qr_bill": False,
        "qr_specimen": None if row.draft_invoice_id is None else specimen_qr(row),
        "draft_invoice_id": row.draft_invoice_id,
        "draft_status": None if row.draft_invoice_id is None else "draft",
        "recipe_contract_fixture": bool(row.recipe_contract_fixture),
        "representative_name": row.representative_name,
        "representative_kind": row.representative_kind,
        "representative_proof_expected": row.representative_proof_expected,
        "representative_powers_verified": False,
        "representative_decision": row.representative_decision,
        "representative_review": _representative_review(row, events),
        "representative_reviewed_by": _representative_reviewer(events),
        "state": row.state,
        "banner": _BANNER,
        "production_activation": False,
        "policy_status": "DRAFT",
        "history": [
            {"action": event.action, "to_state": event.to_state} for event in events
        ],
        "due_chf": None if follow is None else follow.get("due_chf"),
        "recognized_chf": None if follow is None else follow.get("recognized_chf"),
        "balance_chf": None if follow is None else follow.get("balance_chf"),
        "real_payment": False,
        "invoice_confirmation_required": (
            True
            if follow is None
            else bool(follow.get("invoice_confirmation_required"))
        ),
        "mandate_kind": None if follow is None else follow.get("mandate_kind"),
        "mandate_id": None if follow is None else follow.get("mandate_id"),
        "admin_review_required": bool(follow and follow.get("admin_review_required")),
        "preparation_label": None if follow is None else follow.get("label"),
        **legs,
    }


def reconcile_followup_balance(booking: Any) -> bool:
    """Recalcule le solde si un règlement simulé arrive après la préparation.

    Une seule ligne de dossier. Le montant dû ne devient pas une nouvelle créance.
    """
    row = _lock(int(booking.id))
    if row is None:
        return False
    if getattr(booking, "invoice_line_id", None) is not None:
        raise PrepRefusal("facture_deja_emise", 409)
    due = amount_due(booking)
    recognized, refused = _recognized_cash(int(booking.id))
    balance = due - recognized
    if balance < 0:
        balance = Decimal("0.00")
    situation = _situation(booking)
    basis, state, prompt, label = _followup_class(
        situation, recognized=recognized, due=due, refused=refused
    )
    events = db.session.scalars(
        select(PatientBillingEvent)
        .where(PatientBillingEvent.dossier_id == row.id)
        .order_by(PatientBillingEvent.id)
    ).all()
    current = _followup_payload(events)
    due_text = f"{due:.2f}"
    recognized_text = f"{recognized:.2f}"
    balance_text = f"{balance:.2f}"
    if (
        current is not None
        and current.get("due_chf") == due_text
        and current.get("recognized_chf") == recognized_text
        and current.get("balance_chf") == balance_text
        and current.get("label") == label
    ):
        return False
    key = f"followup-reconcile-{int(booking.id)}-{recognized_text}"
    already = db.session.scalar(
        select(PatientBillingEvent.id).where(
            PatientBillingEvent.dossier_id == row.id,
            PatientBillingEvent.idempotency_key == key,
        )
    )
    if already is not None:
        return False
    signature = (
        situation.get("signature")
        if isinstance(situation.get("signature"), dict)
        else {}
    )
    representative = bool(signature.get("admin_review_required"))
    previous = row.state
    row.state = state
    if not representative:
        row.contractual_basis = basis
    _event(
        row,
        key=key,
        action="reconcile",
        before=previous,
        after=state,
        actor_role="company",
        actor_id=None,
        payload={
            "request_fingerprint": fingerprint(
                {"kind": "reconcile", "booking_id": int(booking.id)}
            ),
            "due_chf": due_text,
            "recognized_chf": recognized_text,
            "balance_chf": balance_text,
            "recognition": "simulation",
            "real_payment": False,
            "invoice_issued": False,
            "legally_effective": False,
            "invoice_confirmation_required": prompt,
            "mandate_kind": situation.get("kind"),
            "mandate_id": situation.get("mandate_id"),
            "admin_review_required": representative,
            "label": label,
            "creates_debt": False,
            "counted_once": True,
        },
    )
    db.session.flush()
    return True


def ensure_patient_followup(
    booking_id: int, *, company_id: int, actor_id: int | None
) -> dict[str, Any]:
    """Prépare un seul dossier de suivi. N'insère aucune facture."""
    require_prep()
    from models.booking import Booking

    booking = db.session.get(Booking, int(booking_id))
    if booking is None or int(booking.company_id) != int(company_id):
        raise PrepRefusal("reservation_absente", 404)
    _refuse_followup_target(booking)
    existing = _lock(int(booking.id))
    if existing is not None:
        if int(existing.company_id) != int(company_id):
            raise PrepRefusal("autre_entreprise", 404)
        if reconcile_followup_balance(booking):
            db.session.commit()
        view = _view(existing)
        view["replay"] = True
        return view
    due = amount_due(booking)
    price_before = booking.price_amount
    debtor_before = booking.billed_to_type
    recognized, refused = _recognized_cash(int(booking.id))
    balance = due - recognized
    if balance < 0:
        balance = Decimal("0.00")
    situation = _situation(booking)
    basis, state, prompt, label = _followup_class(
        situation, recognized=recognized, due=due, refused=refused
    )
    if booking.driver_id is None:
        raise PrepRefusal("chauffeur_non_affecte", 403)
    signature = (
        situation.get("signature")
        if isinstance(situation.get("signature"), dict)
        else {}
    )
    representative = bool(signature.get("admin_review_required"))
    row = PatientBillingDossier(
        booking_id=int(booking.id),
        company_id=int(company_id),
        driver_id=int(booking.driver_id),
        debtor_label=debtor_label(booking),
        amount_chf=due,
        currency="CHF",
        contractual_basis="representative" if representative else basis,
        document_version=str(situation.get("document_version") or DOCUMENT_VERSION),
        document_approved=False,
        legally_binding=False,
        real_signature_collected=False,
        invoice_issued=False,
        attestation_used_as_consent=False,
        production_activation=False,
        state=state,
        decision_version=decision_version(booking),
        recipe_contract_fixture=False,
        representative_name=(
            None
            if not signature.get("representative_name")
            else str(signature.get("representative_name"))[:160]
        ),
        representative_kind=(
            None
            if not signature.get("representative_quality")
            else str(signature.get("representative_quality"))[:40]
        ),
        representative_proof_expected="justificatif de pouvoir"
        if representative
        else None,
        representative_powers_verified=False,
        representative_decision="pending_review" if representative else "absent",
    )
    db.session.add(row)
    try:
        db.session.flush()
    except IntegrityError:
        db.session.rollback()
        winner = _lock(int(booking.id))
        if winner is None:
            raise
        view = _view(winner)
        view["replay"] = True
        return view
    _event(
        row,
        key=f"followup-{int(booking.id)}",
        action="followup",
        before=state,
        after=state,
        actor_role="company",
        actor_id=actor_id,
        payload={
            "request_fingerprint": fingerprint(
                {"kind": "followup", "booking_id": int(booking.id)}
            ),
            "due_chf": f"{due:.2f}",
            "recognized_chf": f"{recognized:.2f}",
            "balance_chf": f"{balance:.2f}",
            "recognition": "simulation",
            "real_payment": False,
            "invoice_issued": False,
            "legally_effective": False,
            "invoice_confirmation_required": prompt,
            "mandate_kind": situation.get("kind"),
            "mandate_id": situation.get("mandate_id"),
            "admin_review_required": representative,
            "label": label,
            "creates_debt": False,
        },
    )
    db.session.flush()
    if booking.price_amount != price_before or booking.billed_to_type != debtor_before:
        db.session.rollback()
        raise PrepRefusal("montant_modifie", 500)
    if getattr(booking, "invoice_line_id", None) is not None:
        db.session.rollback()
        raise PrepRefusal("facture_deja_emise", 409)
    db.session.commit()
    view = _view(row)
    view["replay"] = False
    return view


def _dossier_presence(booking_id: int) -> str:
    """absent, prepared, pending. unavailable si l'état persistant n'est pas lisible."""
    row = db.session.scalar(
        select(PatientBillingDossier).where(
            PatientBillingDossier.booking_id == int(booking_id)
        )
    )
    if row is None:
        return "absent"
    if row.representative_decision == "pending_review":
        return "pending"
    if row.state in {"preparing", "administrative", "closed"}:
        return "prepared"
    return "unavailable"


def hint_label(
    situation: dict[str, Any],
    *,
    recognized: Decimal,
    due: Decimal,
    refused: bool,
    dossier_state: str,
) -> str:
    """Un mandat applicable n'est pas un dossier. « Préparé » exige la ligne persistante."""
    _basis, _state, _prompt, persisted = _followup_class(
        situation, recognized=recognized, due=due, refused=refused
    )
    if dossier_state != "absent":
        return persisted
    if "Dossier préparé" in persisted:
        return "Mandat signé. Facturation prévue, dossier non préparé. À facturer."
    if persisted.startswith("Dossier de suivi"):
        balance = due - recognized
        if balance < 0:
            balance = Decimal("0.00")
        return f"Vérification nécessaire. Solde restant {balance:.2f} CHF. À facturer."
    return persisted


def billing_followup_hint(booking_id: int) -> dict[str, Any] | None:
    """Lecture seule. Ne prépare aucun dossier."""
    from domain.end_of_trip_settlement.recipe_simulation import (
        recipe_simulation_read_allowed,
    )
    from models.booking import Booking

    if not recipe_simulation_read_allowed():
        return None
    booking = db.session.get(Booking, int(booking_id))
    if booking is None or _blocked_target(booking) is not None:
        return None
    try:
        due = amount_due(booking)
    except PrepRefusal:
        return None
    recognized, refused = _recognized_cash(int(booking.id))
    balance = due - recognized
    if balance < 0:
        balance = Decimal("0.00")
    situation = _situation(booking)
    dossier_state = _dossier_presence(int(booking.id))
    _basis, _state, prompt, _persisted = _followup_class(
        situation, recognized=recognized, due=due, refused=refused
    )
    label = hint_label(
        situation,
        recognized=recognized,
        due=due,
        refused=refused,
        dossier_state=dossier_state,
    )
    return {
        "invoice_confirmation_required": prompt,
        "label": None if prompt else label,
        "dossier_state": dossier_state,
        "due_chf": f"{due:.2f}",
        "recognized_chf": f"{recognized:.2f}",
        "balance_chf": f"{balance:.2f}",
        "real_payment": False,
        "invoice_issued": False,
        "legally_effective": False,
        "mandate_kind": situation.get("kind"),
    }


def _refuse_followup_target(booking: Any) -> None:
    reason = _blocked_target(booking)
    if reason == "round_trip":
        raise PrepRefusal("aller_retour_reserve", 409)
    if reason == "clinic":
        raise PrepRefusal("institution_non_modifiee", 409)
    if reason == "not_patient":
        raise PrepRefusal("reservation_non_concernee", 409)
    if reason == "cancelled":
        raise PrepRefusal("transport_annule", 409)
    if reason == "issued":
        raise PrepRefusal("facture_deja_emise", 409)


def _blocked_target(booking: Any) -> str | None:
    from application.settlement.leg_settlement import (
        linked_round_trip,
        separate_leg_ready,
    )

    if linked_round_trip(booking) and not separate_leg_ready(booking):
        return "round_trip"
    payer = str(
        getattr(booking.billed_to_type, "value", booking.billed_to_type) or "patient"
    )
    payer = payer.strip().lower()
    if payer == "clinic":
        return "clinic"
    if payer != "patient":
        return "not_patient"
    if status_of(booking) == BookingStatus.CANCELED:
        return "cancelled"
    if getattr(booking, "invoice_line_id", None) is not None:
        return "issued"
    return None


def _recognized_cash(booking_id: int) -> tuple[Decimal, bool]:
    from models.settlement_production import CashCollection

    rows = db.session.scalars(
        select(CashCollection).where(CashCollection.booking_id == int(booking_id))
    ).all()
    if not rows:
        return Decimal("0.00"), False
    refused = any(row.disposition == "refused" for row in rows) and all(
        row.net_chf is None or Decimal(row.net_chf) <= 0 for row in rows
    )
    from application.settlement.cash_ledger import _recognized

    return _recognized(int(booking_id)), refused


def _situation(booking: Any) -> dict[str, Any]:
    from application.settlement.mandate_registry import assess_for_booking

    if booking.client_id is None:
        return {
            "kind": "absent",
            "mandate_id": None,
            "document_version": None,
            "signature": None,
        }
    try:
        assessed = assess_for_booking(int(booking.company_id), int(booking.id))
    except Exception:
        return {
            "kind": "absent",
            "mandate_id": None,
            "document_version": None,
            "signature": None,
        }
    mandate = (
        assessed.get("mandate") if isinstance(assessed.get("mandate"), dict) else {}
    )
    return {
        "kind": str(assessed.get("kind") or "absent"),
        "mandate_id": mandate.get("id"),
        "document_version": mandate.get("document_version"),
        "signature": mandate.get("demonstration_signature"),
    }


def _followup_class(
    situation: dict[str, Any], *, recognized: Decimal, due: Decimal, refused: bool
) -> tuple[str, str, bool, str]:
    kind = str(situation.get("kind") or "absent")
    signature = (
        situation.get("signature")
        if isinstance(situation.get("signature"), dict)
        else {}
    )
    if refused:
        return (
            "refused",
            "administrative",
            False,
            "Refus enregistré. Le solde reste à la décision de l'entreprise. Aucune facture.",
        )
    if recognized >= due:
        return (
            "administrative",
            "administrative",
            False,
            "Solde nul. Le montant reconnu n'est pas facturé de nouveau.",
        )
    if kind == "applicable_demonstration":
        return (
            "prior_presented",
            "preparing",
            False,
            "Mandat signé. Dossier préparé. Facture préparée.",
        )
    if kind == "specimen":
        if signature.get("admin_review_required"):
            return (
                "representative",
                "preparing",
                False,
                "Mandat signé. Vérification nécessaire.",
            )
        return (
            "acceptance_unapproved",
            "preparing",
            False,
            "Mandat signé. Facture préparée.",
        )
    if kind in {"expired", "amendment_required"}:
        return (
            "administrative",
            "administrative",
            False,
            "Couverture inutilisable. L'entreprise décide du solde. Aucune facture.",
        )
    balance = due - recognized
    return (
        "missing",
        "administrative",
        True,
        f"Dossier de suivi. Solde restant {balance:.2f} CHF. À facturer.",
    )
