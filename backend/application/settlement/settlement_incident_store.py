"""Persistance PostgreSQL des signalements de non-règlement et cartes externes."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from sqlalchemy.exc import IntegrityError

from domain.end_of_trip_settlement.settlement_incident import (
    _ADMIN_TRANSITIONS,
    MOTIF_AUTRE,
    MOTIFS,
    STATUS_A_TRAITER,
    STATUSES,
)
from ext import db
from models.settlement_non_payment_incident import (
    ExternalCardSettlement,
    SettlementNonPaymentEvent,
    SettlementNonPaymentIncident,
)


class SettlementStoreRefusal(Exception):
    def __init__(self, code: str, status: int = 409) -> None:
        self.code = code
        self.status = status
        super().__init__(code)


def _view(incident: SettlementNonPaymentIncident) -> dict[str, Any]:
    return {
        "id": incident.id,
        "public_id": incident.public_id,
        "booking_id": incident.booking_id,
        "company_id": incident.company_id,
        "driver_id": incident.driver_id,
        "client_id": incident.client_id,
        "debtor_id": incident.debtor_id,
        "debtor_label": incident.debtor_label,
        "amount_due_chf": f"{Decimal(incident.amount_due_chf):.2f}",
        "currency": incident.currency,
        "transport_date": incident.transport_date,
        "motif": incident.motif,
        "observation": incident.observation,
        "status": incident.status,
        "payment_recorded": False,
        "signature_recorded": False,
        "deferred_billing_authorized": False,
        "mission_completed": True,
        "mission_close_allowed": bool(incident.mission_close_allowed),
        "settlement_pending": True,
        "transport_complete_without_settlement": True,
        "history": [
            {
                "id": event.id,
                "at": event.created_at.isoformat() if event.created_at else None,
                "actor_role": event.actor_role,
                "actor_id": event.actor_id,
                "action": event.action,
                "detail": event.detail,
            }
            for event in incident.events
        ],
    }


def open_incident(
    *,
    booking_id: int,
    company_id: int,
    driver_id: int,
    actor_company_id: int,
    client_id: int | None,
    debtor_id: int | None,
    debtor_label: str | None,
    amount_due_chf: Decimal,
    transport_date: str,
    motif: str,
    observation: str,
    idempotency_key: str,
) -> dict[str, Any]:
    if int(actor_company_id) != int(company_id):
        raise SettlementStoreRefusal("company_isolation_refused", 403)
    key = idempotency_key.strip()
    if not key:
        raise SettlementStoreRefusal("idempotency_requise", 400)
    if motif not in MOTIFS:
        raise SettlementStoreRefusal("motif_inconnu", 400)
    obs = observation.strip()
    if motif == MOTIF_AUTRE and not obs:
        raise SettlementStoreRefusal("observation_requise_pour_autre", 400)
    if amount_due_chf < 0:
        raise SettlementStoreRefusal("montant_invalide", 400)

    existing = (
        db.session.query(SettlementNonPaymentIncident)
        .filter_by(company_id=int(company_id), idempotency_key=key)
        .one_or_none()
    )
    if existing is not None:
        return {**_view(existing), "duplicate": True}

    row = SettlementNonPaymentIncident(
        idempotency_key=key,
        booking_id=int(booking_id),
        company_id=int(company_id),
        driver_id=int(driver_id),
        client_id=client_id,
        debtor_id=debtor_id,
        debtor_label=debtor_label,
        amount_due_chf=amount_due_chf,
        transport_date=transport_date,
        motif=motif,
        observation=obs,
        status=STATUS_A_TRAITER,
        payment_recorded=False,
        signature_recorded=False,
        deferred_billing_authorized=False,
        mission_completed=True,
        mission_close_allowed=True,
    )
    db.session.add(row)
    try:
        db.session.flush()
    except IntegrityError as exc:
        db.session.rollback()
        winner = (
            db.session.query(SettlementNonPaymentIncident)
            .filter_by(company_id=int(company_id), idempotency_key=key)
            .one_or_none()
        )
        if winner is None:
            raise SettlementStoreRefusal("conflit_idempotence", 409) from exc
        return {**_view(winner), "duplicate": True}

    db.session.add(
        SettlementNonPaymentEvent(
            incident_id=int(row.id),
            idempotency_key=f"open:{key}",
            actor_role="driver",
            actor_id=int(driver_id),
            action="opened",
            detail=motif,
        )
    )
    db.session.commit()
    db.session.refresh(row)
    return {**_view(row), "duplicate": False}


def list_company(company_id: int) -> dict[str, Any]:
    rows = (
        db.session.query(SettlementNonPaymentIncident)
        .filter_by(company_id=int(company_id))
        .order_by(SettlementNonPaymentIncident.id.desc())
        .all()
    )
    return {"cases": [_view(row) for row in rows]}


def get_company(company_id: int, incident_id: int) -> dict[str, Any] | None:
    row = db.session.get(SettlementNonPaymentIncident, int(incident_id))
    if row is None or int(row.company_id) != int(company_id):
        return None
    return _view(row)


def admin_set_status(
    *,
    company_id: int,
    incident_id: int,
    new_status: str,
    actor_id: int,
    justification: str,
    idempotency_key: str,
) -> dict[str, Any]:
    row = db.session.get(SettlementNonPaymentIncident, int(incident_id))
    if row is None or int(row.company_id) != int(company_id):
        raise SettlementStoreRefusal("introuvable", 404)
    if new_status not in STATUSES:
        raise SettlementStoreRefusal("statut_inconnu", 400)
    allowed = _ADMIN_TRANSITIONS.get(row.status, frozenset())
    if new_status not in allowed:
        raise SettlementStoreRefusal("transition_refusee", 409)
    note = justification.strip()
    if new_status in {"regularise", "cloture_justifie"} and not note:
        raise SettlementStoreRefusal("justification_requise", 400)
    key = idempotency_key.strip() or f"status:{row.id}:{new_status}:{actor_id}"
    prior = (
        db.session.query(SettlementNonPaymentEvent)
        .filter_by(incident_id=int(row.id), idempotency_key=key)
        .one_or_none()
    )
    if prior is not None:
        return _view(row)
    before = row.status
    row.status = new_status
    db.session.add(
        SettlementNonPaymentEvent(
            incident_id=int(row.id),
            idempotency_key=key,
            actor_role="company_admin",
            actor_id=int(actor_id),
            action=f"status:{new_status}",
            detail=note or f"{before}->{new_status}",
        )
    )
    db.session.commit()
    db.session.refresh(row)
    return _view(row)


def admin_note_later_payment(
    *,
    company_id: int,
    incident_id: int,
    actor_id: int,
    note: str,
    idempotency_key: str,
) -> dict[str, Any]:
    row = db.session.get(SettlementNonPaymentIncident, int(incident_id))
    if row is None or int(row.company_id) != int(company_id):
        raise SettlementStoreRefusal("introuvable", 404)
    text = note.strip()
    if not text:
        raise SettlementStoreRefusal("note_requise", 400)
    key = idempotency_key.strip() or f"later-pay:{row.id}:{actor_id}:{text[:24]}"
    prior = (
        db.session.query(SettlementNonPaymentEvent)
        .filter_by(incident_id=int(row.id), idempotency_key=key)
        .one_or_none()
    )
    if prior is not None:
        return _view(row)
    # Ne jamais basculer payment_recorded (contrainte + règle métier).
    db.session.add(
        SettlementNonPaymentEvent(
            incident_id=int(row.id),
            idempotency_key=key,
            actor_role="company_admin",
            actor_id=int(actor_id),
            action="later_payment_noted",
            detail=text,
        )
    )
    db.session.commit()
    db.session.refresh(row)
    return _view(row)


def record_external_card(
    *,
    booking_id: int,
    company_id: int,
    driver_id: int,
    actor_company_id: int,
    amount_chf: Decimal,
    reference: str,
    receipt_note: str,
    idempotency_key: str,
) -> dict[str, Any]:
    if int(actor_company_id) != int(company_id):
        raise SettlementStoreRefusal("company_isolation_refused", 403)
    key = idempotency_key.strip()
    if not key:
        raise SettlementStoreRefusal("idempotency_requise", 400)
    ref = reference.strip()
    receipt = receipt_note.strip()
    if not ref and not receipt:
        raise SettlementStoreRefusal("preuve_carte_requise", 400)
    if amount_chf <= 0:
        raise SettlementStoreRefusal("montant_invalide", 400)

    existing = (
        db.session.query(ExternalCardSettlement)
        .filter_by(company_id=int(company_id), idempotency_key=key)
        .one_or_none()
    )
    if existing is not None:
        return _card_view(existing, duplicate=True)

    row = ExternalCardSettlement(
        booking_id=int(booking_id),
        company_id=int(company_id),
        driver_id=int(driver_id),
        amount_chf=amount_chf,
        reference=ref,
        receipt_note=receipt,
        confirmed_by_driver=True,
        idempotency_key=key,
        channel="external_terminal",
        tap_to_pay=False,
        provider_name="external_bank_terminal",
    )
    db.session.add(row)
    try:
        db.session.commit()
    except IntegrityError as exc:
        db.session.rollback()
        winner = (
            db.session.query(ExternalCardSettlement)
            .filter_by(company_id=int(company_id), idempotency_key=key)
            .one_or_none()
        )
        if winner is None:
            raise SettlementStoreRefusal("conflit_idempotence", 409) from exc
        return _card_view(winner, duplicate=True)
    db.session.refresh(row)
    return _card_view(row, duplicate=False)


def _card_view(row: ExternalCardSettlement, *, duplicate: bool) -> dict[str, Any]:
    return {
        "id": row.id,
        "booking_id": row.booking_id,
        "company_id": row.company_id,
        "driver_id": row.driver_id,
        "amount_chf": f"{Decimal(row.amount_chf):.2f}",
        "currency": row.currency,
        "channel": "external_terminal",
        "tap_to_pay": False,
        "provider_name": "external_bank_terminal",
        "reference": row.reference,
        "receipt_note": row.receipt_note,
        "payment_recorded": True,
        "confirmed_by_driver": True,
        "duplicate": duplicate,
    }
