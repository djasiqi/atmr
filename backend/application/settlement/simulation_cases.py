"""Persistance du dossier fictif. N'écrit jamais paiement, facture ou caisse."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from application.invoices.issued_invoice_guard import invoice_is_issued
from domain.end_of_trip_settlement.simulation_barrier import ledger_open
from domain.end_of_trip_settlement.simulation_machine import (
    MachineResult,
    SimulationRefusal,
    company_cash,
    declare_cash,
    invoice_event,
    recover_tap,
    resolve_tap,
    shortfall_cash,
    start_tap,
)
from ext import db
from models import Company
from models.enums import BookingStatus, PaymentStatus
from models.invoice import Invoice, InvoiceLine
from models.payment import Payment
from models.settlement_policy import CompanySettlementCapability
from models.settlement_simulation import (
    SettlementSimulationCase,
    SettlementSimulationEvent,
)

_CENT = Decimal("0.01")
_OPERABLE = frozenset({BookingStatus.IN_PROGRESS, BookingStatus.COMPLETED})
_CANCELLED = frozenset({BookingStatus.CANCELED})
DOCUMENT_VERSION = "2026-10-26.v1-draft"
_CASH_STEPS = ("receive", "hold", "remit", "reconcile")
_COMPANY_FOLLOWUP = frozenset(
    {"simulated_confirmed", "administrative_simulated", "closed"}
)


class LedgerClosed(Exception):
    """La barrière est fermée. Aucune ligne n'est écrite."""


def decision_version(booking: Any) -> str:
    raw = "|".join(
        (
            str(booking.id),
            str(booking.company_id),
            str(booking.price_amount),
            str(getattr(booking, "billed_to_type", None)),
            str(getattr(booking, "status", None)),
            str(bool(getattr(booking, "is_round_trip", False))),
            str(getattr(booking, "parent_booking_id", None)),
        )
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _amount(booking: Any) -> Decimal:
    raw = getattr(booking, "price_amount", None)
    if raw is None:
        raise SimulationRefusal("montant_absent", 409)
    amount = Decimal(str(raw)).quantize(_CENT)
    if amount <= 0:
        raise SimulationRefusal("montant_absent", 409)
    return amount


def _status(booking: Any) -> BookingStatus | str:
    status = booking.status
    if isinstance(status, BookingStatus):
        return status
    try:
        return BookingStatus(str(status))
    except ValueError:
        return str(status)


def _round_trip(booking: Any) -> bool:
    return bool(getattr(booking, "is_round_trip", False)) or bool(
        getattr(booking, "parent_booking_id", None)
    )


def _already_paid(booking_id: int) -> bool:
    row = db.session.execute(
        select(Payment.id).where(
            Payment.booking_id == booking_id,
            Payment.status == PaymentStatus.COMPLETED,
        )
    ).first()
    return row is not None


def _issued_invoice(booking_id: int) -> bool:
    invoices = db.session.execute(
        select(Invoice)
        .join(InvoiceLine, InvoiceLine.invoice_id == Invoice.id)
        .where(InvoiceLine.reservation_id == booking_id)
    ).scalars()
    return any(invoice_is_issued(invoice) for invoice in invoices)


def _basis(booking: Any) -> str:
    billed = str(getattr(booking, "billed_to_type", "") or "")
    if billed != "patient":
        return "not_patient"
    return "missing"


def _debtor(booking: Any) -> str:
    billed = str(getattr(booking, "billed_to_type", "") or "")
    if billed == "clinic":
        return "Institution"
    name = str(getattr(booking, "customer_name", "") or "").strip()
    return name or "Patient"


def _capabilities_enabled(company_id: int) -> bool:
    row = db.session.get(CompanySettlementCapability, company_id)
    if row is None:
        return False
    return bool(
        row.tap_to_pay_enabled or row.cash_enabled or row.direct_patient_invoice_enabled
    )


def _refuse_booking(booking: Any, *, driver_id: int, company_id: int) -> None:
    if int(booking.company_id) != int(company_id):
        raise SimulationRefusal("autre_entreprise", 404)
    if booking.driver_id is None or int(booking.driver_id) != int(driver_id):
        raise SimulationRefusal("chauffeur_non_affecte", 403)
    if _round_trip(booking):
        raise SimulationRefusal("aller_retour_bloque", 409)
    status = _status(booking)
    if status in _CANCELLED:
        raise SimulationRefusal("transport_annule", 409)
    if status not in _OPERABLE:
        raise SimulationRefusal("statut_incompatible", 409)
    if str(getattr(booking, "billed_to_type", "") or "") == "clinic":
        raise SimulationRefusal("institution_facturee", 409)
    if _already_paid(int(booking.id)):
        raise SimulationRefusal("deja_paye", 409)
    if _issued_invoice(int(booking.id)):
        raise SimulationRefusal("facture_emise", 409)
    _amount(booking)


def _view(case: SettlementSimulationCase) -> dict[str, Any]:
    events = db.session.scalars(
        select(SettlementSimulationEvent)
        .where(SettlementSimulationEvent.case_id == case.id)
        .order_by(SettlementSimulationEvent.id)
    ).all()
    history = [
        {
            "action": event.action,
            "from_state": event.from_state,
            "to_state": event.to_state,
            "actor_role": event.actor_role,
            "at": event.created_at.isoformat() if event.created_at else None,
            "payload": event.payload,
        }
        for event in events
    ]
    modes = ["tap_to_pay", "cash", "invoice"] if case.state == "preparing" else []
    if case.state == "simulated_refused":
        modes = ["tap_to_pay", "cash", "invoice"]
    company = db.session.get(Company, int(case.company_id))
    return {
        "id": case.id,
        "booking_id": case.booking_id,
        "company_id": case.company_id,
        "company_name": company.name if company is not None else None,
        "driver_id": case.driver_id,
        "debtor_label": case.debtor_label,
        "amount_chf": format(Decimal(case.amount_chf).quantize(_CENT), "f"),
        "currency": "CHF",
        "mode": case.mode,
        "modes": modes,
        "state": case.state,
        "decision_version": case.decision_version,
        "attempt_id": case.attempt_id,
        "contractual_basis": case.contractual_basis,
        "document_version": DOCUMENT_VERSION,
        "document_approved": False,
        "signature_recorded": False,
        "real_financial_write": False,
        "real_capabilities_enabled": _capabilities_enabled(int(case.company_id)),
        "persistent_simulation": True,
        "simulation": True,
        "banner": "SIMULATION — AUCUN PAIEMENT RÉEL",
        "payment_executable": False,
        "persists_payment": False,
        "writes_completed_payment": False,
        "writes_cash_receipt": False,
        "writes_invoice": False,
        "writes_financial_signature": False,
        "provider": "mock",
        "history": history,
        "created_at": case.created_at.isoformat() if case.created_at else None,
        "updated_at": case.updated_at.isoformat() if case.updated_at else None,
    }


def _require_open() -> None:
    if not ledger_open():
        raise LedgerClosed()


def _lock(booking_id: int) -> SettlementSimulationCase | None:
    return db.session.scalars(
        select(SettlementSimulationCase)
        .where(SettlementSimulationCase.booking_id == booking_id)
        .with_for_update()
    ).first()


def _request_fingerprint(action: str, payload: dict[str, Any]) -> str:
    material = {
        "action": action,
        "simulated_result": payload.get("simulated_result"),
        "tendered_chf": payload.get("tendered_chf"),
        "change_chf": payload.get("change_chf"),
        "change_confirmed": payload.get("change_confirmed"),
        "disposition": payload.get("disposition"),
        "kind": payload.get("kind"),
        "step": payload.get("step"),
        "amount_chf": payload.get("amount_chf"),
        "company_id": payload.get("company_id"),
    }
    raw = json.dumps(material, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _replay(
    case: SettlementSimulationCase,
    key: str,
    action: str,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    existing = db.session.scalars(
        select(SettlementSimulationEvent).where(
            SettlementSimulationEvent.case_id == case.id,
            SettlementSimulationEvent.idempotency_key == key,
        )
    ).first()
    if existing is None:
        return None
    current = _request_fingerprint(action, payload)
    stored = (existing.payload or {}).get("request_fingerprint")
    if stored is None:
        if existing.action != action:
            raise SimulationRefusal("conflit_idempotence", 409)
    elif stored != current:
        raise SimulationRefusal("conflit_idempotence", 409)
    view = _view(case)
    view["replay"] = True
    return view


def _guard_company_step(
    case: SettlementSimulationCase, step: str, actor_role: str
) -> None:
    if actor_role != "company":
        raise SimulationRefusal("rapprochement_reserve_entreprise", 403)
    if case.mode != "cash":
        raise SimulationRefusal("suivi_especes_sans_dossier")
    if step not in _CASH_STEPS:
        raise SimulationRefusal("suivi_especes_inconnu", 400)
    if case.state not in _COMPANY_FOLLOWUP:
        raise SimulationRefusal("suivi_premature")
    done = db.session.scalars(
        select(SettlementSimulationEvent.action)
        .where(
            SettlementSimulationEvent.case_id == case.id,
            SettlementSimulationEvent.action.in_(
                [f"company_cash_{name}" for name in _CASH_STEPS]
            ),
        )
        .order_by(SettlementSimulationEvent.id)
    ).all()
    if len(done) >= len(_CASH_STEPS):
        raise SimulationRefusal("etape_caisse_deja_enregistree", 409)
    expected = _CASH_STEPS[len(done)]
    if step != expected:
        if f"company_cash_{step}" in done:
            raise SimulationRefusal("etape_caisse_deja_enregistree", 409)
        raise SimulationRefusal("etape_caisse_hors_ordre", 409)


def _apply(
    case: SettlementSimulationCase,
    result: MachineResult,
    *,
    key: str,
    action: str,
    request_payload: dict[str, Any],
    actor_role: str,
    actor_id: int | None,
) -> dict[str, Any]:
    if result.replay:
        view = _view(case)
        view["replay"] = True
        return view
    booking_id = int(case.booking_id)
    previous = case.state
    stored = dict(result.payload)
    stored["request_fingerprint"] = _request_fingerprint(action, request_payload)
    try:
        with db.session.begin_nested():
            case.state = result.state
            case.mode = result.mode
            case.idempotency_key = key
            if result.attempt_id:
                case.attempt_id = result.attempt_id
            case.signature_recorded = False
            case.real_financial_write = False
            case.updated_at = datetime.now(UTC)
            db.session.add(
                SettlementSimulationEvent(
                    case_id=case.id,
                    idempotency_key=key,
                    action=result.event_action,
                    from_state=previous,
                    to_state=result.state,
                    actor_role=actor_role,
                    actor_id=actor_id,
                    payload=stored,
                )
            )
            db.session.flush()
    except IntegrityError:
        db.session.rollback()
        locked = _lock(booking_id)
        if locked is None:
            raise
        replayed = _replay(locked, key, action, request_payload)
        if replayed is not None:
            return replayed
        raise SimulationRefusal("cle_deja_utilisee") from None
    db.session.commit()
    db.session.refresh(case)
    view = _view(case)
    view["replay"] = False
    return view


def _stale(case: SettlementSimulationCase, booking: Any) -> bool:
    return case.decision_version != decision_version(booking) or Decimal(
        case.amount_chf
    ).quantize(_CENT) != _amount(booking)


def open_case(booking: Any, *, driver_id: int, company_id: int) -> dict[str, Any]:
    _require_open()
    _refuse_booking(booking, driver_id=driver_id, company_id=company_id)
    existing = _lock(int(booking.id))
    if existing is not None:
        if int(existing.company_id) != int(company_id):
            raise SimulationRefusal("autre_entreprise", 404)
        return _view(existing)
    case = SettlementSimulationCase(
        booking_id=int(booking.id),
        company_id=int(company_id),
        driver_id=int(driver_id),
        debtor_label=_debtor(booking),
        amount_chf=_amount(booking),
        currency="CHF",
        mode=None,
        decision_version=decision_version(booking),
        state="preparing",
        contractual_basis=_basis(booking),
        signature_recorded=False,
        real_financial_write=False,
    )
    try:
        with db.session.begin_nested():
            db.session.add(case)
            db.session.flush()
    except IntegrityError:
        db.session.rollback()
        winner = _lock(int(booking.id))
        if winner is None:
            raise
        if int(winner.company_id) != int(company_id):
            raise SimulationRefusal("autre_entreprise", 404) from None
        return _view(winner)
    db.session.commit()
    db.session.refresh(case)
    return _view(case)


def read_case(
    booking: Any, *, driver_id: int, company_id: int
) -> dict[str, Any] | None:
    _require_open()
    if int(booking.company_id) != int(company_id):
        raise SimulationRefusal("autre_entreprise", 404)
    if booking.driver_id is None or int(booking.driver_id) != int(driver_id):
        raise SimulationRefusal("chauffeur_non_affecte", 403)
    case = db.session.scalars(
        select(SettlementSimulationCase).where(
            SettlementSimulationCase.booking_id == int(booking.id)
        )
    ).first()
    if case is None:
        return None
    return _view(case)


def transition_case(
    booking: Any,
    *,
    driver_id: int,
    company_id: int,
    actor_role: str,
    actor_id: int | None,
    action: str,
    idempotency_key: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    _require_open()
    key = (idempotency_key or "").strip()
    if not key:
        raise SimulationRefusal("cle_idempotence_requise", 400)
    if int(booking.company_id) != int(company_id):
        raise SimulationRefusal("autre_entreprise", 404)
    case = _lock(int(booking.id))
    if case is None:
        raise SimulationRefusal("dossier_absent", 404)
    if int(case.company_id) != int(company_id):
        raise SimulationRefusal("autre_entreprise", 404)
    if actor_role == "driver" and (
        booking.driver_id is None or int(booking.driver_id) != int(driver_id)
    ):
        raise SimulationRefusal("chauffeur_non_affecte", 403)
    replayed = _replay(case, key, action, payload)
    if replayed is not None:
        return replayed
    if action == "company_cash":
        _guard_company_step(case, str(payload.get("step") or ""), actor_role)
    if action != "company_cash" and _stale(case, booking):
        previous = case.state
        case.state = "administrative_simulated"
        case.updated_at = datetime.now(UTC)
        db.session.add(
            SettlementSimulationEvent(
                case_id=case.id,
                idempotency_key=key,
                action="decision_obsolete",
                from_state=previous,
                to_state="administrative_simulated",
                actor_role=actor_role,
                actor_id=actor_id,
                payload={
                    "real_financial_write": False,
                    "request_fingerprint": _request_fingerprint(action, payload),
                },
            )
        )
        db.session.commit()
        raise SimulationRefusal("decision_obsolete")
    amount = Decimal(case.amount_chf).quantize(_CENT)
    merchant = int(case.company_id)
    if action == "start_tap":
        result = start_tap(
            state=case.state,
            mode=case.mode,
            attempt_id=case.attempt_id,
            amount_chf=amount,
            merchant_company_id=merchant,
            new_attempt_id=key,
        )
    elif action == "resolve_tap":
        result = resolve_tap(
            state=case.state,
            attempt_id=case.attempt_id,
            amount_chf=amount,
            merchant_company_id=merchant,
            simulated_result=str(payload.get("simulated_result") or ""),
        )
    elif action == "recover_tap":
        result = recover_tap(state=case.state, attempt_id=case.attempt_id)
    elif action == "declare_cash":
        result = declare_cash(
            state=case.state,
            mode=case.mode,
            expected_chf=amount,
            tendered_chf=payload.get("tendered_chf"),
            change_chf=payload.get("change_chf", "0"),
            change_confirmed=bool(payload.get("change_confirmed")),
        )
    elif action == "cash_shortfall":
        result = shortfall_cash(
            state=case.state,
            mode=case.mode,
            expected_chf=amount,
            tendered_chf=payload.get("tendered_chf"),
            disposition=str(payload.get("disposition") or ""),
        )
    elif action == "invoice_event":
        result = invoice_event(
            state=case.state,
            mode=case.mode,
            kind=str(payload.get("kind") or ""),
            contractual_basis=case.contractual_basis
            if case.contractual_basis == "known_draft"
            else "missing",
        )
    elif action == "company_cash":
        result = company_cash(
            state=case.state,
            mode=case.mode,
            step=str(payload.get("step") or ""),
            actor_role=actor_role,
        )
    else:
        raise SimulationRefusal("action_inconnue", 400)
    if payload.get("amount_chf") not in (None, "", format(amount, "f")):
        raise SimulationRefusal("montant_client_ignore", 400)
    if payload.get("company_id") not in (None, merchant):
        raise SimulationRefusal("entreprise_client_ignoree", 400)
    return _apply(
        case,
        result,
        key=key,
        action=action,
        request_payload=payload,
        actor_role=actor_role,
        actor_id=actor_id,
    )


def list_company_cases(company_id: int) -> list[dict[str, Any]]:
    _require_open()
    rows = db.session.scalars(
        select(SettlementSimulationCase)
        .where(SettlementSimulationCase.company_id == company_id)
        .order_by(SettlementSimulationCase.id)
    ).all()
    return [_view(row) for row in rows]


def read_company_case(company_id: int, case_id: int) -> dict[str, Any] | None:
    _require_open()
    case = db.session.get(SettlementSimulationCase, case_id)
    if case is None or int(case.company_id) != int(company_id):
        return None
    return _view(case)
