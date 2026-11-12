"""Registre d'espèces. N'écrit ni paiement, ni facture, ni caisse réelle activée."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from application.settlement.production_gates import (
    amount_due,
    authorize_settlement_history_read,
    debtor_label,
    decision_version,
    fingerprint,
    refuse_booking,
    require_new_cash_declaration,
    require_prep,
)
from domain.end_of_trip_settlement.cash_settlement import evaluate_cash, money
from domain.end_of_trip_settlement.financial_contract import (
    CashLine,
    imputable,
    recognized_net,
    within_cap,
)
from domain.end_of_trip_settlement.production_prep_barrier import PrepRefusal
from ext import db
from models import Company
from models.settlement_production import (
    CashCollection,
    CashCollectionEvent,
    CashHandover,
    CashReceipt,
    CashReconciliation,
)

_BANNER = "PRÉPARATION — ENCAISSEMENT NON ACTIVÉ"
_LOCAL_BANNER = "CAISSE LOCALE — AUCUN PRESTATAIRE EXTERNE"


def _ledger_banner() -> str:
    from domain.end_of_trip_settlement.v1_local_ledger import v1_local_ledger_enabled

    if v1_local_ledger_enabled():
        return _LOCAL_BANNER
    return _BANNER


def _opened_as_exceptional(row: CashCollection) -> bool:
    events = db.session.scalars(
        select(CashCollectionEvent).where(
            CashCollectionEvent.collection_id == row.id,
            CashCollectionEvent.action == "open",
        )
    ).all()
    return any(bool((event.payload or {}).get("exceptional")) for event in events)


def _authorize_cash_write(
    booking: Any, row: CashCollection, *, driver_id: int, company_id: int
) -> None:
    if _opened_as_exceptional(row):
        from application.settlement.leg_settlement import require_exceptional_cash

        require_exceptional_cash(booking)
        return
    require_new_cash_declaration(booking, driver_id=driver_id, company_id=company_id)


def open_collection(
    booking: Any,
    *,
    driver_id: int,
    company_id: int,
    actor_id: int | None,
    exceptional: bool = False,
) -> dict[str, Any]:
    require_prep()
    refuse_booking(booking, driver_id=driver_id, company_id=company_id)
    existing = _lock(int(booking.id))
    if existing is not None:
        if exceptional and not _opened_as_exceptional(existing):
            from application.settlement.leg_settlement import require_exceptional_cash

            require_exceptional_cash(booking)
            raise PrepRefusal("encaissement_exceptionnel_refuse", 409)
        view = _view(existing)
        view["replay"] = True
        return view
    if exceptional:
        from application.settlement.leg_settlement import require_exceptional_cash

        require_exceptional_cash(booking)
    else:
        require_new_cash_declaration(
            booking, driver_id=driver_id, company_id=company_id
        )
    row = CashCollection(
        booking_id=int(booking.id),
        company_id=int(company_id),
        driver_id=int(driver_id),
        debtor_label=debtor_label(booking),
        amount_due_chf=amount_due(booking),
        currency="CHF",
        state="prepared",
        decision_version=decision_version(booking),
        production_activation=False,
        legally_approved=False,
        change_confirmed=False,
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
        before="prepared",
        after="prepared",
        actor_role="driver",
        actor_id=actor_id,
        payload={"request_fingerprint": "open", "exceptional": bool(exceptional)},
    )
    db.session.commit()
    return _view(row)


def record_capped_collection(
    booking: Any,
    payload: dict,
    *,
    driver_id: int,
    company_id: int,
    actor_id: int | None,
    operation_key: str,
) -> dict[str, Any]:
    """Insère un encaissement sans dépasser le dû, sous verrou de réservation.

    Le prix contractuel n'est pas réécrit. Une clé déjà enregistrée est rejouée.
    """
    due = amount_due(booking)
    _lock_booking(int(booking.id))
    existing = _collection_by_key(int(booking.id), operation_key)
    if existing is not None:
        replay = _replay(existing, operation_key, payload)
        if replay is None:
            raise PrepRefusal("conflit_idempotence", 409)
        return _with_position(replay, int(booking.id), due)
    placeholder = _prepared_collection(int(booking.id))
    if placeholder is not None and (
        decision_version(booking) != placeholder.decision_version
        or payload.get("decision_version") not in (None, placeholder.decision_version)
    ):
        _event(
            placeholder,
            key=operation_key,
            action="decision_obsolete",
            before=placeholder.state,
            after="administrative",
            actor_role="driver",
            actor_id=actor_id,
            payload={"request_fingerprint": fingerprint(payload)},
        )
        placeholder.state = "administrative"
        placeholder.client_operation_key = operation_key
        db.session.commit()
        raise PrepRefusal("decision_obsolete", 409)
    if placeholder is not None:
        _authorize_cash_write(
            booking, placeholder, driver_id=driver_id, company_id=company_id
        )
    try:
        decision = evaluate_cash(
            due=due,
            tendered=money(payload.get("tendered_chf")),
            change=money(payload.get("change_chf", "0.00")),
            change_confirmed=bool(payload.get("change_confirmed")),
            disposition=payload.get("disposition"),
        )
    except ValueError as exc:
        raise PrepRefusal(str(exc), 400) from exc
    disposition = str(payload.get("disposition") or decision.outcome)
    candidate_counts = imputable(
        state=decision.outcome,
        disposition=disposition,
        net=decision.net_chf,
        due=due,
    )
    projected = _recognized(int(booking.id))
    if candidate_counts:
        projected = (projected + decision.net_chf).quantize(Decimal("0.01"))
    try:
        if not within_cap(projected, due):
            raise ValueError("depassement_autorise")
    except ValueError as exc:
        db.session.rollback()
        raise PrepRefusal(str(exc), 409) from exc
    if placeholder is None:
        row = CashCollection(
            booking_id=int(booking.id),
            company_id=int(company_id),
            driver_id=int(driver_id),
            debtor_label=debtor_label(booking),
            amount_due_chf=due,
            currency="CHF",
            state=decision.outcome,
            decision_version=decision_version(booking),
            production_activation=False,
            legally_approved=False,
            change_confirmed=bool(payload.get("change_confirmed")),
            kind="collection",
            client_operation_key=operation_key,
        )
        db.session.add(row)
        before = "prepared"
    else:
        row = placeholder
        before = row.state
        row.client_operation_key = operation_key
        row.state = decision.outcome
    row.tendered_chf = money(payload.get("tendered_chf"))
    row.change_chf = decision.change_chf
    row.change_confirmed = bool(payload.get("change_confirmed"))
    row.net_chf = decision.net_chf
    row.disposition = disposition
    row.production_activation = False
    row.legally_approved = False
    try:
        db.session.flush()
    except IntegrityError:
        db.session.rollback()
        winner = _collection_by_key(int(booking.id), operation_key)
        if winner is None:
            raise
        replay = _replay(winner, operation_key, payload)
        if replay is None:
            raise PrepRefusal("conflit_idempotence", 409) from None
        return _with_position(replay, int(booking.id), due)
    if decision.receipt_allowed:
        _receipt(row)
    _recognized_now, remaining = _position(int(booking.id), due)
    _event(
        row,
        key=operation_key,
        action="declare_cash",
        before=before,
        after=row.state,
        actor_role="driver",
        actor_id=actor_id,
        payload={
            "request_fingerprint": fingerprint(payload),
            "decision_h": decision.decision_h,
            "imputed": False,
            "receipt_allowed": decision.receipt_allowed,
            "held_is_not_void": decision.held_is_not_void,
            "full_receivable_created": False,
            "accounted": decision.outcome == "received",
            "invoice_issued": False,
            "balance_chf": f"{remaining:.2f}",
        },
    )
    from application.settlement.patient_billing_prep import reconcile_followup_balance

    reconcile_followup_balance(booking)
    db.session.commit()
    view = _view(row)
    view["accounted"] = decision.outcome == "received"
    return _with_position(view, int(booking.id), due)


def record_compensation(
    booking: Any,
    payload: dict,
    *,
    driver_id: int,
    company_id: int,
    actor_id: int | None,
    operation_key: str,
) -> dict[str, Any]:
    """Ajoute une compensation. La collection d'origine n'est pas réécrite."""
    due = amount_due(booking)
    _lock_booking(int(booking.id))
    existing = _collection_by_key(int(booking.id), operation_key)
    if existing is not None:
        replay = _replay(existing, operation_key, payload)
        if replay is None:
            raise PrepRefusal("conflit_idempotence", 409)
        return _with_position(replay, int(booking.id), due)
    origin_raw = payload.get("compensates_collection_id")
    if origin_raw in (None, ""):
        raise PrepRefusal("origine_absente", 404)
    origin = db.session.scalar(
        select(CashCollection)
        .where(CashCollection.id == int(origin_raw))
        .with_for_update()
    )
    if origin is None:
        db.session.rollback()
        raise PrepRefusal("origine_absente", 404)
    if int(origin.company_id) != int(company_id):
        db.session.rollback()
        raise PrepRefusal("autre_entreprise", 404)
    if int(origin.booking_id) != int(booking.id):
        db.session.rollback()
        raise PrepRefusal("autre_reservation", 409)
    if origin.kind == "compensation" or origin.net_chf is None:
        db.session.rollback()
        raise PrepRefusal("origine_non_compensable", 409)
    try:
        signed = money(payload.get("net_chf"))
    except (ValueError, ArithmeticError) as exc:
        raise PrepRefusal("montant_illisible", 400) from exc
    if signed == 0:
        db.session.rollback()
        raise PrepRefusal("montant_nul", 400)
    if signed > 0:
        db.session.rollback()
        raise PrepRefusal("signe_invalide", 400)
    reason = str(payload.get("reason") or "").strip()
    if len(reason) < 3:
        db.session.rollback()
        raise PrepRefusal("motif_obligatoire", 400)
    linked = db.session.execute(
        text(
            "SELECT 1 FROM invoice_payments "
            "WHERE cash_collection_id = :origin_id LIMIT 1"
        ),
        {"origin_id": int(origin.id)},
    ).first()
    if linked is not None:
        db.session.rollback()
        raise PrepRefusal("correction_apres_imputation", 409)
    already = db.session.execute(
        text(
            "SELECT COALESCE(SUM(net_chf), 0) FROM cash_collection "
            "WHERE compensates_collection_id = :origin_id"
        ),
        {"origin_id": int(origin.id)},
    ).scalar_one()
    origin_net = Decimal(origin.net_chf)
    if origin_net + Decimal(already) + signed < 0:
        db.session.rollback()
        raise PrepRefusal("depassement_autorise", 409)
    projected = (_recognized(int(booking.id)) + signed).quantize(Decimal("0.01"))
    try:
        if not within_cap(projected, due):
            raise ValueError("depassement_autorise")
    except ValueError as exc:
        db.session.rollback()
        raise PrepRefusal(str(exc), 409) from exc
    before_origin = Decimal(origin.net_chf)
    row = CashCollection(
        booking_id=int(booking.id),
        company_id=int(company_id),
        driver_id=int(driver_id),
        debtor_label=debtor_label(booking),
        amount_due_chf=due,
        currency="CHF",
        state="administrative",
        disposition="correction",
        decision_version=decision_version(booking),
        production_activation=False,
        legally_approved=False,
        change_confirmed=False,
        kind="compensation",
        compensates_collection_id=int(origin.id),
        client_operation_key=operation_key,
        net_chf=signed,
        tendered_chf=abs(signed),
        change_chf=Decimal("0.00"),
    )
    db.session.add(row)
    try:
        db.session.flush()
    except IntegrityError:
        db.session.rollback()
        winner = _collection_by_key(int(booking.id), operation_key)
        if winner is None:
            raise
        replay = _replay(winner, operation_key, payload)
        if replay is None:
            raise PrepRefusal("conflit_idempotence", 409) from None
        return _with_position(replay, int(booking.id), due)
    if Decimal(origin.net_chf) != before_origin:
        db.session.rollback()
        raise PrepRefusal("origine_modifiee", 409)
    _recognized_now, remaining = _position(int(booking.id), due)
    _event(
        row,
        key=operation_key,
        action="compensate_cash",
        before="administrative",
        after="administrative",
        actor_role="driver",
        actor_id=actor_id,
        payload={
            "request_fingerprint": fingerprint(payload),
            "reason": reason,
            "compensates_collection_id": int(origin.id),
            "imputed": False,
            "invoice_issued": False,
            "balance_chf": f"{remaining:.2f}",
        },
    )
    db.session.commit()
    view = _view(row)
    view["accounted"] = False
    return _with_position(view, int(booking.id), due)


def compensate_cash(
    booking: Any,
    payload: dict,
    *,
    driver_id: int,
    company_id: int,
    actor_id: int | None,
) -> dict[str, Any]:
    """Corrige une saisie. Ce n'est ni un avoir, ni un remboursement."""
    require_prep()
    refuse_booking(booking, driver_id=driver_id, company_id=company_id)
    key = str(payload.get("idempotency_key") or "").strip()
    if not key:
        raise PrepRefusal("cle_absente", 400)
    return record_compensation(
        booking,
        payload,
        driver_id=driver_id,
        company_id=company_id,
        actor_id=actor_id,
        operation_key=key,
    )


def declare_cash(
    booking: Any,
    payload: dict,
    *,
    driver_id: int,
    company_id: int,
    actor_id: int | None,
) -> dict[str, Any]:
    require_prep()
    refuse_booking(booking, driver_id=driver_id, company_id=company_id)
    from application.settlement.leg_settlement import reject_leg_amount

    reject_leg_amount(booking, payload, company_id=company_id)
    key = str(payload.get("idempotency_key") or "").strip()
    if not key:
        raise PrepRefusal("cle_absente", 400)
    return record_capped_collection(
        booking,
        payload,
        driver_id=driver_id,
        company_id=company_id,
        actor_id=actor_id,
        operation_key=key,
    )


def refuse_payment(
    booking: Any,
    payload: dict,
    *,
    driver_id: int,
    company_id: int,
    actor_id: int | None,
) -> dict[str, Any]:
    """Enregistre un refus. Aucun encaissement, aucune facture, aucun e-mail."""
    require_prep()
    refuse_booking(booking, driver_id=driver_id, company_id=company_id)
    reason = str(payload.get("reason") or "").strip()
    if len(reason) < 3:
        raise PrepRefusal("motif_obligatoire", 400)
    if len(reason) > 500:
        raise PrepRefusal("motif_trop_long", 400)
    key = str(payload.get("idempotency_key") or "").strip()
    if not key:
        raise PrepRefusal("cle_absente", 400)
    row = _lock(int(booking.id))
    if row is None:
        raise PrepRefusal("dossier_absent", 404)
    replay = _replay(row, key, payload)
    if replay is not None:
        return replay
    if row.state != "prepared":
        raise PrepRefusal("operation_close", 409)
    _authorize_cash_write(booking, row, driver_id=driver_id, company_id=company_id)
    before = row.state
    row.tendered_chf = Decimal("0.00")
    row.change_chf = Decimal("0.00")
    row.change_confirmed = False
    row.net_chf = Decimal("0.00")
    row.disposition = "refused"
    row.state = "administrative"
    _event(
        row,
        key=key,
        action="refuse_payment",
        before=before,
        after=row.state,
        actor_role="driver",
        actor_id=actor_id,
        payload={
            "request_fingerprint": fingerprint(payload),
            "reason": reason,
            "invoice_issued": False,
            "accounted": False,
            "balance_chf": f"{Decimal(row.amount_due_chf):.2f}",
        },
    )
    _notify_refusal(row, reason)
    from application.settlement.patient_billing_prep import reconcile_followup_balance

    reconcile_followup_balance(booking)
    db.session.commit()
    view = _view(row)
    view["accounted"] = False
    return view


def mark_uncertain(
    booking: Any,
    payload: dict,
    *,
    driver_id: int,
    company_id: int,
    actor_id: int | None,
) -> dict[str, Any]:
    require_prep()
    refuse_booking(booking, driver_id=driver_id, company_id=company_id)
    key = str(payload.get("idempotency_key") or "").strip()
    if not key:
        raise PrepRefusal("cle_absente", 400)
    row = _lock(int(booking.id))
    if row is None:
        raise PrepRefusal("dossier_absent", 404)
    replay = _replay(row, key, payload)
    if replay is not None:
        return replay
    if row.state != "prepared":
        raise PrepRefusal("aucune_operation_en_cours", 409)
    before = row.state
    row.state = "uncertain"
    _event(
        row,
        key=key,
        action="mark_uncertain",
        before=before,
        after="uncertain",
        actor_role="driver",
        actor_id=actor_id,
        payload={
            "request_fingerprint": fingerprint(payload),
            "accounted": False,
        },
    )
    db.session.commit()
    view = _view(row)
    view["accounted"] = False
    return view


def collection_still_held(origin: CashCollection) -> tuple[Decimal, Decimal]:
    """Espèces encore chez le chauffeur, et total des remises déjà confirmées.

    Les remises seulement déclarées ne sont pas déduites. Les compensations
    de correction et de remboursement le sont, une seule fois.
    """
    zero = Decimal("0.00")
    if origin.kind != "collection" or origin.net_chf is None:
        return zero, zero
    if not imputable(
        state=origin.state,
        disposition=origin.disposition,
        net=Decimal(origin.net_chf),
        due=Decimal(origin.amount_due_chf),
    ):
        return zero, zero
    compensations = db.session.scalars(
        select(CashCollection.net_chf).where(
            CashCollection.compensates_collection_id == int(origin.id)
        )
    ).all()
    handed = db.session.scalars(
        select(CashHandover.amount_chf).where(
            CashHandover.collection_id == int(origin.id),
            CashHandover.state == "confirmed",
        )
    ).all()
    handed_total = sum((Decimal(amount) for amount in handed), zero).quantize(
        Decimal("0.01")
    )
    held = (
        Decimal(origin.net_chf)
        + sum((Decimal(net) for net in compensations), zero)
        - handed_total
    ).quantize(Decimal("0.01"))
    return held, handed_total


def _handover_view(
    row: CashCollection, handover: CashHandover, *, replay: bool
) -> dict[str, Any]:
    view = _view(row)
    view["replay"] = replay
    view["handover_id"] = int(handover.id)
    view["handover_amount_chf"] = f"{Decimal(handover.amount_chf):.2f}"
    view["idempotency_key"] = handover.idempotency_key
    return view


def _handover_public(item: CashHandover) -> dict[str, Any]:
    """Lecture d'une remise. Aucun montant n'est recalculé."""
    return {
        "id": int(item.id),
        "amount_chf": _cents(Decimal(item.amount_chf)),
        "state": item.state,
        "idempotency_key": item.idempotency_key,
        "declared_at": None if item.created_at is None else item.created_at.isoformat(),
        "confirmed_at": None
        if item.confirmed_at is None
        else item.confirmed_at.isoformat(),
    }


def _handovers_of(collection_id: int) -> list[CashHandover]:
    return list(
        db.session.scalars(
            select(CashHandover)
            .where(CashHandover.collection_id == int(collection_id))
            .order_by(CashHandover.id)
        ).all()
    )


def declare_handover(
    booking: Any,
    *,
    driver_id: int,
    company_id: int,
    actor_id: int,
    idempotency_key: str,
    amount_chf: Any = None,
) -> dict[str, Any]:
    """Déclare une remise partielle ou totale. Ne confirme pas et n'écrit aucun paiement."""
    require_prep()
    key = (idempotency_key or "").strip()
    if not key:
        raise PrepRefusal("cle_absente", 400)
    if len(key) > 80:
        raise PrepRefusal("cle_idempotence_trop_longue", 400)
    row = _lock(int(booking.id))
    if row is None or int(row.company_id) != int(company_id):
        raise PrepRefusal("dossier_absent", 404)
    if int(row.driver_id) != int(driver_id):
        raise PrepRefusal("chauffeur_non_affecte", 403)
    if row.state != "received" or row.net_chf is None or row.kind != "collection":
        raise PrepRefusal("remise_sans_reception", 409)
    existing = db.session.scalar(
        select(CashHandover).where(
            CashHandover.collection_id == int(row.id),
            CashHandover.idempotency_key == key,
        )
    )
    held, _handed = collection_still_held(row)
    if amount_chf in (None, ""):
        amount = held
    else:
        try:
            amount = money(amount_chf)
        except ValueError as exc:
            raise PrepRefusal("montant_invalide", 400) from exc
    if amount <= 0:
        raise PrepRefusal("montant_nul", 400)
    if existing is not None:
        if Decimal(existing.amount_chf) != amount:
            raise PrepRefusal("conflit_idempotence", 409)
        view = _handover_view(row, existing, replay=True)
        db.session.rollback()
        return view
    if amount > held:
        raise PrepRefusal("depassement_autorise", 409)
    handover = CashHandover(
        collection_id=row.id,
        company_id=row.company_id,
        driver_id=row.driver_id,
        amount_chf=amount,
        idempotency_key=key,
        state="declared",
        declared_by_user_id=int(actor_id),
    )
    nested = db.session.begin_nested()
    db.session.add(handover)
    try:
        db.session.flush()
    except IntegrityError:
        nested.rollback()
        raced = db.session.scalar(
            select(CashHandover).where(
                CashHandover.collection_id == int(row.id),
                CashHandover.idempotency_key == key,
            )
        )
        if raced is None or Decimal(raced.amount_chf) != amount:
            raise PrepRefusal("conflit_idempotence", 409) from None
        view = _handover_view(row, raced, replay=True)
        db.session.rollback()
        return view
    else:
        nested.commit()
    _event(
        row,
        key=key,
        action="declare_handover",
        before=row.state,
        after=row.state,
        actor_role="driver",
        actor_id=actor_id,
        payload={
            "request_fingerprint": "handover",
            "amount_chf": f"{amount:.2f}",
        },
    )
    db.session.commit()
    stored = db.session.get(CashHandover, handover.id)
    if stored is None:
        raise PrepRefusal("remise_absente", 404)
    return _handover_view(row, stored, replay=False)


def confirm_handover(
    collection_id: int,
    *,
    company_id: int,
    actor_id: int,
    idempotency_key: str = "",
) -> dict[str, Any]:
    """Confirme une remise déjà déclarée. Le chauffeur n'atteint pas cette fonction."""
    require_prep()
    row = db.session.scalar(
        select(CashCollection)
        .where(
            CashCollection.id == int(collection_id),
            CashCollection.company_id == int(company_id),
        )
        .with_for_update()
    )
    if row is None:
        raise PrepRefusal("dossier_absent", 404)
    key = (idempotency_key or "").strip()
    if key:
        handover = db.session.scalar(
            select(CashHandover).where(
                CashHandover.collection_id == int(row.id),
                CashHandover.idempotency_key == key,
            )
        )
        if handover is None:
            raise PrepRefusal("remise_absente", 404)
    else:
        found = _handovers_of(int(row.id))
        pending = [item for item in found if item.state == "declared"]
        done = [item for item in found if item.state == "confirmed"]
        if len(pending) == 1:
            handover = pending[0]
        elif not pending and len(done) == 1:
            view = _handover_view(row, done[0], replay=True)
            db.session.rollback()
            return view
        elif not found:
            raise PrepRefusal("remise_absente", 404)
        else:
            raise PrepRefusal("remise_ambigue", 409)
    if int(handover.company_id) != int(company_id):
        raise PrepRefusal("autre_entreprise", 404)
    if handover.state == "confirmed":
        view = _handover_view(row, handover, replay=True)
        db.session.rollback()
        return view
    held, _handed = collection_still_held(row)
    amount = Decimal(handover.amount_chf)
    if amount > held:
        raise PrepRefusal("depassement_autorise", 409)
    handover.state = "confirmed"
    handover.confirmed_by_user_id = int(actor_id)
    handover.confirmed_at = datetime.now(UTC)
    _event(
        row,
        key=f"confirm-{key or handover.id}",
        action="confirm_handover",
        before=row.state,
        after=row.state,
        actor_role="company",
        actor_id=actor_id,
        payload={
            "request_fingerprint": "confirm-handover",
            "amount_chf": f"{amount:.2f}",
            "idempotency_key": handover.idempotency_key,
        },
    )
    db.session.commit()
    return _handover_view(row, handover, replay=False)


def _reconciliation_lines(collection_id: int) -> list[CashReconciliation]:
    return list(
        db.session.scalars(
            select(CashReconciliation)
            .where(CashReconciliation.collection_id == int(collection_id))
            .order_by(CashReconciliation.id)
        ).all()
    )


def _open_variance(origin: CashReconciliation) -> Decimal:
    children = db.session.scalars(
        select(CashReconciliation.variance_chf).where(
            CashReconciliation.regularizes_id == int(origin.id)
        )
    ).all()
    total = Decimal(origin.variance_chf) + sum(
        (Decimal(amount) for amount in children), Decimal("0.00")
    )
    return total.quantize(Decimal("0.01"))


def _line_payload(item: CashReconciliation) -> dict[str, Any]:
    open_amount = None
    if item.status == "constate":
        open_amount = _cents(_open_variance(item))
    return {
        "id": int(item.id),
        "handover_id": int(item.handover_id),
        "collection_id": int(item.collection_id),
        "company_id": int(item.company_id),
        "driver_id": int(item.driver_id),
        "expected_chf": _cents(Decimal(item.expected_chf)),
        "counted_chf": _cents(Decimal(item.counted_chf)),
        "variance_chf": _cents(Decimal(item.variance_chf)),
        "open_variance_chf": open_amount,
        "reason": item.reason,
        "status": item.status,
        "idempotency_key": item.idempotency_key,
        "regularizes_id": None
        if item.regularizes_id is None
        else int(item.regularizes_id),
        "author_user_id": int(item.author_user_id),
        "created_at": None if item.created_at is None else item.created_at.isoformat(),
    }


def _target_handover(row: CashCollection, payload: dict) -> CashHandover:
    found = _handovers_of(int(row.id))
    raw_id = payload.get("handover_id")
    if raw_id not in (None, ""):
        try:
            wanted = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise PrepRefusal("remise_absente", 404) from exc
        match = next((item for item in found if int(item.id) == wanted), None)
        if match is None:
            raise PrepRefusal("remise_absente", 404)
        if match.state != "confirmed":
            raise PrepRefusal("remise_non_confirmee", 409)
        return match
    confirmed = [item for item in found if item.state == "confirmed"]
    if len(confirmed) == 1:
        return confirmed[0]
    if not confirmed:
        raise PrepRefusal("remise_non_confirmee", 409)
    raise PrepRefusal("remise_ambigue", 409)


def _operation_key(payload: dict, fallback: str) -> str:
    key = str(payload.get("idempotency_key") or fallback).strip()
    if not key:
        raise PrepRefusal("cle_absente", 400)
    if len(key) > 80:
        raise PrepRefusal("cle_idempotence_trop_longue", 400)
    return key


def _reason(payload: dict, *, default: str | None) -> str:
    motive = str(payload.get("reason") or "").strip()
    if not motive:
        if default is None:
            raise PrepRefusal("motif_requis", 400)
        motive = default
    if len(motive) > 400:
        raise PrepRefusal("motif_trop_long", 400)
    return motive


def _same_line(
    stored: CashReconciliation,
    *,
    handover_id: int,
    expected: Decimal,
    counted: Decimal,
    reason: str,
    status: str,
    regularizes_id: int | None,
) -> bool:
    stored_parent = (
        None if stored.regularizes_id is None else int(stored.regularizes_id)
    )
    return (
        int(stored.handover_id) == int(handover_id)
        and Decimal(stored.expected_chf) == expected
        and Decimal(stored.counted_chf) == counted
        and stored.reason == reason
        and stored.status == status
        and stored_parent == regularizes_id
    )


def _replay_locked(row: CashCollection) -> dict[str, Any]:
    view = _view(row)
    view["replay"] = True
    db.session.rollback()
    return view


def _lock_collection(collection_id: int, company_id: int) -> CashCollection | None:
    return db.session.scalar(
        select(CashCollection)
        .where(
            CashCollection.id == int(collection_id),
            CashCollection.company_id == int(company_id),
        )
        .with_for_update()
    )


def _store_line(
    row: CashCollection,
    *,
    handover: CashHandover,
    expected: Decimal,
    counted: Decimal,
    reason: str,
    status: str,
    key: str,
    actor_id: int,
    regularizes_id: int | None,
    action: str,
) -> None:
    variance = (counted - expected).quantize(Decimal("0.01"))
    line = CashReconciliation(
        handover_id=int(handover.id),
        collection_id=int(row.id),
        company_id=int(row.company_id),
        driver_id=int(handover.driver_id),
        expected_chf=expected,
        counted_chf=counted,
        variance_chf=variance,
        reason=reason,
        status=status,
        idempotency_key=key,
        regularizes_id=regularizes_id,
        author_user_id=int(actor_id),
    )
    nested = db.session.begin_nested()
    db.session.add(line)
    try:
        db.session.flush()
    except IntegrityError:
        nested.rollback()
        raced = db.session.scalar(
            select(CashReconciliation).where(
                CashReconciliation.company_id == int(row.company_id),
                CashReconciliation.idempotency_key == key,
            )
        )
        if raced is not None and _same_line(
            raced,
            handover_id=int(handover.id),
            expected=expected,
            counted=counted,
            reason=reason,
            status=status,
            regularizes_id=regularizes_id,
        ):
            return
        existing = db.session.scalar(
            select(CashReconciliation).where(
                CashReconciliation.handover_id == int(handover.id),
                CashReconciliation.status == "constate",
            )
        )
        if existing is not None and status == "constate":
            raise PrepRefusal("rapprochement_deja_enregistre", 409) from None
        raise PrepRefusal("conflit_idempotence", 409) from None
    else:
        nested.commit()
    _event(
        row,
        key=f"{action}-{key}",
        action=action,
        before=row.state,
        after=row.state,
        actor_role="company",
        actor_id=actor_id,
        payload={
            "request_fingerprint": "cash-variance",
            "handover_id": int(handover.id),
            "expected_chf": _cents(expected),
            "counted_chf": _cents(counted),
            "variance_chf": _cents(variance),
            "reason": reason,
            "status": status,
            "regularizes_id": regularizes_id,
        },
    )


def _regularize(
    row: CashCollection,
    payload: dict,
    *,
    actor_id: int,
) -> dict[str, Any]:
    try:
        origin_id = int(payload.get("regularizes_id"))
    except (TypeError, ValueError) as exc:
        raise PrepRefusal("ecart_absent", 404) from exc
    origin = db.session.scalar(
        select(CashReconciliation)
        .where(
            CashReconciliation.id == origin_id,
            CashReconciliation.company_id == int(row.company_id),
            CashReconciliation.collection_id == int(row.id),
        )
        .with_for_update()
    )
    if origin is None or origin.status != "constate":
        raise PrepRefusal("ecart_absent", 404)
    handover = db.session.get(CashHandover, int(origin.handover_id))
    if handover is None or handover.state != "confirmed":
        raise PrepRefusal("remise_non_confirmee", 409)
    try:
        amount = money(payload.get("amount_chf"))
    except ValueError as exc:
        raise PrepRefusal("montant_invalide", 400) from exc
    if amount <= 0:
        raise PrepRefusal("montant_nul", 400)
    reason = _reason(payload, default=None)
    key = _operation_key(payload, f"regularise-{origin.id}")
    stored = db.session.scalar(
        select(CashReconciliation).where(
            CashReconciliation.company_id == int(row.company_id),
            CashReconciliation.idempotency_key == key,
        )
    )
    if stored is not None:
        same_target = (
            stored.status == "regularisation"
            and stored.regularizes_id is not None
            and int(stored.regularizes_id) == int(origin.id)
            and int(stored.handover_id) == int(handover.id)
            and stored.reason == reason
            and abs(Decimal(stored.variance_chf)) == amount
        )
        if not same_target:
            raise PrepRefusal("conflit_idempotence", 409)
        return _replay_locked(row)
    open_amount = _open_variance(origin)
    if open_amount == 0:
        raise PrepRefusal("ecart_deja_solde", 409)
    if amount > abs(open_amount):
        raise PrepRefusal("depassement_autorise", 409)
    signed = amount if open_amount < 0 else -amount
    expected = open_amount
    counted = (open_amount + signed).quantize(Decimal("0.01"))
    due_before = Decimal(row.amount_due_chf)
    net_before = row.net_chf
    kept_variance = Decimal(origin.variance_chf)
    kept_expected = Decimal(origin.expected_chf)
    kept_counted = Decimal(origin.counted_chf)
    kept_reason = origin.reason
    _store_line(
        row,
        handover=handover,
        expected=expected,
        counted=counted,
        reason=reason,
        status="regularisation",
        key=key,
        actor_id=actor_id,
        regularizes_id=int(origin.id),
        action="regularize_variance",
    )
    db.session.commit()
    db.session.refresh(row)
    if Decimal(row.amount_due_chf) != due_before or row.net_chf != net_before:
        raise PrepRefusal("montant_modifie", 500)
    fresh = db.session.get(CashReconciliation, int(origin.id))
    if (
        fresh is None
        or Decimal(fresh.variance_chf) != kept_variance
        or Decimal(fresh.expected_chf) != kept_expected
        or Decimal(fresh.counted_chf) != kept_counted
        or fresh.reason != kept_reason
        or fresh.status != "constate"
    ):
        raise PrepRefusal("ecart_modifie", 500)
    return _view(row)


def reconcile(
    collection_id: int,
    payload: dict,
    *,
    company_id: int,
    actor_id: int,
) -> dict[str, Any]:
    """Enregistre un écart ou une régularisation. N'écrit aucun paiement."""
    require_prep()
    row = _lock_collection(collection_id, company_id)
    if row is None:
        raise PrepRefusal("dossier_absent", 404)
    if payload.get("regularizes_id") not in (None, ""):
        return _regularize(row, payload, actor_id=actor_id)
    handover = _target_handover(row, payload)
    if int(handover.company_id) != int(company_id):
        raise PrepRefusal("autre_entreprise", 404)
    try:
        counted = money(payload.get("counted_chf"))
    except ValueError as exc:
        raise PrepRefusal("montant_invalide", 400) from exc
    if counted < 0:
        raise PrepRefusal("montant_nul", 400)
    expected = Decimal(handover.amount_chf).quantize(Decimal("0.01"))
    reason = _reason(payload, default="rapprochement")
    key = _operation_key(payload, f"reconcile-{handover.id}")
    stored = db.session.scalar(
        select(CashReconciliation).where(
            CashReconciliation.company_id == int(company_id),
            CashReconciliation.idempotency_key == key,
        )
    )
    if stored is not None:
        if not _same_line(
            stored,
            handover_id=int(handover.id),
            expected=expected,
            counted=counted,
            reason=reason,
            status="constate",
            regularizes_id=None,
        ):
            raise PrepRefusal("conflit_idempotence", 409)
        return _replay_locked(row)
    existing = db.session.scalar(
        select(CashReconciliation).where(
            CashReconciliation.handover_id == int(handover.id),
            CashReconciliation.status == "constate",
        )
    )
    if existing is not None:
        raise PrepRefusal("rapprochement_deja_enregistre", 409)
    due_before = Decimal(row.amount_due_chf)
    net_before = row.net_chf
    _store_line(
        row,
        handover=handover,
        expected=expected,
        counted=counted,
        reason=reason,
        status="constate",
        key=key,
        actor_id=actor_id,
        regularizes_id=None,
        action="reconcile",
    )
    db.session.commit()
    db.session.refresh(row)
    if Decimal(row.amount_due_chf) != due_before or row.net_chf != net_before:
        raise PrepRefusal("montant_modifie", 500)
    return _view(row)


def render_control_csv(cases: list[dict[str, Any]]) -> str:
    """Export de contrôle. Les colonnes historiques restent devant les écarts."""
    header = (
        "chauffeur;date;transport;recu;monnaie;net;detention;remise;rapprochement;"
        "exception;document;collection;remise_id;attendu;confirme;ecart;motif;statut"
    )
    lines = [header]
    for item in cases:
        details = item.get("reconciliations") or [None]
        if not details:
            details = [None]
        for detail in details:
            exception = (
                "administrative"
                if item.get("state") in {"administrative", "returned"}
                else ""
            )
            variance = "" if detail is None else detail.get("variance_chf") or ""
            lines.append(
                ";".join(
                    [
                        ""
                        if item.get("driver_id") is None
                        else str(item.get("driver_id")),
                        str(item.get("created_at") or "")[:10],
                        ""
                        if item.get("booking_id") is None
                        else str(item.get("booking_id")),
                        ""
                        if item.get("tendered_chf") is None
                        else str(item.get("tendered_chf")),
                        ""
                        if item.get("change_chf") is None
                        else str(item.get("change_chf")),
                        "" if item.get("net_chf") is None else str(item.get("net_chf")),
                        str(item.get("state") or ""),
                        str(item.get("handover_state") or ""),
                        ""
                        if item.get("reconciliation_variance_chf") is None
                        else str(item.get("reconciliation_variance_chf")),
                        exception,
                        "RECETTE — CONTROLE FICTIF",
                        "" if item.get("id") is None else str(item.get("id")),
                        "" if detail is None else str(detail.get("handover_id") or ""),
                        "" if detail is None else str(detail.get("expected_chf") or ""),
                        "" if detail is None else str(detail.get("counted_chf") or ""),
                        variance,
                        ""
                        if detail is None
                        else str(detail.get("reason") or "").replace(";", ","),
                        "" if detail is None else str(detail.get("status") or ""),
                    ]
                )
            )
    return "\n".join(lines)


def read_collection(
    booking: Any, *, driver_id: int, company_id: int
) -> dict[str, Any] | None:
    """Consulte le registre espèces sans ouvrir une nouvelle collecte.

    Les gardes d'écriture (``facture_emise``, ``deja_paye``, …) n'effacent
    pas l'historique : elles désactivent seulement ``collect_allowed``.
    """
    require_prep()
    authorize_settlement_history_read(
        booking, driver_id=driver_id, company_id=company_id
    )
    existing = db.session.scalar(
        select(CashCollection.id)
        .where(CashCollection.booking_id == int(booking.id))
        .order_by(CashCollection.id)
        .limit(1)
    )
    if existing is None:
        return None
    view = describe_cash(booking)
    view["booking_id"] = int(booking.id)
    view["payment_mode"] = "especes"
    try:
        refuse_booking(booking, driver_id=driver_id, company_id=company_id)
    except PrepRefusal as refusal:
        view["collect_allowed"] = False
        view["refusal_reason"] = str(refusal.code)
        actions = dict(view.get("actions") or {})
        actions["declare_cash"] = False
        view["actions"] = actions
    return view


def _cents(value: Decimal) -> str:
    return f"{value.quantize(Decimal('0.01')):.2f}"


def _empty_bucket() -> dict[str, Decimal]:
    zero = Decimal("0.00")
    return {
        "collected": zero,
        "corrections": zero,
        "refunds": zero,
        "handed": zero,
    }


def _bucket_payload(
    company_id: int, driver_id: int | None, parts: dict[str, Decimal]
) -> dict[str, Any]:
    held = (
        parts["collected"] - parts["corrections"] - parts["refunds"] - parts["handed"]
    ).quantize(Decimal("0.01"))
    body = {
        "company_id": int(company_id),
        "collected_chf": _cents(parts["collected"]),
        "corrections_chf": _cents(parts["corrections"]),
        "refunds_chf": _cents(parts["refunds"]),
        "handed_over_chf": _cents(parts["handed"]),
        "held_chf": _cents(held),
    }
    if driver_id is not None:
        body["driver_id"] = int(driver_id)
        body["driver_label"] = _driver_label(int(driver_id))
    return body


def _movement(
    row: CashCollection,
    *,
    nature: str,
    amount: Decimal,
    counted: bool,
    confirmed: bool,
) -> dict[str, Any]:
    return {
        "company_id": int(row.company_id),
        "driver_id": int(row.driver_id),
        "booking_id": int(row.booking_id),
        "collection_id": int(row.id),
        "compensates_collection_id": None
        if row.compensates_collection_id is None
        else int(row.compensates_collection_id),
        "nature": nature,
        "amount_chf": _cents(amount),
        "state": row.state,
        "disposition": row.disposition,
        "confirmed": confirmed,
        "counted": counted,
        "at": None if row.created_at is None else row.created_at.isoformat(),
    }


def _cash_position(
    company_id: int,
    rows: list[CashCollection],
    handovers: list[CashHandover],
) -> dict[str, Any]:
    """Espèces encore détenues. Le journal de caisse est la seule source.

    Un remboursement confirmé est la compensation ``disposition=refund``.
    La ligne ``invoice_refund`` n'est pas relue : elle décrirait le même
    événement une seconde fois. Une imputation de facture n'entre pas.
    """
    by_collection = {int(row.id): row for row in rows}
    confirmed: dict[int, list[CashHandover]] = {}
    for item in handovers:
        if item.state == "confirmed":
            confirmed.setdefault(int(item.collection_id), []).append(item)
    buckets: dict[int, dict[str, Decimal]] = {}
    movements: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []

    def bucket(driver_id: int) -> dict[str, Decimal]:
        found = buckets.get(int(driver_id))
        if found is None:
            found = _empty_bucket()
            buckets[int(driver_id)] = found
        return found

    for row in rows:
        driver_id = int(row.driver_id)
        bucket(driver_id)
        net = None if row.net_chf is None else Decimal(row.net_chf)
        if row.kind == "compensation" and net is not None and net < 0:
            nature = "remboursement" if row.disposition == "refund" else "correction"
            target = "refunds" if nature == "remboursement" else "corrections"
            bucket(driver_id)[target] += -net
            movements.append(
                _movement(
                    row,
                    nature=nature,
                    amount=net,
                    counted=True,
                    confirmed=True,
                )
            )
            continue
        counts = False
        if row.kind == "collection" and net is not None:
            counts = imputable(
                state=row.state,
                disposition=row.disposition,
                net=net,
                due=Decimal(row.amount_due_chf),
            )
        if counts and net is not None:
            bucket(driver_id)["collected"] += net
            movements.append(
                _movement(
                    row,
                    nature="encaissement",
                    amount=net,
                    counted=True,
                    confirmed=True,
                )
            )
            for handover in confirmed.get(int(row.id), []):
                amount = Decimal(handover.amount_chf)
                bucket(driver_id)["handed"] += amount
                movements.append(
                    {
                        "company_id": int(company_id),
                        "driver_id": driver_id,
                        "booking_id": int(row.booking_id),
                        "collection_id": int(row.id),
                        "compensates_collection_id": None,
                        "handover_id": int(handover.id),
                        "nature": "remise",
                        "amount_chf": _cents(-amount),
                        "state": handover.state,
                        "disposition": None,
                        "confirmed": True,
                        "counted": True,
                        "at": None
                        if handover.confirmed_at is None
                        else handover.confirmed_at.isoformat(),
                    }
                )
            continue
        review.append(
            _movement(
                row,
                nature="a_controler",
                amount=Decimal("0.00") if net is None else net,
                counted=False,
                confirmed=False,
            )
        )

    for handover in handovers:
        if handover.state == "confirmed":
            origin = by_collection.get(int(handover.collection_id))
            if origin is None or origin.kind != "collection":
                review.append(
                    {
                        "company_id": int(company_id),
                        "driver_id": int(handover.driver_id),
                        "booking_id": None
                        if origin is None
                        else int(origin.booking_id),
                        "collection_id": int(handover.collection_id),
                        "compensates_collection_id": None,
                        "handover_id": int(handover.id),
                        "nature": "remise",
                        "amount_chf": _cents(Decimal(handover.amount_chf)),
                        "state": handover.state,
                        "disposition": None,
                        "confirmed": True,
                        "counted": False,
                        "at": None
                        if handover.confirmed_at is None
                        else handover.confirmed_at.isoformat(),
                    }
                )
            continue
        origin = by_collection.get(int(handover.collection_id))
        review.append(
            {
                "company_id": int(company_id),
                "driver_id": int(handover.driver_id),
                "booking_id": None if origin is None else int(origin.booking_id),
                "collection_id": int(handover.collection_id),
                "compensates_collection_id": None,
                "handover_id": int(handover.id),
                "nature": "remise_declaree",
                "amount_chf": _cents(Decimal(handover.amount_chf)),
                "state": handover.state,
                "disposition": None,
                "confirmed": False,
                "counted": False,
                "at": None
                if handover.created_at is None
                else handover.created_at.isoformat(),
            }
        )

    drivers = [
        _bucket_payload(company_id, driver_id, buckets[driver_id])
        for driver_id in sorted(buckets)
    ]
    total = _empty_bucket()
    for parts in buckets.values():
        for key in total:
            total[key] += parts[key]
    position = _bucket_payload(company_id, None, total)
    position["drivers"] = drivers
    position["movements"] = movements
    position["to_review"] = review
    return position


def list_company(company_id: int) -> dict[str, Any]:
    """Lecture du registre. N'écrit aucun paiement, avoir ou mouvement."""
    require_prep()
    cid = int(company_id)
    rows = db.session.scalars(
        select(CashCollection)
        .where(CashCollection.company_id == cid)
        .order_by(CashCollection.id)
    ).all()
    handovers = db.session.scalars(
        select(CashHandover)
        .where(CashHandover.company_id == cid)
        .order_by(CashHandover.id)
    ).all()
    return {
        "banner": _ledger_banner(),
        "production_activation": False,
        "cash_book_activated": False,
        "cases": [_view(row) for row in rows],
        "position": _cash_position(cid, rows, handovers),
    }


def reprint_receipt(collection_id: int, *, company_id: int) -> dict[str, Any]:
    require_prep()
    row = db.session.get(CashCollection, collection_id)
    if row is None or int(row.company_id) != int(company_id):
        raise PrepRefusal("dossier_absent", 404)
    receipt = db.session.scalar(
        select(CashReceipt).where(CashReceipt.collection_id == row.id)
    )
    if receipt is None:
        raise PrepRefusal("recu_absent", 404)
    receipt.duplicate_count = int(receipt.duplicate_count) + 1
    db.session.commit()
    body = _receipt_body(row, receipt, duplicate=True)
    body["pdf_base64"] = specimen_pdf(body["lines"])
    return body


def read_receipt(booking: Any, *, driver_id: int, company_id: int) -> dict[str, Any]:
    """Lecture authentifiée. N'incrémente pas le duplicata et n'écrit rien."""
    require_prep()
    _assigned(booking, driver_id=driver_id, company_id=company_id)
    row = _lock(int(booking.id))
    if row is None:
        raise PrepRefusal("dossier_absent", 404)
    receipt = db.session.scalar(
        select(CashReceipt).where(CashReceipt.collection_id == row.id)
    )
    if receipt is None:
        raise PrepRefusal("quittance_absente", 404)
    return _receipt_body(row, receipt, duplicate=False)


def prepare_receipt(
    booking: Any,
    payload: dict,
    *,
    driver_id: int,
    company_id: int,
    actor_id: int | None,
) -> dict[str, Any]:
    """Crée le spécimen d'une réception déjà confirmée. Aucun montant n'est réécrit."""
    require_prep()
    _assigned(booking, driver_id=driver_id, company_id=company_id)
    key = str(payload.get("idempotency_key") or "").strip()
    if not key:
        raise PrepRefusal("cle_absente", 400)
    row = _lock(int(booking.id))
    if row is None:
        raise PrepRefusal("dossier_absent", 404)
    replay = _replay(row, key, payload)
    if replay is not None:
        return replay
    frozen = _money_snapshot(row)
    if not _reception_confirmed(row):
        raise PrepRefusal("reception_non_confirmee", 409)
    existing = db.session.scalar(
        select(CashReceipt).where(CashReceipt.collection_id == row.id)
    )
    if existing is None:
        _receipt(row)
        _event(
            row,
            key=key,
            action="receipt_prepare",
            before=row.state,
            after=row.state,
            actor_role="driver",
            actor_id=actor_id,
            payload={
                "request_fingerprint": fingerprint(payload),
                "simulated": True,
                "invoice_issued": False,
            },
        )
        _assert_unchanged(row, frozen)
        db.session.commit()
    view = _view(row)
    view["replay"] = existing is not None
    return view


def dispatch_receipt(
    booking: Any,
    payload: dict,
    *,
    driver_id: int,
    company_id: int,
    actor_id: int | None,
) -> dict[str, Any]:
    """Capture un envoi fictif. N'appelle aucun prestataire et ne rouvre pas la caisse."""
    require_prep()
    _assigned(booking, driver_id=driver_id, company_id=company_id)
    channel = str(payload.get("channel") or "").strip()
    if channel not in {"email", "sms", "skip"}:
        raise PrepRefusal("canal_inconnu", 400)
    key = str(payload.get("idempotency_key") or "").strip()
    if not key:
        raise PrepRefusal("cle_absente", 400)
    row = _lock(int(booking.id))
    if row is None:
        raise PrepRefusal("dossier_absent", 404)
    prior = db.session.scalar(
        select(CashCollectionEvent).where(
            CashCollectionEvent.collection_id == row.id,
            CashCollectionEvent.idempotency_key == key,
        )
    )
    if prior is not None and prior.payload.get("channel") not in (None, channel):
        raise PrepRefusal("conflit_idempotence", 409)
    replay = _replay(row, key, payload)
    if replay is not None:
        return replay
    frozen = _money_snapshot(row)
    receipt = db.session.scalar(
        select(CashReceipt).where(CashReceipt.collection_id == row.id)
    )
    if receipt is None:
        raise PrepRefusal("quittance_absente", 409)
    email, phone = _patient_contacts(booking)
    status, destination = _dispatch_status(
        channel,
        email=email,
        phone=phone,
        simulate_failure=bool(payload.get("simulate_failure")),
    )
    _event(
        row,
        key=key,
        action="receipt_dispatch",
        before=row.state,
        after=row.state,
        actor_role="driver",
        actor_id=actor_id,
        payload={
            "request_fingerprint": fingerprint(payload),
            "channel": channel,
            "status": status,
            "sent": False,
            "destination": destination,
            "message": _capture_message(receipt.number),
            "access_path": (
                f"/api/v1/driver/me/bookings/{int(row.booking_id)}/cash-receipt"
            ),
            "public_link": None,
            "invoice_issued": False,
        },
    )
    _assert_unchanged(row, frozen)
    db.session.commit()
    view = _view(row)
    view["replay"] = False
    return view


def list_client_receipts(*, company_id: int, client_id: int) -> dict[str, Any]:
    """Quittances fictives d'un patient de l'entreprise. Lecture seule."""
    require_prep()
    from models.booking import Booking
    from models.client import Client

    client = db.session.get(Client, int(client_id))
    if client is None or int(client.company_id) != int(company_id):
        raise PrepRefusal("patient_absent", 404)
    rows = db.session.execute(
        select(CashCollection, CashReceipt)
        .join(Booking, Booking.id == CashCollection.booking_id)
        .join(CashReceipt, CashReceipt.collection_id == CashCollection.id)
        .where(
            CashCollection.company_id == int(company_id),
            Booking.client_id == int(client_id),
        )
        .order_by(CashReceipt.id)
    ).all()
    return {
        "banner": "QUITTANCES FICTIVES — NON ÉMISES",
        "invoice_issued": False,
        "documents": [
            _receipt_body(collection, receipt, duplicate=False)
            for collection, receipt in rows
        ],
    }


def _receipt(row: CashCollection) -> None:
    number = f"SPECIMEN-{int(row.company_id)}-{int(row.booking_id)}"
    taken = db.session.scalar(
        select(CashReceipt).where(
            CashReceipt.company_id == int(row.company_id),
            CashReceipt.number == number,
        )
    )
    if taken is not None and int(taken.collection_id) != int(row.id):
        number = f"{number}-{int(row.id)}"
    already = db.session.scalar(
        select(CashReceipt).where(CashReceipt.collection_id == int(row.id))
    )
    if already is not None:
        return
    db.session.add(
        CashReceipt(
            collection_id=row.id,
            company_id=row.company_id,
            number=number,
            issued=False,
            legal_status="pending",
            specimen=True,
            duplicate_count=0,
            currency="CHF",
            amount_due_chf=row.amount_due_chf,
            tendered_chf=row.tendered_chf,
            change_chf=row.change_chf,
            net_chf=row.net_chf,
        )
    )


def _receipt_body(
    row: CashCollection, receipt: CashReceipt, *, duplicate: bool
) -> dict[str, Any]:
    company = db.session.get(Company, int(row.company_id))
    due = Decimal(receipt.amount_due_chf)
    received = Decimal(receipt.net_chf)
    balance = (due - received).quantize(Decimal("0.01"))
    partial = balance > 0
    when = "" if receipt.created_at is None else receipt.created_at.isoformat()
    lines = [
        "SIMULATION — QUITTANCE FICTIVE — NON ÉMISE",
        "Ce document n'est ni une facture ni un mandat.",
        "Consultation authentifiée seulement. Aucun lien public.",
        f"Entreprise : {company.name if company is not None else row.company_id}",
        f"Mission : {row.booking_id}",
        f"Bénéficiaire : {row.debtor_label}",
        f"Date : {when}",
        f"Montant dû : {due:.2f} CHF",
        f"Montant remis : {Decimal(receipt.tendered_chf):.2f} CHF",
        f"Monnaie rendue : {Decimal(receipt.change_chf):.2f} CHF",
        f"Montant reçu : {received:.2f} CHF",
        f"Solde restant : {balance:.2f} CHF",
        (
            "Réception partielle. Ce n'est pas un paiement intégral."
            if partial
            else "Réception du montant dû. Quittance fictive, non émise."
        ),
        f"Référence : {receipt.number}",
        "DUPLICATA DE SPÉCIMEN" if duplicate else "SPÉCIMEN",
    ]
    return {
        "issued": False,
        "legal_status": "pending",
        "specimen": True,
        "simulated": True,
        "partial": partial,
        "invoice": False,
        "mandate": False,
        "duplicate": duplicate,
        "number": receipt.number,
        "booking_id": int(row.booking_id),
        "company_name": company.name if company is not None else None,
        "beneficiary": row.debtor_label,
        "amount_due_chf": f"{due:.2f}",
        "amount_received_chf": f"{received:.2f}",
        "balance_chf": f"{balance:.2f}",
        "change_chf": f"{Decimal(receipt.change_chf):.2f}",
        "access_path": f"/api/v1/driver/me/bookings/{int(row.booking_id)}/cash-receipt",
        "public_link": None,
        "lines": lines,
        "production_activation": False,
        "real_collection": False,
    }


def specimen_pdf(lines: list[str]) -> str:
    """PDF minimal, en base64, sans bibliothèque externe."""
    import base64

    text = " | ".join(lines).replace("\\", " ").replace("(", " ").replace(")", " ")
    stream = f"BT /F1 11 Tf 40 780 Td ({text}) Tj ET".encode("latin-1", "replace")
    objects = [
        b"1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n",
        b"2 0 obj<< /Type /Pages /Count 1 /Kids [3 0 R] >>endobj\n",
        (
            b"3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 842 1191] "
            b"/Contents 4 0 R /Resources<< /Font<< /F1 5 0 R >> >> >>endobj\n"
        ),
        (
            f"4 0 obj<< /Length {len(stream)} >>stream\n".encode()
            + stream
            + b"\nendstream endobj\n"
        ),
        b"5 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\n",
    ]
    body = b"%PDF-1.4\n" + b"".join(objects) + b"trailer<< /Root 1 0 R >>\n%%EOF"
    return base64.b64encode(body).decode("ascii")


def _assigned(booking: Any, *, driver_id: int, company_id: int) -> None:
    if int(booking.company_id) != int(company_id):
        raise PrepRefusal("autre_entreprise", 404)
    if booking.driver_id is None or int(booking.driver_id) != int(driver_id):
        raise PrepRefusal("chauffeur_non_affecte", 403)


def _reception_confirmed(row: CashCollection) -> bool:
    if row.disposition == "refused" or row.state in {
        "prepared",
        "uncertain",
        "returned",
    }:
        return False
    if row.net_chf is None or Decimal(row.net_chf) <= 0:
        return False
    return row.state == "received" or row.disposition == "partial"


def _money_snapshot(row: CashCollection) -> tuple[str, str, str, str, str, str | None]:
    return (
        f"{Decimal(row.amount_due_chf):.2f}",
        "" if row.tendered_chf is None else f"{Decimal(row.tendered_chf):.2f}",
        "" if row.change_chf is None else f"{Decimal(row.change_chf):.2f}",
        "" if row.net_chf is None else f"{Decimal(row.net_chf):.2f}",
        row.state,
        row.disposition,
    )


def _assert_unchanged(
    row: CashCollection, frozen: tuple[str, str, str, str, str, str | None]
) -> None:
    if _money_snapshot(row) != frozen:
        raise PrepRefusal("montant_modifie", 500)


def _patient_contacts(booking: Any) -> tuple[str | None, str | None]:
    from models.client import Client
    from models.user import User

    phone = str(getattr(booking, "passenger_phone", "") or "").strip() or None
    if phone is None:
        phone = str(getattr(booking, "requester_phone", "") or "").strip() or None
    email = None
    client_id = getattr(booking, "client_id", None)
    if client_id is None:
        return email, phone
    client = db.session.get(Client, int(client_id))
    if client is None or client.user_id is None:
        return email, phone
    user = db.session.get(User, int(client.user_id))
    if user is None:
        return email, phone
    if user.email:
        email = str(user.email).strip() or None
    if phone is None and getattr(user, "phone", None):
        phone = str(user.phone).strip() or None
    return email, phone


def _fictive_email(value: str) -> bool:
    host = value.strip().lower().split("@")[-1]
    return host in {"example.com", "example.ch"} or host.endswith(".invalid")


def _fictive_phone(value: str) -> bool:
    digits = "".join(character for character in value if character.isdigit())
    return digits.startswith("4100") or digits.startswith("000")


def _mask(value: str | None) -> str | None:
    if not value:
        return None
    if "@" in value:
        local, host = value.split("@", 1)
        return f"{local[:1]}***@{host}"
    digits = "".join(character for character in value if character.isdigit())
    return f"***{digits[-2:]}" if digits else None


def _dispatch_status(
    channel: str, *, email: str | None, phone: str | None, simulate_failure: bool
) -> tuple[str, str | None]:
    if channel == "skip":
        return "skipped", None
    raw = email if channel == "email" else phone
    if not raw:
        return "absent", None
    fictive = _fictive_email(raw) if channel == "email" else _fictive_phone(raw)
    if not fictive:
        return "blocked", _mask(raw)
    if simulate_failure:
        return "failed", _mask(raw)
    return "captured", _mask(raw)


def _last_dispatch(events: list[CashCollectionEvent]) -> dict[str, Any] | None:
    found = [event for event in events if event.action == "receipt_dispatch"]
    if not found:
        return None
    payload = found[-1].payload or {}
    return {
        "channel": payload.get("channel"),
        "status": payload.get("status"),
        "sent": False,
        "destination": payload.get("destination"),
    }


def _capture_message(number: str) -> str:
    return (
        "Quittance fictive de recette. Aucun envoi réel. "
        f"Référence {number}. "
        "Consultez-la dans l'espace authentifié. Aucun lien public."
    )


def _lock_booking(booking_id: int) -> None:
    locked = db.session.execute(
        text("SELECT id FROM booking WHERE id = :booking_id FOR UPDATE"),
        {"booking_id": int(booking_id)},
    ).scalar()
    if locked is None:
        raise PrepRefusal("dossier_absent", 404)


def _collection_by_key(booking_id: int, key: str) -> CashCollection | None:
    return db.session.scalar(
        select(CashCollection).where(
            CashCollection.booking_id == int(booking_id),
            CashCollection.client_operation_key == key,
        )
    )


def _prepared_collection(booking_id: int) -> CashCollection | None:
    return db.session.scalar(
        select(CashCollection)
        .where(
            CashCollection.booking_id == int(booking_id),
            CashCollection.state == "prepared",
            CashCollection.net_chf.is_(None),
        )
        .order_by(CashCollection.id)
        .limit(1)
    )


def _cash_lines(booking_id: int) -> list[CashLine]:
    rows = db.session.scalars(
        select(CashCollection).where(CashCollection.booking_id == int(booking_id))
    ).all()
    lines: list[CashLine] = []
    for row in rows:
        if row.net_chf is None:
            continue
        lines.append(
            CashLine(
                kind=row.kind or "collection",
                state=row.state,
                disposition=row.disposition,
                net_chf=Decimal(row.net_chf),
                due_chf=Decimal(row.amount_due_chf),
            )
        )
    return lines


def _recognized(booking_id: int) -> Decimal:
    lines = _cash_lines(booking_id)
    if not lines:
        return Decimal("0.00")
    return recognized_net(lines)


def _position(booking_id: int, due: Decimal) -> tuple[Decimal, Decimal]:
    recognized = _recognized(booking_id)
    remaining = (due - recognized).quantize(Decimal("0.01"))
    if remaining < 0:
        remaining = Decimal("0.00")
    return recognized, remaining


def _operation_lines(booking_id: int) -> list[dict[str, Any]]:
    rows = db.session.scalars(
        select(CashCollection)
        .where(CashCollection.booking_id == int(booking_id))
        .order_by(CashCollection.id)
    ).all()
    lines: list[dict[str, Any]] = []
    for row in rows:
        if row.net_chf is None:
            continue
        net = Decimal(row.net_chf)
        correction = row.kind == "compensation"
        refund = correction and row.disposition == "refund"
        due = Decimal(row.amount_due_chf)
        line_imputable = False
        if not correction:
            line_imputable = imputable(
                state=row.state,
                disposition=row.disposition,
                net=net,
                due=due,
            )
        if refund:
            label = "Remboursement"
            nature = "remboursement"
        elif correction:
            label = "Correction"
            nature = "correction"
        else:
            label = "Encaissement"
            nature = "encaissement"
        lines.append(
            {
                "id": int(row.id),
                "booking_id": int(row.booking_id),
                "kind": row.kind or "collection",
                "nature": nature,
                "display": f"{label} {net:.2f} CHF",
                "net_chf": f"{net:.2f}",
                "state": row.state,
                "disposition": row.disposition,
                "payment_mode": "especes",
                "reference": f"CASH-{int(row.id)}",
                "created_at": None
                if row.created_at is None
                else row.created_at.isoformat(),
                "imputable": line_imputable,
                "client_operation_key": row.client_operation_key,
                "compensates_collection_id": row.compensates_collection_id,
                "allocated_invoice_payment_id": None,
            }
        )
    return lines


def _booking_history(booking_id: int) -> list[dict[str, Any]]:
    events = db.session.scalars(
        select(CashCollectionEvent)
        .join(CashCollection, CashCollectionEvent.collection_id == CashCollection.id)
        .where(CashCollection.booking_id == int(booking_id))
        .order_by(CashCollectionEvent.id)
    ).all()
    return [
        {
            "action": event.action,
            "to_state": event.to_state,
            "actor_role": event.actor_role,
            "actor_id": event.actor_id,
            "collection_id": event.collection_id,
            "at": None if event.created_at is None else event.created_at.isoformat(),
        }
        for event in events
    ]


def _with_position(
    view: dict[str, Any], booking_id: int, due: Decimal
) -> dict[str, Any]:
    recognized, remaining = _position(booking_id, due)
    operations = _operation_lines(booking_id)
    view["amount_due_chf"] = f"{due:.2f}"
    view["recognized_net_chf"] = f"{recognized:.2f}"
    view["remaining_due_chf"] = f"{remaining:.2f}"
    view["balance_chf"] = f"{remaining:.2f}"
    view["allocated_invoice_payment_id"] = None
    view["collect_allowed"] = remaining > 0
    view["actions"] = {
        "declare_cash": remaining > 0,
        "compensate_cash": recognized > 0,
        "tap_to_pay": False,
        "real_receipt": False,
    }
    view["operations"] = operations
    view["history"] = _booking_history(booking_id)
    view.setdefault("replay", False)
    current = next(
        (line for line in operations if line["id"] == view.get("id")),
        None,
    )
    if current is None:
        view["imputable"] = bool(view.get("imputable"))
    else:
        view["imputable"] = current["imputable"]
        view["net_chf"] = current["net_chf"]
        view["kind"] = current["kind"]
        view["client_operation_key"] = current["client_operation_key"]
    return view


def describe_cash(booking: Any) -> dict[str, Any]:
    """Position lue par le chauffeur. N'écrit ni paiement ni facture."""
    return _with_position(
        {"id": None, "replay": False, "imputable": False},
        int(booking.id),
        amount_due(booking),
    )


def _lock(booking_id: int) -> CashCollection | None:
    return db.session.scalar(
        select(CashCollection)
        .where(CashCollection.booking_id == booking_id)
        .order_by(CashCollection.id)
        .limit(1)
        .with_for_update()
    )


def _replay(row: CashCollection, key: str, payload: dict) -> dict[str, Any] | None:
    event = db.session.scalar(
        select(CashCollectionEvent).where(
            CashCollectionEvent.collection_id == row.id,
            CashCollectionEvent.idempotency_key == key,
        )
    )
    if event is None:
        return None
    if event.payload.get("request_fingerprint") != fingerprint(payload):
        raise PrepRefusal("conflit_idempotence", 409)
    view = _view(row)
    view["replay"] = True
    view["accounted"] = row.state == "received"
    if event.action == "receipt_dispatch":
        payload = event.payload or {}
        view["receipt_transmission"] = {
            "channel": payload.get("channel"),
            "status": payload.get("status"),
            "sent": False,
            "destination": payload.get("destination"),
        }
    return view


def _event(
    row: CashCollection,
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
        CashCollectionEvent(
            collection_id=row.id,
            idempotency_key=key,
            action=action,
            from_state=before,
            to_state=after,
            actor_role=actor_role,
            actor_id=actor_id,
            payload=payload,
        )
    )


def _notify_refusal(row: CashCollection, reason: str) -> None:
    from models.company_notification import CompanyNotification

    dedupe = f"cash_refusal:{row.booking_id}"
    existing = db.session.scalar(
        select(CompanyNotification).where(
            CompanyNotification.company_id == int(row.company_id),
            CompanyNotification.dedupe_key == dedupe,
        )
    )
    if existing is not None:
        return
    db.session.add(
        CompanyNotification(
            company_id=int(row.company_id),
            event_type="cash_payment_refused",
            title="Refus de paiement signalé",
            message=(
                f"Course {row.booking_id}, {row.debtor_label}. "
                f"Motif : {reason}. Aucun encaissement réel, aucune facture."
            ),
            metadata_json={
                "booking_id": int(row.booking_id),
                "driver_id": int(row.driver_id),
                "amount_due_chf": f"{Decimal(row.amount_due_chf):.2f}",
                "simulation": True,
            },
            is_read=False,
            dedupe_key=dedupe,
        )
    )


def _driver_label(driver_id: int) -> str:
    from models import Driver
    from models.user import User

    driver = db.session.get(Driver, int(driver_id))
    if driver is None or driver.user_id is None:
        return str(driver_id)
    user = db.session.get(User, int(driver.user_id))
    if user is None:
        return str(driver_id)
    name = " ".join(part for part in (user.first_name, user.last_name) if part).strip()
    return name or str(user.username or driver_id)


def _invoice_allocation_snapshot(
    collection_id: int, *, booking_id: int | None = None
) -> dict[str, Any] | None:
    """Lecture seule : paiement imputé lié à la collecte, sans recalcul client.

    Retourne ``None`` si aucune imputation. Laisse remonter les erreurs SQL
    pour que l'appelant distingue ``unavailable`` de ``not_allocated``.
    """
    row = (
        db.session.execute(
            text(
                """
                SELECT ip.id AS payment_id,
                       ip.amount AS imputed_amount,
                       i.invoice_number,
                       i.total_amount,
                       i.balance_due,
                       i.amount_paid,
                       il.total_with_vat AS line_ttc
                FROM invoice_payments AS ip
                JOIN invoices AS i ON i.id = ip.invoice_id
                LEFT JOIN invoice_lines AS il
                  ON il.invoice_id = i.id
                 AND (:booking_id IS NULL OR il.reservation_id = :booking_id)
                WHERE ip.cash_collection_id = :collection_id
                ORDER BY il.id NULLS LAST
                LIMIT 1
                """
            ),
            {
                "collection_id": int(collection_id),
                "booking_id": None if booking_id is None else int(booking_id),
            },
        )
        .mappings()
        .first()
    )
    if row is None:
        return None
    invoice_ttc = Decimal(str(row["total_amount"])).quantize(Decimal("0.01"))
    line_raw = row.get("line_ttc")
    line_ttc = (
        Decimal(str(line_raw)).quantize(Decimal("0.01"))
        if line_raw is not None
        else invoice_ttc
    )
    return {
        "payment_id": int(row["payment_id"]),
        "invoice_number": str(row["invoice_number"] or ""),
        "total_amount": invoice_ttc,
        "line_ttc": line_ttc,
        "balance_due": Decimal(str(row["balance_due"])).quantize(Decimal("0.01")),
        "amount_paid": Decimal(str(row["amount_paid"])).quantize(Decimal("0.01")),
        "imputed_amount": Decimal(str(row["imputed_amount"])).quantize(Decimal("0.01")),
    }


def _apply_invoice_allocation_to_view(
    view: dict[str, Any], *, collection_id: int
) -> dict[str, Any]:
    """Aligne Règlements sur la facture TTC lorsque l'imputation existe.

    Sans lien paiement↔collecte : ``not_allocated`` (Non imputé).
    Échec de lecture : ``unavailable`` (information non renseignée).
    """
    booking_raw = view.get("booking_id")
    try:
        booking_id = int(booking_raw) if booking_raw is not None else None
    except (TypeError, ValueError):
        booking_id = None
    try:
        snap = _invoice_allocation_snapshot(collection_id, booking_id=booking_id)
    except Exception:
        view["allocation_status"] = "unavailable"
        view["allocated_invoice_payment_id"] = None
        view["invoice_number"] = None
        return view
    if snap is None:
        view["allocation_status"] = "not_allocated"
        view["allocated_invoice_payment_id"] = None
        view["invoice_number"] = None
        return view
    view["allocation_status"] = "allocated"
    view["allocated_invoice_payment_id"] = snap["payment_id"]
    view["invoice_number"] = snap["invoice_number"] or None
    view["invoice_issued"] = True
    # Contractuel = TTC de la ligne de transport ; solde = solde facture (backend).
    view["amount_due_chf"] = f"{snap['line_ttc']:.2f}"
    view["balance_chf"] = f"{snap['balance_due']:.2f}"
    view["remaining_due_chf"] = f"{snap['balance_due']:.2f}"
    return view


def _view(row: CashCollection) -> dict[str, Any]:
    company = db.session.get(Company, int(row.company_id))
    events = db.session.scalars(
        select(CashCollectionEvent)
        .where(CashCollectionEvent.collection_id == row.id)
        .order_by(CashCollectionEvent.id)
    ).all()
    found_handovers = _handovers_of(int(row.id))
    pending_handovers = [item for item in found_handovers if item.state == "declared"]
    handover = (
        pending_handovers[-1]
        if pending_handovers
        else (found_handovers[-1] if found_handovers else None)
    )
    lines = _reconciliation_lines(int(row.id))
    reconciliation = None
    if handover is not None:
        reconciliation = next(
            (
                item
                for item in lines
                if int(item.handover_id) == int(handover.id)
                and item.status == "constate"
            ),
            None,
        )
    receipt = db.session.scalar(
        select(CashReceipt).where(CashReceipt.collection_id == row.id)
    )
    due = Decimal(row.amount_due_chf)
    if row.net_chf is not None and (
        row.disposition == "partial" or row.state == "received"
    ):
        received = Decimal(row.net_chf)
    else:
        received = Decimal("0.00")
    balance = (due - received).quantize(Decimal("0.01"))
    refusal_reason = next(
        (
            str(event.payload.get("reason"))
            for event in reversed(events)
            if event.action == "refuse_payment" and event.payload.get("reason")
        ),
        None,
    )
    line_imputable = False
    if row.net_chf is not None:
        line_imputable = imputable(
            state=row.state,
            disposition=row.disposition,
            net=Decimal(row.net_chf),
            due=due,
        )
    view = {
        "id": row.id,
        "booking_id": row.booking_id,
        "company_id": row.company_id,
        "company_name": company.name if company is not None else None,
        "driver_id": row.driver_id,
        "driver_label": _driver_label(int(row.driver_id)),
        "debtor_label": row.debtor_label,
        "amount_due_chf": f"{Decimal(row.amount_due_chf):.2f}",
        "tendered_chf": None
        if row.tendered_chf is None
        else f"{Decimal(row.tendered_chf):.2f}",
        "change_chf": None
        if row.change_chf is None
        else f"{Decimal(row.change_chf):.2f}",
        "net_chf": None if row.net_chf is None else f"{Decimal(row.net_chf):.2f}",
        "amount_received_chf": f"{received:.2f}",
        "balance_chf": f"{balance:.2f}",
        "refusal_reason": refusal_reason,
        "invoice_issued": False,
        "invoice_number": None,
        "allocation_status": "not_allocated",
        "real_collection": False,
        "local_ledger": _ledger_banner() == _LOCAL_BANNER,
        "currency": "CHF",
        "state": row.state,
        "disposition": row.disposition,
        "imputable": line_imputable,
        "banner": _ledger_banner(),
        "production_activation": False,
        "legally_approved": False,
        "policy_status": "DRAFT",
        "accounted": row.state == "received",
        "receipt_issued": False,
        "receipt_number": None if receipt is None else receipt.number,
        "receipt_lines": []
        if receipt is None
        else _receipt_body(row, receipt, duplicate=False)["lines"],
        "receipt_partial": False
        if receipt is None
        else Decimal(receipt.amount_due_chf) - Decimal(receipt.net_chf) > 0,
        "receipt_transmission": _last_dispatch(events),
        "handover_state": None if handover is None else handover.state,
        "handovers": [_handover_public(item) for item in found_handovers],
        "reconciliation_id": None if reconciliation is None else int(reconciliation.id),
        "reconciliation_variance_chf": None
        if reconciliation is None
        else f"{Decimal(reconciliation.variance_chf):.2f}",
        "reconciliations": [_line_payload(item) for item in lines],
        "created_at": None if row.created_at is None else row.created_at.isoformat(),
        "handover_declared_at": None
        if handover is None or handover.created_at is None
        else handover.created_at.isoformat(),
        "handover_confirmed_at": None
        if handover is None or handover.confirmed_at is None
        else handover.confirmed_at.isoformat(),
        "reconciliation_at": None
        if reconciliation is None or reconciliation.created_at is None
        else reconciliation.created_at.isoformat(),
        "history": [
            {
                "action": event.action,
                "to_state": event.to_state,
                "actor_role": event.actor_role,
                "actor_id": event.actor_id,
                "at": None
                if event.created_at is None
                else event.created_at.isoformat(),
            }
            for event in events
        ],
    }
    positioned = _with_position(view, int(row.booking_id), due)
    return _apply_invoice_allocation_to_view(positioned, collection_id=int(row.id))
