"""Préparation financière après une acceptation opérationnelle.

Le drapeau est éteint : le parcours réel ne lit ni n'écrit ces tables.
Aucune politique active n'est créée et le payeur n'est pas verrouillé.
"""

from __future__ import annotations

import logging

from sqlalchemy import text

logger = logging.getLogger(__name__)


def note_acceptance_financial_review(booking, *, company_id: int) -> str:
    """Calcule la compatibilité sans empêcher l'acceptation du transport."""
    from domain.end_of_trip_settlement.activation import acceptance_snapshot_enabled

    if not acceptance_snapshot_enabled():
        return "skipped"
    from ext import db

    try:
        db.session.flush()
    except Exception:
        logger.exception("Flush opérationnel avant revue financière")
        raise
    nested = db.session.begin_nested()
    try:
        reason = _review(booking, company_id=int(company_id))
        nested.commit()
        return reason
    except Exception:
        nested.rollback()
        logger.exception("Revue financière ignorée, acceptation conservée")
        return "financial_review_unavailable"


def _review(booking, *, company_id: int) -> str:
    from domain.end_of_trip_settlement.company_configuration import (
        CompanySettlementSettings,
        acceptance_write_plan,
        prepare_acceptance_snapshot,
        settings_for_acceptance,
    )
    from ext import db

    booking_id = getattr(booking, "id", None)
    if booking_id is None:
        return "booking_unknown"
    owner = db.session.execute(
        text("SELECT company_id FROM booking WHERE id = :id FOR UPDATE"),
        {"id": int(booking_id)},
    ).first()
    if owner is None:
        return "booking_unknown"
    current_owner = owner[0]
    if current_owner is not None and int(current_owner) != int(company_id):
        return "company_mismatch"
    active = db.session.execute(
        text(
            "SELECT id FROM booking_settlement_policy "
            "WHERE booking_id = :id AND status = 'active' FOR UPDATE"
        ),
        {"id": int(booking_id)},
    ).first()
    if active is not None:
        return "policy_already_recorded"
    capability = (
        db.session.execute(
            text(
                "SELECT tap_to_pay_enabled, cash_enabled, "
                "direct_patient_invoice_enabled, tap_to_pay_preferred, "
                "cash_preferred, direct_patient_invoice_preferred "
                "FROM company_settlement_capability WHERE company_id = :id"
            ),
            {"id": int(company_id)},
        )
        .mappings()
        .first()
    )
    authorized = CompanySettlementSettings(
        tap_to_pay_enabled=bool(capability["tap_to_pay_enabled"])
        if capability
        else False,
        cash_enabled=bool(capability["cash_enabled"]) if capability else False,
        direct_patient_invoice_enabled=bool(
            capability["direct_patient_invoice_enabled"]
        )
        if capability
        else False,
    )
    preferences = CompanySettlementSettings(
        tap_to_pay_preferred=bool(capability["tap_to_pay_preferred"])
        if capability
        else False,
        cash_preferred=bool(capability["cash_preferred"]) if capability else False,
        direct_patient_invoice_preferred=bool(
            capability["direct_patient_invoice_preferred"]
        )
        if capability
        else False,
    )
    settings = settings_for_acceptance(preferences=preferences, authorized=authorized)
    payer = _payer(getattr(booking, "billed_to_type", None))
    created_via = str(getattr(booking, "created_via", "") or "")
    invoice_issued = getattr(booking, "invoice_line_id", None) is not None
    already_paid = _payment_completed(int(booking_id))
    is_round_trip = bool(getattr(booking, "is_round_trip", False))
    decision = prepare_acceptance_snapshot(
        company_assigned=True,
        payer=payer,
        created_via=created_via,
        financial_responsibility_locked=False,
        client_has_established_account=False,
        settings=settings,
        already_paid=already_paid,
        invoice_issued=invoice_issued,
        is_round_trip=is_round_trip,
        round_trip_collection_approved=False,
        return_time_defined=None,
        patient_notice=_patient_notice(int(booking_id), int(company_id)),
        announced_amount_chf=_amount(getattr(booking, "price_amount", None)),
    )
    if acceptance_write_plan(decision) != "no_policy_row":
        return "refused"
    return decision.reason


def _amount(value: object):
    from decimal import Decimal

    if value is None or value == "":
        return None
    return Decimal(str(value))


def _payer(value: object) -> str:
    text_value = getattr(value, "value", value)
    raw = str(text_value or "").lower()
    if raw == "clinic":
        return "institution"
    if raw == "insurance":
        return "third_party"
    if raw == "patient":
        return "patient"
    return ""


def _payment_completed(booking_id: int) -> bool:
    from ext import db

    row = db.session.execute(
        text(
            "SELECT 1 FROM payment WHERE booking_id = :id "
            "AND status::text = 'COMPLETED' LIMIT 1"
        ),
        {"id": booking_id},
    ).first()
    return row is not None


def _patient_notice(booking_id: int, company_id: int):
    from datetime import datetime
    from decimal import Decimal

    from domain.end_of_trip_settlement.company_configuration import PatientNotice
    from ext import db

    ready = db.session.execute(
        text("SELECT to_regclass('public.booking_settlement_patient_notice')")
    ).scalar()
    if ready is None:
        return None
    row = (
        db.session.execute(
            text(
                "SELECT amount_chf, conditions_version, author_user_id, "
                "confirmed_at, channel FROM booking_settlement_patient_notice "
                "WHERE booking_id = :booking_id AND company_id = :company_id "
                "AND receipt_proven = false "
                "ORDER BY confirmed_at DESC LIMIT 1"
            ),
            {"booking_id": booking_id, "company_id": company_id},
        )
        .mappings()
        .first()
    )
    if row is None:
        return None
    confirmed_at = row["confirmed_at"]
    if isinstance(confirmed_at, datetime) and confirmed_at.tzinfo is None:
        return None
    amount = row["amount_chf"]
    return PatientNotice(
        confirmed=True,
        author_user_id=int(row["author_user_id"]),
        confirmed_at=confirmed_at,
        conditions_version=row["conditions_version"],
        amount_chf=Decimal(str(amount)) if amount is not None else None,
        debtor_label="patient",
        channel=row["channel"],
        auto_checked=False,
    )
