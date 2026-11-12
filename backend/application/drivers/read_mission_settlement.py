"""Lecture seule du règlement d'une mission. Aucune écriture."""

from __future__ import annotations

from decimal import Decimal

from domain.end_of_trip_settlement.mission_facts import MissionFacts, decide_mission
from domain.end_of_trip_settlement.policy import SettlementPolicy
from ext import db


def _text(value: object, default: str = "") -> str:
    raw = getattr(value, "value", value)
    text = str(raw or default).strip()
    return text or default


def _money(value: object) -> Decimal:
    if value is None:
        return Decimal("0.00")
    return Decimal(str(value)).quantize(Decimal("0.01"))


def _policy_from_row(row: object, booking_id: int) -> SettlementPolicy | None:
    if row is None:
        return None
    raw_modes = getattr(row, "authorized_modes", None) or []
    modes = frozenset(str(item) for item in raw_modes) & {
        "tap_to_pay",
        "cash",
        "invoice",
    }
    try:
        return SettlementPolicy(
            booking_id=booking_id,
            revision=int(row.revision),
            modality=str(row.modality),
            modes=modes,
            payment_scope=getattr(row, "payment_scope", None),
            collection_moment=getattr(row, "collection_moment", None),
            amount_chf=_money(row.amount_chf) if row.amount_chf is not None else None,
            currency="CHF",
            decision_origin=str(row.decision_origin),
            policy_version=str(row.policy_version),
            financial_responsibility_locked=bool(row.financial_responsibility_locked),
            invoice_signature_text=row.invoice_signature_text,
            invoice_signature_text_version=row.invoice_signature_text_version,
            validated_at=row.validated_at,
            validated_by_user_id=int(row.validated_by_user_id),
            round_trip_collection_approved=bool(
                getattr(row, "round_trip_collection_approved", False)
            ),
        )
    except (TypeError, ValueError):
        return None


def load_mission_decision(booking_id: int, driver: object) -> dict[str, object] | None:
    """Assemble les faits puis délègue au moteur. N'enregistre rien."""
    from models.booking import Booking
    from models.enums import PaymentStatus
    from models.payment import Payment
    from models.settlement_policy import (
        BookingSettlementPolicy,
        CompanySettlementCapability,
    )

    booking = db.session.get(Booking, booking_id)
    if booking is None:
        return None
    company_id = int(booking.company_id)
    root_id = int(booking.parent_booking_id or booking.id)
    siblings = (
        db.session.query(Booking.id, Booking.amount)
        .filter((Booking.id == root_id) | (Booking.parent_booking_id == root_id))
        .all()
    )
    leg_ids = [int(row.id) for row in siblings] or [booking_id]
    payments = (
        db.session.query(Payment.amount)
        .filter(
            Payment.booking_id.in_(leg_ids),
            Payment.status == PaymentStatus.COMPLETED,
        )
        .all()
    )
    paid = sum((_money(row.amount) for row in payments), Decimal("0.00"))
    group_amount = (
        sum((_money(row.amount) for row in siblings), Decimal("0.00"))
        if len(siblings) > 1
        else None
    )
    policy_row = (
        db.session.query(BookingSettlementPolicy)
        .filter_by(booking_id=booking_id, status="active")
        .one_or_none()
    )
    capability = db.session.get(CompanySettlementCapability, company_id)
    facts = MissionFacts(
        booking_id=booking_id,
        company_id=company_id,
        executing_company_id=int(booking.executing_company_id or company_id),
        reader_driver_id=int(driver.id),
        reader_company_id=int(driver.company_id),
        assigned_driver_id=int(booking.driver_id) if booking.driver_id else None,
        created_via=_text(booking.created_via, "legacy"),
        billed_to_type=_text(booking.billed_to_type, "patient"),
        billing_intent=None,
        billing_party_id=booking.billing_party_id,
        operational_status=_text(booking.status, "COMPLETED"),
        leg_amount_chf=_money(booking.amount),
        group_amount_chf=group_amount,
        is_round_trip=len(siblings) > 1 or bool(booking.is_return),
        is_return_leg=bool(booking.is_return),
        priced_leg_count=max(len(siblings), 1),
        invoice_already_issued=booking.invoice_line_id is not None,
        completed_payment_amount_chf=paid,
        debtor_display_name=getattr(booking, "customer_name", None),
        policy=_policy_from_row(policy_row, booking_id),
        tap_to_pay_enabled=bool(getattr(capability, "tap_to_pay_enabled", False)),
        cash_enabled=bool(getattr(capability, "cash_enabled", False)),
        direct_patient_invoice_enabled=bool(
            getattr(capability, "direct_patient_invoice_enabled", False)
        ),
        tap_to_pay_preferred=bool(getattr(capability, "tap_to_pay_preferred", False)),
        cash_preferred=bool(getattr(capability, "cash_preferred", False)),
        direct_patient_invoice_preferred=bool(
            getattr(capability, "direct_patient_invoice_preferred", False)
        ),
    )
    return decide_mission(facts)
