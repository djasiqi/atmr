"""Préférences de règlement d'une entreprise.

Les colonnes « enabled » restent fausses : une préférence n'autorise
ni Tap to Pay, ni les espèces, ni la facture directe.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import select

from domain.end_of_trip_settlement.access import actor_may
from domain.end_of_trip_settlement.company_configuration import declare_patient_notice
from models.settlement_policy import (
    BookingSettlementPatientNotice,
    CompanySettlementCapability,
    CompanySettlementCapabilityRevision,
)

_ENABLED = (
    "tap_to_pay_enabled",
    "cash_enabled",
    "direct_patient_invoice_enabled",
)


def settlement_view(company_id: int) -> dict:
    row = CompanySettlementCapability.query.filter_by(company_id=company_id).first()
    history = (
        CompanySettlementCapabilityRevision.query.filter_by(company_id=company_id)
        .order_by(CompanySettlementCapabilityRevision.revision.asc())
        .all()
    )
    return {
        "company_id": company_id,
        "preferences": {
            "tap_to_pay": bool(row.tap_to_pay_preferred) if row else False,
            "cash": bool(row.cash_preferred) if row else False,
            "direct_patient_invoice": bool(row.direct_patient_invoice_preferred)
            if row
            else False,
        },
        "effectively_available": {
            "tap_to_pay": False,
            "cash": False,
            "direct_patient_invoice": False,
        },
        "activation_enabled": False,
        "tap_to_pay_provider": "pending_worldline",
        "conventions_unchanged": True,
        "history": [
            {
                "revision": item.revision,
                "tap_to_pay": item.tap_to_pay_preferred,
                "cash": item.cash_preferred,
                "direct_patient_invoice": item.direct_patient_invoice_preferred,
                "author_user_id": item.author_user_id,
                "created_at": item.created_at.isoformat() if item.created_at else None,
            }
            for item in history
        ],
    }


def update_preferences(
    *,
    company_id: int,
    actor_role: str,
    actor_user_id: int,
    same_company: bool,
    payload: dict,
) -> dict:
    if not same_company or not actor_may(actor_role, "configure_capabilities"):
        raise PermissionError("Seul un responsable de cette entreprise peut modifier.")
    if any(payload.get(name) for name in _ENABLED):
        raise ValueError("Une préférence ne rend pas le moyen disponible.")
    if payload.get("activate_collection") or payload.get("activation_enabled"):
        raise ValueError("L'activation réelle des paiements est désactivée.")
    from ext import db

    row = db.session.execute(
        select(CompanySettlementCapability)
        .where(CompanySettlementCapability.company_id == company_id)
        .with_for_update()
    ).scalar_one_or_none()
    if row is None:
        row = CompanySettlementCapability(
            company_id=company_id,
            tap_to_pay_enabled=False,
            cash_enabled=False,
            direct_patient_invoice_enabled=False,
            tap_to_pay_preferred=False,
            cash_preferred=False,
            direct_patient_invoice_preferred=False,
        )
        db.session.add(row)
        db.session.flush()
    row.tap_to_pay_enabled = False
    row.cash_enabled = False
    row.direct_patient_invoice_enabled = False
    preferences = payload.get("preferences")
    if not isinstance(preferences, dict):
        preferences = payload
    wanted_tap = bool(preferences.get("tap_to_pay"))
    wanted_cash = bool(preferences.get("cash"))
    wanted_invoice = bool(preferences.get("direct_patient_invoice"))
    changed = (
        row.tap_to_pay_preferred != wanted_tap
        or row.cash_preferred != wanted_cash
        or row.direct_patient_invoice_preferred != wanted_invoice
    )
    row.tap_to_pay_preferred = wanted_tap
    row.cash_preferred = wanted_cash
    row.direct_patient_invoice_preferred = wanted_invoice
    row.updated_by_user_id = actor_user_id
    if changed:
        last = db.session.execute(
            select(CompanySettlementCapabilityRevision.revision)
            .where(CompanySettlementCapabilityRevision.company_id == company_id)
            .order_by(CompanySettlementCapabilityRevision.revision.desc())
            .limit(1)
        ).scalar_one_or_none()
        db.session.add(
            CompanySettlementCapabilityRevision(
                company_id=company_id,
                revision=int(last or 0) + 1,
                tap_to_pay_preferred=row.tap_to_pay_preferred,
                cash_preferred=row.cash_preferred,
                direct_patient_invoice_preferred=row.direct_patient_invoice_preferred,
                author_user_id=actor_user_id,
            )
        )
    db.session.flush()
    return settlement_view(company_id)


def record_patient_notice(
    *,
    booking_id: int,
    company_id: int,
    actor_role: str,
    actor_user_id: int,
    same_company: bool,
    payload: dict,
) -> dict:
    if not same_company or not actor_may(actor_role, "configure_capabilities"):
        raise PermissionError("Seul un responsable de cette entreprise peut confirmer.")
    notice = declare_patient_notice(
        confirmed=bool(payload.get("confirmed")),
        auto_checked=bool(payload.get("auto_checked")),
        author_user_id=actor_user_id,
        confirmed_at=datetime.now(UTC),
        conditions_version=payload.get("conditions_version"),
        amount_chf=_amount(payload.get("amount_chf")),
        debtor_label=payload.get("debtor_label"),
        channel=payload.get("channel"),
    )
    if not notice.confirmed:
        raise ValueError("La déclaration est incomplète ou précochée.")
    from ext import db

    row = BookingSettlementPatientNotice(
        booking_id=booking_id,
        company_id=company_id,
        amount_chf=notice.amount_chf,
        presented_modes=list(payload.get("presented_modes") or []),
        conditions_version=str(payload.get("conditions_version") or "").strip(),
        author_user_id=actor_user_id,
        confirmed_at=notice.confirmed_at,
        channel=str(payload.get("channel") or "").strip(),
        receipt_proven=False,
    )
    db.session.add(row)
    db.session.flush()
    return {
        "booking_id": booking_id,
        "company_id": company_id,
        "amount_chf": (
            str(notice.amount_chf) if notice.amount_chf is not None else None
        ),
        "presented_modes": row.presented_modes,
        "conditions_version": row.conditions_version,
        "author_user_id": actor_user_id,
        "confirmed_at": notice.confirmed_at.isoformat(),
        "channel": row.channel,
        "receipt_proven": False,
        "patient_acceptance_proven": False,
    }


def _amount(value: object) -> Decimal | None:
    if value is None or value == "":
        return None
    return Decimal(str(value))
