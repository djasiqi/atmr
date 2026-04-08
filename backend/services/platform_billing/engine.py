"""Moteur de calcul des brouillons de facturation plateforme (idempotent par période draft)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import func, or_

from ext import db
from models import Booking, BookingStatus, Company, CompanyPlatformBillingConfig
from models.enums import PlatformBillingLineType, PlatformBillingPeriodStatus
from models.platform_billing import (
    PlatformBillingPeriod,
    PlatformInvoice,
    PlatformInvoiceLine,
    PlatformSubscriptionPricing,
    PlatformSupportEntry,
)
from services.admin_booking_billing_kernel import (
    CLASSIFICATION_VERSION,
    QUALIFICATION_VERSION,
    build_pilotage_payload_for_booking,
    observed_transport_amount,
)
from services.admin_platform_bookings import _batch_list_transfer_flags
from services.platform_billing.eligibility import is_commissionable_platform
from services.platform_billing.money import money_round_chf
from services.platform_billing.time_bounds import zurich_month_bounds_utc

logger = logging.getLogger(__name__)

_SNAPSHOT_SUB_LABEL = (
    "Volume d'abonnement : courses créées sur la période (created_at), hors annulés."
)
_SUBSCRIPTION_LINE_LABEL = (
    "Abonnement plateforme — volume sur courses créées (created_at), hors annulés"
)


def _dispatch_mode_for_company(company: Company, cfg: CompanyPlatformBillingConfig | None) -> str:
    if cfg and cfg.dispatch_mode_override:
        return cfg.dispatch_mode_override
    dm = company.dispatch_mode
    return dm.value if hasattr(dm, "value") else str(dm)


def _active_config(company_id: int, at: datetime) -> CompanyPlatformBillingConfig | None:
    cfg = (
        CompanyPlatformBillingConfig.query.filter(
            CompanyPlatformBillingConfig.company_id == company_id,
            CompanyPlatformBillingConfig.is_active.is_(True),
            CompanyPlatformBillingConfig.is_billing_enabled.is_(True),
        )
        .order_by(CompanyPlatformBillingConfig.id.desc())
        .first()
    )
    if not cfg:
        return None
    if cfg.effective_from is not None:
        ef = cfg.effective_from
        if ef.tzinfo is None:
            ef = ef.replace(tzinfo=UTC)
        if at < ef:
            return None
    if cfg.effective_to is not None:
        et = cfg.effective_to
        if et.tzinfo is None:
            et = et.replace(tzinfo=UTC)
        if at > et:
            return None
    return cfg


def subscription_volume_count(company_id: int, year: int, month: int) -> int:
    start, end = zurich_month_bounds_utc(year, month)
    n = (
        db.session.query(func.count(Booking.id))
        .filter(
            Booking.company_id == company_id,
            Booking.created_at >= start,
            Booking.created_at <= end,
            Booking.status != BookingStatus.CANCELED,
        )
        .scalar()
    )
    return int(n or 0)


def select_subscription_tier(
    dispatch_mode: str, volume: int
) -> PlatformSubscriptionPricing | None:
    rows = (
        PlatformSubscriptionPricing.query.filter_by(dispatch_mode=dispatch_mode)
        .order_by(PlatformSubscriptionPricing.volume_min.asc())
        .all()
    )
    for row in rows:
        if volume < row.volume_min:
            continue
        if row.volume_max is not None and volume > row.volume_max:
            continue
        return row
    return None


def _commission_bookings_for_month(company_id: int, year: int, month: int) -> list[Booking]:
    start, end = zurich_month_bounds_utc(year, month)
    return (
        Booking.query.filter(
            Booking.company_id == company_id,
            Booking.completed_at.isnot(None),
            Booking.completed_at >= start,
            Booking.completed_at <= end,
            Booking.status.in_([BookingStatus.COMPLETED, BookingStatus.RETURN_COMPLETED]),
        )
        .order_by(Booking.id.asc())
        .all()
    )


def recalculate_platform_period_drafts(period_id: int) -> dict[str, Any]:
    """Régénère tous les relevés entreprise pour une période draft (remplace les lignes)."""
    period = db.session.get(PlatformBillingPeriod, period_id)
    if not period:
        raise ValueError("Période introuvable")
    if period.status == PlatformBillingPeriodStatus.LOCKED.value:
        raise ValueError("Période verrouillée — recalcul interdit")

    # Supprimer les factures existantes (cascade lignes)
    PlatformInvoice.query.filter_by(period_id=period_id).delete()
    db.session.flush()

    configs = CompanyPlatformBillingConfig.query.filter(
        CompanyPlatformBillingConfig.is_active.is_(True),
        CompanyPlatformBillingConfig.is_billing_enabled.is_(True),
    ).all()
    at = datetime.now(UTC)
    generated = 0
    for cfg in configs:
        company = db.session.get(Company, cfg.company_id)
        if not company:
            continue
        active = _active_config(cfg.company_id, at)
        if not active:
            continue
        _build_invoice_for_company(period, company, active)
        generated += 1

    db.session.commit()
    return {"period_id": period_id, "invoices_generated": generated}


def _build_invoice_for_company(
    period: PlatformBillingPeriod,
    company: Company,
    cfg: CompanyPlatformBillingConfig,
) -> PlatformInvoice:
    year, month = period.billing_year, period.billing_month
    cid = company.id
    dm = _dispatch_mode_for_company(company, cfg)

    vol = subscription_volume_count(cid, year, month)
    tier = select_subscription_tier(dm, vol)
    sub_amount = tier.price_monthly if tier else Decimal("0.00")
    sub_amount = money_round_chf(sub_amount)

    lines_data: list[dict[str, Any]] = []

    sub_snap = {
        "rule": "subscription_volume_created_at_excluding_cancelled",
        "volume_count": vol,
        "dispatch_mode": dm,
        "subscription_pricing_id": tier.id if tier else None,
        "tier_label": tier.label if tier else None,
        "classification_version": CLASSIFICATION_VERSION,
        "qualification_version": QUALIFICATION_VERSION,
        "transparency_label": _SNAPSHOT_SUB_LABEL,
    }
    lines_data.append(
        {
            "line_type": PlatformBillingLineType.SUBSCRIPTION.value,
            "label": _SUBSCRIPTION_LINE_LABEL,
            "amount": sub_amount,
            "snapshot_json": sub_snap,
            "sort_order": 0,
        }
    )

    rate = cfg.commission_rate or Decimal("0")
    bookings = _commission_bookings_for_month(cid, year, month)
    flags = _batch_list_transfer_flags(bookings)
    commission_total = Decimal("0")
    commission_details: list[dict[str, Any]] = []
    for b in bookings:
        ht, hp = flags[b.id]
        pl = build_pilotage_payload_for_booking(
            b, has_transfer=ht, has_pending_transfer=hp
        )
        if not is_commissionable_platform(b, pl):
            continue
        ota = observed_transport_amount(b)
        if ota is None:
            continue
        raw = Decimal(str(ota)) * rate
        line_amt = money_round_chf(raw)
        commission_total += line_amt
        commission_details.append(
            {
                "booking_id": b.id,
                "commission": str(line_amt),
                "observed_transport_amount": ota,
            }
        )

    comm_snap = {
        "commission_rate": str(rate),
        "booking_count": len(commission_details),
        "details": commission_details[:500],
        "classification_version": CLASSIFICATION_VERSION,
        "qualification_version": QUALIFICATION_VERSION,
    }
    lines_data.append(
        {
            "line_type": PlatformBillingLineType.INSTITUTION_COMMISSION.value,
            "label": "Commission activité institution",
            "amount": money_round_chf(commission_total),
            "snapshot_json": comm_snap,
            "sort_order": 10,
        }
    )

    support_total = Decimal("0")
    support_rows = (
        PlatformSupportEntry.query.filter(
            PlatformSupportEntry.company_id == cid,
            PlatformSupportEntry.validated_at.isnot(None),
            or_(
                PlatformSupportEntry.billing_period_id.is_(None),
                PlatformSupportEntry.billing_period_id == period.id,
            ),
        )
        .all()
    )
    for se in support_rows:
        support_total += se.amount
        se.billing_period_id = period.id

    if support_rows:
        lines_data.append(
            {
                "line_type": PlatformBillingLineType.SUPPORT_TIME.value,
                "label": "Support et prestations",
                "amount": money_round_chf(support_total),
                "snapshot_json": {
                    "entry_ids": [r.id for r in support_rows],
                    "count": len(support_rows),
                },
                "sort_order": 20,
            }
        )

    total = sum(Decimal(str(x["amount"])) for x in lines_data)
    total = money_round_chf(total)

    inv = PlatformInvoice(
        company_id=cid,
        period_id=period.id,
        currency="CHF",
        subtotal_amount=total,
        total_amount=total,
    )
    db.session.add(inv)
    db.session.flush()

    for row in lines_data:
        db.session.add(
            PlatformInvoiceLine(
                invoice_id=inv.id,
                line_type=row["line_type"],
                label=row.get("label"),
                amount=row["amount"],
                snapshot_json=row.get("snapshot_json"),
                sort_order=row.get("sort_order", 0),
            )
        )

    return inv


def lock_platform_billing_period(period_id: int) -> PlatformBillingPeriod:
    period = db.session.get(PlatformBillingPeriod, period_id)
    if not period:
        raise ValueError("Période introuvable")
    period.status = PlatformBillingPeriodStatus.LOCKED.value
    db.session.commit()
    return period


def get_or_create_period(year: int, month: int) -> PlatformBillingPeriod:
    p = PlatformBillingPeriod.query.filter_by(
        billing_year=year, billing_month=month
    ).first()
    if p:
        return p
    p = PlatformBillingPeriod(
        billing_year=year,
        billing_month=month,
        status=PlatformBillingPeriodStatus.DRAFT.value,
    )
    db.session.add(p)
    db.session.commit()
    return p
