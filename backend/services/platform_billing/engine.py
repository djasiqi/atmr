"""Moteur de calcul des brouillons de facturation plateforme (idempotent par période draft)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import func, or_

from ext import db
from models import Booking, BookingStatus, Company, CompanyPlatformBillingConfig
from models.enums import (
    BookingBillingOrigin,
    PlatformBillingLineType,
    PlatformBillingPeriodStatus,
    PlatformStatementItemType,
    PlatformStatementStatus,
    SubscriptionPricingMode,
)
from models.platform_billing import (
    PlatformBillingCreditor,
    PlatformBillingPeriod,
    PlatformBillingStatementItem,
    PlatformInvoice,
    PlatformInvoiceLine,
    PlatformSubscriptionPricing,
    PlatformSubscriptionPricingGrid,
    PlatformSubscriptionPricingTier,
    PlatformSupportEntry,
)
from services.admin_booking_billing_kernel import (
    CLASSIFICATION_VERSION,
    QUALIFICATION_VERSION,
    build_pilotage_payload_for_booking,
)
from services.admin_platform_bookings import _batch_list_transfer_flags
from services.platform_billing.commissionable_amount import (
    AmountConfidence,
    resolve_commissionable_amount,
)
from services.platform_billing.contracts import (
    distinct_billable_company_ids,
    effective_config_for_period,
    month_start_zurich_utc,
)
from services.platform_billing.eligibility import is_commissionable_platform
from services.platform_billing.errors import BillingInvariantError
from services.platform_billing.money import money_round_chf
from services.platform_billing.time_bounds import (
    billing_period_has_ended,
    next_month_start_zurich_utc,
    zurich_month_bounds_utc,
)

logger = logging.getLogger(__name__)

CALCULATION_VERSION = 2

_SNAPSHOT_SUB_LABEL = (
    "Volume d'abonnement : courses créées sur la période, hors annulés."
)
_SUBSCRIPTION_LINE_LABEL = (
    "Abonnement plateforme — volume sur courses créées, hors annulés"
)
# Franchise TVA suisse (< 100'000 CHF CA) : taux 0 par défaut jusqu'à assujettissement.
_DEFAULT_TAX_RATE = Decimal("0.0000")


def _dispatch_mode_for_company(
    company: Company, cfg: CompanyPlatformBillingConfig | None
) -> str:
    if cfg and cfg.dispatch_mode_override:
        return cfg.dispatch_mode_override
    dm = company.dispatch_mode
    return dm.value if hasattr(dm, "value") else str(dm)


def _active_config(
    company_id: int, at: datetime
) -> CompanyPlatformBillingConfig | None:
    """Compatibilité : résolution par instant (fenêtre semi-ouverte [from, to))."""
    return effective_config_for_period(company_id, at)


def _product_flags_explicit(
    cfg: CompanyPlatformBillingConfig,
) -> tuple[bool, bool, bool]:
    """Flags produits tels que stockés (sans inférence legacy)."""
    return (
        bool(getattr(cfg, "own_portfolio_billing_enabled", False)),
        bool(getattr(cfg, "lirie_commission_enabled", False)),
        bool(getattr(cfg, "support_enabled", False)),
    )


def _is_legacy_product_flags_inferred(cfg: CompanyPlatformBillingConfig) -> bool:
    """True si billing actif sans aucun flag produit (compat V1)."""
    own, comm, support = _product_flags_explicit(cfg)
    return bool(cfg.is_billing_enabled) and not own and not comm and not support


def _product_flags(cfg: CompanyPlatformBillingConfig) -> tuple[bool, bool, bool]:
    """Compat V1 : si aucun flag produit mais billing enabled → tous activés."""
    own, comm, support = _product_flags_explicit(cfg)
    if _is_legacy_product_flags_inferred(cfg):
        return True, True, True
    return own, comm, support


def assert_billing_period_has_ended(
    year: int,
    month: int,
    *,
    now_utc: datetime | None = None,
) -> None:
    """Refuse validate / lock / issue tant que le mois Zurich n'est pas terminé."""
    if billing_period_has_ended(year, month, now_utc=now_utc):
        return
    ends_at = next_month_start_zurich_utc(year, month)
    raise BillingInvariantError(
        "PERIOD_STILL_OPEN",
        f"La période {month:02d}/{year} n’est pas terminée.",
        details={
            "billing_year": year,
            "billing_month": month,
            "period_ends_at": ends_at.isoformat(),
        },
    )


def build_platform_billing_period_readiness(
    period_id: int,
    *,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Readiness canonique pour clôturer une période (SSOT du lock)."""
    period = db.session.get(PlatformBillingPeriod, period_id)
    if not period:
        raise BillingInvariantError(
            "PERIOD_NOT_FOUND",
            "Période introuvable",
            status_code=404,
            details={"period_id": period_id},
        )

    year, month = int(period.billing_year), int(period.billing_month)
    period_ended = billing_period_has_ended(year, month, now_utc=now_utc)
    period_start = month_start_zurich_utc(year, month)
    invoices = PlatformInvoice.query.filter_by(period_id=period_id).all()

    counts = {
        "draft": 0,
        "calculated": 0,
        "needs_review": 0,
        "validated": 0,
        "locked": 0,
        "other": 0,
    }
    not_validated_company_ids: list[int] = []
    for inv in invoices:
        status = inv.statement_status or PlatformStatementStatus.DRAFT.value
        if status == PlatformStatementStatus.DRAFT.value:
            counts["draft"] += 1
            not_validated_company_ids.append(inv.company_id)
        elif status == PlatformStatementStatus.CALCULATED.value:
            counts["calculated"] += 1
            not_validated_company_ids.append(inv.company_id)
        elif status == PlatformStatementStatus.NEEDS_REVIEW.value:
            counts["needs_review"] += 1
            not_validated_company_ids.append(inv.company_id)
        elif status == PlatformStatementStatus.VALIDATED.value:
            counts["validated"] += 1
        elif status == PlatformStatementStatus.LOCKED.value:
            counts["locked"] += 1
        else:
            counts["other"] += 1
            not_validated_company_ids.append(inv.company_id)

    review_item_count = (
        db.session.query(func.count(PlatformBillingStatementItem.id))
        .join(
            PlatformInvoice,
            PlatformBillingStatementItem.statement_id == PlatformInvoice.id,
        )
        .filter(
            PlatformInvoice.period_id == period_id,
            PlatformBillingStatementItem.eligibility_status == "needs_review",
        )
        .scalar()
    )
    review_item_count = int(review_item_count or 0)

    invoice_company_ids = {inv.company_id for inv in invoices}
    missing_statement_company_ids: list[int] = []
    legacy_inferred_company_ids: list[int] = []
    warnings: list[dict[str, Any]] = []

    for company_id in distinct_billable_company_ids():
        cfg = effective_config_for_period(company_id, period_start)
        if cfg is None or not cfg.is_billing_enabled:
            continue
        if _is_legacy_product_flags_inferred(cfg):
            legacy_inferred_company_ids.append(company_id)
        own, comm, support = _product_flags_explicit(cfg)
        # Manquant bloquant : config enabled avec produits explicites, sans relevé
        if (own or comm or support) and company_id not in invoice_company_ids:
            missing_statement_company_ids.append(company_id)

    if legacy_inferred_company_ids:
        warnings.append(
            {
                "code": "LEGACY_PRODUCT_FLAGS_INFERRED",
                "company_ids": sorted(legacy_inferred_company_ids),
                "message": (
                    "Certaines configurations actives n’ont aucun produit "
                    "explicite (compatibilité V1)."
                ),
            }
        )

    blocking_reasons: list[dict[str, Any]] = []
    if not period_ended:
        blocking_reasons.append(
            {
                "code": "PERIOD_STILL_OPEN",
                "message": f"La période {month:02d}/{year} n’est pas terminée.",
                "details": {
                    "billing_year": year,
                    "billing_month": month,
                    "period_ends_at": next_month_start_zurich_utc(
                        year, month
                    ).isoformat(),
                },
            }
        )
    if not invoices:
        blocking_reasons.append(
            {
                "code": "NO_STATEMENTS",
                "message": "Aucun relevé à verrouiller",
            }
        )
    if not_validated_company_ids:
        blocking_reasons.append(
            {
                "code": "STATEMENTS_NOT_VALIDATED",
                "company_ids": sorted(set(not_validated_company_ids)),
                "message": "Tous les relevés doivent être VALIDATED avant clôture.",
            }
        )
    if review_item_count > 0:
        blocking_reasons.append(
            {
                "code": "STATEMENT_ITEMS_NEED_REVIEW",
                "count": review_item_count,
                "message": "Des éléments de relevé sont encore à contrôler.",
            }
        )
    if missing_statement_company_ids:
        blocking_reasons.append(
            {
                "code": "MISSING_EXPECTED_STATEMENTS",
                "company_ids": sorted(missing_statement_company_ids),
                "message": (
                    "Des entreprises facturables n’ont pas de relevé pour cette période."
                ),
            }
        )

    return {
        "ready_to_lock": len(blocking_reasons) == 0,
        "period_has_ended": period_ended,
        "period_id": period_id,
        "billing_year": year,
        "billing_month": month,
        "period_status": period.status,
        "statement_counts": counts,
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
    }


def subscription_volume_count(
    company_id: int,
    year: int,
    month: int,
    *,
    own_portfolio_only: bool = False,
) -> int:
    start, end = zurich_month_bounds_utc(year, month)
    q = db.session.query(func.count(Booking.id)).filter(
        Booking.company_id == company_id,
        Booking.created_at >= start,
        Booking.created_at <= end,
        Booking.status != BookingStatus.CANCELED,
    )
    if own_portfolio_only:
        # NULL = legacy non classifié → compté (parité V1 jusqu'au backfill)
        q = q.filter(
            or_(
                Booking.billing_origin.is_(None),
                Booking.billing_origin == BookingBillingOrigin.OWN_PORTFOLIO.value,
            )
        )
    n = q.scalar()
    return int(n or 0)


def _select_tier_from_grid(grid_id: int, volume: int) -> PlatformSubscriptionPricingTier | None:
    rows = (
        PlatformSubscriptionPricingTier.query.filter_by(grid_id=grid_id)
        .order_by(PlatformSubscriptionPricingTier.volume_min.asc())
        .all()
    )
    for row in rows:
        if volume < row.volume_min:
            continue
        if row.volume_max is not None and volume > row.volume_max:
            continue
        return row
    return None


def _active_default_grid(
    period_start: datetime,
) -> PlatformSubscriptionPricingGrid | None:
    grids = (
        PlatformSubscriptionPricingGrid.query.filter_by(
            grid_key="default", is_active=True
        )
        .order_by(PlatformSubscriptionPricingGrid.id.desc())
        .all()
    )
    for g in grids:
        ef = g.valid_from
        et = g.valid_until
        if ef is not None:
            if ef.tzinfo is None:
                ef = ef.replace(tzinfo=UTC)
            if period_start < ef:
                continue
        if et is not None:
            if et.tzinfo is None:
                et = et.replace(tzinfo=UTC)
            if period_start >= et:
                continue
        return g
    return grids[0] if grids else None


def _resolve_tax_rate(cfg: CompanyPlatformBillingConfig) -> Decimal:
    override = getattr(cfg, "tax_rate_override", None)
    if override is not None:
        return Decimal(str(override))
    creditor = PlatformBillingCreditor.query.filter_by(is_active=True).first()
    if creditor and creditor.default_tax_rate is not None:
        return Decimal(str(creditor.default_tax_rate))
    return _DEFAULT_TAX_RATE


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


def _commission_bookings_for_month(
    company_id: int, year: int, month: int
) -> list[Booking]:
    start, end = zurich_month_bounds_utc(year, month)
    return (
        Booking.query.filter(
            Booking.company_id == company_id,
            Booking.completed_at.isnot(None),
            Booking.completed_at >= start,
            Booking.completed_at <= end,
            Booking.status.in_(
                [BookingStatus.COMPLETED, BookingStatus.RETURN_COMPLETED]
            ),
        )
        .order_by(Booking.id.asc())
        .all()
    )


def recalculate_platform_period_drafts(period_id: int) -> dict[str, Any]:
    """Régénère tous les relevés entreprise pour une période draft (remplace les lignes).

    Résolution contrat : début de période Zurich (jamais datetime.now),
    une seule config par company_id distinct.
    """
    period = db.session.get(PlatformBillingPeriod, period_id)
    if not period:
        raise ValueError("Période introuvable")
    if period.status == PlatformBillingPeriodStatus.LOCKED.value:
        raise ValueError("Période verrouillée — recalcul interdit")

    # Ne supprimer que les relevés non validés / non verrouillés (PR3-ready)
    from models.enums import PlatformStatementStatus

    locked_statuses = (
        PlatformStatementStatus.VALIDATED.value,
        PlatformStatementStatus.LOCKED.value,
    )
    existing = PlatformInvoice.query.filter_by(period_id=period_id).all()
    for inv in existing:
        status = getattr(inv, "statement_status", None) or PlatformStatementStatus.DRAFT.value
        if status in locked_statuses:
            continue
        db.session.delete(inv)
    db.session.flush()

    period_start = month_start_zurich_utc(period.billing_year, period.billing_month)
    company_ids = distinct_billable_company_ids()
    generated = 0
    for company_id in company_ids:
        # Déjà un relevé VALIDATED/LOCKED : ne pas régénérer
        kept = (
            PlatformInvoice.query.filter_by(
                period_id=period_id, company_id=company_id
            ).first()
        )
        if kept:
            continue
        company = db.session.get(Company, company_id)
        if not company:
            continue
        active = effective_config_for_period(company_id, period_start)
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
    period_start = month_start_zurich_utc(year, month)
    own_on, comm_on, support_on = _product_flags(cfg)
    dm = _dispatch_mode_for_company(company, cfg)
    tax_rate = _resolve_tax_rate(cfg)
    cancel_policy = getattr(cfg, "commission_cancellation_policy", "exclude") or "exclude"
    pricing_mode = getattr(
        cfg, "subscription_pricing_mode", SubscriptionPricingMode.VOLUME.value
    )

    lines_data: list[dict[str, Any]] = []
    statement_items: list[dict[str, Any]] = []
    vol = 0
    sub_amount = Decimal("0.00")
    grid_id = getattr(cfg, "pricing_grid_id", None)
    tier_label = None

    if own_on:
        vol = subscription_volume_count(
            cid, year, month, own_portfolio_only=True
        )
        if pricing_mode == SubscriptionPricingMode.FREE.value:
            sub_amount = Decimal("0.00")
            tier_label = "Gratuit"
        elif pricing_mode == SubscriptionPricingMode.FIXED.value:
            sub_amount = money_round_chf(
                Decimal(str(cfg.custom_subscription_amount or 0))
            )
            tier_label = "Montant fixe"
        else:
            # volume : grille versionnée si dispo, sinon legacy dispatch
            if getattr(cfg, "use_global_pricing_grid", True) or grid_id:
                grid = None
                if grid_id:
                    grid = db.session.get(PlatformSubscriptionPricingGrid, grid_id)
                if grid is None:
                    grid = _active_default_grid(period_start)
                if grid is not None:
                    grid_id = grid.id
                    tier = _select_tier_from_grid(grid.id, vol)
                    sub_amount = (
                        money_round_chf(tier.price_monthly)
                        if tier
                        else Decimal("0.00")
                    )
                    tier_label = tier.label if tier else None
                else:
                    tier = select_subscription_tier(dm, vol)
                    sub_amount = (
                        money_round_chf(tier.price_monthly)
                        if tier
                        else Decimal("0.00")
                    )
                    tier_label = tier.label if tier else None
            else:
                tier = select_subscription_tier(dm, vol)
                sub_amount = (
                    money_round_chf(tier.price_monthly) if tier else Decimal("0.00")
                )
                tier_label = tier.label if tier else None

        sub_snap = {
            "rule": "subscription_own_portfolio_created_at_excluding_cancelled",
            "volume_count": vol,
            "dispatch_mode": dm,
            "pricing_mode": pricing_mode,
            "pricing_grid_id": grid_id,
            "tier_label": tier_label,
            "classification_version": CLASSIFICATION_VERSION,
            "qualification_version": QUALIFICATION_VERSION,
            "calculation_version": CALCULATION_VERSION,
            "transparency_label": _SNAPSHOT_SUB_LABEL,
            "contract_id": cfg.id,
        }
        lines_data.append(
            {
                "line_type": PlatformBillingLineType.SUBSCRIPTION.value,
                "label": _SUBSCRIPTION_LINE_LABEL,
                "amount": sub_amount,
                # Qté = volume courses ; P.U. = forfait mensuel (pas un prix par course)
                "quantity": Decimal(vol),
                "unit_amount": sub_amount,
                "snapshot_json": sub_snap,
                "sort_order": 0,
            }
        )
        statement_items.append(
            {
                "item_type": PlatformStatementItemType.OWN_PORTFOLIO_USAGE.value,
                "description": f"Abonnement portefeuille — {vol} transports",
                "quantity": Decimal(vol),
                "unit_amount": sub_amount,
                "base_amount": sub_amount,
                "net_amount": sub_amount,
                "eligibility_status": "eligible",
                "eligibility_reason": "PORTFOLIO_VOLUME",
            }
        )

    rate = cfg.commission_rate or Decimal("0")
    commission_total = Decimal("0")
    commission_base = Decimal("0")
    commission_details: list[dict[str, Any]] = []
    needs_review = False

    if comm_on:
        bookings = _commission_bookings_for_month(cid, year, month)
        flags = _batch_list_transfer_flags(bookings)
        for b in bookings:
            origin = getattr(b, "billing_origin", None)
            if origin == BookingBillingOrigin.OWN_PORTFOLIO.value:
                continue
            if origin == BookingBillingOrigin.LIRIE_MARKETPLACE.value:
                pass  # commissionnable candidat
            elif origin is not None:
                continue
            else:
                # Legacy : heuristique institution
                ht, hp = flags[b.id]
                pl = build_pilotage_payload_for_booking(
                    b, has_transfer=ht, has_pending_transfer=hp
                )
                if not is_commissionable_platform(b, pl):
                    continue

            # Exécutant = cette entreprise
            exec_cid = getattr(b, "executing_company_id", None) or b.company_id
            if exec_cid != cid:
                continue

            resolved = resolve_commissionable_amount(
                b, cancellation_policy=cancel_policy
            )
            if resolved.confidence == AmountConfidence.MISSING:
                needs_review = True
                statement_items.append(
                    {
                        "item_type": PlatformStatementItemType.MARKETPLACE_COMMISSION.value,
                        "booking_id": b.id,
                        "service_date": b.completed_at,
                        "description": f"Transport #{b.id} — montant à contrôler",
                        "net_amount": Decimal("0.00"),
                        "eligibility_status": "needs_review",
                        "eligibility_reason": resolved.reason or "MISSING_AMOUNT",
                        "source_snapshot": {
                            "source": resolved.source.value,
                            "confidence": resolved.confidence.value,
                        },
                    }
                )
                continue
            if resolved.amount is None:
                continue
            raw = resolved.amount * rate
            line_amt = money_round_chf(raw)
            commission_total += line_amt
            commission_base += resolved.amount
            commission_details.append(
                {
                    "booking_id": b.id,
                    "commission": str(line_amt),
                    "commissionable_amount": str(resolved.amount),
                    "source": resolved.source.value,
                    "confidence": resolved.confidence.value,
                }
            )
            statement_items.append(
                {
                    "item_type": PlatformStatementItemType.MARKETPLACE_COMMISSION.value,
                    "booking_id": b.id,
                    "service_date": b.completed_at,
                    "description": f"Commission LIRIE — transport #{b.id}",
                    "base_amount": resolved.amount,
                    "rate": rate,
                    "net_amount": line_amt,
                    "eligibility_status": "eligible",
                    "eligibility_reason": "MARKETPLACE_COMMISSIONABLE",
                    "source_snapshot": {
                        "source": resolved.source.value,
                        "confidence": resolved.confidence.value,
                    },
                }
            )

        comm_snap = {
            "commission_rate": str(rate),
            "booking_count": len(commission_details),
            "commission_base": str(money_round_chf(commission_base)),
            "details": commission_details[:500],
            "classification_version": CLASSIFICATION_VERSION,
            "qualification_version": QUALIFICATION_VERSION,
            "calculation_version": CALCULATION_VERSION,
            "contract_id": cfg.id,
            "cancellation_policy": cancel_policy,
        }
        booking_count = len(commission_details)
        rate_pct_s = f"{(rate * Decimal('100')):.4f}".rstrip("0").rstrip(".")
        comm_amount = money_round_chf(commission_total)
        # Qté = nb transports ; prix unit. = commission HT par transport
        unit_comm = (
            money_round_chf(comm_amount / Decimal(booking_count))
            if booking_count > 0
            else None
        )
        comm_label = (
            f"Commission LIRIE sur transports marketplace acceptés "
            f"(taux {rate_pct_s} %)"
        )
        lines_data.append(
            {
                "line_type": PlatformBillingLineType.INSTITUTION_COMMISSION.value,
                "label": comm_label,
                "amount": comm_amount,
                "quantity": Decimal(booking_count),
                "unit_amount": unit_comm,
                "snapshot_json": comm_snap,
                "sort_order": 10,
            }
        )

    support_total = Decimal("0")
    support_rows: list[PlatformSupportEntry] = []
    if support_on:
        support_rows = PlatformSupportEntry.query.filter(
            PlatformSupportEntry.company_id == cid,
            PlatformSupportEntry.validated_at.isnot(None),
            or_(
                PlatformSupportEntry.billing_period_id.is_(None),
                PlatformSupportEntry.billing_period_id == period.id,
            ),
        ).all()
        for se in support_rows:
            support_total += se.amount
            se.billing_period_id = period.id
            statement_items.append(
                {
                    "item_type": PlatformStatementItemType.SUPPORT.value,
                    "support_entry_id": se.id,
                    "service_date": se.occurred_at,
                    "description": se.description or f"Support #{se.id}",
                    "quantity": Decimal(se.duration_minutes) / Decimal(60),
                    "unit_amount": se.hourly_rate_snapshot,
                    "base_amount": se.amount,
                    "net_amount": se.amount,
                    "eligibility_status": "eligible",
                    "eligibility_reason": "SUPPORT_VALIDATED",
                }
            )
        if support_rows:
            total_minutes = sum(int(r.duration_minutes or 0) for r in support_rows)
            hours = (Decimal(total_minutes) / Decimal(60)).quantize(Decimal("0.01"))
            hours_s = f"{hours:.2f}".rstrip("0").rstrip(".")
            rate_ref = support_rows[0].hourly_rate_snapshot
            rate_s = f"{Decimal(str(rate_ref)):.2f}".rstrip("0").rstrip(".")
            support_label = (
                f"Support plateforme — {hours_s} h à {rate_s} CHF/h"
                if len({str(r.hourly_rate_snapshot) for r in support_rows}) == 1
                else f"Support plateforme — {hours_s} h"
            )
            lines_data.append(
                {
                    "line_type": PlatformBillingLineType.SUPPORT_TIME.value,
                    "label": support_label,
                    "amount": money_round_chf(support_total),
                    "quantity": hours,
                    "unit_amount": (
                        Decimal(str(rate_ref))
                        if len({str(r.hourly_rate_snapshot) for r in support_rows})
                        == 1
                        else None
                    ),
                    "snapshot_json": {
                        "entry_ids": [r.id for r in support_rows],
                        "count": len(support_rows),
                        "duration_minutes": total_minutes,
                        "duration_hours": str(hours),
                        "hourly_rate": str(rate_ref) if rate_ref is not None else None,
                    },
                    "sort_order": 20,
                }
            )

    subtotal = money_round_chf(
        sum(Decimal(str(x["amount"])) for x in lines_data)
    )
    tax_amount = money_round_chf(subtotal * tax_rate / Decimal("100"))
    total = money_round_chf(subtotal + tax_amount)

    stmt_status = (
        PlatformStatementStatus.NEEDS_REVIEW.value
        if needs_review
        else PlatformStatementStatus.CALCULATED.value
    )

    inv = PlatformInvoice(
        company_id=cid,
        period_id=period.id,
        currency="CHF",
        subtotal_amount=subtotal,
        tax_rate=tax_rate,
        tax_amount=tax_amount,
        total_amount=total,
        statement_status=stmt_status,
        calculation_version=CALCULATION_VERSION,
        contract_id=cfg.id,
        pricing_grid_id=grid_id,
        own_portfolio_count=vol if own_on else 0,
        subscription_amount=sub_amount if own_on else Decimal("0.00"),
        lirie_transport_count=len(commission_details) if comm_on else 0,
        commission_base=money_round_chf(commission_base) if comm_on else Decimal("0.00"),
        commission_rate_snapshot=rate if comm_on else None,
        commission_amount=money_round_chf(commission_total) if comm_on else Decimal("0.00"),
        support_amount=money_round_chf(support_total) if support_on else Decimal("0.00"),
        snapshot_json={
            "contract_id": cfg.id,
            "pricing_grid_id": grid_id,
            "tax_rate": str(tax_rate),
            "calculation_version": CALCULATION_VERSION,
            "period_timezone": "Europe/Zurich",
            "cancellation_policy": cancel_policy,
        },
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
                quantity=row.get("quantity"),
                unit_amount=row.get("unit_amount"),
                snapshot_json=row.get("snapshot_json"),
                sort_order=row.get("sort_order", 0),
            )
        )

    for item in statement_items:
        db.session.add(
            PlatformBillingStatementItem(
                statement_id=inv.id,
                item_type=item["item_type"],
                booking_id=item.get("booking_id"),
                support_entry_id=item.get("support_entry_id"),
                service_date=item.get("service_date"),
                description=item.get("description"),
                quantity=item.get("quantity"),
                unit_amount=item.get("unit_amount"),
                base_amount=item.get("base_amount"),
                rate=item.get("rate"),
                net_amount=item["net_amount"],
                tax_rate=tax_rate,
                eligibility_status=item.get("eligibility_status"),
                eligibility_reason=item.get("eligibility_reason"),
                source_snapshot=item.get("source_snapshot"),
            )
        )

    return inv


def validate_statement(
    statement_id: int,
    *,
    now_utc: datetime | None = None,
) -> PlatformInvoice:
    """Passe CALCULATED → VALIDATED (NEEDS_REVIEW / DRAFT interdits)."""
    inv = db.session.get(PlatformInvoice, statement_id)
    if not inv:
        raise BillingInvariantError(
            "STATEMENT_NOT_FOUND",
            "Relevé introuvable",
            status_code=404,
            details={"statement_id": statement_id},
        )
    period = inv.period
    if period is None:
        raise BillingInvariantError(
            "PERIOD_NOT_FOUND",
            "Période introuvable pour ce relevé",
            status_code=404,
            details={"statement_id": statement_id},
        )
    assert_billing_period_has_ended(
        int(period.billing_year),
        int(period.billing_month),
        now_utc=now_utc,
    )
    status = inv.statement_status or PlatformStatementStatus.DRAFT.value
    if status == PlatformStatementStatus.NEEDS_REVIEW.value:
        raise BillingInvariantError(
            "STATEMENT_REVIEW_REQUIRED",
            "Ce relevé contient des éléments non résolus. "
            "Corrigez les données sources puis recalculez le relevé.",
            details={"statement_id": statement_id, "statement_status": status},
        )
    if status != PlatformStatementStatus.CALCULATED.value:
        raise BillingInvariantError(
            "INVALID_STATEMENT_TRANSITION",
            f"Validation autorisée uniquement depuis CALCULATED "
            f"(état actuel: {status}).",
            details={"statement_id": statement_id, "statement_status": status},
        )
    inv.statement_status = PlatformStatementStatus.VALIDATED.value
    db.session.commit()
    return inv


def reopen_statement_for_correction(statement_id: int) -> dict[str, Any]:
    """Réouvre un relevé VALIDATED/LOCKED pour ajouter support / recalculer.

    Annule la facture légale non payée/envoyée liée, repasse le relevé en
    CALCULATED, puis régénère les lignes (intégration des heures support).
    """
    from models.enums import PlatformIssuedInvoiceStatus
    from models.platform_billing import PlatformIssuedInvoice

    inv = db.session.get(PlatformInvoice, statement_id)
    if not inv:
        raise ValueError("Relevé introuvable")
    status = inv.statement_status or PlatformStatementStatus.DRAFT.value
    if status not in (
        PlatformStatementStatus.VALIDATED.value,
        PlatformStatementStatus.LOCKED.value,
    ):
        raise ValueError(
            f"Réouverture inutile (état actuel: {status}). "
            "Saisissez les heures puis recalculez."
        )

    issued_rows = (
        PlatformIssuedInvoice.query.filter_by(statement_id=statement_id)
        .order_by(PlatformIssuedInvoice.id.desc())
        .all()
    )
    cancelled_issued_id = None
    for issued in issued_rows:
        if issued.status in (
            PlatformIssuedInvoiceStatus.CANCELLED.value,
            PlatformIssuedInvoiceStatus.DRAFT.value,
        ):
            issued.statement_id = None
            continue
        blocked = {
            PlatformIssuedInvoiceStatus.PAID.value,
            PlatformIssuedInvoiceStatus.SENT.value,
            PlatformIssuedInvoiceStatus.CREDITED.value,
            PlatformIssuedInvoiceStatus.OVERDUE.value,
        }
        if issued.status in blocked or issued.sent_at:
            raise ValueError(
                "Réouverture impossible : facture déjà envoyée ou payée. "
                "Utilisez une note de crédit."
            )
        issued.status = PlatformIssuedInvoiceStatus.CANCELLED.value
        issued.cancelled_at = datetime.now(UTC)
        issued.statement_id = None
        cancelled_issued_id = issued.id

    period_id = inv.period_id
    # Supprimer ce relevé pour forcer une régénération complète (avec support)
    db.session.delete(inv)
    db.session.flush()

    recalc = recalculate_platform_period_drafts(period_id)
    return {
        "ok": True,
        "cancelled_issued_id": cancelled_issued_id,
        "recalculate": recalc,
        "period_id": period_id,
    }


def lock_platform_billing_period(
    period_id: int,
    *,
    now_utc: datetime | None = None,
) -> PlatformBillingPeriod:
    """Verrouille la période seulement si readiness.ready_to_lock."""
    readiness = build_platform_billing_period_readiness(
        period_id, now_utc=now_utc
    )
    if not readiness["ready_to_lock"]:
        first = (readiness.get("blocking_reasons") or [{}])[0]
        raise BillingInvariantError(
            str(first.get("code") or "PERIOD_NOT_READY_TO_LOCK"),
            str(
                first.get("message")
                or "La période ne peut pas être verrouillée."
            ),
            details={"readiness": readiness},
        )
    period = db.session.get(PlatformBillingPeriod, period_id)
    invoices = PlatformInvoice.query.filter_by(period_id=period_id).all()
    for inv in invoices:
        inv.statement_status = PlatformStatementStatus.LOCKED.value
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
