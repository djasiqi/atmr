"""Agrégats légers pour GET /admin/dashboard/summary (orientation admin, pas fourre-tout).

Champs dépréciés (conservés pour compat JSON, non affichés côté FE dashboard) :
- booking_trends → toujours [] (plus de requête mensuelle)
- active_users_30d, invoices_current_month, revenue_current_month_chf
- platform_alerts_open (sémantique réelle = actions plateforme / CR en exécution)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import and_, desc, func, nullslast, or_, select
from sqlalchemy.orm import joinedload

from ext import db
from models import Booking, BookingStatus, User, UserRole
from models.client import Client
from models.company import Company
from models.demo_request import DemoRequest
from models.dispatch import DispatchRun
from models.enums import (
    DispatchStatus,
    PlatformBillingPeriodStatus,
    PlatformIssuedInvoiceStatus,
    PlatformStatementStatus,
)
from models.invoice import Invoice
from models.platform_billing import (
    PlatformBillingPeriod,
    PlatformInvoice,
    PlatformIssuedInvoice,
)
from models.platform_change_request import PlatformChangeRequest
from models.platform_runbook_execution import PlatformRunbookExecution
from services.platform_governance_constants import CHANGE_REQUEST_EXECUTING

# Réservations « à traiter » : non terminées / non annulées (aligné dispatch opérationnel).
BOOKING_PENDING_ACTION_STATUSES: tuple[BookingStatus, ...] = (
    BookingStatus.PENDING,
    BookingStatus.ACCEPTED,
    BookingStatus.ASSIGNED,
    BookingStatus.EN_ROUTE,
    BookingStatus.IN_PROGRESS,
)

# Demandes démo « ouvertes » : même règle que le compteur `new` côté AdminDemoRequests.
DEMO_OPEN_STATUSES: frozenset[str] = frozenset({"new"})
_DECEMBER = 12

_STATEMENT_TO_REVIEW: tuple[str, ...] = (
    PlatformStatementStatus.NEEDS_REVIEW.value,
    PlatformStatementStatus.CALCULATED.value,
)

_ISSUED_EXCLUDED: tuple[str, ...] = (
    PlatformIssuedInvoiceStatus.DRAFT.value,
    PlatformIssuedInvoiceStatus.CANCELLED.value,
    PlatformIssuedInvoiceStatus.CREDITED.value,
)


def _is_synthetic_demo_email_expr(column):
    lowered = func.lower(func.coalesce(column, ""))
    return or_(
        lowered.like("%@demo.local"),
        lowered.like("%@demo.lirie.ch"),
        lowered.like("demo-%@%"),
        lowered.like("%@internal.atmr.local"),
    )


def _month_window(now: datetime) -> tuple[datetime, datetime]:
    start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if now.month == _DECEMBER:
        end_of_month = now.replace(
            year=now.year + 1,
            month=1,
            day=1,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
    else:
        end_of_month = now.replace(
            month=now.month + 1,
            day=1,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
    return start_of_month, end_of_month


def build_admin_dashboard_summary() -> dict[str, Any]:
    """Construit le payload dashboard (une transaction logique, requêtes agrégées)."""
    now = datetime.now(UTC)
    seven_ago = now - timedelta(days=7)
    thirty_ago = now - timedelta(days=30)
    start_of_month, end_of_month = _month_window(now)
    start_of_today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # --- priorities ---
    bookings_pending_action = (
        db.session.scalar(
            select(func.count(Booking.id)).where(
                Booking.status.in_(
                    tuple(s.value for s in BOOKING_PENDING_ACTION_STATUSES)
                )
            )
        )
        or 0
    )

    demo_requests_open = (
        db.session.scalar(
            select(func.count(DemoRequest.id)).where(
                DemoRequest.status.in_(DEMO_OPEN_STATUSES)
            )
        )
        or 0
    )

    tenants_suspended = (
        db.session.scalar(
            select(func.count(Company.id)).where(Company.platform_suspended.is_(True))
        )
        or 0
    )

    # Compat API : clé historique ; sémantique = actions plateforme (CR en exécution).
    platform_alerts_open = (
        db.session.scalar(
            select(func.count(PlatformChangeRequest.id)).where(
                PlatformChangeRequest.status == CHANGE_REQUEST_EXECUTING
            )
        )
        or 0
    )

    billing_to_review = (
        db.session.scalar(
            select(func.count(PlatformInvoice.id))
            .select_from(PlatformInvoice)
            .join(
                PlatformBillingPeriod,
                PlatformInvoice.period_id == PlatformBillingPeriod.id,
            )
            .where(
                PlatformInvoice.statement_status.in_(_STATEMENT_TO_REVIEW),
                PlatformBillingPeriod.status == PlatformBillingPeriodStatus.DRAFT.value,
                PlatformInvoice.cancelled_at.is_(None),
            )
        )
        or 0
    )

    # --- kpi_business ---
    bookings_created_7d = (
        db.session.scalar(
            select(func.count(Booking.id)).where(
                Booking.created_at >= seven_ago,
                Booking.created_at <= now,
            )
        )
        or 0
    )

    bookings_completed_7d = (
        db.session.scalar(
            select(func.count(Booking.id)).where(
                Booking.status.in_(
                    (BookingStatus.COMPLETED, BookingStatus.RETURN_COMPLETED)
                ),
                Booking.completed_at.isnot(None),
                Booking.completed_at >= seven_ago,
                Booking.completed_at <= now,
            )
        )
        or 0
    )

    # Compat : annulations touchées récemment (updated_at) — ne pas utiliser pour le taux.
    bookings_canceled_7d = (
        db.session.scalar(
            select(func.count(Booking.id)).where(
                Booking.status == BookingStatus.CANCELED,
                Booking.updated_at >= seven_ago,
                Booking.updated_at <= now,
            )
        )
        or 0
    )

    # Cohorte : créées dans la fenêtre ET actuellement annulées.
    bookings_canceled_from_created_7d = (
        db.session.scalar(
            select(func.count(Booking.id)).where(
                Booking.created_at >= seven_ago,
                Booking.created_at <= now,
                Booking.status == BookingStatus.CANCELED,
            )
        )
        or 0
    )
    created_for_rate = max(int(bookings_created_7d), 1)
    cancellation_rate_7d = float(bookings_canceled_from_created_7d) / float(
        created_for_rate
    )
    if int(bookings_created_7d) == 0:
        cancellation_rate_7d = 0.0

    # Déprécié FE : conservé temporairement.
    active_users_30d = (
        db.session.scalar(
            select(func.count(User.id)).where(
                User.updated_at >= thirty_ago,
                User.updated_at <= now,
                User.role != UserRole.ADMIN,
                ~_is_synthetic_demo_email_expr(User.email),
            )
        )
        or 0
    )

    invoices_current_month = (
        db.session.scalar(
            select(func.count(Invoice.id)).where(
                Invoice.created_at >= start_of_month,
                Invoice.created_at < end_of_month,
            )
        )
        or 0
    )

    # Déprécié : valeur transports terminés (pas le CA LIRIE).
    revenue_stmt = select(func.coalesce(func.sum(Booking.amount), 0)).where(
        and_(
            Booking.status == BookingStatus.COMPLETED,
            Booking.scheduled_time >= start_of_month,
            Booking.scheduled_time < end_of_month,
        )
    )
    revenue_current_month_chf = float(
        db.session.execute(revenue_stmt).scalar_one() or 0
    )

    # Vrai montant facturé LIRIE (factures légales émises).
    effective_issued_at = func.coalesce(
        PlatformIssuedInvoice.issued_at, PlatformIssuedInvoice.created_at
    )
    platform_invoiced_stmt = select(
        func.coalesce(func.sum(PlatformIssuedInvoice.total_amount), 0)
    ).where(
        effective_issued_at >= start_of_month,
        effective_issued_at < end_of_month,
        PlatformIssuedInvoice.status.notin_(_ISSUED_EXCLUDED),
    )
    platform_invoiced_current_month_chf = float(
        db.session.execute(platform_invoiced_stmt).scalar_one() or 0
    )

    # --- platform_snippet ---
    runbooks_today = (
        db.session.scalar(
            select(func.count(PlatformRunbookExecution.id)).where(
                PlatformRunbookExecution.created_at >= start_of_today,
                PlatformRunbookExecution.created_at <= now,
            )
        )
        or 0
    )

    active_statuses = tuple(s.value for s in BOOKING_PENDING_ACTION_STATUSES)
    tenants_in_drift = (
        db.session.scalar(
            select(func.count(func.distinct(Company.id)))
            .select_from(Company)
            .where(Company.platform_suspended.is_(True))
            .where(
                or_(
                    Company.id.in_(
                        select(Booking.company_id)
                        .where(Booking.status.in_(active_statuses))
                        .distinct()
                    ),
                    Company.id.in_(
                        select(DispatchRun.company_id)
                        .where(DispatchRun.status == DispatchStatus.RUNNING)
                        .distinct()
                    ),
                )
            )
        )
        or 0
    )

    open_governance = int(platform_alerts_open)
    critical_attention_count = int(tenants_in_drift) + open_governance
    overall_status = "degraded" if critical_attention_count > 0 else "ok"

    # Déprécié : plus de requête mensuelle — clé conservée vide.
    booking_trends: list[dict[str, Any]] = []

    # --- recent_activity : 5 dernières réservations ---
    recent_rows = (
        Booking.query.options(joinedload(Booking.client).joinedload(Client.user))
        .order_by(nullslast(desc(Booking.updated_at)), desc(Booking.id))
        .limit(5)
        .all()
    )
    recent_activity: list[dict[str, Any]] = []
    for b in recent_rows:
        client = b.client
        first = (
            getattr(getattr(client, "user", None), "first_name", None)
            if client
            else None
        )
        last = (
            getattr(getattr(client, "user", None), "last_name", None)
            if client
            else None
        )
        name_parts = [p for p in (first, last) if p]
        client_label = " ".join(name_parts) if name_parts else None
        if not client_label and client:
            client_label = str(client.id)
        label = f"Réservation #{b.id}"
        if client_label:
            label = f"{client_label} — #{b.id}"
        st = b.status.value if hasattr(b.status, "value") else str(b.status)
        occurred = b.updated_at or b.created_at
        recent_activity.append(
            {
                "type": "booking",
                "entity_id": int(b.id),
                "label": label,
                "status": st,
                "occurred_at": occurred.isoformat() if occurred else now.isoformat(),
                "action": "open_booking",
            }
        )

    return {
        "generated_at": now.isoformat(),
        "priorities": {
            "bookings_pending_action": int(bookings_pending_action),
            "demo_requests_open": int(demo_requests_open),
            "tenants_suspended": int(tenants_suspended),
            "organizations_suspended": int(tenants_suspended),
            "platform_alerts_open": int(platform_alerts_open),
            "billing_to_review": int(billing_to_review),
            "critical_attention_count": int(critical_attention_count),
        },
        "kpi_business": {
            "bookings_created_7d": int(bookings_created_7d),
            "bookings_completed_7d": int(bookings_completed_7d),
            "bookings_canceled_7d": int(bookings_canceled_7d),
            "bookings_canceled_from_created_7d": int(bookings_canceled_from_created_7d),
            "cancellation_rate_7d": cancellation_rate_7d,
            "active_users_30d": int(active_users_30d),
            "invoices_current_month": int(invoices_current_month),
            "revenue_current_month_chf": revenue_current_month_chf,
            "platform_invoiced_current_month_chf": platform_invoiced_current_month_chf,
        },
        "platform_snippet": {
            "overall_status": overall_status,
            "open_alerts": open_governance,
            "runbooks_today": int(runbooks_today),
            "tenants_in_drift": int(tenants_in_drift),
            "critical_attention_count": int(critical_attention_count),
        },
        # Déprécié : toujours vide — ne plus appeler get_monthly_booking_counts.
        "booking_trends": booking_trends,
        "recent_activity": recent_activity,
    }
