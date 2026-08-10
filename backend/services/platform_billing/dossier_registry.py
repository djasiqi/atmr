"""Registre opérationnel des dossiers facturation (projection company+période)."""

from __future__ import annotations

import csv
import io
import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import joinedload

from ext import db
from models.company import Company
from models.enums import (
    PlatformIssuedInvoiceStatus,
)
from models.platform_billing import (
    PlatformBillingPeriod,
    PlatformInvoice,
    PlatformIssuedInvoice,
)
from services.platform_billing.contracts import distinct_billable_company_ids
from services.platform_billing.decimal_json import decimal_to_str
from services.platform_billing.dossier_status import (
    A_TRAITER_STATUSES,
    compute_issuable,
    dossier_key,
    operational_status,
    resolve_actions,
    zero_charge_flags,
)
from services.platform_billing.issued_status import (
    balance_due_for_registry,
    is_credit_note,
)
from services.platform_billing.money import money_round_chf

logger = logging.getLogger(__name__)


def _period_label(period: PlatformBillingPeriod) -> str:
    return f"{period.billing_month:02d}.{period.billing_year}"


def _load_primary_issued(
    statement: PlatformInvoice | None,
) -> PlatformIssuedInvoice | None:
    if statement is None:
        return None
    inactive = {
        PlatformIssuedInvoiceStatus.CANCELLED.value,
        PlatformIssuedInvoiceStatus.CREDITED.value,
    }
    rows = (
        PlatformIssuedInvoice.query.filter_by(statement_id=statement.id)
        .filter(PlatformIssuedInvoice.document_type != "CREDIT_NOTE")
        .order_by(PlatformIssuedInvoice.id.desc())
        .all()
    )
    for row in rows:
        if is_credit_note(row):
            continue
        if row.status in inactive:
            continue
        return row
    return None


def _credit_note_id(inv: PlatformIssuedInvoice | None) -> int | None:
    if inv is None:
        return None
    cn = getattr(inv, "credit_note", None)
    return cn.id if cn else None


def _composition(statement: PlatformInvoice | None) -> dict[str, Any]:
    if statement is None:
        return {
            "own_portfolio_count": 0,
            "lirie_transport_count": 0,
            "subscription_amount": None,
            "commission_amount": None,
            "support_amount": None,
            "summary": "—",
        }
    own = int(getattr(statement, "own_portfolio_count", None) or 0)
    mkt = int(getattr(statement, "lirie_transport_count", None) or 0)
    parts: list[str] = []
    if own:
        parts.append(f"Portefeuille · {own}")
    if mkt:
        parts.append(f"Marketplace · {mkt} transport{'s' if mkt != 1 else ''}")
    if not parts:
        parts.append("Aucun volume")
    return {
        "own_portfolio_count": own,
        "lirie_transport_count": mkt,
        "subscription_amount": decimal_to_str(
            getattr(statement, "subscription_amount", None)
        ),
        "commission_amount": decimal_to_str(
            getattr(statement, "commission_amount", None)
        ),
        "support_amount": decimal_to_str(getattr(statement, "support_amount", None)),
        "summary": " · ".join(parts),
    }


def build_dossier_row(
    *,
    period: PlatformBillingPeriod,
    company: Company,
    statement: PlatformInvoice | None,
    caps: set[str] | None = None,
    now: datetime | None = None,
    check_qr: bool = False,
) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    primary = _load_primary_issued(statement)
    status = operational_status(
        statement=statement,
        period=period,
        primary_invoice=primary,
        now=now,
    )
    zc, zc_reason = zero_charge_flags(statement)
    qr_errors: list[str] = []
    issuable = False
    if status == "PRETE_A_EMETTRE" and statement is not None and check_qr:
        from services.platform_billing.issuance import statement_qr_ready

        issuable, qr_errors = statement_qr_ready(statement)
    else:
        issuable, qr_errors = compute_issuable(statement, period)

    if zc and status == "PRETE_A_EMETTRE":
        issuable = False
        if "Montant total doit être > 0 pour QR" not in qr_errors:
            qr_errors = [*list(qr_errors), "Montant total doit être > 0 pour QR"]

    credit_id = _credit_note_id(primary)
    replaced_id = (
        getattr(primary, "replaces_issued_invoice_id", None) if primary else None
    )
    actions = resolve_actions(
        status=status,
        statement=statement,
        primary_invoice=primary,
        credit_note_id=credit_id,
        issuable=issuable,
        issuer_errors=qr_errors,
        caps=caps,
        now=now,
    )

    amount: Decimal | None = None
    if primary is not None:
        amount = money_round_chf(Decimal(str(primary.total_amount or 0)))
    elif statement is not None:
        amount = money_round_chf(Decimal(str(statement.total_amount or 0)))

    paid = (
        money_round_chf(Decimal(str(primary.amount_paid or 0)))
        if primary is not None
        else None
    )
    balance = balance_due_for_registry(primary) if primary is not None else None

    return {
        "dossier_key": dossier_key(period.id, company.id),
        "period_id": period.id,
        "billing_year": period.billing_year,
        "billing_month": period.billing_month,
        "period_label": _period_label(period),
        "period_status": period.status,
        "company_id": company.id,
        "company_name": company.name,
        "statement_id": statement.id if statement else None,
        "statement_status": (
            getattr(statement, "statement_status", None) if statement else None
        ),
        "primary_invoice_id": primary.id if primary else None,
        "invoice_number": primary.invoice_number if primary else None,
        "credit_note_id": credit_id,
        "replaced_invoice_id": replaced_id,
        "composition": _composition(statement),
        "amount": decimal_to_str(amount) if amount is not None else None,
        "currency": (primary.currency if primary else None)
        or (statement.currency if statement else "CHF"),
        "due_at": primary.due_at.isoformat() if primary and primary.due_at else None,
        "amount_paid": decimal_to_str(paid) if paid is not None else None,
        "balance_due": decimal_to_str(balance) if balance is not None else None,
        "operational_status": status,
        "zero_charge": zc,
        "issuable": issuable,
        "zero_charge_reason": zc_reason,
        "issuance_errors": qr_errors if not issuable else [],
        **actions,
    }


def _caller_caps(admin_user_id: int | None) -> set[str]:
    if not admin_user_id:
        return set()
    from services.admin_authz import (
        CAP_BILLING_CANCEL,
        CAP_BILLING_CREDIT,
        CAP_BILLING_DUE_DATE,
        CAP_BILLING_ISSUE,
        CAP_BILLING_LOCK,
        CAP_BILLING_PAYMENT,
        CAP_BILLING_READ,
        CAP_BILLING_SEND,
        CAP_BILLING_VALIDATE,
        user_has_admin_capability,
    )

    wanted = [
        CAP_BILLING_READ,
        CAP_BILLING_VALIDATE,
        CAP_BILLING_LOCK,
        CAP_BILLING_ISSUE,
        CAP_BILLING_SEND,
        CAP_BILLING_PAYMENT,
        CAP_BILLING_DUE_DATE,
        CAP_BILLING_CANCEL,
        CAP_BILLING_CREDIT,
    ]
    return {c for c in wanted if user_has_admin_capability(admin_user_id, c)}


def _pairs_for_period(
    period: PlatformBillingPeriod,
) -> list[tuple[Company, PlatformInvoice | None]]:
    billable = set(distinct_billable_company_ids())
    invoices = (
        PlatformInvoice.query.filter_by(period_id=period.id)
        .options(joinedload(PlatformInvoice.issued_invoices))
        .all()
    )
    by_co = {inv.company_id: inv for inv in invoices}
    company_ids = sorted(billable | set(by_co.keys()))
    out: list[tuple[Company, PlatformInvoice | None]] = []
    for cid in company_ids:
        co = db.session.get(Company, cid)
        if not co:
            continue
        # Si pas billable et pas de relevé : ignorer
        if cid not in billable and cid not in by_co:
            continue
        out.append((co, by_co.get(cid)))
    return out


def _pairs_all_periods(
    *,
    year: int | None,
    month: int | None,
) -> list[tuple[PlatformBillingPeriod, Company, PlatformInvoice | None]]:
    """Tous les dossiers existants (relevés) + filtres année/mois optionnels."""
    stmt = select(PlatformInvoice).options(
        joinedload(PlatformInvoice.issued_invoices),
        joinedload(PlatformInvoice.period),
    )
    if year is not None or month is not None:
        stmt = stmt.join(PlatformBillingPeriod)
        if year is not None:
            stmt = stmt.where(PlatformBillingPeriod.billing_year == int(year))
        if month is not None:
            stmt = stmt.where(PlatformBillingPeriod.billing_month == int(month))
    invoices = list(db.session.scalars(stmt).unique().all())
    rows: list[tuple[PlatformBillingPeriod, Company, PlatformInvoice | None]] = []
    for inv in invoices:
        period = inv.period
        if period is None:
            period = db.session.get(PlatformBillingPeriod, inv.period_id)
        if period is None:
            continue
        co = db.session.get(Company, inv.company_id)
        if not co:
            continue
        rows.append((period, co, inv))
    return rows


def list_dossiers(
    *,
    period_id: int | None = None,
    year: int | None = None,
    month: int | None = None,
    q: str | None = None,
    operational_status_filter: str | None = None,
    a_traiter: bool = False,
    page: int = 1,
    per_page: int = 50,
    admin_user_id: int | None = None,
    now: datetime | None = None,
    check_qr: bool = False,
) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    caps = _caller_caps(admin_user_id)
    page = max(1, int(page))
    per_page = min(max(1, int(per_page)), 200)

    period: PlatformBillingPeriod | None = None
    raw_rows: list[tuple[PlatformBillingPeriod, Company, PlatformInvoice | None]] = []

    if period_id is not None:
        period = db.session.get(PlatformBillingPeriod, int(period_id))
        if not period:
            raise ValueError("Période introuvable")
        for co, st in _pairs_for_period(period):
            raw_rows.append((period, co, st))
    elif year is not None and month is not None:
        period = PlatformBillingPeriod.query.filter_by(
            billing_year=int(year), billing_month=int(month)
        ).first()
        if period:
            for co, st in _pairs_for_period(period):
                raw_rows.append((period, co, st))
        # période absente → liste vide (pas de création auto)
    else:
        raw_rows = _pairs_all_periods(year=year, month=month)

    dossiers: list[dict[str, Any]] = []
    q_norm = (q or "").strip().lower()
    status_filter = (operational_status_filter or "").strip().upper() or None

    for per, co, st in raw_rows:
        row = build_dossier_row(
            period=per,
            company=co,
            statement=st,
            caps=caps,
            now=now,
            check_qr=check_qr,
        )
        if q_norm:
            hay = f"{co.name or ''} {row.get('invoice_number') or ''}".lower()
            if q_norm not in hay:
                continue
        if a_traiter and row["operational_status"] not in A_TRAITER_STATUSES:
            continue
        if status_filter and row["operational_status"] != status_filter:
            continue
        dossiers.append(row)

    # Tri : année/mois desc, nom entreprise
    dossiers.sort(
        key=lambda d: (
            -(d.get("billing_year") or 0),
            -(d.get("billing_month") or 0),
            (d.get("company_name") or "").lower(),
        )
    )

    stats = _compute_stats(dossiers)
    total = len(dossiers)
    start = (page - 1) * per_page
    page_items = dossiers[start : start + per_page]

    return {
        "items": page_items,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total": total,
            "pages": (total + per_page - 1) // per_page if per_page else 1,
        },
        "stats": stats,
        "period": (
            {
                "id": period.id,
                "billing_year": period.billing_year,
                "billing_month": period.billing_month,
                "status": period.status,
            }
            if period
            else None
        ),
        "filters": {
            "period_id": period_id,
            "year": year,
            "month": month,
            "q": q,
            "operational_status": status_filter,
            "a_traiter": a_traiter,
        },
    }


def _compute_stats(dossiers: list[dict[str, Any]]) -> dict[str, Any]:
    a_emettre = Decimal("0.00")
    facture_brut = Decimal("0.00")
    avoirs = Decimal("0.00")
    encaisse = Decimal("0.00")
    solde = Decimal("0.00")

    for d in dossiers:
        status = d["operational_status"]
        amt = Decimal(str(d["amount"] or 0))
        paid = Decimal(str(d["amount_paid"] or 0))
        bal = Decimal(str(d["balance_due"] or 0))
        pid = d.get("primary_invoice_id")
        if (
            pid is None
            and d.get("statement_id")
            and status
            in (
                "A_CALCULER",
                "A_CONTROLER",
                "PRETE_A_CLOTURER",
                "PRETE_A_EMETTRE",
            )
        ):
            a_emettre += amt
        elif pid is not None:
            if status == "CREDITED":
                # net 0 après avoir — compté via avoir séparé si besoin
                facture_brut += amt
                avoirs += amt
            elif status != "CANCELLED":
                facture_brut += amt
                encaisse += paid
                solde += bal

    # Avoirs liés : somme des credit_note totals si IDs présents
    credit_ids = [d["credit_note_id"] for d in dossiers if d.get("credit_note_id")]
    if credit_ids:
        credit_invoices = PlatformIssuedInvoice.query.filter(
            PlatformIssuedInvoice.id.in_(credit_ids)
        ).all()
        avoirs = Decimal("0.00")
        for c in credit_invoices:
            avoirs += money_round_chf(abs(Decimal(str(c.total_amount or 0))))

    facture_net = money_round_chf(facture_brut - avoirs)
    return {
        "dossiers_count": len(dossiers),
        "a_emettre": decimal_to_str(money_round_chf(a_emettre)),
        "facture_net": decimal_to_str(facture_net),
        "encaisse": decimal_to_str(money_round_chf(encaisse)),
        "solde_ouvert": decimal_to_str(money_round_chf(solde)),
    }


def get_dossier(
    period_id: int,
    company_id: int,
    *,
    admin_user_id: int | None = None,
    now: datetime | None = None,
    check_qr: bool = True,
) -> dict[str, Any] | None:
    period = db.session.get(PlatformBillingPeriod, int(period_id))
    company = db.session.get(Company, int(company_id))
    if not period or not company:
        return None
    statement = (
        PlatformInvoice.query.filter_by(period_id=period.id, company_id=company.id)
        .options(joinedload(PlatformInvoice.issued_invoices))
        .first()
    )
    return build_dossier_row(
        period=period,
        company=company,
        statement=statement,
        caps=_caller_caps(admin_user_id),
        now=now or datetime.now(UTC),
        check_qr=check_qr,
    )


def export_dossiers_csv(**kwargs: Any) -> str:
    data = list_dossiers(**kwargs, page=1, per_page=10_000)
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=";", quoting=csv.QUOTE_MINIMAL)
    w.writerow(
        [
            "dossier_key",
            "billing_year",
            "billing_month",
            "company_id",
            "company_name",
            "statement_id",
            "invoice_number",
            "amount",
            "amount_paid",
            "balance_due",
            "due_at",
            "operational_status",
            "primary_action",
        ]
    )
    for d in data["items"]:
        w.writerow(
            [
                d["dossier_key"],
                d["billing_year"],
                d["billing_month"],
                d["company_id"],
                d["company_name"],
                d["statement_id"],
                d["invoice_number"],
                d["amount"],
                d["amount_paid"],
                d["balance_due"],
                d["due_at"],
                d["operational_status"],
                d["primary_action"],
            ]
        )
    return buf.getvalue()
