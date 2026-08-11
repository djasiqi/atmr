"""Liste unifiée factures normales + partenaires (Lot 6 perf company-space).

Principe : ``UNION ALL`` SQL (factures normales + factures partenaires) avec
une projection commune (id, issued_at, kind), tri déterministe
``ORDER BY issued_at DESC NULLS LAST, id DESC`` puis pagination ``OFFSET/LIMIT``
exécutée en base. Interdit : fusionner deux listes déjà paginées indépendamment
côté Python (perte de déterminisme + coût mémoire).

Seule la page d'IDs résultante est hydratée (chargement complet des factures/
lignes/paiements) — jamais l'ensemble filtré.

Les statistiques (totaux émis/payé/solde, nb en retard) sont calculées par des
agrégats SQL (SUM/COUNT ... FILTER) en ``Decimal`` — jamais en itérant une
liste Python.

Aucune écriture ni commit dans ce chemin : c'est un GET pur.
"""

from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal
from typing import Any, cast

from sqlalchemy import and_, case, desc, exists, func, literal, or_, select, union_all
from sqlalchemy.orm import aliased, joinedload, subqueryload

from ext import db
from models import BookingTransfer, Client, Company, Invoice, User
from models.enums import InvoiceStatus, TransferStatus
from models.partner_invoice import (
    PartnerInvoice,
    PartnerInvoiceStatus,
    partner_invoice_transfers,
)
from models.partnership import Partnership

logger = logging.getLogger(__name__)


def _d(value: Any) -> Decimal:
    """Convertit une valeur SQL (Decimal/None/int) en Decimal sûr (jamais float)."""
    if value is None:
        return Decimal("0")
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal("0")


def list_company_invoices_unified(
    *,
    company_id: int,
    status_raw: str = "",
    client_id: int | None = None,
    year: int | None = None,
    month: int | None = None,
    q: str = "",
    with_balance: bool = False,
    with_reminders: bool = False,
    page: int = 1,
    per_page: int = 20,
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    """Retourne ``(items_page, total_count, stats)`` pour la liste factures d'une entreprise.

    - ``items_page`` : uniquement la page demandée (factures normales + partenaires
      mélangées, triées par date d'émission décroissante puis id décroissant).
    - ``total_count`` : nombre total d'éléments filtrés (normales + partenaires),
      calculé via ``COUNT(*)`` SQL sur l'union, pas via ``len()`` d'une liste Python.
    - ``stats`` : totaux émis/payé/solde + nb en retard (Decimal en interne,
      float uniquement à la sérialisation finale pour compat JSON/frontend).
    """
    page = max(int(page or 1), 1)
    per_page = min(max(int(per_page or 20), 1), 100)
    status_raw = (status_raw or "").strip().lower()
    q = (q or "").strip()

    status_map = {
        "draft": InvoiceStatus.DRAFT,
        "sent": InvoiceStatus.SENT,
        "partially_paid": InvoiceStatus.PARTIALLY_PAID,
        "paid": InvoiceStatus.PAID,
        "overdue": InvoiceStatus.OVERDUE,
        "cancelled": InvoiceStatus.CANCELLED,
    }
    partner_status_map = {
        "draft": PartnerInvoiceStatus.DRAFT,
        "sent": PartnerInvoiceStatus.SENT,
        "partially_paid": PartnerInvoiceStatus.PARTIALLY_PAID,
        "paid": PartnerInvoiceStatus.PAID,
        "overdue": PartnerInvoiceStatus.OVERDUE,
        "cancelled": PartnerInvoiceStatus.CANCELLED,
    }
    status_enum = status_map.get(status_raw) if status_raw else None
    partner_status_enum = partner_status_map.get(status_raw) if status_raw else None

    overdue_business_filter = and_(
        Invoice.balance_due > 0,
        Invoice.due_date < date.today(),
        Invoice.status.notin_(
            [InvoiceStatus.DRAFT, InvoiceStatus.PAID, InvoiceStatus.CANCELLED]
        ),
    )

    # ------------------------------------------------------------------
    # Branche factures normales (mêmes filtres que l'ancien repository)
    # ------------------------------------------------------------------
    regular_q = db.session.query(
        Invoice.id.label("eid"),
        Invoice.issued_at.label("issued_at"),
        literal("regular").label("kind"),
    ).filter(Invoice.company_id == company_id)

    if status_enum:
        if status_raw == "overdue":
            regular_q = regular_q.filter(overdue_business_filter)
        else:
            regular_q = regular_q.filter(Invoice.status == status_enum)
    if client_id:
        regular_q = regular_q.filter(Invoice.client_id == client_id)
    if year:
        regular_q = regular_q.filter(Invoice.period_year == year)
    if month:
        regular_q = regular_q.filter(Invoice.period_month == month)
    if with_balance:
        regular_q = regular_q.filter(Invoice.balance_due > 0)
    if with_reminders:
        regular_q = regular_q.filter(Invoice.reminder_level > 0)
    if q:
        PatientClient = aliased(Client)
        BillToClient = aliased(Client)
        PatientUser = aliased(User)
        like = f"%{q}%"
        regular_q = (
            regular_q.join(PatientClient, Invoice.client_id == PatientClient.id)
            .join(PatientUser, PatientClient.user_id == PatientUser.id)
            .outerjoin(BillToClient, Invoice.bill_to_client_id == BillToClient.id)
            .filter(
                or_(
                    Invoice.invoice_number.ilike(like),
                    PatientUser.first_name.ilike(like),
                    PatientUser.last_name.ilike(like),
                    PatientUser.username.ilike(like),
                    BillToClient.institution_name.ilike(like),
                )
            )
        )
    regular_q = regular_q.distinct()

    # ------------------------------------------------------------------
    # Branche factures partenaires
    # Une facture partenaire appartient à l'entreprise si :
    # - l'entreprise est owner_company_id ou partner_company_id du partenariat
    # - ET un transfert COMPLETED associé a cette entreprise comme exécutante
    # ✅ EXISTS (au lieu d'une boucle Python + un COUNT par facture partenaire)
    # ------------------------------------------------------------------
    transfer_exists = exists(
        select(1)
        .select_from(partner_invoice_transfers)
        .join(
            BookingTransfer,
            BookingTransfer.id == partner_invoice_transfers.c.booking_transfer_id,
        )
        .where(
            partner_invoice_transfers.c.partner_invoice_id == PartnerInvoice.id,
            BookingTransfer.executing_company_id == company_id,
            BookingTransfer.status == TransferStatus.COMPLETED,
        )
    )

    # Statut effectif "corrigé" pour l'affichage / les stats (ex-mutation Python
    # remplacée par une expression SQL — jamais persistée, purement calculée) :
    # - CANCELLED reste CANCELLED
    # - solde <= 0 -> PAID
    # - solde > 0 et montant payé > 0 -> PARTIALLY_PAID
    # - solde > 0, rien payé, mais statut stocké PAID (incohérent) -> SENT
    # - sinon : statut stocké inchangé
    partner_balance_expr = PartnerInvoice.total_amount - PartnerInvoice.amount_paid
    partner_effective_status = case(
        (
            PartnerInvoice.status == PartnerInvoiceStatus.CANCELLED,
            PartnerInvoice.status,
        ),
        (partner_balance_expr <= 0, PartnerInvoiceStatus.PAID),
        (PartnerInvoice.amount_paid > 0, PartnerInvoiceStatus.PARTIALLY_PAID),
        (PartnerInvoice.status == PartnerInvoiceStatus.PAID, PartnerInvoiceStatus.SENT),
        else_=PartnerInvoice.status,
    )

    partner_q = (
        db.session.query(
            PartnerInvoice.id.label("eid"),
            PartnerInvoice.issued_at.label("issued_at"),
            literal("partner").label("kind"),
        )
        .join(Partnership, PartnerInvoice.partnership_id == Partnership.id)
        .filter(
            or_(
                Partnership.owner_company_id == company_id,
                Partnership.partner_company_id == company_id,
            ),
            transfer_exists,
        )
    )
    if partner_status_enum is not None:
        partner_q = partner_q.filter(PartnerInvoice.status == partner_status_enum)
    if year:
        partner_q = partner_q.filter(PartnerInvoice.period_year == year)
    if month:
        partner_q = partner_q.filter(PartnerInvoice.period_month == month)
    if with_balance:
        partner_q = partner_q.filter(partner_balance_expr > 0)
    if q:
        like = f"%{q}%"
        OwnerCo = aliased(Company)
        PartnerCo = aliased(Company)
        partner_q = (
            partner_q.outerjoin(OwnerCo, Partnership.owner_company_id == OwnerCo.id)
            .outerjoin(PartnerCo, Partnership.partner_company_id == PartnerCo.id)
            .filter(
                or_(
                    PartnerInvoice.invoice_number.ilike(like),
                    OwnerCo.name.ilike(like),
                    PartnerCo.name.ilike(like),
                )
            )
        )

    # ------------------------------------------------------------------
    # UNION ALL + tri déterministe + pagination SQL (pas de fusion Python de
    # deux listes déjà paginées indépendamment)
    # ------------------------------------------------------------------
    # cast(Any): stubs SQLAlchemy refusent Query.statement (Row[...]) pour _TP de union_all
    unified = cast(Any, union_all(regular_q.statement, partner_q.statement)).subquery(
        "inv_union"
    )

    total_count = int(db.session.query(func.count()).select_from(unified).scalar() or 0)

    page_rows = (
        db.session.query(unified.c.eid, unified.c.kind, unified.c.issued_at)
        .order_by(desc(unified.c.issued_at).nulls_last(), desc(unified.c.eid))
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )

    # ------------------------------------------------------------------
    # Hydratation : uniquement les IDs de la page courante (pas le jeu filtré complet)
    # ------------------------------------------------------------------
    reg_ids = [int(r.eid) for r in page_rows if r.kind == "regular"]
    partner_ids = [int(r.eid) for r in page_rows if r.kind == "partner"]

    reg_by_id: dict[int, Invoice] = {}
    if reg_ids:
        loaded = (
            Invoice.query.options(
                joinedload(Invoice.client).joinedload(Client.user),
                joinedload(Invoice.bill_to_client).joinedload(Client.user),
                joinedload(Invoice.billing_party),
                joinedload(Invoice.billed_to_company),
                subqueryload(Invoice.lines),
                subqueryload(Invoice.payments),
                subqueryload(Invoice.reminders),
            )
            .filter(Invoice.id.in_(reg_ids))
            .all()
        )
        reg_by_id = {inv.id: inv for inv in loaded}

    partner_by_id: dict[int, PartnerInvoice] = {}
    if partner_ids:
        loaded_pi = (
            PartnerInvoice.query.options(
                joinedload(PartnerInvoice.partnership).joinedload(
                    Partnership.owner_company
                ),
                joinedload(PartnerInvoice.partnership).joinedload(
                    Partnership.partner_company
                ),
            )
            .filter(PartnerInvoice.id.in_(partner_ids))
            .all()
        )
        partner_by_id = {pi.id: pi for pi in loaded_pi}

    items: list[dict[str, Any]] = []
    for row in page_rows:
        if row.kind == "regular":
            inv = reg_by_id.get(int(row.eid))
            if inv is None:
                continue
            # Brouillon : harmoniser total facture / TTC lignes pour l'affichage
            # (liste vs éditeur / PDF). Correction en mémoire uniquement — un GET
            # ne doit jamais écrire/committer en base (Lot 6).
            if inv.status == InvoiceStatus.DRAFT:
                from application.invoices.edit_draft_invoice import (
                    repair_draft_invoice_if_line_totals_inconsistent,
                )

                repair_draft_invoice_if_line_totals_inconsistent(inv)
            items.append(inv.to_dict(list_view=True))
        else:
            pi = partner_by_id.get(int(row.eid))
            if pi is None:
                continue
            amount_paid = _d(pi.amount_paid)
            total_amount = _d(pi.total_amount)
            balance_due = total_amount - amount_paid

            # Statut effectif pour l'affichage (même logique que partner_effective_status,
            # calculée ici en Python car limitée à la page hydratée — pas de mutation ORM).
            if pi.status == PartnerInvoiceStatus.CANCELLED:
                effective_status = pi.status
            elif balance_due <= 0:
                effective_status = PartnerInvoiceStatus.PAID
            elif amount_paid > 0:
                effective_status = PartnerInvoiceStatus.PARTIALLY_PAID
            elif pi.status == PartnerInvoiceStatus.PAID:
                effective_status = PartnerInvoiceStatus.SENT
            else:
                effective_status = pi.status

            partner_company_name = None
            if pi.partnership:
                if (
                    pi.partnership.owner_company_id == company_id
                    and pi.partnership.partner_company
                ):
                    partner_company_name = pi.partnership.partner_company.name
                elif (
                    pi.partnership.partner_company_id == company_id
                    and pi.partnership.owner_company
                ):
                    partner_company_name = pi.partnership.owner_company.name

            items.append(
                {
                    "id": pi.id,
                    "invoice_number": pi.invoice_number,
                    "period_year": pi.period_year,
                    "period_month": pi.period_month,
                    "total_amount": float(total_amount),
                    "amount_paid": float(amount_paid),
                    "balance_due": float(balance_due),
                    "status": effective_status,
                    "issued_at": pi.issued_at.isoformat() if pi.issued_at else None,
                    "due_date": pi.due_date.isoformat() if pi.due_date else None,
                    "paid_at": pi.paid_at.isoformat() if pi.paid_at else None,
                    "pdf_url": pi.pdf_url,
                    "currency": pi.currency,
                    "client": {
                        "id": None,
                        "first_name": "",
                        "last_name": "",
                        "username": "",
                        "is_institution": True,
                        "institution_name": partner_company_name
                        or "Entreprise partenaire",
                    },
                    "bill_to_client": None,
                    "lines": [],
                    "payments": [],
                    "reminders": [],
                    "reminder_level": 0,  # Les factures partenaires n'ont pas de rappels
                    "last_reminder_at": None,
                    "is_partner_invoice": True,  # Flag pour identifier les factures partenaires
                    "partnership_id": pi.partnership_id,
                }
            )

    # ------------------------------------------------------------------
    # Stats/agrégats SQL (SUM/COUNT ... FILTER), Decimal — jamais en itérant
    # une liste Python ni en chargeant l'ensemble filtré.
    # ------------------------------------------------------------------
    stats_base = db.session.query(Invoice).filter(Invoice.company_id == company_id)
    if status_enum:
        if status_raw == "overdue":
            stats_base = stats_base.filter(overdue_business_filter)
        else:
            stats_base = stats_base.filter(Invoice.status == status_enum)
    if client_id:
        stats_base = stats_base.filter(Invoice.client_id == client_id)
    if year:
        stats_base = stats_base.filter(Invoice.period_year == year)
    if month:
        stats_base = stats_base.filter(Invoice.period_month == month)
    if with_balance:
        stats_base = stats_base.filter(Invoice.balance_due > 0)
    if with_reminders:
        stats_base = stats_base.filter(Invoice.reminder_level > 0)

    regular_stats_row = stats_base.with_entities(
        func.coalesce(
            func.sum(Invoice.total_amount).filter(
                Invoice.status != InvoiceStatus.CANCELLED
            ),
            0,
        ).label("total_issued"),
        func.coalesce(func.sum(Invoice.amount_paid), 0).label("total_paid"),
        func.coalesce(func.sum(Invoice.balance_due), 0).label("total_balance"),
        func.count(Invoice.id).filter(overdue_business_filter).label("overdue_count"),
    ).first()

    # Stats factures partenaires : mêmes filtres d'appartenance/statut/période que
    # la liste (hors with_balance/recherche texte, non appliqués aux stats — parité
    # avec le comportement historique de cet endpoint).
    partner_stats_base = (
        db.session.query(PartnerInvoice)
        .join(Partnership, PartnerInvoice.partnership_id == Partnership.id)
        .filter(
            or_(
                Partnership.owner_company_id == company_id,
                Partnership.partner_company_id == company_id,
            ),
            transfer_exists,
        )
    )
    if partner_status_enum is not None:
        partner_stats_base = partner_stats_base.filter(
            PartnerInvoice.status == partner_status_enum
        )
    if year:
        partner_stats_base = partner_stats_base.filter(
            PartnerInvoice.period_year == year
        )
    if month:
        partner_stats_base = partner_stats_base.filter(
            PartnerInvoice.period_month == month
        )

    partner_stats_row = partner_stats_base.with_entities(
        func.coalesce(
            func.sum(PartnerInvoice.total_amount).filter(
                partner_effective_status != PartnerInvoiceStatus.CANCELLED
            ),
            0,
        ).label("total_issued"),
        func.coalesce(
            func.sum(PartnerInvoice.total_amount).filter(
                partner_effective_status == PartnerInvoiceStatus.PAID
            ),
            0,
        ).label("total_paid"),
        func.coalesce(
            func.sum(PartnerInvoice.total_amount).filter(
                partner_effective_status != PartnerInvoiceStatus.PAID
            ),
            0,
        ).label("total_balance"),
        func.count(PartnerInvoice.id)
        .filter(partner_effective_status == PartnerInvoiceStatus.OVERDUE)
        .label("overdue_count"),
    ).first()

    total_issued = _d(regular_stats_row.total_issued if regular_stats_row else 0) + _d(
        partner_stats_row.total_issued if partner_stats_row else 0
    )
    total_paid = _d(regular_stats_row.total_paid if regular_stats_row else 0) + _d(
        partner_stats_row.total_paid if partner_stats_row else 0
    )
    total_balance = _d(
        regular_stats_row.total_balance if regular_stats_row else 0
    ) + _d(partner_stats_row.total_balance if partner_stats_row else 0)
    overdue_count = int(
        regular_stats_row.overdue_count if regular_stats_row else 0
    ) + int(partner_stats_row.overdue_count if partner_stats_row else 0)

    # Conversion float uniquement à la frontière JSON (le frontend appelle .toFixed(2)) ;
    # tous les calculs internes ci-dessus restent en Decimal.
    stats = {
        "total_issued": float(total_issued),
        "total_paid": float(total_paid),
        "total_balance": float(total_balance),
        "overdue_count": overdue_count,
    }
    return items, total_count, stats
