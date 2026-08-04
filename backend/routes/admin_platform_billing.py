"""Routes admin facturation plateforme LIRIE (relevés dual-produit)."""

from __future__ import annotations

import csv
import io
import json
import logging
import os
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from flask import Response, request
from flask_jwt_extended import jwt_required
from flask_restx import Namespace, Resource
from sqlalchemy import func

from ext import db, limiter, role_required
from models import Company, UserRole
from models.billing_profile import CompanyBillingProfile
from models.platform_billing import (
    CompanyPlatformBillingConfig,
    PlatformBillingCreditor,
    PlatformBillingPeriod,
    PlatformBillingStatementItem,
    PlatformInvoice,
    PlatformInvoiceLine,
    PlatformIssuedInvoice,
    PlatformSubscriptionPricing,
    PlatformSubscriptionPricingGrid,
    PlatformSubscriptionPricingTier,
    PlatformSupportEntry,
)
from security.ip_whitelist import ip_whitelist_required
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
    CAP_CONFIGURATION_MANAGE,
    require_admin_capability,
)

# Admin déjà protégé (JWT + rôle + IP) : plafonds adaptés à l’UI interactive
# (rechargements modal / recalculs), pas aux endpoints publics.
_RL_ADMIN_READ = "300 per hour"
_RL_ADMIN_WRITE = "200 per hour"
_RL_ADMIN_HEAVY = "60 per hour"
from services.platform_billing.contracts import (
    close_contract,
    create_contract_version,
    list_contracts,
    resolve_effective_instant_from_payload,
    serialize_contract,
)
from services.platform_billing.decimal_json import (
    decimal_to_str,
    decimal_to_str_trim,
    parse_decimal,
)
from services.platform_billing.engine import (
    get_or_create_period,
    lock_platform_billing_period,
    recalculate_platform_period_drafts,
    reopen_statement_for_correction,
    validate_statement,
)
from services.platform_billing.errors import BillingInvariantError
from services.platform_billing.issuance import (
    issue_platform_invoice,
    read_issued_invoice_pdf,
    statement_issuance_ready,
    statement_qr_ready,
)
from services.platform_billing.payments import (
    cancel_issued_invoice,
    create_credit_note,
    mark_sent,
    record_payment,
    refresh_overdue_statuses,
    reverse_payment,
)
from services.platform_billing.due_date import update_issued_invoice_due_date
from services.platform_billing.issued_registry import (
    export_issued_invoices_csv,
    get_issued_invoice_detail,
    list_issued_invoices,
    serialize_issued_invoice,
)
from services.platform_billing.readiness import build_company_readiness
from shared.error_handlers import APIErrorHandler

logger = logging.getLogger(__name__)
_MAX_CALENDAR_MONTH = 12


def _admin_user_id_from_jwt() -> int | None:
    """Résout l'identité JWT (public_id UUID) vers user.id entier.

    get_jwt_identity() renvoie un public_id, jamais l'id numérique.
    """
    from flask_jwt_extended import get_jwt_identity

    from models import User

    identity = get_jwt_identity()
    if identity is None or identity == "":
        return None
    # Compat éventuelle si un token legacy porte un id numérique
    try:
        return int(identity)
    except (TypeError, ValueError):
        pass
    user = User.query.filter_by(public_id=str(identity)).first()
    return int(user.id) if user is not None else None


def _serialize_period(p: PlatformBillingPeriod) -> dict[str, Any]:
    return {
        "id": p.id,
        "billing_year": p.billing_year,
        "billing_month": p.billing_month,
        "status": p.status,
        "timezone": p.timezone,
        "created_at": p.created_at.isoformat() if p.created_at else None,
        "updated_at": p.updated_at.isoformat() if p.updated_at else None,
    }


def _dual_product_config_ui_enabled() -> bool:
    """UI dual-produit visible par défaut (désactiver via env = false)."""
    return os.getenv(
        "PLATFORM_BILLING_DUAL_PRODUCT_CONFIG_UI", "true"
    ).lower() in ("1", "true", "yes", "on")


def _serialize_line(ln: PlatformInvoiceLine) -> dict[str, Any]:
    from services.platform_billing.issuance import (
        _enrich_line_label_for_pdf,
        resolve_line_qty_unit,
    )

    resolved = resolve_line_qty_unit(ln)
    qty = resolved["quantity"]
    unit = resolved["unit_amount"]
    rate_pct = resolved["unit_rate_percent"]
    return {
        "id": ln.id,
        "line_type": ln.line_type,
        "label": _enrich_line_label_for_pdf(ln),
        "amount": decimal_to_str(ln.amount),
        "quantity": decimal_to_str_trim(Decimal(str(qty)), places=4)
        if qty is not None
        else None,
        "unit_amount": decimal_to_str_trim(Decimal(str(unit)), places=4)
        if unit is not None
        else None,
        "unit_rate_percent": decimal_to_str_trim(rate_pct, places=4)
        if rate_pct is not None
        else None,
        "snapshot_json": ln.snapshot_json,
        "sort_order": ln.sort_order,
    }


def _serialize_invoice(
    inv: PlatformInvoice, *, company_name: str | None = None
) -> dict[str, Any]:
    lines = sorted(inv.lines, key=lambda x: (x.sort_order, x.id))
    issued = (
        PlatformIssuedInvoice.query.filter_by(statement_id=inv.id)
        .order_by(PlatformIssuedInvoice.id.desc())
        .first()
    )
    name = company_name
    if name is None and inv.company_id:
        co = db.session.get(Company, inv.company_id)
        name = co.name if co else None
    return {
        "id": inv.id,
        "company_id": inv.company_id,
        "company_name": name,
        "period_id": inv.period_id,
        "currency": inv.currency,
        "statement_status": getattr(inv, "statement_status", None),
        "subtotal_amount": decimal_to_str(inv.subtotal_amount),
        "tax_rate": decimal_to_str_trim(getattr(inv, "tax_rate", None), places=4),
        "tax_amount": decimal_to_str(getattr(inv, "tax_amount", None)),
        "total_amount": decimal_to_str(inv.total_amount),
        "own_portfolio_count": getattr(inv, "own_portfolio_count", None),
        "subscription_amount": decimal_to_str(
            getattr(inv, "subscription_amount", None)
        ),
        "lirie_transport_count": getattr(inv, "lirie_transport_count", None),
        "commission_base": decimal_to_str(getattr(inv, "commission_base", None)),
        "commission_rate_snapshot": decimal_to_str_trim(
            getattr(inv, "commission_rate_snapshot", None), places=6
        ),
        "commission_amount": decimal_to_str(getattr(inv, "commission_amount", None)),
        "support_amount": decimal_to_str(getattr(inv, "support_amount", None)),
        "calculation_version": getattr(inv, "calculation_version", None),
        "cancelled_at": inv.cancelled_at.isoformat() if inv.cancelled_at else None,
        "lines": [_serialize_line(x) for x in lines],
        "issued_invoice": _serialize_issued(issued) if issued else None,
    }


def _serialize_company_platform_config(
    cfg: CompanyPlatformBillingConfig,
) -> dict[str, Any]:
    """Legacy + champs dual-produit (legacy UI)."""
    base = serialize_contract(cfg)
    # Compat champs historiques (lecture)
    base["dispatch_mode_override"] = cfg.dispatch_mode_override
    base["commission_rate_legacy_float"] = None  # volontairement absent
    return base


def _serialize_creditor(c: PlatformBillingCreditor) -> dict[str, Any]:
    return {
        "id": c.id,
        "legal_name": c.legal_name,
        "street_name": c.street_name,
        "building_number": c.building_number,
        "postal_code": c.postal_code,
        "city": c.city,
        "country_code": c.country_code,
        "uid_ide": c.uid_ide,
        "vat_number": c.vat_number,
        "default_tax_rate": decimal_to_str_trim(c.default_tax_rate, places=4),
        "iban": c.iban,
        "qr_iban": c.qr_iban,
        "payment_reference_mode": c.payment_reference_mode,
        "payment_terms_days_default": c.payment_terms_days_default,
        "legal_form": c.legal_form,
        "signatory_name": c.signatory_name,
        "signatory_title": c.signatory_title,
        "is_active": c.is_active,
    }


def _serialize_debtor_address(
    company: Company, profile: CompanyBillingProfile | None
) -> dict[str, Any]:
    """Adresse débiteur (profil facturation ou domicile entreprise)."""
    if profile is not None:
        return {
            "source": "billing_profile",
            "legal_name": profile.legal_name,
            "street_name": profile.street_name,
            "building_number": profile.building_number,
            "postal_code": profile.postal_code,
            "city": profile.city,
            "country_code": profile.country_code or "CH",
        }
    return {
        "source": "company_domicile",
        "legal_name": company.name if company else None,
        "street_name": getattr(company, "domicile_address_line1", None) if company else None,
        "building_number": None,
        "postal_code": getattr(company, "domicile_zip", None) if company else None,
        "city": getattr(company, "domicile_city", None) if company else None,
        "country_code": (getattr(company, "domicile_country", None) or "CH")
        if company
        else "CH",
    }


def _serialize_issued(inv: PlatformIssuedInvoice) -> dict[str, Any]:
    return serialize_issued_invoice(inv)


def register_platform_billing_routes(admin_ns: Namespace) -> None:
    """Enregistre les routes sous /admin/platform-billing/."""
    if getattr(admin_ns, "_lirie_platform_billing_registered", False):
        return
    admin_ns._lirie_platform_billing_registered = True

    @admin_ns.route("/platform-billing/periods")
    class PlatformBillingPeriods(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @limiter.limit(_RL_ADMIN_READ)
        def get(self):
            rows = (
                PlatformBillingPeriod.query.order_by(
                    PlatformBillingPeriod.billing_year.desc(),
                    PlatformBillingPeriod.billing_month.desc(),
                )
                .limit(120)
                .all()
            )
            return {"periods": [_serialize_period(p) for p in rows]}, 200

        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @limiter.limit(_RL_ADMIN_WRITE)
        def post(self):
            data = request.get_json(silent=True) or {}
            try:
                year = int(data.get("billing_year") or data.get("year"))
                month = int(data.get("billing_month") or data.get("month"))
            except (TypeError, ValueError):
                return APIErrorHandler.handle_validation_error(
                    "billing_year et billing_month requis (entiers)",
                    logger_instance=logger,
                )
            if month < 1 or month > _MAX_CALENDAR_MONTH:
                return APIErrorHandler.handle_validation_error(
                    "billing_month invalide", logger_instance=logger
                )
            p = get_or_create_period(year, month)
            return _serialize_period(p), 201

    @admin_ns.route("/platform-billing/periods/<int:period_id>")
    class PlatformBillingPeriodDetail(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @limiter.limit(_RL_ADMIN_READ)
        def get(self, period_id: int):
            p = db.session.get(PlatformBillingPeriod, period_id)
            if not p:
                admin_ns.abort(404, "Période introuvable")
            return _serialize_period(p), 200

    @admin_ns.route("/platform-billing/periods/<int:period_id>/recalculate")
    class PlatformBillingRecalculate(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @limiter.limit(_RL_ADMIN_HEAVY)
        def post(self, period_id: int):
            try:
                out = recalculate_platform_period_drafts(period_id)
                return out, 200
            except ValueError as e:
                return {"error": str(e)}, 400

    @admin_ns.route("/platform-billing/periods/<int:period_id>/lock")
    class PlatformBillingLock(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @require_admin_capability(CAP_BILLING_LOCK)
        @limiter.limit(_RL_ADMIN_HEAVY)
        def post(self, period_id: int):
            try:
                p = lock_platform_billing_period(period_id)
                return _serialize_period(p), 200
            except BillingInvariantError as e:
                return e.to_response()
            except ValueError as e:
                return {"error": str(e)}, 400

    @admin_ns.route("/platform-billing/periods/<int:period_id>/export")
    class PlatformBillingPeriodExport(Resource):
        """Export CSV des relevés de la période (une ligne par ligne de facture)."""

        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @limiter.limit(_RL_ADMIN_WRITE)
        def get(self, period_id: int):
            p = db.session.get(PlatformBillingPeriod, period_id)
            if not p:
                admin_ns.abort(404, "Période introuvable")
            invs = (
                PlatformInvoice.query.filter_by(period_id=period_id)
                .order_by(PlatformInvoice.company_id.asc())
                .all()
            )
            buf = io.StringIO()
            w = csv.writer(buf, delimiter=";", quoting=csv.QUOTE_MINIMAL)
            w.writerow(
                [
                    "billing_year",
                    "billing_month",
                    "period_status",
                    "company_id",
                    "invoice_id",
                    "currency",
                    "line_id",
                    "line_type",
                    "label",
                    "amount",
                    "sort_order",
                    "snapshot_json",
                ]
            )
            for inv in invs:
                lines = sorted(inv.lines, key=lambda x: (x.sort_order, x.id))
                for ln in lines:
                    snap = ln.snapshot_json
                    w.writerow(
                        [
                            p.billing_year,
                            p.billing_month,
                            p.status,
                            inv.company_id,
                            inv.id,
                            inv.currency,
                            ln.id,
                            ln.line_type,
                            ln.label or "",
                            str(ln.amount),
                            ln.sort_order,
                            json.dumps(snap, ensure_ascii=False) if snap else "",
                        ]
                    )
            csv_bytes = ("\ufeff" + buf.getvalue()).encode("utf-8")
            fname = f"platform-billing-{p.billing_year}-{p.billing_month:02d}.csv"
            return Response(
                csv_bytes,
                mimetype="text/csv; charset=utf-8",
                headers={
                    "Content-Disposition": f'attachment; filename="{fname}"',
                },
            )

    @admin_ns.route("/platform-billing/periods/<int:period_id>/invoices")
    class PlatformBillingPeriodInvoices(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @limiter.limit(_RL_ADMIN_READ)
        def get(self, period_id: int):
            p = db.session.get(PlatformBillingPeriod, period_id)
            if not p:
                admin_ns.abort(404, "Période introuvable")
            invs = PlatformInvoice.query.filter_by(period_id=period_id).all()
            company_ids = {i.company_id for i in invs if i.company_id}
            names: dict[int, str] = {}
            if company_ids:
                for co in Company.query.filter(Company.id.in_(company_ids)).all():
                    names[co.id] = co.name
            return {
                "invoices": [
                    _serialize_invoice(i, company_name=names.get(i.company_id))
                    for i in invs
                ],
            }, 200

    @admin_ns.route("/platform-billing/invoices/<int:invoice_id>")
    class PlatformBillingInvoiceDetail(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @limiter.limit(_RL_ADMIN_READ)
        def get(self, invoice_id: int):
            inv = db.session.get(PlatformInvoice, invoice_id)
            if not inv:
                admin_ns.abort(404, "Relevé introuvable")
            return _serialize_invoice(inv), 200

    @admin_ns.route("/platform-billing/subscription-pricing")
    class PlatformSubscriptionPricingResource(Resource):
        """Grille d'abonnement plateforme (lecture)."""

        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @limiter.limit(_RL_ADMIN_READ)
        def get(self):
            rows = PlatformSubscriptionPricing.query.order_by(
                PlatformSubscriptionPricing.dispatch_mode.asc(),
                PlatformSubscriptionPricing.volume_min.asc(),
            ).all()
            return {
                "items": [
                    {
                        "id": r.id,
                        "dispatch_mode": r.dispatch_mode,
                        "volume_min": r.volume_min,
                        "volume_max": r.volume_max,
                        "price_monthly": decimal_to_str(r.price_monthly),
                        "label": r.label,
                    }
                    for r in rows
                ]
            }, 200

    @admin_ns.route("/platform-billing/companies/config")
    class CompaniesPlatformBillingConfigList(Resource):
        """Liste des entreprises avec la dernière ligne de config (paramètres transporteurs)."""

        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @limiter.limit(_RL_ADMIN_READ)
        def get(self):
            q = (request.args.get("q") or "").strip()
            subq = (
                db.session.query(
                    CompanyPlatformBillingConfig.company_id.label("cid"),
                    func.max(CompanyPlatformBillingConfig.id).label("mid"),
                )
                .group_by(CompanyPlatformBillingConfig.company_id)
                .subquery()
            )
            latest_cfgs = (
                db.session.query(CompanyPlatformBillingConfig)
                .join(subq, CompanyPlatformBillingConfig.id == subq.c.mid)
                .all()
            )
            cfg_map = {c.company_id: c for c in latest_cfgs}
            # Par défaut : entreprises approuvées uniquement (évite les clones e2e).
            include_unapproved = (request.args.get("include_unapproved") or "").lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            co_query = Company.query.order_by(Company.name.asc())
            if not include_unapproved:
                co_query = co_query.filter(Company.is_approved.is_(True))
            if q:
                co_query = co_query.filter(Company.name.ilike(f"%{q}%"))
            companies = co_query.limit(500).all()
            items = [
                {
                    "company_id": co.id,
                    "company_name": co.name,
                    "config": _serialize_company_platform_config(cfg_map[co.id])
                    if co.id in cfg_map
                    else None,
                }
                for co in companies
            ]
            return {"items": items}, 200

    @admin_ns.route("/platform-billing/companies/<int:company_id>/config")
    class CompanyPlatformBillingConfigResource(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @limiter.limit(_RL_ADMIN_READ)
        def get(self, company_id: int):
            co = db.session.get(Company, company_id)
            if not co:
                admin_ns.abort(404, "Entreprise introuvable")
            cfg = (
                CompanyPlatformBillingConfig.query.filter_by(company_id=company_id)
                .order_by(CompanyPlatformBillingConfig.id.desc())
                .first()
            )
            if not cfg:
                return {"config": None, "company_id": company_id}, 200
            return {
                "config": _serialize_company_platform_config(cfg),
                "company_id": company_id,
            }, 200

        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @limiter.limit(_RL_ADMIN_READ)
        def put(self, company_id: int):
            co = db.session.get(Company, company_id)
            if not co:
                admin_ns.abort(404, "Entreprise introuvable")
            data = request.get_json(silent=True) or {}
            cfg = (
                CompanyPlatformBillingConfig.query.filter_by(company_id=company_id)
                .order_by(CompanyPlatformBillingConfig.id.desc())
                .first()
            )
            if cfg is not None:
                from services.platform_billing.partner_agreement import (
                    PartnerAgreementError,
                    assert_config_mutable,
                )

                try:
                    assert_config_mutable(cfg.id)
                except PartnerAgreementError as exc:
                    return {"ok": False, "error": exc.message}, exc.status_code
            if not cfg:
                cfg = CompanyPlatformBillingConfig(company_id=company_id)
                db.session.add(cfg)
            if "is_billing_enabled" in data:
                cfg.is_billing_enabled = bool(data["is_billing_enabled"])
            if "dispatch_mode_override" in data:
                v = data["dispatch_mode_override"]
                cfg.dispatch_mode_override = v if v else None
            if "commission_rate" in data and data["commission_rate"] is not None:
                cfg.commission_rate = Decimal(str(data["commission_rate"]))
            elif "commission_rate" in data:
                cfg.commission_rate = None
            if (
                "support_hourly_rate_default" in data
                and data["support_hourly_rate_default"] is not None
            ):
                cfg.support_hourly_rate_default = Decimal(
                    str(data["support_hourly_rate_default"])
                )
            elif "support_hourly_rate_default" in data:
                cfg.support_hourly_rate_default = None
            if "is_active" in data:
                cfg.is_active = bool(data["is_active"])
            if "notes" in data:
                cfg.notes = data.get("notes")
            try:
                if any(
                    k in data
                    for k in (
                        "effective_year",
                        "effective_month",
                        "effective_from",
                    )
                ):
                    cfg.effective_from = resolve_effective_instant_from_payload(
                        data,
                        year_key="effective_year",
                        month_key="effective_month",
                        iso_key="effective_from",
                    )
                if any(
                    k in data
                    for k in (
                        "effective_to_year",
                        "effective_to_month",
                        "effective_to",
                    )
                ):
                    cfg.effective_to = resolve_effective_instant_from_payload(
                        data,
                        year_key="effective_to_year",
                        month_key="effective_to_month",
                        iso_key="effective_to",
                    )
            except BillingInvariantError as e:
                return e.to_response()
            db.session.commit()
            db.session.refresh(cfg)
            return {
                "ok": True,
                "company_id": company_id,
                "config": _serialize_company_platform_config(cfg),
            }, 200

    @admin_ns.route("/platform-billing/support-entries")
    class PlatformSupportEntries(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @limiter.limit(_RL_ADMIN_READ)
        def get(self):
            from services.platform_billing.support_entries import (
                serialize_support_entry,
            )

            cid = request.args.get("company_id", type=int)
            q = PlatformSupportEntry.query
            if cid:
                q = q.filter(PlatformSupportEntry.company_id == cid)
            rows = q.order_by(PlatformSupportEntry.occurred_at.desc()).limit(200).all()
            return {
                "entries": [serialize_support_entry(e) for e in rows]
            }, 200

        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @limiter.limit(_RL_ADMIN_WRITE)
        def post(self):
            from services.platform_billing.support_entries import (
                create_support_entry,
                serialize_support_entry,
            )

            data = request.get_json(silent=True) or {}
            user_id = _admin_user_id_from_jwt()
            try:
                se = create_support_entry(data, validated_by_user_id=user_id)
            except ValueError as e:
                return APIErrorHandler.handle_validation_error(
                    str(e), logger_instance=logger
                )
            # Recalcul optionnel du mois pour intégrer la ligne au relevé
            recalc = data.get("recalculate_period", True)
            if isinstance(recalc, str):
                recalc = recalc.lower() in ("1", "true", "yes", "on")
            period_id = data.get("billing_period_id") or se.billing_period_id
            recalc_result = None
            if recalc and period_id:
                try:
                    recalc_result = recalculate_platform_period_drafts(int(period_id))
                except ValueError as e:
                    return {
                        "ok": True,
                        "entry": serialize_support_entry(se),
                        "recalculate_error": str(e),
                    }, 201
            return {
                "ok": True,
                "entry": serialize_support_entry(se),
                "recalculate": recalc_result,
            }, 201

    @admin_ns.route("/platform-billing/support-entries/<int:entry_id>")
    class PlatformSupportEntryDetail(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @limiter.limit(_RL_ADMIN_WRITE)
        def patch(self, entry_id: int):
            from services.platform_billing.support_entries import (
                serialize_support_entry,
                update_support_entry,
            )

            data = request.get_json(silent=True) or {}
            user_id = _admin_user_id_from_jwt()
            try:
                se = update_support_entry(
                    entry_id, data, validated_by_user_id=user_id
                )
            except LookupError:
                admin_ns.abort(404, "Entrée introuvable")
            except ValueError as e:
                return APIErrorHandler.handle_validation_error(
                    str(e), logger_instance=logger
                )

            recalc = data.get("recalculate_period", True)
            if isinstance(recalc, str):
                recalc = recalc.lower() in ("1", "true", "yes", "on")
            period_id = data.get("billing_period_id") or se.billing_period_id
            recalc_result = None
            if recalc and period_id:
                try:
                    recalc_result = recalculate_platform_period_drafts(int(period_id))
                except ValueError as e:
                    return {
                        "ok": True,
                        "entry": serialize_support_entry(se),
                        "recalculate_error": str(e),
                    }, 200
            return {
                "ok": True,
                "entry": serialize_support_entry(se),
                "recalculate": recalc_result,
            }, 200

        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @limiter.limit(_RL_ADMIN_WRITE)
        def delete(self, entry_id: int):
            from services.platform_billing.support_entries import delete_support_entry

            data = request.get_json(silent=True) or {}
            try:
                snapshot, entry_period_id = delete_support_entry(entry_id)
            except LookupError:
                admin_ns.abort(404, "Entrée introuvable")

            recalc = data.get("recalculate_period", True)
            if isinstance(recalc, str):
                recalc = recalc.lower() in ("1", "true", "yes", "on")
            # query param fallback (DELETE sans body)
            if "recalculate_period" in request.args:
                recalc = request.args.get("recalculate_period", "true").lower() in (
                    "1",
                    "true",
                    "yes",
                    "on",
                )
            period_id = (
                data.get("billing_period_id")
                or request.args.get("billing_period_id", type=int)
                or entry_period_id
            )
            recalc_result = None
            if recalc and period_id:
                try:
                    recalc_result = recalculate_platform_period_drafts(int(period_id))
                except ValueError as e:
                    return {
                        "ok": True,
                        "deleted_id": entry_id,
                        "entry": snapshot,
                        "recalculate_error": str(e),
                    }, 200
            return {
                "ok": True,
                "deleted_id": entry_id,
                "entry": snapshot,
                "recalculate": recalc_result,
            }, 200

    @admin_ns.route("/platform-billing/support-entries/<int:entry_id>/validate")
    class PlatformSupportEntryValidate(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @limiter.limit(_RL_ADMIN_READ)
        def post(self, entry_id: int):
            se = db.session.get(PlatformSupportEntry, entry_id)
            if not se:
                admin_ns.abort(404, "Entrée introuvable")
            user_id = _admin_user_id_from_jwt()
            se.validated_at = datetime.now(UTC)
            se.validated_by_user_id = user_id
            db.session.commit()
            return {"ok": True, "id": se.id}, 200

    @admin_ns.route("/platform-billing/feature-flags")
    class PlatformBillingFeatureFlags(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        def get(self):
            return {
                "PLATFORM_BILLING_DUAL_PRODUCT_CONFIG_UI": _dual_product_config_ui_enabled()
            }, 200

    @admin_ns.route(
        "/platform-billing/companies/<int:company_id>/billing-contracts"
    )
    class CompanyBillingContracts(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @limiter.limit(_RL_ADMIN_READ)
        def get(self, company_id: int):
            if not db.session.get(Company, company_id):
                admin_ns.abort(404, "Entreprise introuvable")
            rows = list_contracts(company_id)
            profile = CompanyBillingProfile.query.filter_by(
                company_id=company_id
            ).first()
            company = db.session.get(Company, company_id)
            creditor = PlatformBillingCreditor.query.filter_by(is_active=True).first()
            latest = rows[0] if rows else None
            from services.platform_billing.partner_agreement import (
                get_active_agreement,
                serialize_agreement,
            )
            from services.platform_billing.partner_identity import (
                serialize_partner_identity_payload,
            )

            readiness = build_company_readiness(
                company=company,
                contract=latest,
                profile=profile,
                creditor=creditor,
            )
            contracts_payload = []
            for c in rows:
                item = serialize_contract(c)
                active_agr = get_active_agreement(c.id)
                item["active_agreement"] = (
                    serialize_agreement(active_agr) if active_agr else None
                )
                contracts_payload.append(item)
            return {
                "company_id": company_id,
                "contracts": contracts_payload,
                "readiness": readiness,
                "debtor_address": _serialize_debtor_address(company, profile),
                "partner_identity": serialize_partner_identity_payload(
                    company, profile
                ),
            }, 200

        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @require_admin_capability(CAP_CONFIGURATION_MANAGE)
        @limiter.limit(_RL_ADMIN_READ)
        def post(self, company_id: int):
            if not _dual_product_config_ui_enabled():
                # API disponible pour tests/staging ; écriture OK côté API
                pass
            if not db.session.get(Company, company_id):
                admin_ns.abort(404, "Entreprise introuvable")
            data = request.get_json(silent=True) or {}
            try:
                cfg = create_contract_version(company_id, data)
            except BillingInvariantError as e:
                return e.to_response()
            except ValueError as e:
                return APIErrorHandler.handle_validation_error(
                    str(e), logger_instance=logger
                )
            return {"ok": True, "contract": serialize_contract(cfg)}, 201

    @admin_ns.route(
        "/platform-billing/companies/<int:company_id>/debtor-address"
    )
    class CompanyDebtorAddressResource(Resource):
        """Adresse + identité contractuelle partenaire (sans création de profil)."""

        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @require_admin_capability(CAP_CONFIGURATION_MANAGE)
        @limiter.limit(_RL_ADMIN_READ)
        def put(self, company_id: int):
            company = db.session.get(Company, company_id)
            if not company:
                admin_ns.abort(404, "Entreprise introuvable")
            data = request.get_json(silent=True) or {}
            street = (data.get("street_name") or "").strip()
            building = (data.get("building_number") or "").strip() or None
            postal = (data.get("postal_code") or "").strip()
            city = (data.get("city") or "").strip()
            country = ((data.get("country_code") or "CH").strip() or "CH").upper()
            legal_name = (data.get("legal_name") or "").strip() or company.name

            if not street or not postal or not city:
                return APIErrorHandler.handle_validation_error(
                    "Rue, NPA et localité sont obligatoires",
                    logger_instance=logger,
                )

            from services.platform_billing.partner_identity import (
                serialize_partner_identity_payload,
                validate_legal_form,
            )

            try:
                if "legal_form" in data:
                    company.legal_form = validate_legal_form(data.get("legal_form"))
            except ValueError as e:
                return APIErrorHandler.handle_validation_error(
                    str(e), logger_instance=logger
                )

            if "signatory_name" in data:
                company.signatory_name = (
                    (data.get("signatory_name") or "").strip() or None
                )
            if "signatory_title" in data:
                company.signatory_title = (
                    (data.get("signatory_title") or "").strip() or None
                )
            if "uid_ide" in data:
                company.uid_ide = (data.get("uid_ide") or "").strip() or None

            profile = CompanyBillingProfile.query.filter_by(
                company_id=company_id
            ).first()
            if legal_name:
                # Raison sociale contractuelle = nom entreprise (ex. Emmenez-moi Sàrl)
                company.name = legal_name[:100]

            if profile is not None:
                profile.legal_name = legal_name
                profile.street_name = street
                # building_number NOT NULL au modèle : chaîne vide si absent
                profile.building_number = building or ""
                profile.postal_code = postal
                profile.city = city
                profile.country_code = country
                if "uid_ide" in data and data.get("uid_ide"):
                    profile.uid_ide = (data.get("uid_ide") or "").strip()
            else:
                # Fallback domicile entreprise (pas de création de profil)
                company.domicile_address_line1 = street
                company.domicile_address_line2 = building
                company.domicile_zip = postal
                company.domicile_city = city
                company.domicile_country = country

            db.session.commit()
            db.session.refresh(company)
            if profile is not None:
                db.session.refresh(profile)

            creditor = PlatformBillingCreditor.query.filter_by(is_active=True).first()
            contracts = list_contracts(company_id)
            latest = contracts[0] if contracts else None
            readiness = build_company_readiness(
                company=company,
                contract=latest,
                profile=profile,
                creditor=creditor,
            )
            return {
                "ok": True,
                "debtor_address": _serialize_debtor_address(company, profile),
                "partner_identity": serialize_partner_identity_payload(
                    company, profile
                ),
                "readiness": readiness,
            }, 200

    @admin_ns.route("/platform-billing/billing-contracts/<int:contract_id>")
    class BillingContractDetail(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        def get(self, contract_id: int):
            cfg = db.session.get(CompanyPlatformBillingConfig, contract_id)
            if not cfg:
                admin_ns.abort(404, "Contrat introuvable")
            return {"contract": serialize_contract(cfg)}, 200

    @admin_ns.route(
        "/platform-billing/billing-contracts/<int:contract_id>/close"
    )
    class BillingContractClose(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @require_admin_capability(CAP_CONFIGURATION_MANAGE)
        @limiter.limit(_RL_ADMIN_READ)
        def post(self, contract_id: int):
            data = request.get_json(silent=True) or {}
            try:
                cfg = close_contract(contract_id, data=data)
            except BillingInvariantError as e:
                return e.to_response()
            except ValueError as e:
                msg = str(e)
                if "gelée" in msg.lower() or "gelé" in msg.lower():
                    return {"ok": False, "error": msg}, 409
                return APIErrorHandler.handle_validation_error(
                    msg, logger_instance=logger
                )
            return {"ok": True, "contract": serialize_contract(cfg)}, 200

    @admin_ns.route(
        "/platform-billing/billing-contracts/<int:contract_id>/agreements"
    )
    class BillingContractAgreements(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @limiter.limit(_RL_ADMIN_READ)
        def get(self, contract_id: int):
            cfg = db.session.get(CompanyPlatformBillingConfig, contract_id)
            if not cfg:
                admin_ns.abort(404, "Contrat commercial introuvable")
            from services.platform_billing.partner_agreement import (
                list_agreements_for_config,
                serialize_agreement,
            )

            rows = list_agreements_for_config(contract_id)
            return {
                "billing_config_id": contract_id,
                "agreements": [serialize_agreement(a) for a in rows],
            }, 200

        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @require_admin_capability(CAP_CONFIGURATION_MANAGE)
        @limiter.limit(_RL_ADMIN_WRITE)
        def post(self, contract_id: int):
            """Génère ou régénère le DOCX (brouillon)."""
            from services.platform_billing.partner_agreement import (
                PartnerAgreementError,
                generate_agreement,
                serialize_agreement,
            )

            try:
                agr = generate_agreement(
                    contract_id,
                    user_id=_admin_user_id_from_jwt(),
                )
            except PartnerAgreementError as exc:
                return {"ok": False, "error": exc.message}, exc.status_code
            return {"ok": True, "agreement": serialize_agreement(agr)}, 201

    @admin_ns.route(
        "/platform-billing/agreements/<int:agreement_id>"
    )
    class PartnerAgreementDetail(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @limiter.limit(_RL_ADMIN_READ)
        def get(self, agreement_id: int):
            from models.platform_billing import PlatformPartnerAgreement
            from services.platform_billing.partner_agreement import (
                serialize_agreement,
            )

            agr = db.session.get(PlatformPartnerAgreement, agreement_id)
            if not agr:
                admin_ns.abort(404, "Accord introuvable")
            return {"agreement": serialize_agreement(agr)}, 200

    @admin_ns.route(
        "/platform-billing/agreements/<int:agreement_id>/docx"
    )
    class PartnerAgreementDocxDownload(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @limiter.limit(_RL_ADMIN_READ)
        def get(self, agreement_id: int):
            from models.platform_billing import PlatformPartnerAgreement
            from security.audit_log import AuditLogger
            from shared.upload_path_resolver import serve_stored_upload

            agr = db.session.get(PlatformPartnerAgreement, agreement_id)
            if not agr or not agr.generated_storage_key:
                admin_ns.abort(404, "Document généré introuvable")
            AuditLogger.log_action(
                action_type="partner_agreement_downloaded",
                action_category="platform_billing",
                user_id=_admin_user_id_from_jwt(),
                user_type="admin",
                company_id=agr.company_id,
                action_details={
                    "agreement_id": agr.id,
                    "kind": "generated_docx",
                    "reference": agr.reference,
                },
                resource_type="platform_partner_agreement",
                resource_id=str(agr.id),
            )
            return serve_stored_upload(
                agr.generated_storage_key,
                as_attachment=True,
                download_filename=f"{agr.reference.replace('/', '_')}.docx",
            )

    @admin_ns.route(
        "/platform-billing/agreements/<int:agreement_id>/mark-sent"
    )
    class PartnerAgreementMarkSent(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @require_admin_capability(CAP_CONFIGURATION_MANAGE)
        @limiter.limit(_RL_ADMIN_WRITE)
        def post(self, agreement_id: int):
            from services.platform_billing.partner_agreement import (
                PartnerAgreementError,
                mark_agreement_sent,
                serialize_agreement,
            )

            try:
                agr = mark_agreement_sent(
                    agreement_id,
                    user_id=_admin_user_id_from_jwt(),
                )
            except PartnerAgreementError as exc:
                return {"ok": False, "error": exc.message}, exc.status_code
            return {"ok": True, "agreement": serialize_agreement(agr)}, 200

    @admin_ns.route(
        "/platform-billing/agreements/<int:agreement_id>/void"
    )
    class PartnerAgreementVoid(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @require_admin_capability(CAP_CONFIGURATION_MANAGE)
        @limiter.limit(_RL_ADMIN_WRITE)
        def post(self, agreement_id: int):
            from services.platform_billing.partner_agreement import (
                PartnerAgreementError,
                serialize_agreement,
                void_agreement,
            )

            data = request.get_json(silent=True) or {}
            try:
                agr = void_agreement(
                    agreement_id,
                    reason=str(data.get("reason") or ""),
                    user_id=_admin_user_id_from_jwt(),
                )
            except PartnerAgreementError as exc:
                return {"ok": False, "error": exc.message}, exc.status_code
            return {"ok": True, "agreement": serialize_agreement(agr)}, 200

    @admin_ns.route(
        "/platform-billing/agreements/<int:agreement_id>/upload-signed"
    )
    class PartnerAgreementUploadSigned(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @require_admin_capability(CAP_CONFIGURATION_MANAGE)
        @limiter.limit(_RL_ADMIN_WRITE)
        def post(self, agreement_id: int):
            from datetime import date as date_cls

            from services.platform_billing.partner_agreement import (
                PartnerAgreementError,
                serialize_agreement,
                upload_signed_pdf,
            )

            if "file" not in request.files:
                return APIErrorHandler.handle_validation_error(
                    "Fichier PDF requis (champ file)",
                    logger_instance=logger,
                )
            f = request.files["file"]
            content = f.read()
            signed_on_raw = (
                request.form.get("agreement_signed_on")
                or (request.get_json(silent=True) or {}).get("agreement_signed_on")
            )
            if not signed_on_raw:
                return APIErrorHandler.handle_validation_error(
                    "agreement_signed_on (YYYY-MM-DD) est obligatoire",
                    logger_instance=logger,
                )
            try:
                signed_on = date_cls.fromisoformat(str(signed_on_raw)[:10])
            except ValueError:
                return APIErrorHandler.handle_validation_error(
                    "agreement_signed_on invalide (attendu YYYY-MM-DD)",
                    logger_instance=logger,
                )
            try:
                agr = upload_signed_pdf(
                    agreement_id,
                    content=content,
                    original_filename=f.filename,
                    agreement_signed_on=signed_on,
                    user_id=_admin_user_id_from_jwt(),
                )
            except PartnerAgreementError as exc:
                return {"ok": False, "error": exc.message}, exc.status_code
            return {"ok": True, "agreement": serialize_agreement(agr)}, 200

    @admin_ns.route(
        "/platform-billing/agreements/<int:agreement_id>/signed"
    )
    class PartnerAgreementSignedDownload(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @limiter.limit(_RL_ADMIN_READ)
        def get(self, agreement_id: int):
            from models.platform_billing import PlatformPartnerAgreement
            from security.audit_log import AuditLogger
            from shared.upload_path_resolver import serve_stored_upload

            agr = db.session.get(PlatformPartnerAgreement, agreement_id)
            if not agr or not agr.signed_storage_key:
                admin_ns.abort(404, "PDF signé introuvable")
            AuditLogger.log_action(
                action_type="partner_agreement_downloaded",
                action_category="platform_billing",
                user_id=_admin_user_id_from_jwt(),
                user_type="admin",
                company_id=agr.company_id,
                action_details={
                    "agreement_id": agr.id,
                    "kind": "signed_pdf",
                    "reference": agr.reference,
                },
                resource_type="platform_partner_agreement",
                resource_id=str(agr.id),
            )
            return serve_stored_upload(
                agr.signed_storage_key,
                as_attachment=True,
                download_filename=(
                    agr.signed_original_filename
                    or f"{agr.reference.replace('/', '_')}-signed.pdf"
                ),
            )

    @admin_ns.route("/platform-billing/creditor")
    class PlatformBillingCreditorResource(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        def get(self):
            c = PlatformBillingCreditor.query.filter_by(is_active=True).first()
            return {"creditor": _serialize_creditor(c) if c else None}, 200

        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @limiter.limit(_RL_ADMIN_WRITE)
        def put(self):
            data = request.get_json(silent=True) or {}
            legal_name = (data.get("legal_name") or "").strip()
            if not legal_name:
                return APIErrorHandler.handle_validation_error(
                    "Nom / raison sociale du créancier obligatoire "
                    "(personne physique si indépendant, pas l'enseigne LIRIE)",
                    logger_instance=logger,
                )
            # Typographie adresses (ex. Ernest- Pictet → Ernest-Pictet)
            if "street_name" in data and data.get("street_name"):
                import re as _re

                data["street_name"] = _re.sub(
                    r"\s*-\s*", "-", str(data["street_name"]).strip()
                )
            c = PlatformBillingCreditor.query.filter_by(is_active=True).first()
            if not c:
                c = PlatformBillingCreditor(
                    legal_name=legal_name,
                    street_name=data.get("street_name") or "",
                    postal_code=data.get("postal_code") or "",
                    city=data.get("city") or "",
                    # Franchise TVA par défaut (< 100'000 CHF)
                    default_tax_rate=Decimal("0.0000"),
                )
                db.session.add(c)
            for field in (
                "legal_name",
                "street_name",
                "building_number",
                "postal_code",
                "city",
                "country_code",
                "uid_ide",
                "vat_number",
                "iban",
                "qr_iban",
                "payment_reference_mode",
                "creditor_reference_base",
                "signatory_name",
                "signatory_title",
            ):
                if field in data:
                    setattr(c, field, data.get(field))
            c.legal_name = legal_name
            if "legal_form" in data:
                from services.platform_billing.partner_identity import (
                    validate_legal_form,
                )

                try:
                    c.legal_form = validate_legal_form(data.get("legal_form"))
                except ValueError as e:
                    return APIErrorHandler.handle_validation_error(
                        str(e), logger_instance=logger
                    )
            if "default_tax_rate" in data:
                c.default_tax_rate = parse_decimal(
                    data.get("default_tax_rate"),
                    field="default_tax_rate",
                    min_value=Decimal("0"),
                    max_value=Decimal("100"),
                    allow_none=False,
                )
            if "payment_terms_days_default" in data:
                c.payment_terms_days_default = int(
                    data["payment_terms_days_default"]
                )
            if "is_active" in data:
                c.is_active = bool(data["is_active"])
            db.session.commit()
            db.session.refresh(c)
            return {"ok": True, "creditor": _serialize_creditor(c)}, 200

    @admin_ns.route("/platform-billing/pricing-grids")
    class PlatformPricingGrids(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        def get(self):
            grids = PlatformSubscriptionPricingGrid.query.order_by(
                PlatformSubscriptionPricingGrid.id.desc()
            ).all()
            items = []
            for g in grids:
                tiers = (
                    PlatformSubscriptionPricingTier.query.filter_by(grid_id=g.id)
                    .order_by(PlatformSubscriptionPricingTier.volume_min.asc())
                    .all()
                )
                items.append(
                    {
                        "id": g.id,
                        "grid_key": g.grid_key,
                        "label": g.label,
                        "currency": g.currency,
                        "valid_from": g.valid_from.isoformat()
                        if g.valid_from
                        else None,
                        "valid_until": g.valid_until.isoformat()
                        if g.valid_until
                        else None,
                        "is_active": g.is_active,
                        "tiers": [
                            {
                                "id": t.id,
                                "volume_min": t.volume_min,
                                "volume_max": t.volume_max,
                                "price_monthly": decimal_to_str(t.price_monthly),
                                "label": t.label,
                            }
                            for t in tiers
                        ],
                    }
                )
            return {"items": items}, 200

    @admin_ns.route(
        "/platform-billing/invoices/<int:invoice_id>/validate"
    )
    class PlatformInvoiceValidate(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @require_admin_capability(CAP_BILLING_VALIDATE)
        def post(self, invoice_id: int):
            try:
                inv = validate_statement(invoice_id)
            except BillingInvariantError as e:
                return e.to_response()
            except ValueError as e:
                return APIErrorHandler.handle_validation_error(
                    str(e), logger_instance=logger
                )
            return {"ok": True, "invoice": _serialize_invoice(inv)}, 200

    @admin_ns.route(
        "/platform-billing/invoices/<int:invoice_id>/reopen"
    )
    class PlatformInvoiceReopen(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @limiter.limit(_RL_ADMIN_WRITE)
        def post(self, invoice_id: int):
            try:
                result = reopen_statement_for_correction(invoice_id)
            except ValueError as e:
                return APIErrorHandler.handle_validation_error(
                    str(e), logger_instance=logger
                )
            return result, 200

    @admin_ns.route(
        "/platform-billing/invoices/<int:invoice_id>/statement-items"
    )
    class PlatformInvoiceStatementItems(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        def get(self, invoice_id: int):
            inv = db.session.get(PlatformInvoice, invoice_id)
            if not inv:
                admin_ns.abort(404, "Relevé introuvable")
            items = (
                PlatformBillingStatementItem.query.filter_by(statement_id=invoice_id)
                .order_by(PlatformBillingStatementItem.id.asc())
                .all()
            )
            return {
                "items": [
                    {
                        "id": i.id,
                        "item_type": i.item_type,
                        "booking_id": i.booking_id,
                        "support_entry_id": i.support_entry_id,
                        "description": i.description,
                        "base_amount": decimal_to_str(i.base_amount),
                        "rate": decimal_to_str(i.rate, places=6),
                        "net_amount": decimal_to_str(i.net_amount),
                        "eligibility_status": i.eligibility_status,
                        "eligibility_reason": i.eligibility_reason,
                    }
                    for i in items
                ]
            }, 200

    @admin_ns.route(
        "/platform-billing/invoices/<int:invoice_id>/issue"
    )
    class PlatformInvoiceIssue(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @require_admin_capability(CAP_BILLING_ISSUE)
        @limiter.limit(_RL_ADMIN_WRITE)
        def post(self, invoice_id: int):
            try:
                issued = issue_platform_invoice(invoice_id)
            except BillingInvariantError as e:
                return e.to_response()
            except ValueError as e:
                return APIErrorHandler.handle_validation_error(
                    str(e), logger_instance=logger
                )
            return {"ok": True, "issued_invoice": _serialize_issued(issued)}, 201

    @admin_ns.route("/platform-billing/issued-invoices")
    class PlatformIssuedInvoicesList(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @require_admin_capability(CAP_BILLING_READ)
        @ip_whitelist_required()
        @limiter.limit(_RL_ADMIN_READ)
        def get(self):
            args = request.args
            try:
                company_id = (
                    int(args["company_id"]) if args.get("company_id") else None
                )
                year = int(args["year"]) if args.get("year") else None
                month = int(args["month"]) if args.get("month") else None
                page = int(args.get("page") or 1)
                per_page = int(args.get("per_page") or 20)
            except (TypeError, ValueError):
                return APIErrorHandler.handle_validation_error(
                    "Paramètres de pagination/filtre invalides",
                    logger_instance=logger,
                )
            return list_issued_invoices(
                q=args.get("q"),
                company_id=company_id,
                status=args.get("status"),
                payment_state=args.get("payment_state"),
                year=year,
                month=month,
                with_balance=args.get("with_balance") in ("1", "true", "yes"),
                overdue_only=args.get("overdue_only") in ("1", "true", "yes"),
                with_dunning=args.get("with_dunning") in ("1", "true", "yes"),
                document_type=args.get("document_type"),
                page=page,
                per_page=per_page,
                sort_by=args.get("sort_by") or "issued_at",
                sort_order=args.get("sort_order") or "desc",
            ), 200

    @admin_ns.route("/platform-billing/issued-invoices/export")
    class PlatformIssuedInvoicesExport(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @require_admin_capability(CAP_BILLING_READ)
        @ip_whitelist_required()
        @limiter.limit(_RL_ADMIN_READ)
        def get(self):
            args = request.args
            try:
                company_id = (
                    int(args["company_id"]) if args.get("company_id") else None
                )
                year = int(args["year"]) if args.get("year") else None
                month = int(args["month"]) if args.get("month") else None
            except (TypeError, ValueError):
                return APIErrorHandler.handle_validation_error(
                    "Paramètres de filtre invalides",
                    logger_instance=logger,
                )
            csv_text, filename = export_issued_invoices_csv(
                q=args.get("q"),
                company_id=company_id,
                status=args.get("status"),
                payment_state=args.get("payment_state"),
                year=year,
                month=month,
                with_balance=args.get("with_balance") in ("1", "true", "yes"),
                overdue_only=args.get("overdue_only") in ("1", "true", "yes"),
                with_dunning=args.get("with_dunning") in ("1", "true", "yes"),
                document_type=args.get("document_type"),
                sort_by=args.get("sort_by") or "issued_at",
                sort_order=args.get("sort_order") or "desc",
            )
            return Response(
                csv_text,
                mimetype="text/csv; charset=utf-8",
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}"',
                },
            )

    @admin_ns.route("/platform-billing/issued-invoices/<int:issued_id>")
    class PlatformIssuedInvoiceDetail(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @require_admin_capability(CAP_BILLING_READ)
        @ip_whitelist_required()
        @limiter.limit(_RL_ADMIN_READ)
        def get(self, issued_id: int):
            try:
                return get_issued_invoice_detail(issued_id), 200
            except ValueError as e:
                return APIErrorHandler.handle_validation_error(
                    str(e), logger_instance=logger
                )

    @admin_ns.route(
        "/platform-billing/issued-invoices/<int:issued_id>/due-date"
    )
    class PlatformIssuedDueDate(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @require_admin_capability(CAP_BILLING_DUE_DATE)
        @ip_whitelist_required()
        @limiter.limit(_RL_ADMIN_WRITE)
        def patch(self, issued_id: int):
            data = request.get_json(silent=True) or {}
            reason = (data.get("reason") or "").strip()
            due_raw = data.get("due_at")
            if not due_raw:
                return APIErrorHandler.handle_validation_error(
                    "due_at est requis",
                    logger_instance=logger,
                )
            try:
                new_due = datetime.fromisoformat(
                    str(due_raw).replace("Z", "+00:00")
                )
                inv = update_issued_invoice_due_date(
                    issued_id,
                    new_due_at=new_due,
                    reason=reason,
                    admin_user_id=_admin_user_id_from_jwt(),
                )
            except ValueError as e:
                return APIErrorHandler.handle_validation_error(
                    str(e), logger_instance=logger
                )
            return {"ok": True, "issued_invoice": _serialize_issued(inv)}, 200

    @admin_ns.route(
        "/platform-billing/issued-invoices/<int:issued_id>/pdf"
    )
    class PlatformIssuedInvoicePdf(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @limiter.limit(_RL_ADMIN_READ)
        def get(self, issued_id: int):
            try:
                data, filename = read_issued_invoice_pdf(issued_id)
            except ValueError as e:
                return APIErrorHandler.handle_validation_error(
                    str(e), logger_instance=logger
                )
            return Response(
                data,
                mimetype="application/pdf",
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}"',
                },
            )

    @admin_ns.route(
        "/platform-billing/invoices/<int:invoice_id>/readiness"
    )
    class PlatformInvoiceReadiness(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        def get(self, invoice_id: int):
            inv = db.session.get(PlatformInvoice, invoice_id)
            if not inv:
                admin_ns.abort(404, "Relevé introuvable")
            iss_ok, iss_err = statement_issuance_ready(inv)
            qr_ok, qr_err = statement_qr_ready(inv)
            return {
                "statement_id": invoice_id,
                "statement_status": inv.statement_status,
                "issuance_ready": iss_ok,
                "issuance_errors": iss_err,
                "qr_ready": qr_ok,
                "qr_errors": qr_err,
            }, 200

    @admin_ns.route(
        "/platform-billing/issued-invoices/<int:issued_id>/send"
    )
    class PlatformIssuedSend(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @require_admin_capability(CAP_BILLING_SEND)
        @ip_whitelist_required()
        def post(self, issued_id: int):
            try:
                inv = mark_sent(issued_id)
            except ValueError as e:
                return APIErrorHandler.handle_validation_error(
                    str(e), logger_instance=logger
                )
            return {"ok": True, "issued_invoice": _serialize_issued(inv)}, 200

    @admin_ns.route(
        "/platform-billing/issued-invoices/<int:issued_id>/payments"
    )
    class PlatformIssuedPayments(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @require_admin_capability(CAP_BILLING_PAYMENT)
        @ip_whitelist_required()
        def post(self, issued_id: int):
            data = request.get_json(silent=True) or {}
            try:
                amount = parse_decimal(
                    data.get("amount"),
                    field="amount",
                    min_value=Decimal("0.01"),
                    allow_none=False,
                )
                paid_at = data.get("paid_at")
                dt = (
                    datetime.fromisoformat(str(paid_at).replace("Z", "+00:00"))
                    if paid_at
                    else None
                )
                user_id = _admin_user_id_from_jwt()
                inv = record_payment(
                    issued_id,
                    amount=amount,
                    paid_at=dt,
                    method=data.get("method"),
                    reference=data.get("reference"),
                    notes=data.get("notes"),
                    created_by_user_id=user_id,
                    idempotency_key=data.get("idempotency_key"),
                )
            except ValueError as e:
                return APIErrorHandler.handle_validation_error(
                    str(e), logger_instance=logger
                )
            return {"ok": True, "issued_invoice": _serialize_issued(inv)}, 201

    @admin_ns.route(
        "/platform-billing/issued-invoices/<int:issued_id>/payments/"
        "<int:payment_id>/reverse"
    )
    class PlatformIssuedPaymentReverse(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @require_admin_capability(CAP_BILLING_PAYMENT)
        @ip_whitelist_required()
        def post(self, issued_id: int, payment_id: int):
            data = request.get_json(silent=True) or {}
            try:
                inv = reverse_payment(
                    issued_id,
                    payment_id,
                    reason=str(data.get("reason") or ""),
                    created_by_user_id=_admin_user_id_from_jwt(),
                )
            except ValueError as e:
                return APIErrorHandler.handle_validation_error(
                    str(e), logger_instance=logger
                )
            return {"ok": True, "issued_invoice": _serialize_issued(inv)}, 200

    @admin_ns.route(
        "/platform-billing/issued-invoices/<int:issued_id>/cancel"
    )
    class PlatformIssuedCancel(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @require_admin_capability(CAP_BILLING_CANCEL)
        @ip_whitelist_required()
        def post(self, issued_id: int):
            try:
                inv = cancel_issued_invoice(issued_id)
            except ValueError as e:
                return APIErrorHandler.handle_validation_error(
                    str(e), logger_instance=logger
                )
            return {"ok": True, "issued_invoice": _serialize_issued(inv)}, 200

    @admin_ns.route(
        "/platform-billing/issued-invoices/<int:issued_id>/credit-note"
    )
    class PlatformIssuedCredit(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @require_admin_capability(CAP_BILLING_CREDIT)
        @ip_whitelist_required()
        def post(self, issued_id: int):
            data = request.get_json(silent=True) or {}
            try:
                inv = create_credit_note(
                    issued_id,
                    reason=str(data.get("reason") or ""),
                    created_by_user_id=_admin_user_id_from_jwt(),
                )
            except ValueError as e:
                return APIErrorHandler.handle_validation_error(
                    str(e), logger_instance=logger
                )
            return {"ok": True, "issued_invoice": _serialize_issued(inv)}, 201

    @admin_ns.route("/platform-billing/issued-invoices/refresh-overdue")
    class PlatformIssuedRefreshOverdue(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        def post(self):
            n = refresh_overdue_statuses()
            return {"ok": True, "updated": n}, 200

    @admin_ns.route(
        "/platform-billing/bookings/<int:booking_id>/billing-origin"
    )
    class BookingBillingOriginCorrect(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        def post(self, booking_id: int):
            from models import Booking
            from services.platform_billing.billing_origin import correct_billing_origin

            booking = db.session.get(Booking, booking_id)
            if not booking:
                admin_ns.abort(404, "Réservation introuvable")
            data = request.get_json(silent=True) or {}
            user_id = _admin_user_id_from_jwt()
            try:
                correct_billing_origin(
                    booking,
                    str(data.get("billing_origin") or ""),
                    reason=str(data.get("reason") or ""),
                    author_user_id=user_id,
                )
            except ValueError as e:
                return APIErrorHandler.handle_validation_error(
                    str(e), logger_instance=logger
                )
            return {
                "ok": True,
                "booking_id": booking_id,
                "billing_origin": booking.billing_origin,
                "billing_origin_source": booking.billing_origin_source,
            }, 200

    @admin_ns.route(
        "/platform-billing/companies/<int:company_id>/dunning/pause"
    )
    class PlatformBillingDunningPause(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @require_admin_capability(CAP_BILLING_LOCK)
        @ip_whitelist_required()
        @limiter.limit(_RL_ADMIN_WRITE)
        def post(self, company_id: int):
            from datetime import datetime, timedelta, timezone

            from services.platform_billing.capabilities import pause_dunning

            company = db.session.get(Company, company_id)
            if not company:
                admin_ns.abort(404, "Entreprise introuvable")
            data = request.get_json(silent=True) or {}
            reason = (data.get("reason") or "").strip()
            if len(reason) < 5:
                return APIErrorHandler.handle_validation_error(
                    "Une raison d'au moins 5 caractères est requise.",
                    logger_instance=logger,
                )
            until = None
            paused_until_raw = data.get("paused_until")
            if paused_until_raw:
                try:
                    until = datetime.fromisoformat(
                        str(paused_until_raw).replace("Z", "+00:00")
                    )
                except ValueError:
                    return APIErrorHandler.handle_validation_error(
                        "paused_until invalide (ISO 8601 attendu).",
                        logger_instance=logger,
                    )
                if until.tzinfo is None:
                    until = until.replace(tzinfo=timezone.utc)
            else:
                days = int(data.get("days") or 14)
                until = datetime.now(timezone.utc) + timedelta(days=max(1, days))
            if until <= datetime.now(timezone.utc):
                return APIErrorHandler.handle_validation_error(
                    "paused_until doit être dans le futur.",
                    logger_instance=logger,
                )
            pause_dunning(
                company_id,
                until=until,
                reason=reason,
                user_id=_admin_user_id_from_jwt(),
            )
            db.session.commit()
            return {
                "ok": True,
                "dunning_paused_until": until.isoformat(),
                "dunning_pause_reason": reason,
            }, 200

    @admin_ns.route(
        "/platform-billing/companies/<int:company_id>/dunning/resume"
    )
    class PlatformBillingDunningResume(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @require_admin_capability(CAP_BILLING_LOCK)
        @ip_whitelist_required()
        @limiter.limit(_RL_ADMIN_WRITE)
        def post(self, company_id: int):
            from services.platform_billing.capabilities import clear_dunning_pause

            company = db.session.get(Company, company_id)
            if not company:
                admin_ns.abort(404, "Entreprise introuvable")
            data = request.get_json(silent=True) or {}
            reason = (data.get("reason") or "").strip()
            if len(reason) < 5:
                return APIErrorHandler.handle_validation_error(
                    "Une raison d'au moins 5 caractères est requise.",
                    logger_instance=logger,
                )
            clear_dunning_pause(company_id)
            db.session.commit()
            return {
                "ok": True,
                "dunning_paused_until": None,
                "dunning_pause_reason": None,
                "reason": reason,
            }, 200

    @admin_ns.route(
        "/platform-billing/companies/<int:company_id>/billing-access"
    )
    class PlatformBillingAccessStateResource(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @require_admin_capability(CAP_BILLING_LOCK)
        @ip_whitelist_required()
        @limiter.limit(_RL_ADMIN_WRITE)
        def put(self, company_id: int):
            from datetime import datetime, timedelta, timezone

            from models.enums import (
                PlatformBillingAccessState,
                PlatformBillingStateSource,
            )
            from services.platform_billing.capabilities import (
                pause_dunning,
                set_billing_access_state,
            )

            company = db.session.get(Company, company_id)
            if not company:
                admin_ns.abort(404, "Entreprise introuvable")
            data = request.get_json(silent=True) or {}
            state = (data.get("state") or "").strip()
            reason = (data.get("reason_code") or data.get("reason") or "").strip()
            if len(reason) < 5:
                return APIErrorHandler.handle_validation_error(
                    "Une raison d'au moins 5 caractères est requise.",
                    logger_instance=logger,
                )
            pause_days = data.get("pause_days_after_lift")
            try:
                set_billing_access_state(
                    company_id,
                    state,
                    source=PlatformBillingStateSource.ADMIN_MANUAL.value,
                    reason_code=reason
                    if state != PlatformBillingAccessState.ACTIVE.value
                    else "admin_lift",
                    force=True,
                )
                if (
                    state == PlatformBillingAccessState.ACTIVE.value
                    and pause_days is not None
                ):
                    until = datetime.now(timezone.utc) + timedelta(
                        days=max(1, int(pause_days))
                    )
                    pause_dunning(
                        company_id,
                        until=until,
                        reason="pause_after_admin_lift",
                        user_id=_admin_user_id_from_jwt(),
                    )
                db.session.commit()
            except ValueError as e:
                return APIErrorHandler.handle_validation_error(
                    str(e), logger_instance=logger
                )
            db.session.refresh(company)
            return {
                "ok": True,
                "platform_billing_access_state": company.platform_billing_access_state,
                "platform_billing_state_source": company.platform_billing_state_source,
                "dunning_paused_until": company.dunning_paused_until.isoformat()
                if company.dunning_paused_until
                else None,
            }, 200

    @admin_ns.route(
        "/platform-billing/issued-invoices/<int:issued_id>/dunning-hold"
    )
    class PlatformInvoiceDunningHoldResource(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @limiter.limit(_RL_ADMIN_WRITE)
        def post(self, issued_id: int):
            from decimal import Decimal

            from services.platform_billing.dunning import create_dunning_hold

            data = request.get_json(silent=True) or {}
            try:
                hold = create_dunning_hold(
                    issued_id,
                    reason=str(data.get("reason") or "contestation"),
                    disputed_amount=Decimal(str(data.get("disputed_amount") or 0)),
                    hold_until=None,
                    user_id=_admin_user_id_from_jwt(),
                )
            except ValueError as e:
                return APIErrorHandler.handle_validation_error(
                    str(e), logger_instance=logger
                )
            return {"ok": True, "hold_id": hold.id}, 201
