"""Routes admin facturation plateforme LIRIE (distinct pilotage / Invoice)."""

from __future__ import annotations

import csv
import io
import json
import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from flask import Response, request
from flask_jwt_extended import jwt_required
from flask_restx import Namespace, Resource
from sqlalchemy import func

from ext import db, limiter, role_required
from models import Company, UserRole
from models.platform_billing import (
    CompanyPlatformBillingConfig,
    PlatformBillingPeriod,
    PlatformInvoice,
    PlatformInvoiceLine,
    PlatformSubscriptionPricing,
    PlatformSupportEntry,
)
from security.ip_whitelist import ip_whitelist_required
from services.platform_billing.engine import (
    get_or_create_period,
    lock_platform_billing_period,
    recalculate_platform_period_drafts,
)
from shared.error_handlers import APIErrorHandler

logger = logging.getLogger(__name__)


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


def _serialize_line(ln: PlatformInvoiceLine) -> dict[str, Any]:
    return {
        "id": ln.id,
        "line_type": ln.line_type,
        "label": ln.label,
        "amount": float(ln.amount),
        "quantity": float(ln.quantity) if ln.quantity is not None else None,
        "unit_amount": float(ln.unit_amount) if ln.unit_amount is not None else None,
        "snapshot_json": ln.snapshot_json,
        "sort_order": ln.sort_order,
    }


def _serialize_invoice(inv: PlatformInvoice) -> dict[str, Any]:
    lines = sorted(inv.lines, key=lambda x: (x.sort_order, x.id))
    return {
        "id": inv.id,
        "company_id": inv.company_id,
        "period_id": inv.period_id,
        "currency": inv.currency,
        "subtotal_amount": float(inv.subtotal_amount),
        "total_amount": float(inv.total_amount),
        "cancelled_at": inv.cancelled_at.isoformat() if inv.cancelled_at else None,
        "lines": [_serialize_line(x) for x in lines],
    }


def _serialize_company_platform_config(cfg: CompanyPlatformBillingConfig) -> dict[str, Any]:
    return {
        "id": cfg.id,
        "company_id": cfg.company_id,
        "is_billing_enabled": cfg.is_billing_enabled,
        "dispatch_mode_override": cfg.dispatch_mode_override,
        "commission_rate": float(cfg.commission_rate) if cfg.commission_rate is not None else None,
        "support_hourly_rate_default": float(cfg.support_hourly_rate_default)
        if cfg.support_hourly_rate_default is not None
        else None,
        "effective_from": cfg.effective_from.isoformat() if cfg.effective_from else None,
        "effective_to": cfg.effective_to.isoformat() if cfg.effective_to else None,
        "is_active": cfg.is_active,
        "notes": cfg.notes,
    }


def register_platform_billing_routes(admin_ns: Namespace) -> None:
    """Enregistre les routes sous /admin/platform-billing/."""
    if getattr(admin_ns, "_lirie_platform_billing_registered", False):
        return
    setattr(admin_ns, "_lirie_platform_billing_registered", True)

    @admin_ns.route("/platform-billing/periods")
    class PlatformBillingPeriods(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @limiter.limit("60 per hour")
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
        @limiter.limit("30 per hour")
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
            if month < 1 or month > 12:
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
        @limiter.limit("120 per hour")
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
        @limiter.limit("20 per hour")
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
        @limiter.limit("20 per hour")
        def post(self, period_id: int):
            try:
                p = lock_platform_billing_period(period_id)
                return _serialize_period(p), 200
            except ValueError as e:
                return {"error": str(e)}, 400

    @admin_ns.route("/platform-billing/periods/<int:period_id>/export")
    class PlatformBillingPeriodExport(Resource):
        """Export CSV des relevés de la période (une ligne par ligne de facture)."""

        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @limiter.limit("30 per hour")
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
        @limiter.limit("120 per hour")
        def get(self, period_id: int):
            p = db.session.get(PlatformBillingPeriod, period_id)
            if not p:
                admin_ns.abort(404, "Période introuvable")
            invs = PlatformInvoice.query.filter_by(period_id=period_id).all()
            return {
                "invoices": [_serialize_invoice(i) for i in invs],
            }, 200

    @admin_ns.route("/platform-billing/invoices/<int:invoice_id>")
    class PlatformBillingInvoiceDetail(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @limiter.limit("120 per hour")
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
        @limiter.limit("120 per hour")
        def get(self):
            rows = (
                PlatformSubscriptionPricing.query.order_by(
                    PlatformSubscriptionPricing.dispatch_mode.asc(),
                    PlatformSubscriptionPricing.volume_min.asc(),
                ).all()
            )
            return {
                "items": [
                    {
                        "id": r.id,
                        "dispatch_mode": r.dispatch_mode,
                        "volume_min": r.volume_min,
                        "volume_max": r.volume_max,
                        "price_monthly": float(r.price_monthly),
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
        @limiter.limit("60 per hour")
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
            co_query = Company.query.order_by(Company.name.asc())
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
        @limiter.limit("120 per hour")
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
            return {"config": _serialize_company_platform_config(cfg), "company_id": company_id}, 200

        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @limiter.limit("60 per hour")
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
            if "support_hourly_rate_default" in data and data["support_hourly_rate_default"] is not None:
                cfg.support_hourly_rate_default = Decimal(
                    str(data["support_hourly_rate_default"])
                )
            elif "support_hourly_rate_default" in data:
                cfg.support_hourly_rate_default = None
            if "is_active" in data:
                cfg.is_active = bool(data["is_active"])
            if "notes" in data:
                cfg.notes = data.get("notes")
            if "effective_from" in data:
                v = data.get("effective_from")
                if v:
                    cfg.effective_from = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
                else:
                    cfg.effective_from = None
            if "effective_to" in data:
                v = data.get("effective_to")
                if v:
                    cfg.effective_to = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
                else:
                    cfg.effective_to = None
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
        @limiter.limit("120 per hour")
        def get(self):
            cid = request.args.get("company_id", type=int)
            q = PlatformSupportEntry.query
            if cid:
                q = q.filter(PlatformSupportEntry.company_id == cid)
            rows = q.order_by(PlatformSupportEntry.occurred_at.desc()).limit(200).all()
            return {
                "entries": [
                    {
                        "id": e.id,
                        "company_id": e.company_id,
                        "occurred_at": e.occurred_at.isoformat() if e.occurred_at else None,
                        "duration_minutes": e.duration_minutes,
                        "category": e.category,
                        "description": e.description,
                        "hourly_rate_snapshot": float(e.hourly_rate_snapshot),
                        "amount": float(e.amount),
                        "validated_at": e.validated_at.isoformat()
                        if e.validated_at
                        else None,
                        "billing_period_id": e.billing_period_id,
                    }
                    for e in rows
                ]
            }, 200

        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @limiter.limit("60 per hour")
        def post(self):
            data = request.get_json(silent=True) or {}
            try:
                company_id = int(data["company_id"])
                occurred_at = datetime.fromisoformat(
                    str(data["occurred_at"]).replace("Z", "+00:00")
                )
                duration_minutes = int(data["duration_minutes"])
                category = str(data.get("category") or "support")
                hourly = Decimal(str(data["hourly_rate_snapshot"]))
                amount = Decimal(str(data["amount"]))
            except (KeyError, TypeError, ValueError) as e:
                return APIErrorHandler.handle_validation_error(
                    f"Payload invalide: {e}", logger_instance=logger
                )
            se = PlatformSupportEntry(
                company_id=company_id,
                occurred_at=occurred_at,
                duration_minutes=duration_minutes,
                category=category,
                description=data.get("description"),
                hourly_rate_snapshot=hourly,
                amount=amount,
            )
            db.session.add(se)
            db.session.commit()
            return {"id": se.id}, 201

    @admin_ns.route("/platform-billing/support-entries/<int:entry_id>/validate")
    class PlatformSupportEntryValidate(Resource):
        @jwt_required()
        @role_required(UserRole.admin)
        @ip_whitelist_required()
        @limiter.limit("60 per hour")
        def post(self, entry_id: int):
            from flask_jwt_extended import get_jwt_identity

            se = db.session.get(PlatformSupportEntry, entry_id)
            if not se:
                admin_ns.abort(404, "Entrée introuvable")
            uid = get_jwt_identity()
            try:
                user_id = int(uid) if uid is not None else None
            except (TypeError, ValueError):
                user_id = None
            se.validated_at = datetime.now(UTC)
            se.validated_by_user_id = user_id
            db.session.commit()
            return {"ok": True, "id": se.id}, 200
