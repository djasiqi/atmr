"""Routes transporteur pour les créances PORTAL."""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal, InvalidOperation

from flask import request
from flask_jwt_extended import get_jwt_identity, jwt_required
from flask_restx import Namespace, Resource

from ext import db, role_required
from models.enums import UserRole
from models.portal_receivable import PortalReceivable
from models.user import User
from routes.api_error_utils import auth_error
from routes.companies import _get_current_company_via_use_case
from services.billing.portal_receivable import (
    PortalReceivableError,
    ReceivableLineInput,
    add_portal_receivable_payment,
    cancel_portal_receivable,
    create_portal_receivable,
    dispute_portal_receivable,
    serialize_portal_receivable,
)

logger = logging.getLogger(__name__)

portal_receivables_ns = Namespace(
    "portal-receivables",
    description="Créances PORTAL enregistrées par le transporteur",
)


def _parse_datetime(raw: object, field: str) -> datetime:
    if isinstance(raw, datetime):
        return raw
    text = str(raw or "").strip()
    if not text:
        raise PortalReceivableError(
            f"Le champ {field} est obligatoire.",
            code=f"{field}_required",
        )
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PortalReceivableError(
            f"Format invalide pour {field}.",
            code=f"{field}_invalid",
        ) from exc


def _current_user() -> User | None:
    identity = get_jwt_identity()
    if not identity:
        return None
    return User.query.filter_by(public_id=str(identity)).first()


def _owned_receivable(company_id: int, receivable_id: int) -> PortalReceivable:
    receivable = (
        PortalReceivable.query.filter_by(
            id=receivable_id, creditor_company_id=company_id
        )
        .one_or_none()
    )
    if receivable is None:
        raise PortalReceivableError(
            "Créance introuvable.",
            code="receivable_not_found",
        )
    return receivable


@portal_receivables_ns.route("")
class PortalReceivableCollection(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        company, error, status = _get_current_company_via_use_case()
        if error or company is None:
            return error or {"error": "Entreprise non trouvée"}, status or 404
        rows = (
            PortalReceivable.query.filter_by(creditor_company_id=company.id)
            .order_by(PortalReceivable.id.desc())
            .all()
        )
        return {"data": [serialize_portal_receivable(row) for row in rows]}, 200

    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        company, error, status = _get_current_company_via_use_case()
        if error or company is None:
            return error or {"error": "Entreprise non trouvée"}, status or 404
        user = _current_user()
        if user is None:
            return auth_error("unauthorized", "Utilisateur introuvable.", 401)
        data = request.get_json(silent=True) or {}
        if any(
            key in data
            for key in (
                "debtor_user_id",
                "debtor_name",
                "debtor_email",
                "debtor_phone",
                "debtor_billing_address",
            )
        ):
            return {
                "error": "client_supplied_debtor_forbidden",
                "message": "Le débiteur est fixé par la preuve de commande.",
            }, 400
        try:
            raw_lines = data.get("lines") or []
            if not isinstance(raw_lines, list):
                raise PortalReceivableError(
                    "Les lignes doivent être une liste.",
                    code="lines_invalid",
                )
            lines: list[ReceivableLineInput] = []
            for item in raw_lines:
                if not isinstance(item, dict):
                    raise PortalReceivableError(
                        "Chaque ligne doit être un objet.",
                        code="line_invalid",
                    )
                try:
                    amount = Decimal(str(item.get("invoiced_amount")))
                except (InvalidOperation, TypeError) as exc:
                    raise PortalReceivableError(
                        "Montant de ligne invalide.",
                        code="invoiced_amount_invalid",
                    ) from exc
                lines.append(
                    ReceivableLineInput(
                        booking_id=int(item["booking_id"]),
                        invoiced_amount=amount,
                        description=item.get("description"),
                    )
                )
            receivable = create_portal_receivable(
                company=company,
                recorded_by_user_id=int(user.id),
                external_invoice_number=str(data.get("external_invoice_number") or ""),
                issued_at=_parse_datetime(data.get("issued_at"), "issued_at"),
                due_date=_parse_datetime(data.get("due_date"), "due_date"),
                lines=lines,
                currency=str(data.get("currency") or "CHF"),
            )
            db.session.commit()
            return {"data": serialize_portal_receivable(receivable)}, 201
        except PortalReceivableError as exc:
            db.session.rollback()
            return {"error": exc.code, "message": exc.message}, 400
        except Exception as exc:
            db.session.rollback()
            logger.exception("Création créance PORTAL échouée: %s", exc)
            return {"error": "server_error", "message": "Erreur serveur."}, 500


@portal_receivables_ns.route("/<int:receivable_id>")
class PortalReceivableDetail(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self, receivable_id: int):
        company, error, status = _get_current_company_via_use_case()
        if error or company is None:
            return error or {"error": "Entreprise non trouvée"}, status or 404
        try:
            receivable = _owned_receivable(int(company.id), receivable_id)
            return {"data": serialize_portal_receivable(receivable)}, 200
        except PortalReceivableError as exc:
            return {"error": exc.code, "message": exc.message}, 404


@portal_receivables_ns.route("/<int:receivable_id>/payments")
class PortalReceivablePayments(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, receivable_id: int):
        company, error, status = _get_current_company_via_use_case()
        if error or company is None:
            return error or {"error": "Entreprise non trouvée"}, status or 404
        user = _current_user()
        if user is None:
            return auth_error("unauthorized", "Utilisateur introuvable.", 401)
        data = request.get_json(silent=True) or {}
        try:
            receivable = _owned_receivable(int(company.id), receivable_id)
            add_portal_receivable_payment(
                receivable=receivable,
                amount=Decimal(str(data.get("amount"))),
                paid_at=_parse_datetime(data.get("paid_at"), "paid_at"),
                method=str(data.get("method") or ""),
                recorded_by_user_id=int(user.id),
                reference=data.get("reference"),
            )
            db.session.commit()
            return {"data": serialize_portal_receivable(receivable)}, 201
        except (PortalReceivableError, InvalidOperation, TypeError, ValueError) as exc:
            db.session.rollback()
            if isinstance(exc, PortalReceivableError):
                return {"error": exc.code, "message": exc.message}, 400
            return {
                "error": "payment_invalid",
                "message": "Paiement invalide.",
            }, 400


@portal_receivables_ns.route("/<int:receivable_id>/dispute")
class PortalReceivableDispute(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, receivable_id: int):
        company, error, status = _get_current_company_via_use_case()
        if error or company is None:
            return error or {"error": "Entreprise non trouvée"}, status or 404
        user = _current_user()
        if user is None:
            return auth_error("unauthorized", "Utilisateur introuvable.", 401)
        data = request.get_json(silent=True) or {}
        try:
            receivable = _owned_receivable(int(company.id), receivable_id)
            dispute_portal_receivable(
                receivable=receivable,
                reason=str(data.get("reason") or ""),
                actor_user_id=int(user.id),
            )
            db.session.commit()
            return {"data": serialize_portal_receivable(receivable)}, 200
        except PortalReceivableError as exc:
            db.session.rollback()
            return {"error": exc.code, "message": exc.message}, 400


@portal_receivables_ns.route("/<int:receivable_id>/cancel")
class PortalReceivableCancel(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, receivable_id: int):
        company, error, status = _get_current_company_via_use_case()
        if error or company is None:
            return error or {"error": "Entreprise non trouvée"}, status or 404
        user = _current_user()
        if user is None:
            return auth_error("unauthorized", "Utilisateur introuvable.", 401)
        data = request.get_json(silent=True) or {}
        try:
            receivable = _owned_receivable(int(company.id), receivable_id)
            cancel_portal_receivable(
                receivable=receivable,
                reason=str(data.get("reason") or ""),
                actor_user_id=int(user.id),
            )
            db.session.commit()
            return {"data": serialize_portal_receivable(receivable)}, 200
        except PortalReceivableError as exc:
            db.session.rollback()
            return {"error": exc.code, "message": exc.message}, 400
