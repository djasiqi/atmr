"""Lecture et contestation des créances PORTAL côté client."""

from __future__ import annotations

from flask import request
from flask_jwt_extended import get_jwt_identity, jwt_required
from flask_restx import Namespace, Resource

from ext import db, role_required
from models.enums import UserRole
from models.portal_receivable import PortalReceivable
from models.user import User
from routes.api_error_utils import auth_error
from services.billing.client_carrier_invoices import (
    company_invoice_visible_to_client,
    list_company_invoices_for_client_user,
)
from services.billing.portal_receivable import (
    PortalReceivableError,
    dispute_portal_receivable,
    serialize_portal_receivable_for_client,
)

client_portal_receivables_ns = Namespace(
    "client-portal-receivables",
    description="Créances PORTAL visibles par le client débiteur",
)


def _current_user() -> User | None:
    identity = get_jwt_identity()
    if not identity:
        return None
    return User.query.filter_by(public_id=str(identity)).first()


@client_portal_receivables_ns.route("")
class ClientPortalReceivableCollection(Resource):
    @jwt_required()
    @role_required(UserRole.client)
    def get(self):
        user = _current_user()
        if user is None:
            return auth_error("unauthorized", "Utilisateur introuvable.", 401)
        rows = (
            PortalReceivable.query.filter_by(debtor_user_id=int(user.id))
            .order_by(PortalReceivable.due_date.desc(), PortalReceivable.id.desc())
            .all()
        )
        receivables = [serialize_portal_receivable_for_client(row) for row in rows]
        known_numbers = {
            str(item.get("external_invoice_number") or "").strip()
            for item in receivables
        }
        invoices = [
            item
            for item in list_company_invoices_for_client_user(int(user.id))
            if str(item.get("external_invoice_number") or "").strip() not in known_numbers
        ]
        return {"data": [*receivables, *invoices]}, 200


@client_portal_receivables_ns.route("/invoices/<int:invoice_id>/pdf")
class ClientCarrierInvoicePdf(Resource):
    @jwt_required()
    @role_required(UserRole.client)
    def get(self, invoice_id: int):
        from werkzeug.exceptions import NotFound as WzNotFound

        from shared.invoice_pdf_filename import build_invoice_pdf_download_filename
        from shared.upload_path_resolver import serve_stored_upload

        user = _current_user()
        if user is None:
            return auth_error("unauthorized", "Utilisateur introuvable.", 401)
        invoice = company_invoice_visible_to_client(int(user.id), int(invoice_id))
        if invoice is None or not str(invoice.pdf_url or "").strip():
            return {
                "error": "invoice_pdf_not_found",
                "message": "Le PDF de cette facture n’est pas disponible.",
            }, 404
        try:
            return serve_stored_upload(
                invoice.pdf_url,
                as_attachment=True,
                download_filename=build_invoice_pdf_download_filename(invoice),
            )
        except WzNotFound:
            return {
                "error": "invoice_pdf_not_found",
                "message": "Le PDF de cette facture n’est pas disponible.",
            }, 404


@client_portal_receivables_ns.route("/<int:receivable_id>/dispute")
class ClientPortalReceivableDispute(Resource):
    @jwt_required()
    @role_required(UserRole.client)
    def post(self, receivable_id: int):
        user = _current_user()
        if user is None:
            return auth_error("unauthorized", "Utilisateur introuvable.", 401)
        data = request.get_json(silent=True) or {}
        receivable = PortalReceivable.query.filter_by(
            id=int(receivable_id), debtor_user_id=int(user.id)
        ).one_or_none()
        if receivable is None:
            return {
                "error": "receivable_not_found",
                "message": "Créance introuvable.",
            }, 404
        try:
            dispute_portal_receivable(
                receivable=receivable,
                reason=str(data.get("reason") or ""),
                actor_user_id=int(user.id),
            )
            db.session.commit()
            return {"data": serialize_portal_receivable_for_client(receivable)}, 200
        except PortalReceivableError as exc:
            db.session.rollback()
            return {"error": exc.code, "message": exc.message}, 400
