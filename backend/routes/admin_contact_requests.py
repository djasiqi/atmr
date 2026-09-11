from __future__ import annotations

from flask import request
from flask_jwt_extended import jwt_required
from flask_restx import Namespace, Resource

from ext import db, limiter, role_required
from models import ContactRequest, UserRole
from services.admin_authz import CAP_PARTNERS_READ, require_admin_capability
from services.contact.retry import retry_internal_notification

admin_contact_requests_ns = Namespace(
    "admin_contact_requests",
    description="Suivi admin des demandes de contact",
)


@admin_contact_requests_ns.route("")
class AdminContactRequestsList(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    @require_admin_capability(CAP_PARTNERS_READ)
    def get(self):
        status_filter = str(request.args.get("status") or "").strip().lower()
        query = ContactRequest.query
        if status_filter == "failed":
            query = query.filter(ContactRequest.email_delivery_status == "failed")
        elif status_filter == "new":
            query = query.filter(ContactRequest.status == "new")
        rows = query.order_by(ContactRequest.created_at.desc()).limit(200).all()
        failed_count = ContactRequest.query.filter(
            ContactRequest.email_delivery_status == "failed"
        ).count()
        return {
            "ok": True,
            "items": [row.serialize for row in rows],
            "failed_count": failed_count,
        }, 200


@admin_contact_requests_ns.route("/<int:contact_request_id>/retry-notification")
class AdminRetryContactNotification(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    @require_admin_capability(CAP_PARTNERS_READ)
    @limiter.limit("30 per hour")
    def post(self, contact_request_id: int):
        row = db.session.get(ContactRequest, contact_request_id)
        if row is None:
            return {"error": "not_found", "message": "Demande introuvable."}, 404
        result = retry_internal_notification(row, ignore_cooldown=True)
        db.session.commit()
        return {
            "ok": bool(result.get("ok")),
            "skipped": bool(result.get("skipped")),
            "item": row.serialize,
            "error": result.get("error"),
        }, 200
