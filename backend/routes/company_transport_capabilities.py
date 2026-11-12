"""Déclarations de transport de l'entreprise connectée."""

from flask import request
from flask_jwt_extended import get_jwt_identity, jwt_required
from flask_restx import Resource

from application.companies.transport_capabilities import (
    apply_transport_capabilities,
    capability_view,
)
from ext import db, role_required
from models import User, UserRole
from routes.companies import companies_ns
from shared.response_helpers import success_response


@companies_ns.route("/me/transport-capabilities")
class CompanyTransportCapabilities(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Lit les déclarations de l'entreprise courante."""
        company, error, status = _current_company()
        if error:
            return error, status
        return success_response(data=capability_view(company))

    @jwt_required()
    @role_required(UserRole.company)
    def put(self):
        """Enregistre une déclaration volontaire, sans géocodage."""
        company, error, status = _current_company()
        if error:
            return error, status
        user = User.query.filter_by(public_id=get_jwt_identity()).first()
        if user is None:
            return {"error": "Accès non autorisé"}, 401
        payload = request.get_json(silent=True) or {}
        try:
            view = apply_transport_capabilities(
                company,
                payload,
                user_id=int(user.id),
            )
            db.session.commit()
        except ValueError as exc:
            return {"error": str(exc)}, 400
        return success_response(data=view)


def _current_company():
    from routes.companies import _get_current_company_via_use_case

    company, error, status = _get_current_company_via_use_case()
    if error or company is None:
        return None, error or {"error": "Entreprise non trouvée"}, status or 404
    return company, None, None
