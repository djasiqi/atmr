"""Préférences de règlement de l'entreprise connectée.

La rubrique répond 404 tant que le drapeau de recette est éteint.
Aucune préférence n'active un moyen de paiement.
"""

from flask import request
from flask_jwt_extended import get_jwt_identity, jwt_required
from flask_restx import Resource

from application.companies.settlement_settings import (
    record_patient_notice,
    settlement_view,
    update_preferences,
)
from domain.end_of_trip_settlement import activation
from ext import db, role_required
from models import User, UserRole
from routes.companies import companies_ns
from shared.response_helpers import success_response


def _disabled():
    return {"error": "Rubrique indisponible"}, 404


def _settings_enabled() -> bool:
    return activation.company_settings_enabled()


@companies_ns.route("/me/settlement-settings")
class CompanySettlementSettings(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        if not _settings_enabled():
            return _disabled()
        company, error, status = _current_company()
        if error:
            return error, status
        return success_response(data=settlement_view(int(company.id)))

    @jwt_required()
    @role_required(UserRole.company)
    def put(self):
        if not _settings_enabled():
            return _disabled()
        company, error, status = _current_company()
        if error:
            return error, status
        user = _actor()
        if user is None:
            return {"error": "Accès non autorisé"}, 401
        payload = request.get_json(silent=True) or {}
        if payload.get("company_id") not in (None, int(company.id)):
            return {"error": "Accès non autorisé"}, 403
        try:
            view = update_preferences(
                company_id=int(company.id),
                actor_role="company",
                actor_user_id=int(user.id),
                same_company=True,
                payload=payload,
            )
            db.session.commit()
        except PermissionError as exc:
            db.session.rollback()
            return {"error": str(exc)}, 403
        except ValueError as exc:
            db.session.rollback()
            return {"error": str(exc)}, 400
        return success_response(data=view)


@companies_ns.route("/me/settlement-settings/patient-notice/<int:booking_id>")
class CompanySettlementPatientNotice(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, booking_id: int):
        if not _settings_enabled():
            return _disabled()
        company, error, status = _current_company()
        if error:
            return error, status
        user = _actor()
        if user is None:
            return {"error": "Accès non autorisé"}, 401
        payload = request.get_json(silent=True) or {}
        try:
            view = record_patient_notice(
                booking_id=int(booking_id),
                company_id=int(company.id),
                actor_role="company",
                actor_user_id=int(user.id),
                same_company=True,
                payload=payload,
            )
            db.session.commit()
        except PermissionError as exc:
            db.session.rollback()
            return {"error": str(exc)}, 403
        except ValueError as exc:
            db.session.rollback()
            return {"error": str(exc)}, 400
        return success_response(data=view)


def _actor():
    return User.query.filter_by(public_id=get_jwt_identity()).first()


def _current_company():
    from routes.companies import _get_current_company_via_use_case

    company, error, status = _get_current_company_via_use_case()
    if error or company is None:
        return None, error or {"error": "Entreprise non trouvée"}, status or 404
    return company, None, None
