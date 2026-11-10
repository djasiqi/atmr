"""Invitations chauffeur : portail entreprise et lien public.

Le lien public ne crée pas de chauffeur tant que le verrou d'activation
est fermé. Aucune inscription chauffeur publique n'est exposée ici.
"""

from __future__ import annotations

from typing import Any

from flask import request
from flask_jwt_extended import get_jwt_identity, jwt_required, verify_jwt_in_request
from flask_restx import Resource

from ext import limiter, role_required
from models import User, UserRole
from routes.companies import companies_ns
from routes.driver import driver_ns
from services.driver_invitations import (
    InvitationRejected,
    accept_invitation,
    cancel_invitation,
    create_invitation,
    list_invitations,
    open_invitation,
    resend_invitation,
    revoke_attachment,
)
from shared.response_helpers import json_response


def _body() -> dict[str, Any]:
    data = request.get_json(silent=True)
    if isinstance(data, dict):
        return data
    return {}


def _actor_and_company():
    from application.users.get_current_company import GetCurrentCompanyUseCase

    result = GetCurrentCompanyUseCase().execute()
    if result.error or not result.company:
        raise InvitationRejected(
            "acces_refuse",
            (result.error or {}).get("error") or "Accès refusé.",
            status=result.status_code or 403,
        )
    identity = get_jwt_identity()
    actor = User.query.filter_by(public_id=str(identity or "")).one_or_none()
    if actor is None:
        raise InvitationRejected("acces_refuse", "Accès refusé.", status=403)
    return result.company, actor


def _authenticated_user_id() -> int | None:
    try:
        verify_jwt_in_request(optional=True)
    except Exception:
        return None
    identity = get_jwt_identity()
    if not identity:
        return None
    user = User.query.filter_by(public_id=str(identity)).one_or_none()
    if user is None:
        return None
    return int(user.id)


def _rejected(error: InvitationRejected):
    return json_response(
        {"error": error.code, "message": error.message},
        error.status,
    )


@companies_ns.route("/me/driver-invitations")
class CompanyDriverInvitations(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        try:
            company, actor = _actor_and_company()
            payload = list_invitations(company, actor)
        except InvitationRejected as error:
            return _rejected(error)
        return json_response(payload, 200)

    @jwt_required()
    @role_required(UserRole.company)
    @limiter.limit("20 per hour")
    def post(self):
        try:
            company, actor = _actor_and_company()
            payload = create_invitation(company, actor, _body())
        except InvitationRejected as error:
            return _rejected(error)
        return json_response(payload, 201)


@companies_ns.route("/me/driver-invitations/<string:public_id>/resend")
class CompanyDriverInvitationResend(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @limiter.limit("10 per hour")
    def post(self, public_id: str):
        try:
            company, actor = _actor_and_company()
            payload = resend_invitation(company, actor, public_id)
        except InvitationRejected as error:
            return _rejected(error)
        return json_response(payload, 200)


@companies_ns.route("/me/driver-invitations/<string:public_id>/cancel")
class CompanyDriverInvitationCancel(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, public_id: str):
        try:
            company, actor = _actor_and_company()
            payload = cancel_invitation(company, actor, public_id)
        except InvitationRejected as error:
            return _rejected(error)
        return json_response(payload, 200)


@companies_ns.route("/me/driver-invitations/<string:public_id>/revoke")
class CompanyDriverInvitationRevoke(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, public_id: str):
        try:
            company, actor = _actor_and_company()
            payload = revoke_attachment(company, actor, public_id)
        except InvitationRejected as error:
            return _rejected(error)
        return json_response(payload, 200)


@driver_ns.route("/invitations/open")
class DriverInvitationOpen(Resource):
    @limiter.limit("30 per hour")
    def post(self):
        try:
            payload = open_invitation(str(_body().get("token") or ""))
        except InvitationRejected as error:
            return _rejected(error)
        return json_response(payload, 200)


@driver_ns.route("/invitations/accept")
class DriverInvitationAccept(Resource):
    @limiter.limit("12 per hour")
    def post(self):
        try:
            payload = accept_invitation(_body(), _authenticated_user_id())
        except InvitationRejected as error:
            return _rejected(error)
        return json_response(payload, 200)
