"""Administration des préinscriptions. Aucune activation de portail."""

from __future__ import annotations

from flask import request
from flask_jwt_extended import get_jwt_identity, jwt_required
from flask_restx import Namespace, Resource

from ext import role_required
from models.enums import UserRole
from services.admin_authz import CAP_PARTNERS_READ, require_admin_capability
from services.organization_application_approval import (
    approve_application,
    revoke_approved_grant,
)
from services.organization_application_review import (
    application_detail,
    apply_decision,
    list_applications,
    load_actor,
    search_organizations,
)
from services.organization_applications import ApplicationRejected

admin_organization_applications_ns = Namespace(
    "admin_organization_applications",
    description="Instruction des demandes d'inscription professionnelle",
)


def _actor():
    identity = get_jwt_identity()
    return load_actor(str(identity or ""))


def _optional_int(value):
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ApplicationRejected(
            "identifiant_invalide",
            "L'identifiant d'organisation est invalide.",
        ) from error


def _optional_bool(value):
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "oui"}
    return bool(value)


def _rejected(error: ApplicationRejected):
    return {"code": error.code, "message": error.message}, error.status


@admin_organization_applications_ns.route("")
class AdminOrganizationApplicationList(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    @require_admin_capability(CAP_PARTNERS_READ)
    def get(self):
        return list_applications(
            query=str(request.args.get("q") or ""),
            kind=str(request.args.get("kind") or ""),
            status=str(request.args.get("status") or ""),
        )


@admin_organization_applications_ns.route("/<string:public_id>")
class AdminOrganizationApplicationDetail(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    @require_admin_capability(CAP_PARTNERS_READ)
    def get(self, public_id: str):
        try:
            return application_detail(public_id)
        except ApplicationRejected as error:
            return _rejected(error)


@admin_organization_applications_ns.route("/<string:public_id>/organizations")
class AdminOrganizationApplicationSearch(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    @require_admin_capability(CAP_PARTNERS_READ)
    def get(self, public_id: str):
        try:
            application_detail(public_id)
        except ApplicationRejected as error:
            return _rejected(error)
        return search_organizations(str(request.args.get("q") or ""))


@admin_organization_applications_ns.route("/<string:public_id>/decisions")
class AdminOrganizationApplicationDecision(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    @require_admin_capability(CAP_PARTNERS_READ)
    def post(self, public_id: str):
        payload = request.get_json(silent=True) or {}
        try:
            actor = _actor()
            return apply_decision(
                public_id,
                actor,
                str(payload.get("action") or ""),
                payload.get("note"),
                confirmed=_optional_bool(payload.get("confirmed")),
                company_id=_optional_int(payload.get("company_id")),
                institution_id=_optional_int(payload.get("institution_id")),
            )
        except ApplicationRejected as error:
            return _rejected(error)


@admin_organization_applications_ns.route("/<string:public_id>/approve")
class AdminOrganizationApplicationApprove(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    @require_admin_capability(CAP_PARTNERS_READ)
    def post(self, public_id: str):
        try:
            return approve_application(public_id, _actor())
        except ApplicationRejected as error:
            return _rejected(error)


@admin_organization_applications_ns.route("/<string:public_id>/revoke")
class AdminOrganizationApplicationRevoke(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    @require_admin_capability(CAP_PARTNERS_READ)
    def post(self, public_id: str):
        try:
            return revoke_approved_grant(public_id, _actor())
        except ApplicationRejected as error:
            return _rejected(error)
