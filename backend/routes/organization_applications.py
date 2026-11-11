"""Préinscription publique Entreprise et Institution.

Le dépôt ne crée ni tenant, ni habilitation, ni rôle professionnel.
"""

from __future__ import annotations

from typing import Any

from flask import request
from flask_restx import Namespace, Resource

from ext import limiter
from services.organization_application_approval import accept_institution_invitation
from services.organization_applications import (
    ApplicationRejected,
    read_with_tracking_token,
    submit_application,
    verify_email_token,
)
from shared.response_helpers import json_response

organization_applications_ns = Namespace(
    "organization_applications",
    description="Préinscription entreprise et institution",
)

_COOKIE = "lirie_organization_application"
_COOKIE_MAX_AGE = 14 * 24 * 3600


def _body() -> dict[str, Any]:
    data = request.get_json(silent=True)
    if isinstance(data, dict):
        return data
    return {}


def _tracking_secret() -> str:
    header = request.headers.get("Authorization", "")
    if header.startswith("Bearer "):
        return header[7:].strip()
    return str(request.cookies.get(_COOKIE) or "")


@organization_applications_ns.route("")
class OrganizationApplicationDeposit(Resource):
    @limiter.limit("8 per hour")
    def post(self):
        try:
            payload = submit_application(_body())
        except ApplicationRejected as error:
            return json_response(
                {"error": error.code, "message": error.message},
                error.status,
            )
        return json_response(payload, 202)


@organization_applications_ns.route("/verify")
class OrganizationApplicationVerify(Resource):
    @limiter.limit("12 per hour")
    def post(self):
        try:
            payload = verify_email_token(str(_body().get("token") or ""))
        except ApplicationRejected as error:
            return json_response(
                {"error": error.code, "message": error.message},
                error.status,
            )
        response = json_response(payload, 200)
        response.set_cookie(
            _COOKIE,
            str(payload["tracking_token"]),
            max_age=_COOKIE_MAX_AGE,
            httponly=True,
            samesite="Lax",
            path="/",
        )
        return response


@organization_applications_ns.route("/institution-activation")
class OrganizationApplicationInstitutionActivation(Resource):
    @limiter.limit("12 per hour")
    def post(self):
        try:
            payload = accept_institution_invitation(str(_body().get("token") or ""))
        except ApplicationRejected as error:
            return json_response(
                {"error": error.code, "message": error.message},
                error.status,
            )
        return json_response(payload, 200)


@organization_applications_ns.route("/mine")
class OrganizationApplicationMine(Resource):
    def get(self):
        try:
            payload = read_with_tracking_token(_tracking_secret())
        except ApplicationRejected as error:
            return json_response(
                {"error": error.code, "message": error.message},
                error.status,
            )
        return json_response(payload, 200)
