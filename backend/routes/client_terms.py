"""Acceptation des CGU et CGV du client privé. Insertion et lecture seules."""

from __future__ import annotations

import logging

from flask import request
from flask_jwt_extended import jwt_required
from flask_restx import Resource

from ext import db, limiter, role_required
from models.enums import UserRole
from repositories.client_repository import ClientRepository
from routes.clients import clients_ns
from services.legal.portal_terms_catalog import (
    CatalogIntegrityError,
    current_portal_terms,
)
from services.legal.record_terms_acceptance import (
    ClientSuppliedTermsError,
    PortalTermsContextError,
    list_portal_terms_acceptances,
    record_portal_terms_acceptance,
    reject_client_supplied_terms,
)
from shared.error_handlers import APIErrorHandler
from shared.infrastructure.adapters.auth_adapter import get_current_user_via_use_case
from shared.response_helpers import created_response, success_response

logger = logging.getLogger(__name__)
client_repo = ClientRepository()


def _iso(value: object) -> str | None:
    if value is None:
        return None
    iso = getattr(value, "isoformat", None)
    return iso() if callable(iso) else None


def _serialize(row: object) -> dict[str, object]:
    return {
        "id": row.id,
        "document_type": row.document_type,
        "terms_version": row.terms_version,
        "terms_hash": row.terms_hash,
        "accepted_at": _iso(row.accepted_at),
        "email_snapshot": row.email_snapshot,
        "email_verified_at_snapshot": _iso(row.email_verified_at_snapshot),
        "phone_snapshot": row.phone_snapshot,
        "phone_verified_at_snapshot": _iso(row.phone_verified_at_snapshot),
        "verification_method": row.verification_method,
    }


@clients_ns.route("/me/terms-acceptances")
class ClientMyTermsAcceptances(Resource):
    @jwt_required()
    @role_required(UserRole.client)
    @limiter.limit("60 per hour")
    def get(self):
        current_user = get_current_user_via_use_case()
        if not current_user:
            return APIErrorHandler.handle_permission_error(
                "Utilisateur introuvable ou jeton invalide",
                logger_instance=logger,
            )
        rows = list_portal_terms_acceptances(current_user.id)
        return success_response(data=[_serialize(row) for row in rows])

    @jwt_required()
    @role_required(UserRole.client)
    @limiter.limit("20 per hour")
    def post(self):
        current_user = get_current_user_via_use_case()
        if not current_user:
            return APIErrorHandler.handle_permission_error(
                "Utilisateur introuvable ou jeton invalide",
                logger_instance=logger,
            )
        client = client_repo.find_by_user_id(current_user.id)
        if client is None:
            return APIErrorHandler.handle_permission_error(
                "Profil client introuvable",
                logger_instance=logger,
            )
        try:
            reject_client_supplied_terms(request.get_json(silent=True) or {})
            rows = record_portal_terms_acceptance(current_user, client)
            db.session.commit()
        except ClientSuppliedTermsError as exc:
            return APIErrorHandler.handle_validation_error(
                str(exc),
                logger_instance=logger,
            )
        except PortalTermsContextError as exc:
            return APIErrorHandler.handle_permission_error(
                str(exc),
                logger_instance=logger,
            )
        except CatalogIntegrityError as exc:
            return APIErrorHandler.handle_exception(exc, logger)
        return created_response(data=[_serialize(row) for row in rows])


@clients_ns.route("/me/portal-terms")
class ClientMyPortalTerms(Resource):
    """Textes canoniques actuellement opposables. Lecture seule, sans acceptation."""

    @jwt_required()
    @role_required(UserRole.client)
    @limiter.limit("60 per hour")
    def get(self):
        current_user = get_current_user_via_use_case()
        if not current_user:
            return APIErrorHandler.handle_permission_error(
                "Utilisateur introuvable ou jeton invalide",
                logger_instance=logger,
            )
        client = client_repo.find_by_user_id(current_user.id)
        if client is None:
            return APIErrorHandler.handle_permission_error(
                "Profil client introuvable",
                logger_instance=logger,
            )
        try:
            from services.auth.portal_phone_verification import is_portal_client

            if not is_portal_client(client):
                raise PortalTermsContextError(
                    "Ces conditions concernent le compte client privé."
                )
            documents = current_portal_terms()
        except PortalTermsContextError as exc:
            return APIErrorHandler.handle_permission_error(
                str(exc),
                logger_instance=logger,
            )
        except CatalogIntegrityError as exc:
            return APIErrorHandler.handle_exception(exc, logger)
        return success_response(
            data=[
                {
                    "document_type": spec.document_type,
                    "terms_version": spec.terms_version,
                    "terms_hash": spec.terms_hash,
                    "canonical_body": spec.canonical_body,
                    "locale": spec.locale,
                }
                for spec in documents
            ]
        )
