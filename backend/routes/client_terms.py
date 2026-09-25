"""Acceptation des CGU et CGV du client privé. Insertion et lecture seules."""

from __future__ import annotations

import logging
from io import BytesIO

from flask import request, send_file
from flask_jwt_extended import jwt_required
from flask_restx import Resource

from ext import db, limiter, role_required
from models.client_terms_acceptance import (
    DOCUMENT_TERMS_OF_SERVICE,
    DOCUMENT_TRANSPORT_TERMS,
)
from models.enums import UserRole
from repositories.client_repository import ClientRepository
from routes.clients import clients_ns
from services.legal.portal_terms_catalog import (
    CatalogIntegrityError,
    current_portal_terms,
)
from services.legal.portal_terms_pdf import (
    build_portal_terms_pdf_bytes,
    portal_terms_pdf_filename,
)
from services.legal.portal_terms_status import (
    accept_current_required_portal_terms,
    resolve_portal_terms_status,
)
from services.legal.record_terms_acceptance import (
    ClientSuppliedTermsError,
    PortalTermsContextError,
    list_portal_terms_acceptances,
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
        payload = request.get_json(silent=True) or {}
        try:
            reject_client_supplied_terms(payload)
            if payload.get("accept_current_required_terms") is not True:
                return {
                    "error": "terms_acceptance_required",
                    "message": (
                        "L'acceptation des conditions exigées doit être explicite."
                    ),
                }, 400
            rows, inserted = accept_current_required_portal_terms(current_user, client)
            db.session.commit()
        except ClientSuppliedTermsError as exc:
            db.session.rollback()
            return {"error": exc.code, "message": str(exc)}, 400
        except PortalTermsContextError as exc:
            db.session.rollback()
            return APIErrorHandler.handle_permission_error(
                str(exc),
                logger_instance=logger,
            )
        except CatalogIntegrityError as exc:
            db.session.rollback()
            return APIErrorHandler.handle_exception(exc, logger)
        except Exception as exc:
            db.session.rollback()
            return APIErrorHandler.handle_exception(exc, logger)
        body = [_serialize(row) for row in rows]
        if inserted:
            return created_response(data=body)
        return success_response(data=body)


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


@clients_ns.route("/me/portal-terms/<string:document_type>/pdf")
class ClientMyPortalTermsPdf(Resource):
    """PDF officiel du document courant (logo LIRIE + texte canonique)."""

    @jwt_required()
    @role_required(UserRole.client)
    @limiter.limit("60 per hour")
    def get(self, document_type: str):
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
        if document_type not in (
            DOCUMENT_TERMS_OF_SERVICE,
            DOCUMENT_TRANSPORT_TERMS,
        ):
            return APIErrorHandler.handle_validation_error(
                "Type de document inconnu.",
                logger_instance=logger,
            )
        try:
            from services.auth.portal_phone_verification import is_portal_client

            if not is_portal_client(client):
                raise PortalTermsContextError(
                    "Ces conditions concernent le compte client privé."
                )
            specs = current_portal_terms()
            match = next(
                (spec for spec in specs if spec.document_type == document_type),
                None,
            )
            if match is None:
                return APIErrorHandler.handle_not_found_error(
                    "Document introuvable.",
                    logger_instance=logger,
                )
            pdf_bytes = build_portal_terms_pdf_bytes(match)
            filename = portal_terms_pdf_filename(match)
        except PortalTermsContextError as exc:
            return APIErrorHandler.handle_permission_error(
                str(exc),
                logger_instance=logger,
            )
        except CatalogIntegrityError as exc:
            return APIErrorHandler.handle_exception(exc, logger)
        except Exception as exc:  # noqa: BLE001
            return APIErrorHandler.handle_exception(exc, logger)
        return send_file(
            BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename,
        )


def _serialize_status(user: object) -> dict[str, object]:
    resolved = resolve_portal_terms_status(user)
    return {
        "status": resolved.status,
        "documents": [
            {
                "document_type": document.document_type,
                "current_version": document.current_version,
                "current_hash": document.current_hash,
                "requires_reacceptance": document.requires_reacceptance,
                "accepted_version": document.accepted_version,
                "accepted_at": _iso(document.accepted_at),
                "acceptance_id": document.acceptance_id,
                "acceptance_required": document.acceptance_required,
                "contractual_basis": document.contractual_basis,
                "canonical_body": document.canonical_body,
            }
            for document in resolved.documents
        ],
    }


@clients_ns.route("/me/portal-terms-status")
class ClientMyPortalTermsStatus(Resource):
    """Statut des acceptations exigées. Lecture seule, sans écriture."""

    @jwt_required()
    @role_required(UserRole.client)
    @limiter.limit("120 per hour")
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
            payload = _serialize_status(current_user)
        except PortalTermsContextError as exc:
            return APIErrorHandler.handle_permission_error(
                str(exc),
                logger_instance=logger,
            )
        except CatalogIntegrityError as exc:
            return APIErrorHandler.handle_exception(exc, logger)
        return success_response(data=payload)
