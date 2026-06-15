# routes/institutions.py
"""Routes pour le portail institutionnel (cliniques, EMS, IMAD, hôpitaux).

Ce module fournit les endpoints API pour les institutions:
- /me: Informations de l'institution courante
- /api-keys: Gestion des clés API pour DPI
- /dpi/probe: Endpoint probe pour tester l'auth API Key
"""

# pyright: reportGeneralTypeIssues=false, reportArgumentType=false, reportOperatorIssue=false
import logging
import secrets
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, TypedDict

import sentry_sdk
from flask import current_app, request
from flask_jwt_extended import get_jwt, jwt_required
from flask_restx import Namespace, Resource, fields

from application.institutions.institution_settings_service import (
    get_or_create_settings,
)
from ext import db, limiter, role_required
from models import DemoAccess, UserRole
from models.enums import InstitutionRole
from models.institution_api_key import (
    VALID_SCOPES,
    InstitutionApiKey,
    generate_api_key,
    validate_scopes,
)
from models.user import User
from routes.api_error_models import (
    create_api_error_model,
    create_not_found_error_model,
    create_permission_error_model,
    create_validation_error_model,
)
from schemas.institution_schemas import (
    InstitutionMyProfileUpdateSchema,
    InstitutionSettingsUpdateSchema,
    InstitutionUpdateSchema,
    InstitutionUserInviteSchema,
    InstitutionUserUpdateProfileSchema,
    InstitutionUserUpdateRoleSchema,
    PermissionRequestCreateSchema,
)
from security.api_key_auth import api_key_required
from security.audit_log import AuditLogger
from security.authorization import AuthorizationService
from shared.error_handlers import APIErrorHandler
from shared.upload_validation import ALLOWED_LOGO_EXT, validate_file_upload

logger = logging.getLogger(__name__)

MAX_LOGO_MB = 2
MAX_LOGO_BYTES = MAX_LOGO_MB * 1024 * 1024

# Namespace pour les institutions
institutions_ns = Namespace(
    "institutions",
    description="Opérations pour le portail institutionnel",
)

# Modèles Swagger
api_error_model = create_api_error_model(institutions_ns)
not_found_error_model = create_not_found_error_model(institutions_ns)
permission_error_model = create_permission_error_model(institutions_ns)
validation_error_model = create_validation_error_model(institutions_ns)

# Modèle de réponse pour /me
institution_me_model = institutions_ns.model(
    "InstitutionMe",
    {
        "id": fields.Integer(required=True, description="ID de l'institution"),
        "public_id": fields.String(required=True, description="Public ID"),
        "name": fields.String(required=True, description="Nom de l'institution"),
        "institution_type": fields.String(
            description="Type (clinic, ems, imad, hospital)"
        ),
        "address": fields.String(description="Adresse"),
        "logo_url": fields.String(description="URL du logo institution"),
        "institution_role": fields.String(
            description="Rôle de l'utilisateur dans l'institution"
        ),
        "user": fields.Raw(description="Informations de l'utilisateur"),
    },
)


def _resolve_me_identity_for_response(
    user: User, jwt_claims: Mapping[str, Any]
) -> tuple[str | None, str]:
    institution_role_raw = jwt_claims.get("institution_role")
    if institution_role_raw is None:
        institution_role_raw = getattr(user, "institution_role", None)
    institution_role = (
        str(institution_role_raw) if institution_role_raw is not None else None
    )

    email = str(user.email or "")
    if email.startswith("demo-"):
        access = (
            DemoAccess.query.filter(
                DemoAccess.demo_user_id == user.id,
                DemoAccess.status == "active",
            )
            .order_by(DemoAccess.created_at.desc())
            .first()
        )
        request_email = getattr(getattr(access, "demo_request", None), "email", None)
        if request_email:
            email = request_email
    return institution_role, email


def _build_institution_me_response(
    institution: Any, user: User, jwt_claims: Mapping[str, Any]
) -> dict[str, Any]:
    """Payload standard GET/PUT /institutions/me."""
    institution_role, display_email = _resolve_me_identity_for_response(user, jwt_claims)
    return {
        "id": institution.id,
        "public_id": institution.public_id,
        "name": institution.name,
        "institution_type": institution.institution_type,
        "address": institution.address,
        "contact_email": institution.contact_email,
        "contact_phone": institution.contact_phone,
        "notes": institution.notes,
        "logo_url": getattr(institution, "logo_url", None),
        "institution_role": institution_role,
        "user": {
            "id": user.id,
            "public_id": user.public_id,
            "username": user.username,
            "email": display_email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "phone": user.phone,
        },
    }


@institutions_ns.route("/me")
class InstitutionMe(Resource):
    """Endpoint pour récupérer les informations de l'institution courante."""

    @institutions_ns.doc(
        description="Retourne les informations de l'institution de l'utilisateur connecté.",
        security="Bearer",
    )
    @institutions_ns.response(200, "Succès", institution_me_model)
    @institutions_ns.response(401, "Non authentifié", permission_error_model)
    @institutions_ns.response(403, "Accès refusé", permission_error_model)
    @institutions_ns.response(404, "Institution non trouvée", not_found_error_model)
    @institutions_ns.response(500, "Erreur serveur", api_error_model)
    @jwt_required()
    @role_required(UserRole.INSTITUTION)
    def get(self):
        """Retourne les informations de l'institution de l'utilisateur connecté.

        Endpoint probe pour valider le tenant institution.
        """
        try:
            # Récupérer l'institution et l'utilisateur via le service d'autorisation
            institution, user = AuthorizationService.require_institution()

            # Récupérer le rôle institution depuis le JWT
            jwt_claims = get_jwt()

            # Audit log
            try:
                AuditLogger.log_institution_action(
                    action_type="institution_me_viewed",
                    institution_id=institution.id,
                    user_id=user.id,
                    result_status="success",
                    action_details={
                        "institution_role": jwt_claims.get("institution_role"),
                    },
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
            except Exception as audit_error:
                # Ne pas bloquer la requête si l'audit log échoue
                logger.warning(
                    "[Institution] Échec audit log institution_me_viewed: %s",
                    audit_error,
                )

            # Construire la réponse
            response_data = _build_institution_me_response(
                institution, user, jwt_claims
            )

            logger.info(
                "[Institution] GET /me - institution_id=%s, user_id=%s",
                institution.id,
                user.id,
            )

            return response_data, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error(
                "[Institution] Erreur GET /me: %s - %s",
                type(e).__name__,
                str(e),
            )
            return APIErrorHandler.handle_exception(e, logger)

    @institutions_ns.doc(
        description="Met à jour le profil de l'institution (admin only).",
        security="Bearer",
    )
    @institutions_ns.response(200, "Succès", institution_me_model)
    @institutions_ns.response(400, "Données invalides", validation_error_model)
    @institutions_ns.response(401, "Non authentifié", permission_error_model)
    @institutions_ns.response(403, "Accès refusé", permission_error_model)
    @jwt_required()
    @role_required(UserRole.INSTITUTION)
    def put(self):
        """Met à jour le profil de l'institution.

        Requiert le rôle institution_admin.
        Seuls les champs fournis sont modifiés (update partiel).
        """
        try:
            # Vérifier rôle admin institution
            institution, user = AuthorizationService.require_institution_role(
                InstitutionRole.ADMIN.value
            )

            data = request.get_json() or {}

            # Valider avec Marshmallow
            schema = InstitutionUpdateSchema()
            errors = schema.validate(data)
            if errors:
                return {"error": "Données invalides", "details": errors}, 400

            validated = schema.load(data)

            if not validated:
                return {"error": "Aucun champ à mettre à jour"}, 400

            # Appliquer les modifications (update partiel)
            old_values = {}
            new_values = {}
            updatable_fields = [
                "name",
                "institution_type",
                "address",
                "contact_email",
                "contact_phone",
                "notes",
            ]

            for field in updatable_fields:
                if field in validated:
                    old_val = getattr(institution, field, None)
                    new_val = validated.get(field)
                    # Trim les strings
                    if isinstance(new_val, str):
                        new_val = new_val.strip()
                    if old_val != new_val:
                        old_values[field] = old_val
                        new_values[field] = new_val
                        setattr(institution, field, new_val)

            if not new_values:
                jwt_claims = get_jwt()
                return _build_institution_me_response(institution, user, jwt_claims), 200

            db.session.commit()

            # ── Sync BillingParty liés (P0: adresse facture toujours à jour) ──
            try:
                from services.billing.institution_billing_resolver import (
                    sync_institution_to_billing_parties,
                )

                bp_synced = sync_institution_to_billing_parties(institution)
                if bp_synced > 0:
                    db.session.commit()
                    logger.info(
                        "[Institution] Synced %d BillingParty(s) after profile update",
                        bp_synced,
                    )
            except Exception as sync_err:
                logger.warning(
                    "[Institution] BillingParty sync error (non-blocking): %s",
                    sync_err,
                )

            # Audit log
            try:
                AuditLogger.log_action(
                    action_type="institution_profile_updated",
                    action_category="institution",
                    user_id=user.id,
                    user_type="institution",
                    institution_id=institution.id,
                    result_status="success",
                    action_details={
                        "old_values": old_values,
                        "new_values": new_values,
                    },
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
            except Exception as audit_error:
                logger.warning(
                    "[Institution] Échec audit log institution_profile_updated: %s",
                    audit_error,
                )

            logger.info(
                "[Institution] PUT /me - institution_id=%s, fields=%s",
                institution.id,
                list(new_values.keys()),
            )

            jwt_claims = get_jwt()
            return _build_institution_me_response(institution, user, jwt_claims), 200

        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            logger.error(
                "[Institution] Erreur PUT /me: %s - %s",
                type(e).__name__,
                str(e),
            )
            return APIErrorHandler.handle_exception(e, logger)


@institutions_ns.route("/me/logo")
class InstitutionLogo(Resource):
    """Upload / suppression du logo institution (admin only)."""

    @institutions_ns.doc(
        description="Retourne l'URL du logo actuel (ou null).",
        security="Bearer",
    )
    @jwt_required()
    @role_required(UserRole.INSTITUTION)
    def get(self):
        try:
            institution, _user = AuthorizationService.require_institution_role(
                InstitutionRole.ADMIN.value
            )
            return {"logo_url": getattr(institution, "logo_url", None)}, 200
        except Exception as e:
            return APIErrorHandler.handle_exception(e, logger)

    @institutions_ns.doc(
        description="Upload d'un logo institution (PNG/JPG/JPEG/SVG, max 2 Mo).",
        security="Bearer",
    )
    @jwt_required()
    @role_required(UserRole.INSTITUTION)
    def post(self):
        try:
            institution, user = AuthorizationService.require_institution_role(
                InstitutionRole.ADMIN.value
            )
            institution_id = int(getattr(institution, "id", 0) or 0)
            if institution_id <= 0:
                return APIErrorHandler.handle_exception(
                    Exception("Institution introuvable (ID invalide)."),
                    logger,
                )

            file = request.files.get("file")
            if not file or not file.filename:
                return APIErrorHandler.handle_validation_error(
                    "Fichier vide.",
                    field="file",
                    logger_instance=logger,
                )

            is_valid, error_msg = validate_file_upload(
                file=file,
                filename=file.filename,
                allowed_extensions=ALLOWED_LOGO_EXT,
                max_size_bytes=MAX_LOGO_BYTES,
                validate_content=True,
            )
            if not is_valid:
                return APIErrorHandler.handle_validation_error(
                    error_msg or "Fichier invalide.",
                    field="file",
                    logger_instance=logger,
                )

            content = file.read()
            ext = (file.filename or "").rsplit(".", 1)[1].lower()

            upload_root = Path(
                current_app.config.get(
                    "UPLOADS_DIR", str(Path(current_app.root_path) / "uploads")
                )
            ).resolve()
            public_base = current_app.config.get("UPLOADS_PUBLIC_BASE", "/uploads")

            from application.institutions.logo.upload_institution_logo import (
                UploadInstitutionLogoUseCase,
            )
            from infrastructure.files.institution_logo_storage import (
                FileSystemInstitutionLogoStorage,
            )

            storage = FileSystemInstitutionLogoStorage(base_uploads_dir=upload_root)
            uc = UploadInstitutionLogoUseCase(storage=storage, public_base=public_base)
            result = uc.execute(
                institution=institution,
                institution_id=institution_id,
                extension=ext,
                content=content,
            )
            if not result.ok:
                return result.error or {"error": "Bad request"}, result.status_code or 400

            db.session.commit()

            try:
                AuditLogger.log_action(
                    action_type="institution_logo_uploaded",
                    action_category="institution",
                    user_id=user.id,
                    user_type="institution",
                    institution_id=institution_id,
                    result_status="success",
                )
            except Exception as audit_err:
                logger.warning("Échec audit log institution_logo_uploaded: %s", audit_err)

            return {
                "logo_url": getattr(institution, "logo_url", None),
                "size_bytes": result.size_bytes,
            }, 200
        except Exception as e:
            db.session.rollback()
            return APIErrorHandler.handle_exception(e, logger)

    @institutions_ns.doc(
        description="Supprime le logo institution (fichier + champ DB).",
        security="Bearer",
    )
    @jwt_required()
    @role_required(UserRole.INSTITUTION)
    def delete(self):
        try:
            institution, user = AuthorizationService.require_institution_role(
                InstitutionRole.ADMIN.value
            )
            institution_id = int(getattr(institution, "id", 0) or 0)

            upload_root = Path(
                current_app.config.get(
                    "UPLOADS_DIR", str(Path(current_app.root_path) / "uploads")
                )
            ).resolve()

            from application.institutions.logo.delete_institution_logo import (
                DeleteInstitutionLogoUseCase,
            )
            from infrastructure.files.institution_logo_storage import (
                FileSystemInstitutionLogoStorage,
            )

            storage = FileSystemInstitutionLogoStorage(base_uploads_dir=upload_root)
            uc = DeleteInstitutionLogoUseCase(storage=storage)
            _ = uc.execute(institution=institution, institution_id=institution_id)
            db.session.commit()

            try:
                AuditLogger.log_action(
                    action_type="institution_logo_deleted",
                    action_category="institution",
                    user_id=user.id,
                    user_type="institution",
                    institution_id=institution_id,
                    result_status="success",
                )
            except Exception as audit_err:
                logger.warning("Échec audit log institution_logo_deleted: %s", audit_err)

            return {"message": "Logo supprimé."}, 200
        except Exception as e:
            db.session.rollback()
            return APIErrorHandler.handle_exception(e, logger)


# =============================================================================
# API Keys Management (admin institution only)
# =============================================================================

# Modèles Swagger pour API Keys
api_key_create_model = institutions_ns.model(
    "ApiKeyCreate",
    {
        "name": fields.String(
            required=True,
            description="Nom descriptif de la clé (ex: 'DPI Prod')",
            min_length=1,
            max_length=100,
        ),
        "scopes": fields.List(
            fields.String(),
            required=True,
            description="Liste des scopes autorisés",
            example=["requests:read", "requests:write"],
        ),
    },
)

api_key_response_model = institutions_ns.model(
    "ApiKeyResponse",
    {
        "id": fields.Integer(description="ID de la clé"),
        "name": fields.String(description="Nom de la clé"),
        "key_prefix": fields.String(description="Préfixe de la clé"),
        "scopes": fields.List(fields.String(), description="Scopes autorisés"),
        "last_used_at": fields.String(description="Dernière utilisation (ISO8601)"),
        "created_at": fields.String(description="Date de création (ISO8601)"),
        "revoked_at": fields.String(description="Date de révocation (ISO8601)"),
        "is_active": fields.Boolean(description="True si la clé est active"),
    },
)

api_key_created_response_model = institutions_ns.model(
    "ApiKeyCreatedResponse",
    {
        "id": fields.Integer(description="ID de la clé"),
        "name": fields.String(description="Nom de la clé"),
        "key_prefix": fields.String(description="Préfixe de la clé"),
        "scopes": fields.List(fields.String(), description="Scopes autorisés"),
        "created_at": fields.String(description="Date de création (ISO8601)"),
        "key": fields.String(
            description="⚠️ Clé brute (affichée UNE SEULE FOIS, à sauvegarder)",
        ),
    },
)

api_key_list_response_model = institutions_ns.model(
    "ApiKeyListResponse",
    {
        "api_keys": fields.List(
            fields.Nested(api_key_response_model),
            description="Liste des clés API",
        ),
        "valid_scopes": fields.List(
            fields.String(),
            description="Liste des scopes valides",
        ),
    },
)


# ============================================================================
# Settings (billing + notifications + timeouts)
# ============================================================================


@institutions_ns.route("/settings")
class InstitutionSettingsResource(Resource):
    """GET/PUT des paramètres institution (admin only)."""

    @institutions_ns.doc(
        description="Retourne les paramètres de l'institution (billing, notifications, timeouts).",
        security="Bearer",
    )
    @jwt_required()
    @role_required(UserRole.INSTITUTION)
    def get(self):
        """GET /institutions/settings — tous rôles institution."""
        try:
            institution, _user = AuthorizationService.require_institution()
            settings = get_or_create_settings(institution.id)
            db.session.commit()  # persister le lazy-create éventuel

            return {
                "institution": {
                    "billing_email": institution.billing_email,
                    "billing_address": institution.billing_address,
                    "vat_number": institution.vat_number,
                },
                "settings": settings.serialize,
            }, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error("[Institution] Erreur GET /settings: %s", e)
            return APIErrorHandler.handle_exception(e, logger)

    @institutions_ns.doc(
        description="Met à jour les paramètres de l'institution (admin ou billing).",
        security="Bearer",
    )
    @jwt_required()
    @role_required(UserRole.INSTITUTION)
    def put(self):
        """PUT /institutions/settings — admin ou billing."""
        try:
            institution, user = AuthorizationService.require_institution_role(
                InstitutionRole.ADMIN.value,
                InstitutionRole.BILLING.value,
            )

            data = request.get_json() or {}
            schema = InstitutionSettingsUpdateSchema()
            errors = schema.validate(data)
            if errors:
                return {"error": "Données invalides", "details": errors}, 400

            validated = schema.load(data)
            if not validated:
                return {"error": "Aucun champ à mettre à jour"}, 400

            settings = get_or_create_settings(institution.id)

            # Séparer les champs Institution vs InstitutionSettings
            institution_fields = ["billing_email", "billing_address", "vat_number"]
            settings_fields = [
                "timeout_same_day_minutes",
                "timeout_default_minutes",
                "offer_dispatch_mode",
                "default_billing_intent",
                "default_vat_rate",
                "default_payment_terms_days",
                "notification_emails",
                "notify_request_sent",
                "notify_offer_accepted",
                "notify_request_expired",
                "timezone",
                "default_pickup_mode",
                "entry_points",
                "default_contact_phone",
            ]

            old_values = {}
            new_values = {}

            # Appliquer les champs Institution
            for field in institution_fields:
                if field in validated:
                    old_val = getattr(institution, field, None)
                    new_val = validated.get(field)
                    if isinstance(new_val, str):
                        new_val = new_val.strip()
                    if old_val != new_val:
                        old_values[field] = old_val
                        new_values[field] = new_val
                        setattr(institution, field, new_val)

            # Appliquer les champs InstitutionSettings
            for field in settings_fields:
                if field in validated:
                    old_val = getattr(settings, field, None)
                    new_val = validated.get(field)
                    if isinstance(new_val, str):
                        new_val = new_val.strip()
                    # Normaliser notification_emails: trim + lowercase + dedup + tri
                    if field == "notification_emails" and isinstance(new_val, list):
                        seen: set[str] = set()
                        normalized: list[str] = []
                        for email in new_val:
                            e = email.strip().lower()
                            if e and e not in seen:
                                normalized.append(e)
                                seen.add(e)
                        new_val = sorted(normalized)
                    # Normaliser entry_points: trim + dedup (ordre conservé)
                    if field == "entry_points" and isinstance(new_val, list):
                        seen_ep: set[str] = set()
                        norm_ep: list[str] = []
                        for ep in new_val:
                            trimmed = ep.strip()
                            if trimmed and trimmed not in seen_ep:
                                norm_ep.append(trimmed)
                                seen_ep.add(trimmed)
                        new_val = norm_ep
                    # Comparer proprement (list vs list)
                    if old_val != new_val:
                        old_values[field] = (
                            old_val if not isinstance(old_val, list) else list(old_val)
                        )
                        new_values[field] = new_val
                        setattr(settings, field, new_val)

            if not new_values:
                return {
                    "institution": {
                        "billing_email": institution.billing_email,
                        "billing_address": institution.billing_address,
                        "vat_number": institution.vat_number,
                    },
                    "settings": settings.serialize,
                }, 200

            db.session.commit()

            # ── Sync BillingParty si billing_address/email changé ──
            billing_fields_changed = any(
                f in new_values for f in ("billing_email", "billing_address")
            )
            if billing_fields_changed:
                try:
                    from services.billing.institution_billing_resolver import (
                        sync_institution_to_billing_parties,
                    )

                    bp_synced = sync_institution_to_billing_parties(institution)
                    if bp_synced > 0:
                        db.session.commit()
                        logger.info(
                            "[Institution] Synced %d BillingParty(s) after settings update",
                            bp_synced,
                        )
                except Exception as sync_err:
                    logger.warning(
                        "[Institution] BillingParty sync error (non-blocking): %s",
                        sync_err,
                    )

            # Audit log
            try:
                AuditLogger.log_action(
                    action_type="institution_settings_updated",
                    action_category="institution",
                    user_id=user.id,
                    user_type="institution",
                    institution_id=institution.id,
                    result_status="success",
                    action_details={
                        "old_values": {k: str(v) for k, v in old_values.items()},
                        "new_values": {k: str(v) for k, v in new_values.items()},
                    },
                )
            except Exception as audit_err:
                logger.warning("[Institution] Audit log settings error: %s", audit_err)

            logger.info(
                "[Institution] PUT /settings - institution_id=%s, changed=%s",
                institution.id,
                list(new_values.keys()),
            )

            return {
                "institution": {
                    "billing_email": institution.billing_email,
                    "billing_address": institution.billing_address,
                    "vat_number": institution.vat_number,
                },
                "settings": settings.serialize,
            }, 200

        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            logger.error("[Institution] Erreur PUT /settings: %s", e)
            return APIErrorHandler.handle_exception(e, logger)


# ============================================================================
# API Keys
# ============================================================================


@institutions_ns.route("/api-keys")
class InstitutionApiKeyList(Resource):
    """Endpoints pour lister et créer des clés API."""

    @institutions_ns.doc(
        description="Liste les clés API de l'institution (admin only).",
        security="Bearer",
    )
    @institutions_ns.response(200, "Succès", api_key_list_response_model)
    @institutions_ns.response(401, "Non authentifié", permission_error_model)
    @institutions_ns.response(403, "Accès refusé", permission_error_model)
    @jwt_required()
    @role_required(UserRole.INSTITUTION)
    def get(self):
        """Liste les clés API de l'institution.

        Requiert le rôle institution_admin.
        Ne retourne jamais la clé brute.
        """
        try:
            # Vérifier rôle admin institution
            institution, _user = AuthorizationService.require_institution_role(
                InstitutionRole.ADMIN.value
            )

            # Récupérer les clés API
            api_keys = (
                InstitutionApiKey.query.filter_by(institution_id=institution.id)
                .order_by(InstitutionApiKey.created_at.desc())
                .all()
            )

            return {
                "api_keys": [key.serialize for key in api_keys],
                "valid_scopes": list(VALID_SCOPES),
            }, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error(
                "[Institution] Erreur GET /api-keys: %s - %s",
                type(e).__name__,
                str(e),
            )
            return APIErrorHandler.handle_exception(e, logger)

    @institutions_ns.doc(
        description="Crée une nouvelle clé API (admin only).",
        security="Bearer",
    )
    @institutions_ns.expect(api_key_create_model, validate=True)
    @institutions_ns.response(201, "Clé créée", api_key_created_response_model)
    @institutions_ns.response(400, "Erreur de validation", validation_error_model)
    @institutions_ns.response(401, "Non authentifié", permission_error_model)
    @institutions_ns.response(403, "Accès refusé", permission_error_model)
    @institutions_ns.response(422, "Scopes invalides", validation_error_model)
    @jwt_required()
    @role_required(UserRole.INSTITUTION)
    def post(self):
        """Crée une nouvelle clé API pour l'institution.

        Requiert le rôle institution_admin.
        ⚠️ IMPORTANT: La clé brute n'est retournée qu'UNE SEULE FOIS.
        """
        try:
            # Vérifier rôle admin institution
            institution, user = AuthorizationService.require_institution_role(
                InstitutionRole.ADMIN.value
            )

            data = request.get_json() or {}

            # Valider les données
            name = data.get("name", "").strip()
            if not name:
                return {"error": "Le nom est requis"}, 400

            scopes = data.get("scopes", [])
            if not isinstance(scopes, list):
                return {"error": "scopes doit être une liste"}, 400

            # Valider les scopes
            is_valid, invalid_scopes = validate_scopes(scopes)
            if not is_valid:
                return {
                    "error": f"Scopes invalides: {', '.join(invalid_scopes)}",
                    "valid_scopes": list(VALID_SCOPES),
                }, 422

            # Générer la clé
            raw_key, key_prefix, key_hash = generate_api_key()

            # Créer l'entrée en DB
            api_key = InstitutionApiKey()
            api_key.institution_id = institution.id
            api_key.name = name
            api_key.key_prefix = key_prefix
            api_key.key_hash = key_hash
            api_key.set_scopes(scopes)
            api_key.created_by_user_id = user.id

            db.session.add(api_key)
            db.session.commit()

            # Audit log
            try:
                AuditLogger.log_action(
                    action_type="api_key_created",
                    action_category="institution_api",
                    user_id=user.id,
                    user_type="institution",
                    institution_id=institution.id,
                    result_status="success",
                    action_details={
                        "api_key_id": api_key.id,
                        "api_key_name": name,
                        "scopes": scopes,
                        "key_prefix": key_prefix,
                    },
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
            except Exception as audit_error:
                logger.warning(
                    "[Institution] Échec audit log api_key_created: %s",
                    audit_error,
                )

            logger.info(
                "[Institution] Clé API créée: id=%s, name=%s, institution_id=%s",
                api_key.id,
                name,
                institution.id,
            )

            # Retourner avec la clé brute (UNE SEULE FOIS)
            return {
                "id": api_key.id,
                "name": api_key.name,
                "key_prefix": api_key.key_prefix,
                "scopes": api_key.get_scopes(),
                "created_at": api_key.created_at.isoformat()
                if api_key.created_at
                else None,
                "key": raw_key,  # ⚠️ Clé brute, affichée UNE SEULE FOIS
            }, 201

        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            logger.error(
                "[Institution] Erreur POST /api-keys: %s - %s",
                type(e).__name__,
                str(e),
            )
            return APIErrorHandler.handle_exception(e, logger)


@institutions_ns.route("/api-keys/<int:key_id>/revoke")
class InstitutionApiKeyRevoke(Resource):
    """Endpoint pour révoquer une clé API."""

    @institutions_ns.doc(
        description="Révoque une clé API (admin only).",
        security="Bearer",
    )
    @institutions_ns.response(200, "Clé révoquée", api_key_response_model)
    @institutions_ns.response(401, "Non authentifié", permission_error_model)
    @institutions_ns.response(403, "Accès refusé", permission_error_model)
    @institutions_ns.response(404, "Clé non trouvée", not_found_error_model)
    @jwt_required()
    @role_required(UserRole.INSTITUTION)
    def post(self, key_id: int):
        """Révoque une clé API.

        Requiert le rôle institution_admin.
        Une clé révoquée ne peut plus être utilisée.
        """
        try:
            # Vérifier rôle admin institution
            institution, user = AuthorizationService.require_institution_role(
                InstitutionRole.ADMIN.value
            )

            # Récupérer la clé
            api_key = InstitutionApiKey.query.filter_by(
                id=key_id,
                institution_id=institution.id,
            ).first()

            if not api_key:
                return {"error": "Clé API non trouvée"}, 404

            if api_key.is_revoked:
                return {"error": "Clé déjà révoquée"}, 400

            # Révoquer
            api_key.revoke()
            db.session.commit()

            # Audit log
            try:
                AuditLogger.log_action(
                    action_type="api_key_revoked",
                    action_category="institution_api",
                    user_id=user.id,
                    user_type="institution",
                    institution_id=institution.id,
                    result_status="success",
                    action_details={
                        "api_key_id": api_key.id,
                        "api_key_name": api_key.name,
                        "key_prefix": api_key.key_prefix,
                    },
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
            except Exception as audit_error:
                logger.warning(
                    "[Institution] Échec audit log api_key_revoked: %s",
                    audit_error,
                )

            logger.info(
                "[Institution] Clé API révoquée: id=%s, institution_id=%s",
                api_key.id,
                institution.id,
            )

            return api_key.serialize, 200

        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            logger.error(
                "[Institution] Erreur POST /api-keys/%s/revoke: %s - %s",
                key_id,
                type(e).__name__,
                str(e),
            )
            return APIErrorHandler.handle_exception(e, logger)


# =============================================================================
# DPI Endpoints (API Key authentication)
# =============================================================================

dpi_probe_response_model = institutions_ns.model(
    "DpiProbeResponse",
    {
        "status": fields.String(description="Statut de la connexion"),
        "institution_id": fields.Integer(description="ID de l'institution"),
        "api_key_id": fields.Integer(description="ID de la clé API utilisée"),
        "scopes": fields.List(fields.String(), description="Scopes de la clé"),
        "timestamp": fields.String(description="Timestamp ISO8601"),
    },
)


@institutions_ns.route("/dpi/probe")
class DpiProbe(Resource):
    """Endpoint probe pour tester l'authentification API Key."""

    @institutions_ns.doc(
        description="Endpoint probe pour tester l'authentification API Key (DPI).",
        params={
            "X-API-Key": {
                "in": "header",
                "description": "Clé API (format: lir_...)",
                "required": True,
            },
        },
    )
    @institutions_ns.response(200, "Authentification réussie", dpi_probe_response_model)
    @institutions_ns.response(401, "Clé API invalide", permission_error_model)
    @institutions_ns.response(403, "Scope manquant", permission_error_model)
    @institutions_ns.response(429, "Rate limit dépassé", api_error_model)
    @api_key_required(scopes=["requests:read"])
    def get(self):
        """Endpoint probe pour tester l'authentification API Key.

        Requiert le scope 'requests:read'.
        Retourne les informations de la clé API utilisée.
        """
        from datetime import UTC, datetime

        api_key = g.institution_api_key

        return {
            "status": "ok",
            "institution_id": g.institution_id,
            "api_key_id": api_key.id,
            "scopes": list(g.scopes),
            "timestamp": datetime.now(UTC).isoformat(),
        }, 200


# ============================================================================
# USERS MANAGEMENT (institution_admin only)
# ============================================================================


def _serialize_institution_user(user: User, institution=None) -> dict[str, object]:
    """Sérialise un utilisateur institution pour la réponse API."""
    status = user.account_status or "active"
    if (
        status == "invited"
        and user.invite_expires_at
        and user.invite_expires_at < datetime.now(UTC)
    ):
        status = "expired"

    return {
        "id": user.id,
        "public_id": user.public_id,
        "email": user.email,
        "username": user.username,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "institution_role": user.institution_role,
        "job_title": getattr(user, "job_title", None),
        "account_status": status,
        "authentication_method": getattr(user, "authentication_method", None) or "email",
        "force_password_change": bool(getattr(user, "force_password_change", False)),
        "first_login_completed_at": user.first_login_completed_at.isoformat()
        if getattr(user, "first_login_completed_at", None)
        else None,
        "invite_sent_at": user.invite_sent_at.isoformat()
        if user.invite_sent_at
        else None,
        "invite_expires_at": user.invite_expires_at.isoformat()
        if user.invite_expires_at
        else None,
        "password_expires_at": user.password_expires_at.isoformat()
        if getattr(user, "password_expires_at", None)
        else None,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


def _record_institution_user_audit(
    *,
    institution_id: int,
    target_user_id: int,
    performed_by_user_id: int | None,
    event_type: str,
    metadata: dict | None = None,
) -> None:
    try:
        from models.institution_user_audit_event import InstitutionUserAuditEvent

        InstitutionUserAuditEvent.record(
            institution_id=institution_id,
            target_user_id=target_user_id,
            performed_by_user_id=performed_by_user_id,
            event_type=event_type,
            ip_address=request.remote_addr,
            user_agent=request.headers.get("User-Agent"),
            metadata=metadata,
        )
        db.session.commit()
    except Exception as audit_err:
        db.session.rollback()
        logger.warning("[Institution] Audit event failed: %s", audit_err)


def _invite_response_payload(result, institution) -> dict[str, object]:
    from application.institutions.invitation_service import build_invite_url

    payload: dict[str, object] = {
        "message": result.message,
        "user": _serialize_institution_user(result.user, institution)
        if result.user
        else None,
        "email_sent": result.email_result.success if result.email_result else False,
        "email_type": result.email_type,
        "email_error": result.email_result.error
        if result.email_result and not result.email_result.success
        else None,
        "creation_mode": result.creation_mode,
    }
    if result.raw_token:
        payload["invite_link"] = build_invite_url(result.raw_token)
    if result.temporary_credentials:
        payload["temporary_credentials"] = result.temporary_credentials
        payload["credentials_shown_once"] = result.credentials_shown_once
    if result.email_result is None and result.creation_mode == "username":
        payload["email_sent"] = False
    return payload


def _count_admins(institution_id: int) -> int:
    """Compte le nombre d'admins actifs dans l'institution (exclut disabled/archived)."""
    return User.query.filter(
        User.institution_id == institution_id,
        User.institution_role == InstitutionRole.ADMIN.value,
        User.archived_at.is_(None),
        db.or_(User.account_status.is_(None), User.account_status != "disabled"),
    ).count()


def _pending_activation_users(institution_id: int) -> list[User]:
    """Utilisateurs en attente d'activation pour une institution."""
    now = datetime.now(UTC)
    return (
        User.query.filter_by(institution_id=institution_id)
        .filter(User.archived_at.is_(None))
        .filter(
            db.or_(
                db.and_(
                    User.authentication_method == "username",
                    User.first_login_completed_at.is_(None),
                ),
                db.and_(
                    User.force_password_change.is_(True),
                    User.password_expires_at.isnot(None),
                    User.password_expires_at < now,
                ),
                User.account_status == "invited",
            )
        )
        .order_by(User.created_at.asc())
        .all()
    )


@institutions_ns.route("/users")
class InstitutionUsers(Resource):
    """Gestion des utilisateurs de l'institution."""

    @institutions_ns.doc(
        description="Liste les utilisateurs de l'institution.",
        security="Bearer",
    )
    @jwt_required()
    @role_required(UserRole.INSTITUTION)
    def get(self):
        """Liste les utilisateurs de l'institution (admin only)."""
        try:
            institution, _user = AuthorizationService.require_institution_role(
                InstitutionRole.ADMIN.value
            )

            users = (
                User.query.filter_by(institution_id=institution.id)
                .filter(User.archived_at.is_(None))
                .order_by(User.created_at.asc())
                .all()
            )

            return {
                "users": [_serialize_institution_user(u, institution) for u in users],
                "total": len(users),
            }, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error("[Institution] Erreur GET /users: %s", e)
            return APIErrorHandler.handle_exception(e, logger)

    @institutions_ns.doc(
        description="Invite un utilisateur dans l'institution par email.",
        security="Bearer",
    )
    @limiter.limit("10 per hour")
    @jwt_required()
    @role_required(UserRole.INSTITUTION)
    def post(self):
        """Invite ou ajoute un utilisateur à l'institution (admin only)."""
        from application.institutions.invitation_service import invite_or_attach_institution_user

        try:
            institution, admin_user = AuthorizationService.require_institution_role(
                InstitutionRole.ADMIN.value
            )

            data = request.get_json() or {}
            schema = InstitutionUserInviteSchema()
            errors = schema.validate(data)
            if errors:
                return {"error": "Données invalides", "details": errors}, 400

            cross_errors = InstitutionUserInviteSchema.validate_payload(data)
            if cross_errors:
                return {"error": "Données invalides", "details": cross_errors}, 400

            validated = schema.load(data) or {}
            creation_mode = str(validated.get("creation_mode") or "email").strip().lower()
            email_raw = validated.get("email")
            email = str(email_raw).strip().lower() if email_raw else None
            role_value = str(validated.get("institution_role", ""))
            first_name = validated.get("first_name")
            last_name = validated.get("last_name")
            local_username = validated.get("username")
            if local_username:
                local_username = str(local_username).strip().lower()
            job_title = validated.get("job_title")

            result = invite_or_attach_institution_user(
                institution=institution,
                admin_user=admin_user,
                email=email,
                role_value=role_value,
                first_name=str(first_name) if first_name else None,
                last_name=str(last_name) if last_name else None,
                creation_mode=creation_mode,
                local_username=local_username,
                job_title=str(job_title) if job_title else None,
            )

            if result.error and result.http_status >= 400:
                return {"error": result.error}, result.http_status

            if result.user and result.audit_details:
                event_type = (
                    "user_created"
                    if result.path in ("new_user", "new_username")
                    else "institution_user_added"
                )
                try:
                    AuditLogger.log_institution_action(
                        action_type=f"institution_{result.path}",
                        institution_id=institution.id,
                        user_id=admin_user.id,
                        result_status="success",
                        action_details=result.audit_details,
                        ip_address=request.remote_addr,
                        user_agent=request.headers.get("User-Agent"),
                    )
                except Exception as audit_err:
                    logger.warning("[Institution] Audit log failed: %s", audit_err)

                if result.path in ("new_user", "new_username", "existing_user"):
                    audit_event = (
                        "user_created"
                        if result.path in ("new_user", "new_username")
                        else "institution_user_added"
                    )
                    _record_institution_user_audit(
                        institution_id=institution.id,
                        target_user_id=result.user.id,
                        performed_by_user_id=admin_user.id,
                        event_type=audit_event,
                        metadata=result.audit_details,
                    )

            return _invite_response_payload(result, institution), result.http_status

        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            logger.error("[Institution] Erreur POST /users: %s", e)
            return APIErrorHandler.handle_exception(e, logger)


@institutions_ns.route("/users/<int:user_id>/role")
class InstitutionUserRole(Resource):
    """Modifier le rôle d'un utilisateur de l'institution."""

    @institutions_ns.doc(
        description="Modifie le rôle d'un utilisateur dans l'institution.",
        security="Bearer",
    )
    @jwt_required()
    @role_required(UserRole.INSTITUTION)
    def put(self, user_id):
        """Met à jour le rôle institution d'un utilisateur (admin only)."""
        try:
            institution, admin_user = AuthorizationService.require_institution_role(
                InstitutionRole.ADMIN.value
            )

            data = request.get_json() or {}
            schema = InstitutionUserUpdateRoleSchema()
            errors = schema.validate(data)
            if errors:
                return {"error": "Données invalides", "details": errors}, 400

            validated = schema.load(data) or {}
            new_role = str(validated.get("institution_role", ""))

            # Trouver l'utilisateur cible dans cette institution
            target_user = User.query.filter_by(
                id=user_id,
                institution_id=institution.id,
            ).first()

            if not target_user:
                return {"error": "Utilisateur non trouvé dans cette institution"}, 404

            old_role = target_user.institution_role

            # Règle "dernier admin" : on ne peut pas rétrograder le dernier admin
            if (
                old_role == InstitutionRole.ADMIN.value
                and new_role != InstitutionRole.ADMIN.value
            ):
                admin_count = _count_admins(institution.id)
                if admin_count <= 1:
                    return {
                        "error": "Impossible de modifier le rôle du dernier administrateur"
                    }, 400

            target_user.institution_role = new_role
            db.session.commit()

            # Audit log
            try:
                AuditLogger.log_institution_action(
                    action_type="institution_user_role_changed",
                    institution_id=institution.id,
                    user_id=admin_user.id,
                    result_status="success",
                    action_details={
                        "target_user_id": target_user.id,
                        "target_email": target_user.email,
                        "old_role": old_role,
                        "new_role": new_role,
                    },
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
            except Exception as audit_err:
                logger.warning("[Institution] Audit log failed: %s", audit_err)

            return {
                "message": "Rôle mis à jour",
                "user": _serialize_institution_user(target_user),
            }, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error("[Institution] Erreur PUT /users/%s/role: %s", user_id, e)
            return APIErrorHandler.handle_exception(e, logger)


@institutions_ns.route("/users/<int:user_id>")
class InstitutionUserRemove(Resource):
    """Retirer un utilisateur de l'institution."""

    @institutions_ns.doc(
        description="Met à jour les champs descriptifs (fonction/métier) d'un utilisateur.",
        security="Bearer",
    )
    @jwt_required()
    @role_required(UserRole.INSTITUTION)
    def patch(self, user_id):
        """Met à jour les champs descriptifs d'un utilisateur (admin only).

        Champs : first_name, last_name, email (contact), job_title.
        Ne modifie jamais username, authentication_method, institution_role.
        Autorisé même pour les comptes désactivés/archivés.
        """
        from application.institutions.invitation_service import _normalize_job_title
        from application.users.normalization import (
            find_user_by_normalized_email,
            normalize_contact_email,
        )

        def _normalize_name(value: str | None) -> str | None:
            if value is None:
                return None
            stripped = str(value).strip()
            return stripped or None

        try:
            institution, admin_user = AuthorizationService.require_institution_role(
                InstitutionRole.ADMIN.value
            )

            raw_data = request.get_json() or {}
            forbidden_errors = InstitutionUserUpdateProfileSchema.validate_forbidden_fields(
                raw_data
            )
            if forbidden_errors:
                return {"error": "Données invalides", "details": forbidden_errors}, 400

            schema = InstitutionUserUpdateProfileSchema()
            errors = schema.validate(raw_data)
            if errors:
                return {"error": "Données invalides", "details": errors}, 400

            validated = schema.load(raw_data) or {}

            target_user = User.query.filter_by(
                id=user_id,
                institution_id=institution.id,
            ).first()

            if not target_user:
                return {"error": "Utilisateur non trouvé dans cette institution"}, 404

            profile_changes: dict[str, dict[str, object | None]] = {}
            job_title_changed = False
            old_job_title = getattr(target_user, "job_title", None)
            new_job_title = old_job_title

            if "first_name" in raw_data:
                new_first = _normalize_name(validated.get("first_name"))
                old_first = _normalize_name(target_user.first_name)
                if new_first != old_first:
                    profile_changes["first_name"] = {"old": old_first, "new": new_first}
                    target_user.first_name = new_first

            if "last_name" in raw_data:
                new_last = _normalize_name(validated.get("last_name"))
                old_last = _normalize_name(target_user.last_name)
                if new_last != old_last:
                    profile_changes["last_name"] = {"old": old_last, "new": new_last}
                    target_user.last_name = new_last

            if "email" in raw_data:
                new_email = normalize_contact_email(validated.get("email"))
                old_email = normalize_contact_email(target_user.email)
                if new_email != old_email:
                    if new_email:
                        conflict = find_user_by_normalized_email(
                            new_email, exclude_user_id=target_user.id
                        )
                        if conflict:
                            return {
                                "error": "Cet email est déjà utilisé sur la plateforme"
                            }, 409
                    profile_changes["email"] = {"old": old_email, "new": new_email}
                    target_user.email = new_email

            if "job_title" in raw_data:
                new_job_title = _normalize_job_title(validated.get("job_title"))
                if old_job_title != new_job_title:
                    job_title_changed = True
                    target_user.job_title = new_job_title

            if not profile_changes and not job_title_changed:
                if "job_title" in raw_data and "first_name" not in raw_data and "last_name" not in raw_data and "email" not in raw_data:
                    message = "Fonction inchangée"
                else:
                    message = "Profil inchangé"
                return {
                    "message": message,
                    "user": _serialize_institution_user(target_user, institution),
                }, 200

            db.session.commit()

            if profile_changes:
                _record_institution_user_audit(
                    institution_id=institution.id,
                    target_user_id=target_user.id,
                    performed_by_user_id=admin_user.id,
                    event_type="profile_updated",
                    metadata={"changes": profile_changes},
                )
                try:
                    AuditLogger.log_institution_action(
                        action_type="institution_user_profile_updated",
                        institution_id=institution.id,
                        user_id=admin_user.id,
                        result_status="success",
                        action_details={
                            "target_user_id": target_user.id,
                            "target_email": target_user.email,
                            "changes": profile_changes,
                        },
                        ip_address=request.remote_addr,
                        user_agent=request.headers.get("User-Agent"),
                    )
                except Exception as audit_err:
                    logger.warning("[Institution] Audit log failed: %s", audit_err)

            if job_title_changed:
                _record_institution_user_audit(
                    institution_id=institution.id,
                    target_user_id=target_user.id,
                    performed_by_user_id=admin_user.id,
                    event_type="job_title_updated",
                    metadata={"old_value": old_job_title, "new_value": new_job_title},
                )
                try:
                    AuditLogger.log_institution_action(
                        action_type="institution_user_job_title_updated",
                        institution_id=institution.id,
                        user_id=admin_user.id,
                        result_status="success",
                        action_details={
                            "target_user_id": target_user.id,
                            "target_email": target_user.email,
                            "old_value": old_job_title,
                            "new_value": new_job_title,
                        },
                        ip_address=request.remote_addr,
                        user_agent=request.headers.get("User-Agent"),
                    )
                except Exception as audit_err:
                    logger.warning("[Institution] Audit log failed: %s", audit_err)

            if profile_changes and job_title_changed:
                message = "Profil mis à jour"
            elif profile_changes:
                message = "Profil mis à jour"
            else:
                message = "Fonction mise à jour"

            return {
                "message": message,
                "user": _serialize_institution_user(target_user, institution),
            }, 200

        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            logger.error("[Institution] Erreur PATCH /users/%s: %s", user_id, e)
            return APIErrorHandler.handle_exception(e, logger)

    @institutions_ns.doc(
        description="Retire un utilisateur de l'institution.",
        security="Bearer",
    )
    @jwt_required()
    @role_required(UserRole.INSTITUTION)
    def delete(self, user_id):
        """Retire un utilisateur de l'institution (admin only)."""
        try:
            institution, admin_user = AuthorizationService.require_institution_role(
                InstitutionRole.ADMIN.value
            )

            target_user = User.query.filter_by(
                id=user_id,
                institution_id=institution.id,
            ).first()

            if not target_user:
                return {"error": "Utilisateur non trouvé dans cette institution"}, 404

            # Interdire de se retirer soi-même
            if target_user.id == admin_user.id:
                return {
                    "error": "Vous ne pouvez pas vous retirer vous-même de l'institution"
                }, 400

            # Règle "dernier admin"
            if target_user.institution_role == InstitutionRole.ADMIN.value:
                admin_count = _count_admins(institution.id)
                if admin_count <= 1:
                    return {
                        "error": "Impossible de retirer le dernier administrateur"
                    }, 400

            old_role = target_user.institution_role
            old_email = target_user.email
            now = datetime.now(UTC)

            target_user.archived_at = now
            target_user.account_status = "disabled"
            target_user.disabled_at = now
            target_user.invite_token_hash = None

            db.session.commit()

            _record_institution_user_audit(
                institution_id=institution.id,
                target_user_id=target_user.id,
                performed_by_user_id=admin_user.id,
                event_type="user_archived",
                metadata={"old_role": old_role, "old_email": old_email},
            )

            # Audit log
            try:
                AuditLogger.log_institution_action(
                    action_type="institution_user_archived",
                    institution_id=institution.id,
                    user_id=admin_user.id,
                    result_status="success",
                    action_details={
                        "target_user_id": target_user.id,
                        "target_email": old_email,
                        "old_role": old_role,
                    },
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
            except Exception as audit_err:
                logger.warning("[Institution] Audit log failed: %s", audit_err)

            return {"message": "Utilisateur archivé"}, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error("[Institution] Erreur DELETE /users/%s: %s", user_id, e)
            return APIErrorHandler.handle_exception(e, logger)


@institutions_ns.route("/users/import")
class InstitutionUsersImport(Resource):
    """Import CSV collaborateurs (roadmap post-Sprint 3)."""

    @institutions_ns.doc(
        description="Import CSV collaborateurs — non implémenté (roadmap).",
        security="Bearer",
    )
    @jwt_required()
    @role_required(UserRole.INSTITUTION)
    def post(self):
        return {
            "error": "Import CSV non disponible — fonctionnalité roadmap post-Sprint 3",
            "status": "not_implemented",
        }, 501


@institutions_ns.route("/users/pending-activation")
class InstitutionUsersPendingActivation(Resource):
    """Collaborateurs en attente d'activation."""

    @institutions_ns.doc(
        description="Liste les utilisateurs en attente d'activation.",
        security="Bearer",
    )
    @jwt_required()
    @role_required(UserRole.INSTITUTION)
    def get(self):
        try:
            institution, _user = AuthorizationService.require_institution_role(
                InstitutionRole.ADMIN.value
            )
            now = datetime.now(UTC)
            users = _pending_activation_users(institution.id)
            items = []
            for u in users:
                reason = "invited"
                if getattr(u, "authentication_method", "email") == "username" and not u.first_login_completed_at:
                    reason = "never_connected"
                elif u.force_password_change and u.password_expires_at and u.password_expires_at < now:
                    reason = "password_expired"
                elif u.account_status == "invited":
                    if u.invite_expires_at and u.invite_expires_at < now:
                        reason = "invitation_expired"
                    else:
                        reason = "invitation_pending"
                serialized = _serialize_institution_user(u, institution)
                serialized["pending_reason"] = reason
                items.append(serialized)
            return {"users": items, "total": len(items)}, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            return APIErrorHandler.handle_exception(e, logger)


@institutions_ns.route("/users/<int:user_id>/reset-password")
class InstitutionUserResetPassword(Resource):
    """Réinitialiser le mot de passe temporaire (Mode B)."""

    @institutions_ns.doc(
        description="Génère un nouveau mot de passe temporaire (admin only).",
        security="Bearer",
    )
    @limiter.limit("5 per hour")
    @jwt_required()
    @role_required(UserRole.INSTITUTION)
    def post(self, user_id):
        from datetime import timedelta

        from application.institutions.invitation_service import (
            TEMP_PASSWORD_EXPIRY_DAYS,
            _generate_strong_password,
        )
        from models.institution_user_audit_event import InstitutionUserAuditEvent

        try:
            institution, admin_user = AuthorizationService.require_institution_role(
                InstitutionRole.ADMIN.value
            )

            target_user = User.query.filter_by(
                id=user_id,
                institution_id=institution.id,
            ).filter(User.archived_at.is_(None)).first()

            if not target_user:
                return {"error": "Utilisateur non trouvé dans cette institution"}, 404

            if getattr(target_user, "authentication_method", "email") != "username":
                return {
                    "error": "Seuls les comptes créés par identifiant peuvent être réinitialisés ici"
                }, 400

            recent_resets = (
                InstitutionUserAuditEvent.query.filter_by(
                    target_user_id=target_user.id,
                    event_type="password_reset",
                )
                .filter(
                    InstitutionUserAuditEvent.performed_at
                    > datetime.now(UTC) - timedelta(hours=2)
                )
                .count()
            )
            if recent_resets >= 10:
                return {"error": "Trop de réinitialisations récentes pour cet utilisateur"}, 429

            temp_password = _generate_strong_password()
            now = datetime.now(UTC)
            target_user.force_password_change = True
            target_user.password_expires_at = now + timedelta(days=TEMP_PASSWORD_EXPIRY_DAYS)
            target_user.temporary_password_created_at = now
            target_user.last_password_reset_at = now
            target_user.temp_password_generation_count = (
                int(getattr(target_user, "temp_password_generation_count", 0) or 0) + 1
            )
            target_user.set_password(temp_password, force_change=True)
            db.session.commit()

            _record_institution_user_audit(
                institution_id=institution.id,
                target_user_id=target_user.id,
                performed_by_user_id=admin_user.id,
                event_type="password_reset",
            )

            return {
                "message": "Mot de passe temporaire régénéré",
                "user": _serialize_institution_user(target_user, institution),
                "temporary_credentials": {
                    "username": target_user.username,
                    "temporary_password": temp_password,
                },
                "credentials_shown_once": True,
                "email_sent": False,
                "email_type": None,
            }, 200

        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            return APIErrorHandler.handle_exception(e, logger)


@institutions_ns.route("/users/<int:user_id>/resend-invite")
class InstitutionUserResendInvite(Resource):
    """Renvoyer une invitation par email."""

    @institutions_ns.doc(
        description="Renvoie l'invitation par email (admin only). Régénère le token.",
        security="Bearer",
    )
    @limiter.limit("5 per hour")
    @jwt_required()
    @role_required(UserRole.INSTITUTION)
    def post(self, user_id):
        """Renvoie l'invitation par email pour un utilisateur invité ou désactivé."""
        from datetime import UTC, datetime

        from application.institutions.invitation_service import (
            build_invite_url,
            generate_invite_token,
            get_invite_expiry,
            send_invitation_email,
        )

        try:
            institution, admin_user = AuthorizationService.require_institution_role(
                InstitutionRole.ADMIN.value
            )

            target_user = User.query.filter_by(
                id=user_id,
                institution_id=institution.id,
            ).first()

            if not target_user:
                return {"error": "Utilisateur non trouvé dans cette institution"}, 404

            if getattr(target_user, "authentication_method", "email") == "username":
                return {
                    "error": "Utilisez 'Réinitialiser le mot de passe' pour les comptes identifiant"
                }, 400

            admin_name = str(
                admin_user.full_name or admin_user.email or "L'administrateur"
            )
            email_type = "invitation"
            raw_token = None
            invite_result = None

            # Compte actif sans historique d'invitation → notification d'accès
            if (
                target_user.account_status in ("active", None)
                and not target_user.invite_sent_at
                and target_user.email
            ):
                from application.institutions.invitation_service import (
                    send_institution_access_email,
                )

                invite_result = send_institution_access_email(
                    to_email=str(target_user.email),
                    first_name=str(target_user.first_name)
                    if target_user.first_name
                    else None,
                    institution_name=str(institution.name or "Institution"),
                    inviter_name=admin_name,
                    role=str(target_user.institution_role or ""),
                )
                email_type = "access_notification"
            else:
                if target_user.account_status not in ("invited", "disabled", None):
                    if not (
                        target_user.account_status in ("active", None)
                        and target_user.invite_sent_at
                    ):
                        return {
                            "error": "Seuls les utilisateurs invités, désactivés ou actifs sans notification peuvent recevoir un renvoi"
                        }, 400

                raw_token, token_hash = generate_invite_token()
                target_user.invite_token_hash = token_hash
                target_user.invite_expires_at = get_invite_expiry()
                target_user.invite_sent_at = datetime.now(UTC)
                if target_user.account_status == "disabled":
                    target_user.account_status = "invited"
                    target_user.disabled_at = None

                db.session.commit()

                invite_result = send_invitation_email(
                    to_email=str(target_user.email or ""),
                    first_name=str(target_user.first_name)
                    if target_user.first_name
                    else None,
                    institution_name=str(institution.name or "Institution"),
                    inviter_name=admin_name,
                    role=str(target_user.institution_role or ""),
                    raw_token=raw_token,
                )

            if email_type == "access_notification":
                db.session.commit()

            invite_link = build_invite_url(raw_token) if raw_token else None

            _record_institution_user_audit(
                institution_id=institution.id,
                target_user_id=target_user.id,
                performed_by_user_id=admin_user.id,
                event_type="invite_resent",
                metadata={"email_type": email_type, "email_sent": invite_result.success if invite_result else False},
            )

            # Audit log
            try:
                AuditLogger.log_institution_action(
                    action_type="institution_invite_resent",
                    institution_id=institution.id,
                    user_id=admin_user.id,
                    result_status="success",
                    action_details={
                        "target_user_id": target_user.id,
                        "target_email": target_user.email,
                        "email_sent": invite_result.success if invite_result else False,
                        "email_type": email_type,
                    },
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
            except Exception as audit_err:
                logger.warning("[Institution] Audit log resend failed: %s", audit_err)

            message = (
                "Notification d'accès renvoyée par email"
                if email_type == "access_notification" and invite_result and invite_result.success
                else "Invitation renvoyée par email"
                if invite_result and invite_result.success
                else "L'email n'a pas pu être envoyé"
            )

            payload = {
                "message": message,
                "user": _serialize_institution_user(target_user, institution),
                "email_sent": invite_result.success if invite_result else False,
                "email_type": email_type,
                "email_error": invite_result.error
                if invite_result and not invite_result.success
                else None,
            }
            if invite_link:
                payload["invite_link"] = invite_link
            return payload, 200

        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            logger.error("[Institution] Erreur resend-invite user %s: %s", user_id, e)
            return APIErrorHandler.handle_exception(e, logger)


@institutions_ns.route("/users/<int:user_id>/disable")
class InstitutionUserDisable(Resource):
    """Désactiver un utilisateur de l'institution."""

    @institutions_ns.doc(
        description="Désactive un utilisateur (admin only). Le compte reste mais ne peut plus se connecter.",
        security="Bearer",
    )
    @jwt_required()
    @role_required(UserRole.INSTITUTION)
    def post(self, user_id):
        """Désactive un utilisateur de l'institution (admin only)."""
        try:
            institution, admin_user = AuthorizationService.require_institution_role(
                InstitutionRole.ADMIN.value
            )

            target_user = User.query.filter_by(
                id=user_id,
                institution_id=institution.id,
            ).first()

            if not target_user:
                return {"error": "Utilisateur non trouvé dans cette institution"}, 404

            # Interdire de se désactiver soi-même
            if target_user.id == admin_user.id:
                return {"error": "Vous ne pouvez pas vous désactiver vous-même"}, 400

            # Règle "dernier admin"
            if target_user.institution_role == InstitutionRole.ADMIN.value:
                admin_count = _count_admins(institution.id)
                if admin_count <= 1:
                    return {
                        "error": "Impossible de désactiver le dernier administrateur"
                    }, 400

            if target_user.account_status == "disabled":
                return {"error": "Cet utilisateur est déjà désactivé"}, 400

            old_status = target_user.account_status
            target_user.account_status = "disabled"
            target_user.disabled_at = datetime.now(UTC)
            target_user.invite_token_hash = None
            target_user.invite_expires_at = None
            db.session.commit()

            _record_institution_user_audit(
                institution_id=institution.id,
                target_user_id=target_user.id,
                performed_by_user_id=admin_user.id,
                event_type="user_disabled",
                metadata={"old_status": old_status},
            )

            # Audit log
            try:
                AuditLogger.log_institution_action(
                    action_type="institution_user_disabled",
                    institution_id=institution.id,
                    user_id=admin_user.id,
                    result_status="success",
                    action_details={
                        "target_user_id": target_user.id,
                        "target_email": target_user.email,
                        "old_status": old_status,
                    },
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
            except Exception as audit_err:
                logger.warning("[Institution] Audit log disable failed: %s", audit_err)

            return {
                "message": "Utilisateur désactivé",
                "user": _serialize_institution_user(target_user),
            }, 200

        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            logger.error("[Institution] Erreur disable user %s: %s", user_id, e)
            return APIErrorHandler.handle_exception(e, logger)


# ─── Mon profil utilisateur (tous rôles institution) ────────────────────────


@institutions_ns.route("/me/profile")
class InstitutionMyProfile(Resource):
    """Permet à chaque utilisateur institution de gérer son propre profil."""

    @institutions_ns.doc(
        description="Met à jour le profil personnel de l'utilisateur connecté.",
        security="Bearer",
    )
    @institutions_ns.response(200, "Profil mis à jour")
    @institutions_ns.response(400, "Données invalides", validation_error_model)
    @institutions_ns.response(401, "Non authentifié", permission_error_model)
    @jwt_required()
    @role_required(UserRole.INSTITUTION)
    def put(self):
        """Met à jour les informations personnelles (prénom, nom, téléphone).

        Accessible à tous les rôles institution.
        """
        try:
            institution, user = AuthorizationService.require_institution()

            data = request.get_json() or {}
            schema = InstitutionMyProfileUpdateSchema()
            errors = schema.validate(data)
            if errors:
                return {"error": "Données invalides", "details": errors}, 400

            validated_raw = schema.load(data) or {}
            validated: dict[str, Any] = (
                validated_raw if isinstance(validated_raw, dict) else {}
            )

            if not validated:
                return {"error": "Aucun champ à mettre à jour"}, 400

            updated_fields = []
            first_name = validated.get("first_name")
            if first_name is not None:
                user.first_name = str(first_name)
                updated_fields.append("first_name")
            last_name = validated.get("last_name")
            if last_name is not None:
                user.last_name = str(last_name)
                updated_fields.append("last_name")
            if "phone" in validated:
                user.phone = (
                    str(validated.get("phone"))
                    if validated.get("phone") is not None
                    else None
                )
                updated_fields.append("phone")

            db.session.commit()

            try:
                AuditLogger.log_action(
                    action_type="institution_user_profile_updated",
                    action_category="institution",
                    user_id=user.id,
                    user_type="institution",
                    institution_id=institution.id,
                    result_status="success",
                    action_details={"updated_fields": updated_fields},
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
            except Exception as audit_err:
                logger.warning(
                    "[Institution] Audit log profile update failed: %s", audit_err
                )

            return {
                "message": "Profil mis à jour",
                "user": {
                    "id": user.id,
                    "public_id": user.public_id,
                    "username": user.username,
                    "email": user.email,
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                    "phone": user.phone,
                },
            }, 200

        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            logger.error("[Institution] Erreur PUT /me/profile: %s", e)
            return APIErrorHandler.handle_exception(e, logger)


# ─── Demandes de droits (permission requests) ───────────────────────────────


class PermissionRequestRecord(TypedDict):
    id: int
    user_id: int
    user_email: str
    user_name: str
    current_role: str
    requested_role: str
    message: str
    status: str
    created_at: str
    resolved_at: str | None
    resolved_by: int | None


# Table en mémoire pour les demandes de droits (simple dict en attendant un modèle DB)
# En production, utiliser un modèle SQLAlchemy dédié.
_permission_requests_store: dict[int, list[PermissionRequestRecord]] = {}


def _get_permission_requests(institution_id: int) -> list[PermissionRequestRecord]:
    """Retourne la liste des demandes de droits pour une institution."""
    return _permission_requests_store.get(institution_id, [])


def _add_permission_request(
    institution_id: int,
    user_id: int,
    user_email: str,
    user_name: str,
    current_role: str,
    requested_role: str,
    message: str,
) -> PermissionRequestRecord:
    """Ajoute une demande de droits."""
    if institution_id not in _permission_requests_store:
        _permission_requests_store[institution_id] = []

    pr: PermissionRequestRecord = {
        "id": len(_permission_requests_store[institution_id]) + 1,
        "user_id": user_id,
        "user_email": user_email,
        "user_name": user_name,
        "current_role": current_role,
        "requested_role": requested_role,
        "message": message,
        "status": "pending",
        "created_at": datetime.now(UTC).isoformat(),
        "resolved_at": None,
        "resolved_by": None,
    }
    _permission_requests_store[institution_id].append(pr)
    return pr


@institutions_ns.route("/permission-requests")
class PermissionRequests(Resource):
    """Demandes de droits d'utilisation."""

    @institutions_ns.doc(
        description="Crée une demande de droits auprès de l'admin.",
        security="Bearer",
    )
    @institutions_ns.response(201, "Demande créée")
    @institutions_ns.response(400, "Données invalides", validation_error_model)
    @institutions_ns.response(409, "Demande déjà en cours")
    @jwt_required()
    @role_required(UserRole.INSTITUTION)
    def post(self):
        """Envoie une demande de droits à l'admin de l'institution.

        Accessible à tous les rôles institution (sauf admin qui a déjà tous les droits).
        """
        try:
            institution, user = AuthorizationService.require_institution()
            jwt_claims = get_jwt()
            current_role = str(jwt_claims.get("institution_role", "") or "")

            # L'admin n'a pas besoin de demander des droits
            if current_role == InstitutionRole.ADMIN.value:
                return {"error": "Les administrateurs ont déjà tous les droits"}, 400

            data = request.get_json() or {}
            schema = PermissionRequestCreateSchema()
            errors = schema.validate(data)
            if errors:
                return {"error": "Données invalides", "details": errors}, 400

            validated_raw = schema.load(data) or {}
            if not isinstance(validated_raw, dict):
                return {"error": "Données invalides"}, 400

            requested_role = str(validated_raw.get("requested_role") or "")
            message = str(validated_raw.get("message") or "")
            if not requested_role:
                return {
                    "error": "Données invalides",
                    "details": {"requested_role": ["Champ requis"]},
                }, 400
            if not message:
                return {
                    "error": "Données invalides",
                    "details": {"message": ["Champ requis"]},
                }, 400

            # Vérifier qu'il n'y a pas déjà une demande pending
            existing = _get_permission_requests(institution.id)
            has_pending = any(
                pr["user_id"] == user.id and pr["status"] == "pending"
                for pr in existing
            )
            if has_pending:
                return {
                    "error": (
                        "Vous avez déjà une demande en attente. "
                        "Veuillez attendre la réponse de l'administrateur."
                    )
                }, 409

            user_name = f"{user.first_name or ''} {user.last_name or ''}".strip()
            pr = _add_permission_request(
                institution_id=institution.id,
                user_id=user.id,
                user_email=str(user.email or ""),
                user_name=user_name or user.username,
                current_role=current_role,
                requested_role=requested_role,
                message=message,
            )

            try:
                AuditLogger.log_action(
                    action_type="permission_request_created",
                    action_category="institution",
                    user_id=user.id,
                    user_type="institution",
                    institution_id=institution.id,
                    result_status="success",
                    action_details={
                        "current_role": current_role,
                        "requested_role": requested_role,
                    },
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
            except Exception as audit_err:
                logger.warning(
                    "[Institution] Audit log permission request failed: %s", audit_err
                )

            return {"message": "Demande envoyée", "request": pr}, 201

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error("[Institution] Erreur POST /permission-requests: %s", e)
            return APIErrorHandler.handle_exception(e, logger)

    @institutions_ns.doc(
        description="Liste les demandes de droits (admin: toutes, autres: les siennes).",
        security="Bearer",
    )
    @institutions_ns.response(200, "Succès")
    @jwt_required()
    @role_required(UserRole.INSTITUTION)
    def get(self):
        """Liste les demandes de droits.

        - Admin: voit toutes les demandes de l'institution
        - Autres: voient uniquement leurs propres demandes
        """
        try:
            institution, user = AuthorizationService.require_institution()
            jwt_claims = get_jwt()
            current_role = jwt_claims.get("institution_role", "")

            all_requests = _get_permission_requests(institution.id)

            if current_role == InstitutionRole.ADMIN.value:
                # Admin voit tout
                return {"requests": all_requests}, 200

            # Autres: seulement les siennes
            my_requests = [pr for pr in all_requests if pr["user_id"] == user.id]
            return {"requests": my_requests}, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error("[Institution] Erreur GET /permission-requests: %s", e)
            return APIErrorHandler.handle_exception(e, logger)


@institutions_ns.route("/permission-requests/<int:request_id>/resolve")
class PermissionRequestResolve(Resource):
    """Résoudre une demande de droits (admin only)."""

    @institutions_ns.doc(
        description="Approuve ou refuse une demande de droits.",
        security="Bearer",
    )
    @institutions_ns.response(200, "Demande résolue")
    @institutions_ns.response(404, "Demande non trouvée")
    @jwt_required()
    @role_required(UserRole.INSTITUTION)
    def put(self, request_id: int):
        """Approuve ou refuse une demande de droits.

        Body: { "action": "approve" | "deny" }
        Si approve, le rôle de l'utilisateur est automatiquement mis à jour.
        """
        try:
            institution, admin_user = AuthorizationService.require_institution_role(
                InstitutionRole.ADMIN.value
            )

            data = request.get_json() or {}
            action = data.get("action")
            if action not in ("approve", "deny"):
                return {"error": "Action invalide. Utilisez 'approve' ou 'deny'."}, 400

            all_requests = _get_permission_requests(institution.id)
            pr = next((r for r in all_requests if r["id"] == request_id), None)
            if not pr:
                return {"error": "Demande non trouvée"}, 404

            if pr["status"] != "pending":
                return {"error": "Cette demande a déjà été traitée"}, 400

            pr["status"] = "approved" if action == "approve" else "denied"
            pr["resolved_at"] = datetime.now(UTC).isoformat()
            pr["resolved_by"] = admin_user.id

            # Si approuvé, mettre à jour le rôle de l'utilisateur
            if action == "approve":
                target_user = User.query.get(pr["user_id"])
                if target_user:
                    target_user.institution_role = pr["requested_role"]
                    db.session.commit()
                    logger.info(
                        "[Institution] Permission request approved: user %s -> %s",
                        target_user.email,
                        pr["requested_role"],
                    )

            try:
                AuditLogger.log_action(
                    action_type=f"permission_request_{pr['status']}",
                    action_category="institution",
                    user_id=admin_user.id,
                    user_type="institution",
                    institution_id=institution.id,
                    result_status="success",
                    action_details={
                        "request_id": pr["id"],
                        "target_user_id": pr["user_id"],
                        "requested_role": pr["requested_role"],
                    },
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
            except Exception as audit_err:
                logger.warning(
                    "[Institution] Audit log permission resolve failed: %s",
                    audit_err,
                )

            return {"message": f"Demande {pr['status']}", "request": pr}, 200

        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            logger.error(
                "[Institution] Erreur PUT /permission-requests/%s/resolve: %s",
                request_id,
                e,
            )
            return APIErrorHandler.handle_exception(e, logger)
