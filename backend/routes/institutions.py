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

import sentry_sdk
from flask import g, request
from flask_jwt_extended import get_jwt, jwt_required
from flask_restx import Namespace, Resource, fields

from application.institutions.institution_settings_service import (
    get_or_create_settings,
)
from ext import db, limiter, role_required
from models import UserRole
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
    InstitutionUserUpdateRoleSchema,
    PermissionRequestCreateSchema,
)
from security.api_key_auth import api_key_required
from security.audit_log import AuditLogger
from security.authorization import AuthorizationService
from shared.error_handlers import APIErrorHandler

logger = logging.getLogger(__name__)

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
        "institution_role": fields.String(
            description="Rôle de l'utilisateur dans l'institution"
        ),
        "user": fields.Raw(description="Informations de l'utilisateur"),
    },
)


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
            institution_role = jwt_claims.get("institution_role")

            # Audit log
            try:
                AuditLogger.log_institution_action(
                    action_type="institution_me_viewed",
                    institution_id=institution.id,
                    user_id=user.id,
                    result_status="success",
                    action_details={
                        "institution_role": institution_role,
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
            response_data = {
                "id": institution.id,
                "public_id": institution.public_id,
                "name": institution.name,
                "institution_type": institution.institution_type,
                "address": institution.address,
                "contact_email": institution.contact_email,
                "contact_phone": institution.contact_phone,
                "notes": institution.notes,
                "institution_role": institution_role,
                "user": {
                    "id": user.id,
                    "public_id": user.public_id,
                    "username": user.username,
                    "email": user.email,
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                    "phone": user.phone,
                },
            }

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
                # Aucun changement réel
                jwt_claims = get_jwt()
                institution_role = jwt_claims.get("institution_role")
                return {
                    "id": institution.id,
                    "public_id": institution.public_id,
                    "name": institution.name,
                    "institution_type": institution.institution_type,
                    "address": institution.address,
                    "contact_email": institution.contact_email,
                    "contact_phone": institution.contact_phone,
                    "notes": institution.notes,
                    "institution_role": institution_role,
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
            institution_role = jwt_claims.get("institution_role")

            return {
                "id": institution.id,
                "public_id": institution.public_id,
                "name": institution.name,
                "institution_type": institution.institution_type,
                "address": institution.address,
                "contact_email": institution.contact_email,
                "contact_phone": institution.contact_phone,
                "notes": institution.notes,
                "institution_role": institution_role,
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
            logger.error(
                "[Institution] Erreur PUT /me: %s - %s",
                type(e).__name__,
                str(e),
            )
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


def _serialize_institution_user(user: User) -> dict[str, object]:
    """Sérialise un utilisateur institution pour la réponse API."""
    from datetime import UTC, datetime

    status = user.account_status or "active"
    # Si invited et token expiré → marquer comme "expired"
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
        "account_status": status,
        "invite_sent_at": user.invite_sent_at.isoformat()
        if user.invite_sent_at
        else None,
        "invite_expires_at": user.invite_expires_at.isoformat()
        if user.invite_expires_at
        else None,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


def _count_admins(institution_id: int) -> int:
    """Compte le nombre d'admins actifs dans l'institution (exclut disabled)."""
    return User.query.filter(
        User.institution_id == institution_id,
        User.institution_role == InstitutionRole.ADMIN.value,
        db.or_(User.account_status.is_(None), User.account_status != "disabled"),
    ).count()


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
                .order_by(User.created_at.asc())
                .all()
            )

            return {
                "users": [_serialize_institution_user(u) for u in users],
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
        """Invite un utilisateur dans l'institution par email (admin only).

        - Si l'email existe déjà dans cette institution → 409
        - Si l'email existe dans une autre institution → 409
        - Si l'email existe mais libre → rattache + envoie invitation
        - Si l'email n'existe pas → crée le user (status=invited) + envoie invitation
        """
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

            data = request.get_json() or {}
            schema = InstitutionUserInviteSchema()
            errors = schema.validate(data)
            if errors:
                return {"error": "Données invalides", "details": errors}, 400

            validated = schema.load(data) or {}
            email = str(validated.get("email", "")).strip().lower()
            role_value = str(validated.get("institution_role", ""))
            first_name = validated.get("first_name")
            last_name = validated.get("last_name")

            # Vérifier si l'utilisateur existe déjà
            existing_user = User.query.filter_by(email=email).first()

            if existing_user:
                # Vérifier s'il est déjà dans cette institution
                if existing_user.institution_id == institution.id:
                    error_msg = (
                        "Cet utilisateur est désactivé. Utilisez 'Renvoyer invitation' pour le réactiver."
                        if existing_user.account_status == "disabled"
                        else "Cet utilisateur fait déjà partie de l'institution"
                    )
                    return {"error": error_msg}, 409

                # Vérifier s'il est dans une autre institution
                if existing_user.institution_id is not None:
                    return {
                        "error": "Cet utilisateur appartient déjà à une autre institution"
                    }, 409

                # User existant sans institution → rattacher + invitation
                existing_user.institution_id = institution.id
                existing_user.institution_role = role_value
                existing_user.role = UserRole.INSTITUTION
                existing_user.account_status = "active"  # Déjà un compte
                if first_name:
                    existing_user.first_name = first_name
                if last_name:
                    existing_user.last_name = last_name

                db.session.commit()

                # Audit log
                try:
                    AuditLogger.log_institution_action(
                        action_type="institution_user_added",
                        institution_id=institution.id,
                        user_id=admin_user.id,
                        result_status="success",
                        action_details={
                            "target_user_id": existing_user.id,
                            "target_email": email,
                            "institution_role": role_value,
                            "method": "existing_user",
                        },
                        ip_address=request.remote_addr,
                        user_agent=request.headers.get("User-Agent"),
                    )
                except Exception as audit_err:
                    logger.warning("[Institution] Audit log failed: %s", audit_err)

                return {
                    "message": "Utilisateur existant ajouté à l'institution",
                    "user": _serialize_institution_user(existing_user),
                }, 200

            # Créer un nouveau utilisateur avec invitation par email
            raw_token, token_hash = generate_invite_token()
            placeholder_password = secrets.token_urlsafe(32)

            new_user = User()
            new_user.public_id = str(uuid.uuid4())
            new_user.username = email
            new_user.email = email
            new_user.first_name = first_name or ""
            new_user.last_name = last_name or ""
            new_user.role = UserRole.INSTITUTION
            new_user.institution_id = institution.id
            new_user.institution_role = role_value
            new_user.account_status = "invited"
            new_user.invite_token_hash = token_hash
            new_user.invite_expires_at = get_invite_expiry()
            new_user.invite_sent_at = datetime.now(UTC)
            new_user.force_password_change = True
            new_user.set_password(placeholder_password)

            db.session.add(new_user)
            db.session.commit()

            # Envoyer l'email d'invitation (ne bloque pas si échec)
            admin_name = str(
                admin_user.full_name or admin_user.email or "L'administrateur"
            )
            invite_result = send_invitation_email(
                to_email=email,
                first_name=str(first_name) if first_name else None,
                institution_name=str(institution.name or "Institution"),
                inviter_name=admin_name,
                role=role_value,
                raw_token=raw_token,
            )

            if not invite_result.success:
                logger.warning(
                    "[Institution] Invitation créée mais email non envoyé: %s",
                    invite_result.error,
                )

            # Construire le lien d'invitation (fallback si email échoue)
            invite_link = build_invite_url(raw_token)

            # Audit log
            try:
                AuditLogger.log_institution_action(
                    action_type="institution_user_invited",
                    institution_id=institution.id,
                    user_id=admin_user.id,
                    result_status="success",
                    action_details={
                        "target_user_id": new_user.id,
                        "target_email": email,
                        "institution_role": role_value,
                        "method": "email_invitation",
                        "email_sent": invite_result.success,
                    },
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
            except Exception as audit_err:
                logger.warning("[Institution] Audit log failed: %s", audit_err)

            return {
                "message": "Invitation envoyée par email"
                if invite_result.success
                else "Utilisateur créé mais l'email n'a pas pu être envoyé",
                "user": _serialize_institution_user(new_user),
                "email_sent": invite_result.success,
                "email_error": invite_result.error
                if not invite_result.success
                else None,
                "invite_link": invite_link,
            }, 201

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

            # Retirer de l'institution (soft: on ne supprime pas le user)
            target_user.institution_id = None
            target_user.institution_role = None
            db.session.commit()

            # Audit log
            try:
                AuditLogger.log_institution_action(
                    action_type="institution_user_removed",
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

            return {"message": "Utilisateur retiré de l'institution"}, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error("[Institution] Erreur DELETE /users/%s: %s", user_id, e)
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

            # Seuls les utilisateurs "invited" ou "disabled" peuvent recevoir un resend
            if target_user.account_status not in ("invited", "disabled", None):
                return {
                    "error": "Seuls les utilisateurs invités ou désactivés peuvent recevoir une nouvelle invitation"
                }, 400

            # Générer un nouveau token
            raw_token, token_hash = generate_invite_token()

            target_user.invite_token_hash = token_hash
            target_user.invite_expires_at = get_invite_expiry()
            target_user.invite_sent_at = datetime.now(UTC)
            # Réactiver si disabled
            if target_user.account_status == "disabled":
                target_user.account_status = "invited"

            db.session.commit()

            # Envoyer l'email
            admin_name = str(
                admin_user.full_name or admin_user.email or "L'administrateur"
            )
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

            # Construire le lien d'invitation (fallback)
            invite_link = build_invite_url(raw_token)

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
                        "email_sent": invite_result.success,
                    },
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
            except Exception as audit_err:
                logger.warning("[Institution] Audit log resend failed: %s", audit_err)

            return {
                "message": "Invitation renvoyée par email"
                if invite_result.success
                else "Le token a été régénéré mais l'email n'a pas pu être envoyé",
                "user": _serialize_institution_user(target_user),
                "email_sent": invite_result.success,
                "email_error": invite_result.error
                if not invite_result.success
                else None,
                "invite_link": invite_link,
            }, 200

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
            # Invalider le token d'invitation s'il existe
            target_user.invite_token_hash = None
            target_user.invite_expires_at = None
            db.session.commit()

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

            validated = schema.load(data) or {}

            if not validated:
                return {"error": "Aucun champ à mettre à jour"}, 400

            updated_fields = []
            if "first_name" in validated and validated["first_name"] is not None:
                user.first_name = validated["first_name"]
                updated_fields.append("first_name")
            if "last_name" in validated and validated["last_name"] is not None:
                user.last_name = validated["last_name"]
                updated_fields.append("last_name")
            if "phone" in validated:
                user.phone = validated["phone"]
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

# Table en mémoire pour les demandes de droits (simple dict en attendant un modèle DB)
# En production, utiliser un modèle SQLAlchemy dédié.
_permission_requests_store: dict[int, list] = {}


def _get_permission_requests(institution_id: int) -> list:
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
) -> dict:
    """Ajoute une demande de droits."""
    from datetime import datetime, UTC

    if institution_id not in _permission_requests_store:
        _permission_requests_store[institution_id] = []

    pr = {
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
            current_role = jwt_claims.get("institution_role", "")

            # L'admin n'a pas besoin de demander des droits
            if current_role == InstitutionRole.ADMIN.value:
                return {
                    "error": "Les administrateurs ont déjà tous les droits"
                }, 400

            data = request.get_json() or {}
            schema = PermissionRequestCreateSchema()
            errors = schema.validate(data)
            if errors:
                return {"error": "Données invalides", "details": errors}, 400

            validated = schema.load(data)
            requested_role = validated["requested_role"]
            message = validated["message"]

            # Vérifier qu'il n'y a pas déjà une demande pending
            existing = _get_permission_requests(institution.id)
            has_pending = any(
                pr["user_id"] == user.id and pr["status"] == "pending"
                for pr in existing
            )
            if has_pending:
                return {
                    "error": "Vous avez déjà une demande en attente. "
                    "Veuillez attendre la réponse de l'administrateur."
                }, 409

            user_name = f"{user.first_name or ''} {user.last_name or ''}".strip()
            pr = _add_permission_request(
                institution_id=institution.id,
                user_id=user.id,
                user_email=user.email,
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
            else:
                # Autres: seulement les siennes
                my_requests = [
                    pr for pr in all_requests if pr["user_id"] == user.id
                ]
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
                return {
                    "error": "Action invalide. Utilisez 'approve' ou 'deny'."
                }, 400

            all_requests = _get_permission_requests(institution.id)
            pr = next(
                (r for r in all_requests if r["id"] == request_id), None
            )
            if not pr:
                return {"error": "Demande non trouvée"}, 404

            if pr["status"] != "pending":
                return {"error": "Cette demande a déjà été traitée"}, 400

            from datetime import datetime, UTC

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
