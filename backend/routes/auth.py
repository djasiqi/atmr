import hashlib
import json
import logging
import os
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import sentry_sdk  # pyright: ignore[reportMissingImports]
from flask import (  # pyright: ignore[reportMissingImports]
    current_app,
    make_response,
    request,
)
from flask_jwt_extended import (  # pyright: ignore[reportMissingImports]
    create_access_token,
    create_refresh_token,
    decode_token,
    get_jwt,
    get_jwt_identity,
    jwt_required,
)
from flask_mail import Message  # pyright: ignore[reportMissingImports]
from flask_restx import (  # pyright: ignore[reportMissingImports]
    Namespace,
    Resource,
    fields,
)
from itsdangerous import URLSafeTimedSerializer  # pyright: ignore[reportMissingImports]
from marshmallow import (  # pyright: ignore[reportMissingImports]
    Schema,
    ValidationError,
)
from marshmallow import fields as ma_fields  # pyright: ignore[reportMissingImports]

from application.users import (
    AuthenticateUserInput,
    AuthenticateUserUseCase,
    GetCurrentUserUseCase,
    RegisterUserInput,
    RegisterUserUseCase,
)
from ext import db, limiter, mail, role_required
from middleware.trace_id import get_trace_id
from models import (
    Client,
    User,
)  # Client utilisé pour création directe, User pour type annotations
from models.enums import UserRole
from repositories.user_repository import UserRepository
from routes.api_error_models import (
    create_api_error_model,
    create_not_found_error_model,
    create_permission_error_model,
    create_validation_error_model,
)
from schemas.auth_schemas import LoginSchema, RegisterSchema
from schemas.validation_utils import handle_validation_error, validate_request
from security.audit_log import AuditLogger
from security.refresh_token_service import (
    get_user_active_sessions,
    is_token_revoked,
    revoke_all_user_tokens,
    revoke_refresh_token,
    store_refresh_token,
    update_token_last_used,
)
from security.security_metrics import (
    security_login_attempts_total,
    security_login_failures_total,
    security_logout_total,
    security_token_refreshes_total,
)
from services.security.csrf import generate_csrf_token
from services.security.authentication import RefreshTokenService
from shared.error_handlers import APIErrorHandler
from shared.logging_utils import mask_email

logger = logging.getLogger(__name__)

# Initialisation des repositories
user_repo = UserRepository()

auth_ns = Namespace("auth", description="Opérations liées à l'authentification")

# ✅ S1: Modèle Swagger pour la réponse CSRF token
csrf_token_response_model = auth_ns.model(
    "CSRFTokenResponse",
    {
        "csrf_token": fields.String(
            required=True, description="Token CSRF à inclure dans les requêtes mutantes"
        ),
        "ttl": fields.Integer(
            required=True, description="Durée de vie du token en secondes"
        ),
    },
)

# ✅ S1: Modèle Swagger pour la réponse CSRF token
csrf_token_response_model = auth_ns.model(
    "CSRFTokenResponse",
    {
        "csrf_token": fields.String(
            required=True, description="Token CSRF à inclure dans les requêtes mutantes"
        ),
        "ttl": fields.Integer(
            required=True, description="Durée de vie du token en secondes"
        ),
    },
)

# Constante pour la longueur du hash de version du mot de passe
PASSWORD_HASH_VERSION_LENGTH = 16

# Modèle Swagger pour la connexion (login)
login_model = auth_ns.model(
    "Login",
    {
        "email": fields.String(
            required=True,
            description="L'adresse email de l'utilisateur (format email valide)",
        ),
        "password": fields.String(
            required=True, description="Le mot de passe de l'utilisateur", min_length=6
        ),
    },
)

# ✅ P0: Modèles d'erreur standardisés
api_error_model = create_api_error_model(auth_ns)
validation_error_model = create_validation_error_model(auth_ns)
not_found_error_model = create_not_found_error_model(auth_ns)
permission_error_model = create_permission_error_model(auth_ns)

# ✅ P0: Modèle de réponse succès pour login
login_success_model = auth_ns.model(
    "LoginSuccess",
    {
        "message": fields.String(
            required=True,
            description="Message de confirmation",
            example="Connexion réussie",
        ),
        "token": fields.String(
            required=True, description="Token d'accès JWT (pour mobile)"
        ),
        "refresh_token": fields.String(
            required=True, description="Token de rafraîchissement (pour mobile)"
        ),
        "user": fields.Raw(
            required=True,
            description="Informations utilisateur",
            example={
                "id": 1,
                "public_id": "abc123",
                "username": "john.doe",
                "email": "john.doe@example.com",
                "role": "client",
                "force_password_change": False,
            },
        ),
        "trace_id": fields.String(
            required=False,
            description="ID de traçage pour le support",
            example="a1b2c3d4",
        ),
    },
)

# Modèle Swagger pour obtenir un token fresh
fresh_token_request_model = auth_ns.model(
    "FreshTokenRequest",
    {
        "password": fields.String(
            required=True,
            description="Le mot de passe de l'utilisateur pour vérification",
            min_length=6,
        ),
    },
)

# Modèle Swagger pour la réponse de fresh token
fresh_token_response_model = auth_ns.model(
    "FreshTokenResponse",
    {
        "access_token": fields.String(
            required=True, description="Nouveau token d'accès 'fresh'"
        ),
        "message": fields.String(description="Message de confirmation"),
    },
)

# Modèle Swagger pour l'inscription (register)
register_model = auth_ns.model(
    "Register",
    {
        "username": fields.String(
            required=True,
            description="Le nom d'utilisateur",
            min_length=3,
            max_length=50,
        ),
        "email": fields.String(
            required=True,
            description="L'adresse email de l'utilisateur (format email valide)",
        ),
        "password": fields.String(
            required=True, description="Le mot de passe de l'utilisateur", min_length=6
        ),
        "first_name": fields.String(description="Prénom", default=None, max_length=100),
        "last_name": fields.String(description="Nom", default=None, max_length=100),
        "phone": fields.String(
            description="Numéro de téléphone", default=None, max_length=20
        ),
        "address": fields.String(description="Adresse", default=None, max_length=500),
        "birth_date": fields.String(
            description="Date de naissance (YYYY-MM-DD)",
            default=None,
            pattern="^\\d{4}-\\d{2}-\\d{2}$",
        ),
        "gender": fields.String(
            description="Genre (male|female|other)",
            default=None,
            enum=["male", "female", "other"],
        ),
        "profile_image": fields.String(
            description="URL ou données base64 de l'image de profil", default=None
        ),
    },
)

# Modèle Swagger pour la réponse de logout (succès)
logout_response_model = auth_ns.model(
    "LogoutResponse",
    {
        "message": fields.String(description="Message de confirmation de déconnexion"),
    },
)

# Modèle Swagger pour la réponse d'erreur de logout
logout_error_model = auth_ns.model(
    "LogoutError",
    {
        "error": fields.String(description="Message d'erreur"),
    },
)

# Modèle Swagger pour la réponse de révocation de sessions
revoke_sessions_response_model = auth_ns.model(
    "RevokeSessionsResponse",
    {
        "message": fields.String(
            required=True,
            description="Message de confirmation avec nombre de sessions révoquées",
        ),
        "sessions_revoked": fields.Integer(
            required=True, description="Nombre de sessions révoquées"
        ),
    },
)

# Modèle Swagger pour une session
session_model = auth_ns.model(
    "Session",
    {
        "id": fields.Integer(required=True, description="ID de la session"),
        "device_id": fields.String(required=False, description="ID de l'appareil"),
        "device_name": fields.String(required=False, description="Nom de l'appareil"),
        "created_at": fields.String(
            required=True, description="Date de création (ISO8601)"
        ),
        "expires_at": fields.String(
            required=True, description="Date d'expiration (ISO8601)"
        ),
        "last_used_at": fields.String(
            required=False, description="Date de dernière utilisation (ISO8601)"
        ),
        "is_revoked": fields.Boolean(
            required=True, description="Si la session est révoquée"
        ),
        "revoked_at": fields.String(
            required=False, description="Date de révocation (ISO8601)"
        ),
    },
)

# Modèle Swagger pour la réponse de liste de sessions
list_sessions_response_model = auth_ns.model(
    "ListSessionsResponse",
    {
        "sessions": fields.List(
            fields.Nested(session_model),
            required=True,
            description="Liste des sessions actives",
        ),
        "count": fields.Integer(
            required=True, description="Nombre total de sessions actives"
        ),
    },
)

# Schéma Marshmallow pour valider les données d'inscription


class UserSchema(Schema):
    username = ma_fields.String(required=True)
    email = ma_fields.Email(required=True)
    password = ma_fields.String(required=True)

    # --- CORRECTION APPLIQUÉE ICI ---
    # Remplacement de 'missing' par 'load_default'
    first_name = ma_fields.String(load_default=None)
    last_name = ma_fields.String(load_default=None)
    phone = ma_fields.String(load_default=None)
    address = ma_fields.String(load_default=None)
    birth_date = ma_fields.Date(load_default=None)
    gender = ma_fields.String(load_default=None)
    profile_image = ma_fields.String(load_default=None)


# ========================
# 1. Connexion / Login
# ========================
@auth_ns.route("/login")
class Login(Resource):
    @auth_ns.expect(login_model)
    @auth_ns.response(200, "Connexion réussie", login_success_model)
    @auth_ns.response(400, "Erreur de validation", validation_error_model)
    @auth_ns.response(401, "Credentials invalides", permission_error_model)
    @auth_ns.response(403, "Compte désactivé ou non autorisé", permission_error_model)
    @auth_ns.response(429, "Rate limit dépassé", api_error_model)
    @auth_ns.response(500, "Erreur serveur", api_error_model)
    # Limite d'appels pour éviter le brute force
    @limiter.limit("5 per minute")
    def post(self):
        """Authentifie un utilisateur et renvoie un token d'accès."""
        # #region agent log
        log_path = Path(r"c:\Users\jasiq\atmr\.cursor\debug.log")
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "location": "auth.py:Login.post",
                            "message": "POST /auth/login entry",
                            "data": {
                                "headers": {
                                    k: v
                                    for k, v in request.headers
                                    if k.lower()
                                    in [
                                        "authorization",
                                        "x-device-id",
                                        "x-requested-with",
                                        "content-type",
                                    ]
                                },
                                "has_json": request.is_json,
                                "x_requested_with": request.headers.get(
                                    "X-Requested-With"
                                ),
                                "is_mobile_request": request.headers.get(
                                    "X-Requested-With"
                                )
                                == "Expo",
                            },
                            "timestamp": datetime.now(UTC).isoformat(),
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "B",
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion
        try:
            # #region agent log
            try:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "location": "auth.py:Login.post",
                                "message": "before request.get_json",
                                "data": {
                                    "is_json": request.is_json,
                                    "content_type": request.content_type,
                                    "content_length": request.content_length,
                                },
                                "timestamp": datetime.now(UTC).isoformat(),
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "A",
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
            try:
                data = request.get_json() or {}
            except Exception as json_error:
                # Gérer spécifiquement les erreurs de parsing JSON (BadRequest 400)
                # pour éviter qu'elles soient transformées en 500 par le gestionnaire global
                from werkzeug.exceptions import (  # pyright: ignore[reportMissingImports]
                    BadRequest,
                )

                if isinstance(json_error, BadRequest):
                    logger.warning("Erreur parsing JSON dans login: %s", json_error)
                    return APIErrorHandler.handle_validation_error(
                        "Format JSON invalide dans la requête",
                        logger_instance=logger,
                    )
                # Si ce n'est pas une BadRequest, laisser le gestionnaire global la gérer
                raise
            # #region agent log
            try:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "location": "auth.py:Login.post",
                                "message": "after request.get_json",
                                "data": {
                                    "has_email": "email" in data,
                                    "has_password": "password" in data,
                                    "data_type": type(data).__name__,
                                },
                                "timestamp": datetime.now(UTC).isoformat(),
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "A",
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion

            # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
            # #region agent log
            try:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "location": "auth.py:Login.post",
                                "message": "before validate_request",
                                "data": {
                                    "data_keys": list(data.keys()) if data else []
                                },
                                "timestamp": datetime.now(UTC).isoformat(),
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "B",
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
            try:
                validated_data = validate_request(LoginSchema(), data)
            except ValidationError as e:
                return handle_validation_error(e)
            # #region agent log
            try:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "location": "auth.py:Login.post",
                                "message": "after validate_request",
                                "data": {
                                    "validated_keys": list(validated_data.keys())
                                    if validated_data
                                    else []
                                },
                                "timestamp": datetime.now(UTC).isoformat(),
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "B",
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion

            email = validated_data["email"]
            password = validated_data["password"]

            # ✅ DDD: Utiliser le use case pour authentifier l'utilisateur
            # #region agent log
            try:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "location": "auth.py:Login.post",
                                "message": "before AuthenticateUserUseCase.execute",
                                "data": {"email": mask_email(email) if email else None},
                                "timestamp": datetime.now(UTC).isoformat(),
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "C",
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
            uc = AuthenticateUserUseCase()
            input_data = AuthenticateUserInput(email=email, password=password)
            auth_result = uc.execute(input_data)
            # #region agent log
            try:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "location": "auth.py:Login.post",
                                "message": "after AuthenticateUserUseCase.execute",
                                "data": {
                                    "success": auth_result.success
                                    if auth_result
                                    else None,
                                    "has_user": auth_result.user is not None
                                    if auth_result
                                    else None,
                                    "error": auth_result.error if auth_result else None,
                                },
                                "timestamp": datetime.now(UTC).isoformat(),
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "C",
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion

            user = None if not auth_result.success else auth_result.user

            if not user:
                # ✅ Priorité 7: Audit logging pour login échoué
                try:
                    AuditLogger.log_action(
                        action_type="login_failed",
                        action_category="security",
                        user_type="unknown",
                        result_status="failure",
                        result_message="Email ou mot de passe invalide",
                        action_details={
                            "email": mask_email(email),
                            "reason": "invalid_credentials",
                        },
                        ip_address=request.remote_addr,
                        user_agent=request.headers.get("User-Agent"),
                    )
                    # ✅ Priorité 7: Métriques Prometheus pour login échoué
                    security_login_attempts_total.labels(type="failed").inc()
                    security_login_failures_total.inc()
                except Exception as audit_error:
                    # Ne pas bloquer la réponse si l'audit logging échoue
                    logger.warning("Échec audit logging login_failed: %s", audit_error)

                # ✅ S3: Enregistrer tentative échouée pour détection d'alertes
                try:
                    from security.security_alerts import SecurityAlertService

                    SecurityAlertService.record_login_failure(
                        ip_address=request.remote_addr or "unknown", email=email
                    )
                except Exception as alert_error:
                    logger.debug(
                        "[SecurityAlerts] Failed to record login failure: %s",
                        alert_error,
                    )

                # ✅ P0: Ajouter trace_id dans l'erreur
                trace_id = get_trace_id()
                logger.warning(
                    "Login failed",
                    extra={"trace_id": trace_id, "email": mask_email(email)},
                )
                error_response, status_code = APIErrorHandler.handle_permission_error(
                    "Email ou mot de passe invalide.",
                    logger_instance=logger,
                )
                # Ajouter trace_id à la réponse d'erreur
                error_response["trace_id"] = trace_id
                return error_response, status_code

            # Création du token avec le rôle dans additional_claims
            # ✅ SECURITY: Ajout claim 'aud' (audience) pour prévenir token replay
            # #region agent log
            try:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "location": "auth.py:Login.post",
                                "message": "before create_access_token",
                                "data": {
                                    "user_id": user.id if user else None,
                                    "user_public_id": user.public_id if user else None,
                                    "user_role": user.role.value
                                    if user and user.role
                                    else None,
                                },
                                "timestamp": datetime.now(UTC).isoformat(),
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "D",
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
            claims = {
                "role": user.role.value,
                "company_id": getattr(user, "company_id", None),
                "driver_id": getattr(user, "driver_id", None),
                "aud": "atmr-api",  # Audience claim pour sécurité
            }
            access_token = create_access_token(
                identity=str(user.public_id),
                # ⚠️ ID numérique attendu par dispatch_routes
                additional_claims=claims,
                expires_delta=current_app.config["JWT_ACCESS_TOKEN_EXPIRES"],
                fresh=True,  # ✅ Token fresh lors de la connexion initiale
            )
            # #region agent log
            try:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "location": "auth.py:Login.post",
                                "message": "after create_access_token",
                                "data": {
                                    "has_access_token": access_token is not None,
                                    "token_length": len(access_token)
                                    if access_token
                                    else 0,
                                },
                                "timestamp": datetime.now(UTC).isoformat(),
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "D",
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion

            # Création du refresh token
            # (durée configurée dans JWT_REFRESH_TOKEN_EXPIRES)
            # ✅ SECURITY: Ajouter la claim 'aud' et 'pwd_hash' pour invalidation après changement de mot de passe
            pwd_hash_version = _get_password_hash_version(user)
            refresh_expires_delta = current_app.config["JWT_REFRESH_TOKEN_EXPIRES"]
            refresh_token = create_refresh_token(
                identity=str(user.public_id),
                additional_claims={
                    "aud": "atmr-api",  # Audience claim pour sécurité
                    "pwd_hash": pwd_hash_version,  # Hash pour invalider après changement de mot de passe
                },
                expires_delta=refresh_expires_delta,
            )

            # ✅ PHASE 2: Stocker le refresh token dans Redis et DB
            try:
                # Stocker dans Redis pour rotation et limitation
                token_service = RefreshTokenService()
                token_service.store_token(user.id, refresh_token)

                # Stocker aussi dans la DB pour compatibilité et audit
                refresh_expires_at = datetime.now(UTC) + refresh_expires_delta
                store_refresh_token(
                    token=refresh_token,
                    user_id=user.id,
                    expires_at=refresh_expires_at,
                    device_id=request.headers.get("X-Device-ID"),
                    device_name=request.headers.get("X-Device-Name"),
                )

                # Limiter le nombre de tokens actifs (max 5 par défaut)
                max_active_tokens = int(os.getenv("MAX_ACTIVE_REFRESH_TOKENS", "5"))
                token_service.limit_active_tokens(user.id, max_active_tokens)
            except Exception as store_error:
                # Ne pas bloquer le login si le stockage échoue (fallback)
                logger.warning(
                    "Échec stockage refresh token: %s - %s",
                    type(store_error).__name__,
                    str(store_error),
                )
                # Le token sera toujours retourné au client, mais ne sera pas révocable

            # ✅ Priorité 7: Audit logging pour login réussi
            try:
                AuditLogger.log_action(
                    action_type="login_success",
                    action_category="security",
                    user_id=user.id,
                    user_type=user.role.value if user.role else "unknown",
                    result_status="success",
                    action_details={
                        "email": mask_email(email),
                        "username": user.username,
                        "role": user.role.value if user.role else None,
                    },
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
                # ✅ Priorité 7: Métrique Prometheus pour login réussi
                security_login_attempts_total.labels(type="success").inc()
            except Exception as audit_error:
                # Ne pas bloquer le login si l'audit logging échoue
                logger.warning("Échec audit logging login_success: %s", audit_error)

            # ✅ Migration localStorage → cookies httpOnly
            # ✅ P0: Ajouter trace_id dans la réponse
            trace_id = get_trace_id()
            logger.info(
                "Login success",
                extra={
                    "trace_id": trace_id,
                    "user_id": user.id,
                    "email": mask_email(email),
                },
            )

            # Créer la réponse JSON
            response_data = {
                "message": "Connexion réussie",
                "user": {
                    "id": user.id,
                    "public_id": user.public_id,
                    "username": user.username,
                    "email": user.email,
                    "role": user.role.value,
                    "force_password_change": user.force_password_change,
                },
                "trace_id": trace_id,
            }

            # ✅ Compatibilité mobile : retourner tokens en JSON (même modèle que company_mobile)
            # Toujours retourner les tokens dans le JSON pour les applications mobiles
            # Le header X-Requested-With: Expo est optionnel mais recommandé pour identifier les requêtes mobiles
            is_mobile_request = request.headers.get("X-Requested-With") == "Expo"
            # ✅ Même modèle que company_mobile : toujours retourner les tokens dans le JSON
            response_data["token"] = access_token
            response_data["refresh_token"] = refresh_token
            # #region agent log
            try:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "location": "auth.py:Login.post",
                                "message": "response_data before make_response",
                                "data": {
                                    "is_mobile_request": is_mobile_request,
                                    "has_token_in_response": "token" in response_data,
                                    "has_refresh_token_in_response": "refresh_token"
                                    in response_data,
                                    "response_data_keys": list(response_data.keys()),
                                },
                                "timestamp": datetime.now(UTC).isoformat(),
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "B",
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion

            # Créer la réponse avec make_response pour pouvoir définir les cookies
            response = make_response(response_data, 200)

            # ✅ Définir cookies httpOnly pour web (pas pour mobile)
            if not is_mobile_request:
                # Cookie access_token
                response.set_cookie(
                    current_app.config["COOKIE_ACCESS_TOKEN_NAME"],
                    access_token,
                    httponly=current_app.config["COOKIE_HTTP_ONLY"],
                    secure=current_app.config["COOKIE_SECURE"],
                    samesite=current_app.config["COOKIE_SAME_SITE"],
                    max_age=int(
                        current_app.config["JWT_ACCESS_TOKEN_EXPIRES"].total_seconds()
                    ),
                    path=current_app.config["COOKIE_PATH"],
                    domain=current_app.config["COOKIE_DOMAIN"],
                )

                # Cookie refresh_token
                response.set_cookie(
                    current_app.config["COOKIE_REFRESH_TOKEN_NAME"],
                    refresh_token,
                    httponly=current_app.config["COOKIE_HTTP_ONLY"],
                    secure=current_app.config["COOKIE_SECURE"],
                    samesite=current_app.config["COOKIE_SAME_SITE"],
                    max_age=int(
                        current_app.config["JWT_REFRESH_TOKEN_EXPIRES"].total_seconds()
                    ),
                    path=current_app.config["COOKIE_PATH"],
                    domain=current_app.config["COOKIE_DOMAIN"],
                )
                # #region agent log
                try:
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                {
                                    "location": "auth.py:Login.post",
                                    "message": "cookies set for web",
                                    "data": {
                                        "has_access_token_cookie": bool(access_token),
                                        "has_refresh_token_cookie": bool(refresh_token),
                                        "cookie_secure": current_app.config[
                                            "COOKIE_SECURE"
                                        ],
                                        "cookie_httponly": current_app.config[
                                            "COOKIE_HTTP_ONLY"
                                        ],
                                        "cookie_samesite": current_app.config[
                                            "COOKIE_SAME_SITE"
                                        ],
                                        "cookie_path": current_app.config[
                                            "COOKIE_PATH"
                                        ],
                                        "cookie_domain": current_app.config[
                                            "COOKIE_DOMAIN"
                                        ],
                                        "response_headers_set_cookie": "Set-Cookie"
                                        in dict(response.headers),
                                    },
                                    "timestamp": datetime.now(UTC).isoformat(),
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "F",
                                }
                            )
                            + "\n"
                        )
                except Exception:
                    pass
                # #endregion

            return response

        except Exception as e:
            # #region agent log
            try:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "location": "auth.py:Login.post",
                                "message": "EXCEPTION caught in login",
                                "data": {
                                    "exception_type": type(e).__name__,
                                    "exception_message": str(e),
                                    "exception_args": str(e.args)
                                    if hasattr(e, "args")
                                    else None,
                                },
                                "timestamp": datetime.now(UTC).isoformat(),
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "A,B,C,D,E",
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
            sentry_sdk.capture_exception(e)
            logger.error("❌ ERREUR login: %s - %s", type(e).__name__, str(e))
            # ✅ Priorité 7: Audit logging pour erreur interne login
            try:
                data = request.get_json() or {}
                AuditLogger.log_action(
                    action_type="login_error",
                    action_category="security",
                    user_type="unknown",
                    result_status="failure",
                    result_message=f"Erreur interne: {type(e).__name__}",
                    action_details={
                        "email": mask_email(data.get("email", "")) if data else ""
                    },
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
            except Exception:
                # Ignorer les erreurs d'audit logging
                # dans le gestionnaire d'erreurs
                pass
            return APIErrorHandler.handle_exception(
                Exception("Erreur lors de la connexion"),
                logger,
            )


# ========================
# Helper Functions pour Refresh Token
# ========================
def _get_password_hash_version(user: User) -> str:
    """Génère un hash de version basé sur le hash du mot de passe.

    Utilisé pour invalider automatiquement les refresh tokens
    après un changement de mot de passe sans stockage server-side.

    Args:
        user: L'utilisateur dont on veut obtenir la version du mot de passe

    Returns:
        Hash SHA256 des 16 premiers caractères du hash du mot de passe
    """
    password_hash = getattr(user, "password", "")
    if not password_hash:
        return ""
    # Prendre les premiers caractères du hash pour créer une version stable
    # Utiliser SHA256 pour créer un hash plus court et sécurisé
    hash_snippet = (
        str(password_hash)[:PASSWORD_HASH_VERSION_LENGTH]
        if len(str(password_hash)) > PASSWORD_HASH_VERSION_LENGTH
        else str(password_hash)
    )
    return hashlib.sha256(hash_snippet.encode()).hexdigest()[
        :PASSWORD_HASH_VERSION_LENGTH
    ]


def _check_user_profile_active(user: User) -> tuple[bool, str | None]:
    """Vérifie si le profil associé à l'utilisateur est actif.

    Args:
        user: L'utilisateur à vérifier

    Returns:
        Tuple (is_active, error_message):
        - Si actif: (True, None)
        - Si inactif: (False, "Compte désactivé")
        - Si pas de profil: (True, None) - on considère comme actif par défaut
    """
    if user.role == UserRole.driver and user.driver and not user.driver.is_active:
        return False, "Compte désactivé"

    if user.role == UserRole.client and user.clients:
        # Un utilisateur peut avoir plusieurs clients (1-N)
        # On vérifie qu'au moins un client est actif
        active_clients = [c for c in user.clients if c.is_active]
        if not active_clients:
            return False, "Compte désactivé"

    # Pour les autres rôles (admin, company) ou si pas de profil, on considère comme actif
    return True, None


def _validate_refresh_token(
    refresh_token: str,
) -> tuple[str | None, dict[str, str] | None]:
    """Valide un refresh token et retourne l'user_public_id ou une erreur.

    Vérifie aussi que le mot de passe n'a pas changé depuis l'émission du token
    et que le token n'est pas révoqué dans la DB.

    Args:
        refresh_token: Le token JWT à valider

    Returns:
        Tuple (user_public_id, error_response):
        - Si valide: (user_public_id, None)
        - Si invalide: (None, {error: "...", status_code: 401})
    """
    try:
        decoded = decode_token(refresh_token, allow_expired=False)
        user_public_id = decoded.get("sub")

        # Vérifier que c'est bien un refresh token (pas un access token)
        # Les refresh tokens n'ont pas les claims "role" ou "company_id"
        token_type = decoded.get("type", "")
        is_access_token = "role" in decoded or "company_id" in decoded

        if token_type != "refresh" and is_access_token:
            error_response, _ = APIErrorHandler.handle_validation_error(
                "Le token fourni n'est pas un refresh token",
                logger_instance=logger,
            )
            return None, error_response

        if not user_public_id:
            error_response, _ = APIErrorHandler.handle_validation_error(
                "Refresh token invalide (identity manquante)",
                logger_instance=logger,
            )
            return None, error_response

        # ✅ SECURITY: Vérifier que le mot de passe n'a pas changé
        # Récupérer l'utilisateur pour vérifier le hash du mot de passe
        user_dto = user_repo.find_by_public_id(user_public_id)
        if user_dto:
            # Récupérer le modèle User pour accéder aux méthodes de hash
            user = user_repo.find_model_by_public_id(user_public_id)
            if user:
                # Vérifier si le token a un claim pwd_hash (nouveaux tokens)
                token_pwd_hash = decoded.get("pwd_hash")
                if token_pwd_hash:
                    current_pwd_hash = _get_password_hash_version(user)
                    if token_pwd_hash != current_pwd_hash:
                        logger.warning(
                            "Refresh token rejeté : mot de passe modifié pour user %s",
                            user_public_id,
                        )
                        return None, {
                            "error": "Refresh token invalide (mot de passe modifié)"
                        }

        # ✅ SECURITY: Vérifier si le token est révoqué dans la DB (Phase 2)
        # Cette vérification permet la déconnexion forcée par l'admin
        try:
            if is_token_revoked(refresh_token):
                logger.warning(
                    "Refresh token rejeté : token révoqué pour user %s",
                    user_public_id,
                )
                error_response, _ = APIErrorHandler.handle_permission_error(
                    "Refresh token révoqué",
                    logger_instance=logger,
                )
                return None, error_response
        except Exception as revoke_check_error:
            # Si la vérification DB échoue, on continue quand même
            # (pour rétrocompatibilité avec les tokens non stockés)
            logger.debug(
                "Erreur vérification révocation token (ignorée): %s",
                str(revoke_check_error),
            )

        # ✅ SECURITY: Mettre à jour la date de dernière utilisation
        try:
            update_token_last_used(refresh_token)
        except Exception as update_error:
            # Ne pas bloquer le refresh si la mise à jour échoue
            logger.debug(
                "Erreur mise à jour last_used_at (ignorée): %s", str(update_error)
            )

        return user_public_id, None

    except Exception as decode_error:
        logger.warning("Erreur décodage refresh token: %s", str(decode_error))
        error_response, _ = APIErrorHandler.handle_validation_error(
            "Refresh token invalide ou expiré",
            logger_instance=logger,
        )
        return None, error_response


# ========================
# 2. Refresh Token
# ========================
# Modèle Swagger pour la requête de refresh token
refresh_token_request_model = auth_ns.model(
    "RefreshTokenRequest",
    {
        "refresh_token": fields.String(
            required=False,
            description="Refresh token JWT (optionnel si fourni dans Authorization header)",
        ),
    },
)

# Modèle Swagger pour la réponse de refresh token
refresh_token_response_model = auth_ns.model(
    "RefreshTokenResponse",
    {
        "access_token": fields.String(
            required=True, description="Nouveau token d'accès JWT"
        ),
        "refresh_token": fields.String(
            required=True, description="Nouveau refresh token (rotation automatique)"
        ),
        "user": fields.Nested(
            auth_ns.model(
                "RefreshTokenUserInfo",
                {
                    "public_id": fields.String(
                        required=True, description="ID public de l'utilisateur"
                    ),
                    "role": fields.String(
                        required=True, description="Rôle de l'utilisateur"
                    ),
                    "company_id": fields.Integer(
                        required=False, description="ID de la compagnie (si applicable)"
                    ),
                    "driver_id": fields.Integer(
                        required=False, description="ID du chauffeur (si applicable)"
                    ),
                },
            ),
            required=True,
            description="Informations minimales de l'utilisateur",
        ),
        "trace_id": fields.String(
            required=False,
            description="ID de traçage pour le support",
            example="a1b2c3d4",
        ),
    },
)


# ========================
# Login Test (sans CSRF pour tests de charge)
# ========================
@auth_ns.route("/login-test")
class LoginTest(Resource):
    """Endpoint de login pour tests de charge (dev/test uniquement, sans CSRF)."""

    @auth_ns.expect(login_model)
    @auth_ns.response(200, "Connexion réussie", login_success_model)
    @auth_ns.response(401, "Credentials invalides")
    @auth_ns.response(403, "Endpoint disponible uniquement en dev/test")
    def post(self):
        """Authentifie un utilisateur sans vérification CSRF (tests uniquement)."""
        import os

        from flask import abort, current_app, request
        from flask_jwt_extended import create_access_token, create_refresh_token

        from models.user import User

        # ⚠️ Sécurité : Disponible uniquement en environnement dev/test
        flask_env = os.getenv("FLASK_ENV", "production")
        if flask_env not in ["development", "testing"]:
            abort(
                403,
                description="Endpoint disponible uniquement en environnement dev/test",
            )

        # Récupérer les données
        data = request.get_json()
        if not data:
            return {"error": "Pas de données JSON fournies"}, 400

        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            return {"error": "Email et mot de passe requis"}, 400

        # Chercher l'utilisateur
        user = User.query.filter_by(email=email).first()

        if not user or not user.check_password(password):
            return {"error": "Identifiants invalides"}, 401

        # Note: Pas de vérification is_active car c'est un endpoint de test simplifié

        # Générer les tokens JWT
        access_token = create_access_token(identity=user.public_id)
        refresh_token = create_refresh_token(identity=user.public_id)

        # Retourner la réponse (format simplifié pour tests)
        return {
            "message": "Connexion réussie",
            "token": access_token,
            "access_token": access_token,  # Compatibilité
            "refresh_token": refresh_token,
            "user": {
                "id": user.id,
                "public_id": user.public_id,
                "username": user.username,
                "email": user.email,
                "role": user.role.value,
            },
        }, 200


@auth_ns.route("/refresh-token")
class RefreshToken(Resource):
    @auth_ns.expect(refresh_token_request_model)
    @auth_ns.response(200, "Token rafraîchi avec succès", refresh_token_response_model)
    @auth_ns.response(400, "Requête invalide", validation_error_model)
    @auth_ns.response(401, "Refresh token invalide ou expiré", permission_error_model)
    @auth_ns.response(403, "Compte désactivé", permission_error_model)
    @auth_ns.response(404, "Utilisateur non trouvé", not_found_error_model)
    @auth_ns.response(429, "Rate limit dépassé", api_error_model)
    @auth_ns.response(500, "Erreur interne", api_error_model)
    # ✅ S2: Rate limiting plus strict pour refresh token (protection contre abus)
    @limiter.limit("20 per minute")
    def post(self):  # noqa: PLR0911
        """Rafraîchit l'access token à partir d'un refresh token.

        Accepte le refresh_token en body (JSON) ou dans le header Authorization Bearer.

        ✅ Rotation automatique : Génère toujours un nouveau refresh_token et révoque l'ancien.
        Retourne un nouveau access_token, un nouveau refresh_token et les informations
        minimales de l'utilisateur.
        """
        try:
            # ✅ Migration localStorage → cookies httpOnly
            # 1. Récupérer le refresh_token depuis cookie (priorité), body ou header
            refresh_token = None
            is_mobile_request = request.headers.get("X-Requested-With") == "Expo"

            # Priorité 1 : Cookie (pour web)
            if not is_mobile_request:
                refresh_token = request.cookies.get(
                    current_app.config["COOKIE_REFRESH_TOKEN_NAME"]
                )

            # Priorité 2 : Body JSON (pour mobile ou fallback)
            if not refresh_token:
                data = request.get_json(silent=True) or {}
                refresh_token = data.get("refresh_token")

            # Priorité 3 : Header Authorization (rétrocompatibilité)
            if not refresh_token:
                auth_header = request.headers.get("Authorization", "")
                if auth_header and auth_header.startswith("Bearer "):
                    refresh_token = auth_header.split(" ", 1)[1].strip()

            # 3. Validation : refresh_token requis
            if not refresh_token:
                trace_id = get_trace_id()
                logger.warning(
                    "Refresh token missing",
                    extra={"trace_id": trace_id},
                )
                error_response, _ = APIErrorHandler.handle_validation_error(
                    "refresh_token requis (cookie, body ou Authorization header)",
                    logger_instance=logger,
                )
                error_response["trace_id"] = trace_id
                return error_response, 400

            # 4. Valider le refresh token (inclut vérification révocation, pwd_hash, etc.)
            user_public_id, error_response = _validate_refresh_token(refresh_token)
            if error_response or not user_public_id:
                trace_id = get_trace_id()
                logger.warning(
                    "Refresh token invalid",
                    extra={"trace_id": trace_id},
                )
                if error_response:
                    error_response["trace_id"] = trace_id
                    return error_response, 401
                error_response = {
                    "error": "Refresh token invalide",
                    "trace_id": trace_id,
                }
                return error_response, 401

            # 5. Vérifier que l'utilisateur existe
            user_dto = user_repo.find_by_public_id(user_public_id)
            if not user_dto:
                return APIErrorHandler.handle_not_found(
                    "Utilisateur",
                    user_public_id,
                    logger,
                )

            # ✅ SECURITY: Vérifier que le profil (Driver/Client) est actif
            # Récupérer le modèle User pour accéder aux méthodes de profil
            user = user_repo.find_model_by_public_id(user_public_id)
            if not user:
                return APIErrorHandler.handle_not_found(
                    "Utilisateur",
                    user_public_id,
                    logger,
                )
            is_active, error_message = _check_user_profile_active(user)
            if not is_active:
                logger.warning(
                    "Refresh token rejeté : compte désactivé pour user %s (role: %s)",
                    user_public_id,
                    user.role.value if user.role else "unknown",
                )
                return APIErrorHandler.handle_permission_error(
                    error_message or "Compte désactivé",
                    logger_instance=logger,
                )

            # ✅ SECURITY: Ajout claim 'aud' (audience) pour prévenir token replay
            claims = {
                "role": user.role.value,
                "company_id": getattr(user, "company_id", None),
                "driver_id": getattr(user, "driver_id", None),
                "aud": "atmr-api",  # Audience claim pour sécurité
            }

            # 6. Générer nouveau access_token
            new_access_token = create_access_token(
                identity=str(user.public_id),
                additional_claims=claims,
                expires_delta=current_app.config["JWT_ACCESS_TOKEN_EXPIRES"],
            )

            # 7. ✅ ROTATION AUTOMATIQUE : Générer toujours un nouveau refresh_token
            # Utiliser le modèle User déjà récupéré pour accéder aux méthodes de hash
            pwd_hash_version = _get_password_hash_version(user)
            refresh_expires_delta = current_app.config["JWT_REFRESH_TOKEN_EXPIRES"]
            new_refresh_token = create_refresh_token(
                identity=str(user.public_id),
                additional_claims={
                    "aud": "atmr-api",
                    "pwd_hash": pwd_hash_version,
                },
                expires_delta=refresh_expires_delta,
            )

            # ✅ PHASE 2: Utiliser RefreshTokenService pour rotation et limitation
            token_service = RefreshTokenService()

            # ✅ SECURITY: Révoquer l'ancien token via Redis (rotation automatique)
            try:
                token_service.revoke_token(refresh_token)
                # Révoquer aussi dans la DB pour compatibilité
                revoke_refresh_token(
                    refresh_token, reason="Rotation automatique du token"
                )

                # ✅ PHASE 3: Métrique Prometheus pour rotation
                try:
                    from security.security_metrics import tokens_rotation_total

                    tokens_rotation_total.inc()
                except Exception:
                    pass  # Ne pas bloquer si métriques indisponibles
            except Exception as revoke_error:
                # Ne pas bloquer la rotation si la révocation échoue
                logger.warning(
                    "Échec révocation ancien token lors rotation automatique: %s",
                    str(revoke_error),
                )

            # ✅ SECURITY: Stocker le nouveau token dans Redis et DB
            try:
                # Stocker dans Redis pour rotation et limitation
                token_service.store_token(user.id, new_refresh_token)

                # Stocker aussi dans la DB pour compatibilité et audit
                refresh_expires_at = datetime.now(UTC) + refresh_expires_delta
                store_refresh_token(
                    token=new_refresh_token,
                    user_id=user.id,
                    expires_at=refresh_expires_at,
                    device_id=request.headers.get("X-Device-ID"),
                    device_name=request.headers.get("X-Device-Name"),
                )

                # Limiter le nombre de tokens actifs (max 5 par défaut)
                max_active_tokens = int(os.getenv("MAX_ACTIVE_REFRESH_TOKENS", "5"))
                token_service.limit_active_tokens(user.id, max_active_tokens)
            except Exception as store_error:
                # Ne pas bloquer la rotation si le stockage échoue
                logger.warning(
                    "Échec stockage nouveau refresh token lors rotation automatique: %s",
                    str(store_error),
                )

            # 8. ✅ Priorité 7: Audit logging pour token refresh
            try:
                AuditLogger.log_action(
                    action_type="token_refresh",
                    action_category="security",
                    user_id=user.id,
                    user_type=user.role.value if user.role else "unknown",
                    result_status="success",
                    action_details={
                        "rotation_automatic": True,
                        "token_source": (
                            "cookie"
                            if not is_mobile_request
                            and request.cookies.get(
                                current_app.config["COOKIE_REFRESH_TOKEN_NAME"]
                            )
                            else "body"
                            if (request.get_json() or {}).get("refresh_token")
                            else "header"
                        ),
                    },
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
                # ✅ Priorité 7: Métrique Prometheus pour token refresh
                security_token_refreshes_total.inc()
            except Exception as audit_error:
                logger.warning("Échec audit logging token_refresh: %s", audit_error)

            # 9. ✅ Migration localStorage → cookies httpOnly
            # ✅ P0: Ajouter trace_id dans la réponse
            trace_id = get_trace_id()
            logger.info(
                "Token refresh success",
                extra={"trace_id": trace_id, "user_id": user.id},
            )

            # Construire la réponse JSON
            response_data = {
                "user": {
                    "public_id": user.public_id,
                    "role": user.role.value,
                    "company_id": getattr(user, "company_id", None),
                    "driver_id": getattr(user, "driver_id", None),
                },
                "trace_id": trace_id,
            }

            # ✅ Compatibilité mobile : retourner tokens en JSON si header X-Requested-With: Expo
            if is_mobile_request:
                response_data["access_token"] = new_access_token
                response_data["refresh_token"] = new_refresh_token

            # Créer la réponse avec make_response pour pouvoir définir les cookies
            response = make_response(response_data, 200)

            # ✅ Définir cookies httpOnly pour web (pas pour mobile)
            if not is_mobile_request:
                # Cookie access_token
                response.set_cookie(
                    current_app.config["COOKIE_ACCESS_TOKEN_NAME"],
                    new_access_token,
                    httponly=current_app.config["COOKIE_HTTP_ONLY"],
                    secure=current_app.config["COOKIE_SECURE"],
                    samesite=current_app.config["COOKIE_SAME_SITE"],
                    max_age=int(
                        current_app.config["JWT_ACCESS_TOKEN_EXPIRES"].total_seconds()
                    ),
                    path=current_app.config["COOKIE_PATH"],
                    domain=current_app.config["COOKIE_DOMAIN"],
                )

                # Cookie refresh_token (rotation automatique)
                response.set_cookie(
                    current_app.config["COOKIE_REFRESH_TOKEN_NAME"],
                    new_refresh_token,
                    httponly=current_app.config["COOKIE_HTTP_ONLY"],
                    secure=current_app.config["COOKIE_SECURE"],
                    samesite=current_app.config["COOKIE_SAME_SITE"],
                    max_age=int(
                        current_app.config["JWT_REFRESH_TOKEN_EXPIRES"].total_seconds()
                    ),
                    path=current_app.config["COOKIE_PATH"],
                    domain=current_app.config["COOKIE_DOMAIN"],
                )

            return response

        except Exception as e:
            sentry_sdk.capture_exception(e)
            return APIErrorHandler.handle_exception(e, logger)


# ========================
# 3. Obtenir un token "fresh"
# ========================
@auth_ns.route("/fresh-token")
class FreshToken(Resource):
    @auth_ns.expect(fresh_token_request_model)
    @auth_ns.response(200, "Token fresh obtenu avec succès", fresh_token_response_model)
    @auth_ns.response(401, "Mot de passe incorrect ou utilisateur non authentifié")
    @jwt_required()  # Nécessite un token valide (mais pas fresh)
    @limiter.limit("5 per minute")  # Protection contre brute force
    def post(self):  # noqa: PLR0911
        """Obtient un token 'fresh' en vérifiant le mot de passe de l'utilisateur.

        Permet d'obtenir un token 'fresh' sans se déconnecter complètement.
        Utile pour effectuer des actions sensibles qui nécessitent un token fresh.
        """
        try:
            # 1. Récupérer l'utilisateur actuel
            user_public_id = get_jwt_identity()
            if not user_public_id:
                return {"error": "Utilisateur non authentifié"}, 401

            user_dto = user_repo.find_by_public_id(user_public_id)
            if not user_dto or not user_dto.email:
                return {"error": "Utilisateur non trouvé"}, 401

            user = user_repo.find_model_by_email(user_dto.email)
            if not user:
                return {"error": "Utilisateur non trouvé"}, 401

            # 2. Récupérer le mot de passe depuis la requête
            data = request.get_json(silent=True) or {}
            password = data.get("password")
            if not password:
                return {"error": "Mot de passe requis"}, 400

            # 3. Vérifier le mot de passe
            if not user.check_password(password):
                logger.warning(
                    "[Auth] Échec vérification mot de passe pour fresh token (user: %s)",
                    user_public_id,
                )
                return {"error": "Mot de passe incorrect"}, 401

            # 4. Créer un token fresh
            claims = {
                "role": user.role.value
                if hasattr(user.role, "value")
                else str(user.role),
                "company_id": getattr(user, "company_id", None),
                "driver_id": getattr(user, "driver_id", None),
                "aud": "atmr-api",
            }
            fresh_token = create_access_token(
                identity=str(user.public_id),
                additional_claims=claims,
                expires_delta=current_app.config["JWT_ACCESS_TOKEN_EXPIRES"],
                fresh=True,  # ✅ Token fresh
            )

            logger.info(
                "[Auth] Token fresh généré pour user %s (public_id: %s)",
                user.id,
                user_public_id,
            )

            # 5. Retourner le token
            response_data = {
                "access_token": fresh_token,
                "message": "Token fresh obtenu avec succès",
            }

            # Si on utilise des cookies, mettre à jour le cookie
            is_mobile_request = request.headers.get("X-Requested-With") == "Expo"
            if not is_mobile_request:
                response = make_response(response_data)
                response.set_cookie(
                    current_app.config["COOKIE_ACCESS_TOKEN_NAME"],
                    fresh_token,
                    httponly=current_app.config["COOKIE_HTTP_ONLY"],
                    secure=current_app.config["COOKIE_SECURE"],
                    samesite=current_app.config["COOKIE_SAME_SITE"],
                    max_age=int(
                        current_app.config["JWT_ACCESS_TOKEN_EXPIRES"].total_seconds()
                    ),
                    path=current_app.config["COOKIE_PATH"],
                    domain=current_app.config["COOKIE_DOMAIN"],
                )
                return response

            return response_data, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error("[Auth] Erreur lors de la génération du token fresh: %s", e)
            return APIErrorHandler.handle_exception(e, logger)


# ========================
# 4. Logout / Révoquer Token
# ========================
@auth_ns.route("/logout")
class Logout(Resource):
    @auth_ns.doc(
        description=(
            "Révoque le token JWT actuel et l'ajoute à la blacklist. "
            "Après la déconnexion, le token ne pourra plus être utilisé "
            "pour accéder aux endpoints protégés."
        ),
        summary="Déconnexion utilisateur",
    )
    @auth_ns.response(200, "Déconnexion réussie", logout_response_model)
    @auth_ns.response(401, "Token manquant ou invalide")
    @auth_ns.response(500, "Erreur lors de la révocation du token", logout_error_model)
    @jwt_required()
    def post(self):
        """Révoque le token JWT actuel (logout)."""
        try:
            from security.token_blacklist import revoke_token

            # ✅ Priorité 7: Récupérer user_id pour audit logging
            current_user_id = get_jwt_identity()
            user = None
            if current_user_id:
                user = user_repo.find_by_public_id(current_user_id)

            # ✅ Migration localStorage → cookies httpOnly
            # Récupérer le refresh token depuis cookie (priorité), body ou header
            refresh_token = None
            is_mobile_request = request.headers.get("X-Requested-With") == "Expo"

            # Priorité 1 : Cookie (pour web)
            if not is_mobile_request:
                refresh_token = request.cookies.get(
                    current_app.config["COOKIE_REFRESH_TOKEN_NAME"]
                )

            # Priorité 2 : Body JSON (pour mobile ou fallback)
            if not refresh_token:
                data = request.get_json(silent=True) or {}
                refresh_token = data.get("refresh_token")

            # Priorité 3 : Header Authorization (rétrocompatibilité)
            if not refresh_token:
                auth_header = request.headers.get("Authorization", "")
                if auth_header and auth_header.startswith("Bearer "):
                    refresh_token = auth_header.split(" ", 1)[1].strip()

            # ✅ PHASE 2: Révoquer tous les tokens via RefreshTokenService
            token_service = RefreshTokenService()
            try:
                if user:
                    # Révoquer tous les tokens de l'utilisateur dans Redis
                    token_service.revoke_all_user_tokens(user.id)

                    # Révoquer aussi dans la DB pour compatibilité
                    revoke_all_user_tokens(user.id, reason="Logout utilisateur")

                # Si un refresh token spécifique est présent, le révoquer aussi
                if refresh_token:
                    token_service.revoke_token(refresh_token)
                    revoke_refresh_token(refresh_token, reason="Logout utilisateur")
                    logger.debug(
                        "Refresh token révoqué lors du logout pour user %s",
                        current_user_id,
                    )
            except Exception as revoke_error:
                # Ne pas bloquer le logout si la révocation échoue
                logger.warning(
                    "Échec révocation refresh tokens lors logout (ignoré): %s",
                    str(revoke_error),
                )

            # ✅ PHASE 3: Révoquer l'access token actuel (optionnel mais recommandé)
            try:
                from datetime import UTC, datetime

                from services.security.authentication import AccessTokenService

                access_token_service = AccessTokenService()
                jwt_claims = get_jwt()

                # Obtenir le jti (JWT ID) et l'expiration
                token_jti = jwt_claims.get("jti")
                exp = jwt_claims.get("exp")

                if token_jti and exp:
                    # Calculer le temps restant avant expiration
                    now = datetime.now(UTC).timestamp()
                    ttl = int(exp - now)

                    if ttl > 0:
                        access_token_service.revoke_token(token_jti, ttl)
                        logger.debug(
                            "Access token révoqué lors du logout pour user %s, jti=%s",
                            current_user_id,
                            token_jti,
                        )
            except Exception as access_revoke_error:
                # Ne pas bloquer le logout si la révocation de l'access token échoue
                logger.warning(
                    "Échec révocation access token lors logout (ignoré): %s",
                    str(access_revoke_error),
                )

            if revoke_token():
                # ✅ S3: Métrique Prometheus pour invalidation de token
                try:
                    from security.security_metrics import (
                        security_token_invalidations_total,
                    )

                    security_token_invalidations_total.labels(reason="logout").inc()
                except Exception:
                    pass  # Ne pas bloquer si métriques indisponibles

                # ✅ Priorité 7: Audit logging pour logout réussi
                try:
                    AuditLogger.log_action(
                        action_type="logout",
                        action_category="security",
                        user_id=user.id if user else None,
                        user_type=user.role.value if user and user.role else "unknown",
                        result_status="success",
                        action_details={
                            "refresh_token_revoked": refresh_token is not None,
                        },
                        ip_address=request.remote_addr,
                        user_agent=request.headers.get("User-Agent"),
                    )
                    # ✅ Priorité 7: Métrique Prometheus pour logout
                    security_logout_total.inc()
                except Exception as audit_error:
                    logger.warning("Échec audit logging logout: %s", audit_error)

                # ✅ Migration localStorage → cookies httpOnly
                # Créer la réponse avec make_response pour pouvoir supprimer les cookies
                response = make_response({"message": "Déconnexion réussie"}, 200)

                # Supprimer les cookies (web uniquement)
                if not is_mobile_request:
                    response.set_cookie(
                        current_app.config["COOKIE_ACCESS_TOKEN_NAME"],
                        "",
                        expires=0,
                        path=current_app.config["COOKIE_PATH"],
                        domain=current_app.config["COOKIE_DOMAIN"],
                    )
                    response.set_cookie(
                        current_app.config["COOKIE_REFRESH_TOKEN_NAME"],
                        "",
                        expires=0,
                        path=current_app.config["COOKIE_PATH"],
                        domain=current_app.config["COOKIE_DOMAIN"],
                    )

                return response
            return APIErrorHandler.handle_exception(
                Exception("Impossible de révoquer le token"),
                logger,
            )

        except Exception as e:
            sentry_sdk.capture_exception(e)
            return APIErrorHandler.handle_exception(e, logger)


# ========================
# 4. Informations Utilisateur
# ========================
@auth_ns.route("/me")
class UserInfo(Resource):
    @jwt_required()
    def get(self):
        """Retourne les informations de l'utilisateur connecté."""
        try:
            # ✅ DDD: Utiliser le use case pour récupérer l'utilisateur courant
            uc = GetCurrentUserUseCase()
            result = uc.execute()

            if not result.found:
                return result.error, result.status_code or 404

            if result.user is None:
                return APIErrorHandler.handle_not_found(
                    "User",
                    None,
                    logger,
                )

            user = result.user
            return {
                "id": user.id,
                "public_id": user.public_id,
                "username": user.username,
                "email": user.email,
                "role": user.role.value,
            }, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            return APIErrorHandler.handle_exception(e, logger)


# ========================
# 5. Inscription
# ========================
@auth_ns.route("/register")
class Register(Resource):
    @auth_ns.expect(register_model, validate=True)
    @limiter.limit(
        "10 per minute"
    )  # ✅ SECURITY: Rate limiting pour prévenir spam d'inscriptions
    def post(self):
        """Inscrit un nouvel utilisateur avec le rôle 'client' par défaut
        et crée un profil client associé.
        """
        # Gérer la requête OPTIONS pour CORS si nécessaire
        if request.method == "OPTIONS":
            response = make_response("")
            response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Headers"] = (
                "Content-Type, Authorization"
            )
            response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
            return response, 204

        try:
            data = request.get_json() or {}
            logger.info("Données reçues dans /auth/register : %s", data)

            # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
            try:
                validated_data = validate_request(RegisterSchema(), data, strict=False)
            except ValidationError as e:
                # Utiliser abort au lieu de return pour réduire le nombre de returns
                body, code = handle_validation_error(e)
                auth_ns.abort(code or 400, body.get("error", "Validation error"))
                validated_data = {}  # Never reached, but satisfies type checker

            logger.info("Données validées : %s", validated_data)

            # ✅ DDD: Utiliser le use case pour enregistrer l'utilisateur
            username: str = cast("str", validated_data.get("username"))
            password: str = cast("str", validated_data.get("password"))
            email: str = cast("str", validated_data.get("email"))

            # ✅ S3: Validation explicite du mot de passe avec politique renforcée
            from security.password_policy import (
                PasswordPolicyError,
                PasswordPolicyService,
            )

            try:
                # Valider avec la politique stricte (complexité + HIBP)
                PasswordPolicyService.validate_password(
                    password, user_id=None, check_history=False
                )
            except PasswordPolicyError as e:
                auth_ns.abort(400, e.message)

            uc = RegisterUserUseCase()
            input_data = RegisterUserInput(
                username=username,
                email=email,
                password=password,
                first_name=validated_data.get("first_name"),
                last_name=validated_data.get("last_name"),
                phone=validated_data.get("phone"),
                address=validated_data.get("address"),
                birth_date=validated_data.get("birth_date"),
                gender=validated_data.get("gender"),
                profile_image=validated_data.get("profile_image"),
            )
            register_result = uc.execute(input_data)

            if not register_result.success:
                HTTP_BAD_REQUEST = 400
                if register_result.status_code == HTTP_BAD_REQUEST:
                    error_msg = (
                        register_result.error.get("error", "Validation error")
                        if register_result.error
                        else "Validation error"
                    )
                    auth_ns.abort(HTTP_BAD_REQUEST, error_msg)
                error_msg = (
                    register_result.error.get("error", "Registration error")
                    if register_result.error
                    else "Registration error"
                )
                auth_ns.abort(register_result.status_code or 500, error_msg)

            user = register_result.user
            if not user:
                auth_ns.abort(500, "User created but not returned")

            # Création du profil client associé
            # user est garanti non-None après la vérification ci-dessus
            assert user is not None  # Pour le type checker
            client = Client()
            client.user_id = user.id
            client.is_active = True
            client.contact_email = email
            db.session.add(client)
            db.session.commit()
            logger.info("Client créé : user_id=%s, client_id=%s", user.id, client.id)

            logger.info("Utilisateur et client enregistrés avec succès : %s", user.id)
            return {
                "message": "User registered successfully!",
                "user_id": user.public_id,
                "username": user.username,
            }, 201

        except ValidationError as e:
            logger.error("Erreur de validation : %s", e.messages)
            auth_ns.abort(400, "Validation failed")
        except Exception as e:
            # Flask-RESTX abort() lève une HTTPException (ex: 400/409).
            # Ne pas transformer ces erreurs attendues en 500.
            from werkzeug.exceptions import (  # pyright: ignore[reportMissingImports]
                HTTPException,
            )

            if isinstance(e, HTTPException):
                raise
            sentry_sdk.capture_exception(e)
            # Utiliser repr() pour éviter les problèmes de formatage avec %
            exception_message = repr(e) if "%" in str(e) else str(e)
            logger.exception(
                "❌ ERREUR register_user: %s - %s", type(e).__name__, exception_message
            )
            auth_ns.abort(500, "Une erreur interne est survenue.")


# ========================
# 5. Mot de Passe Oublié
# ========================
@auth_ns.route("/forgot-password")
class ForgotPassword(Resource):
    # ✅ S2: Rate limiting strict pour forgot-password (protection brute force)
    @limiter.limit("3 per minute")
    def post(self):
        """Envoie un email de réinitialisation de mot de passe."""
        try:
            data = request.get_json() or {}
            email = data.get("email")
            if not email:
                return APIErrorHandler.handle_validation_error(
                    "Email is required",
                    field="email",
                    logger_instance=logger,
                )

            user = user_repo.find_by_email(email)
            if not user:
                return APIErrorHandler.handle_not_found(
                    "Account",
                    email,
                    logger,
                )

            # Accéder explicitement à la configuration via current_app
            secret_key = current_app.config.get("SECRET_KEY")
            if not secret_key:
                return APIErrorHandler.handle_exception(
                    Exception("Configuration error: SECRET_KEY not set"),
                    logger,
                )

            serializer = URLSafeTimedSerializer(secret_key)
            reset_token = serializer.dumps(user.email, salt="password-reset-salt")

            msg = Message(
                subject="Réinitialisation de votre mot de passe",
                recipients=[email],
                body=(
                    f"Cliquez sur ce lien pour réinitialiser votre mot de passe : "
                    f"http://localhost:3000/reset-password/{reset_token}"
                ),
            )
            mail.send(msg)
            return {"message": "Password reset email sent successfully"}, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            return APIErrorHandler.handle_exception(e, logger)


# ========================
# 6. Réinitialisation via Lien
# ========================


@auth_ns.route("/reset-password/<string:public_id>")
class ResetPassword(Resource):
    # ✅ S2: Rate limiting strict pour reset-password (protection brute force)
    @limiter.limit("3 per minute")
    def post(self, public_id):
        """Réinitialise le mot de passe via un lien contenant le public_id."""
        try:
            data = request.get_json() or {}
            new_password = data.get("new_password")
            if not new_password:
                return APIErrorHandler.handle_validation_error(
                    "Un nouveau mot de passe est requis.",
                    field="new_password",
                    logger_instance=logger,
                )

            user_dto = user_repo.find_by_public_id(public_id)
            if not user_dto:
                return APIErrorHandler.handle_not_found(
                    "Utilisateur",
                    public_id,
                    logger,
                )

            # Obtenir le modèle User pour les opérations de modification
            user = User.query.filter(User.public_id == public_id).first()
            if not user:
                return APIErrorHandler.handle_not_found(
                    "Utilisateur",
                    public_id,
                    logger,
                )

            # ✅ S3: Validation avec politique renforcée (complexité + HIBP + historique)
            from security.password_policy import (
                PasswordPolicyError,
                PasswordPolicyService,
            )

            try:
                PasswordPolicyService.validate_password(
                    new_password, user_id=user.id, check_history=True
                )
            except PasswordPolicyError as e:
                return APIErrorHandler.handle_validation_error(
                    e.message,
                    field="new_password",
                    logger_instance=logger,
                )

            # Le mot de passe est validé explicitement par validate_password()
            # avant set_password() - satisfait les exigences de sécurité
            user.set_password(new_password)  # nosem
            user.force_password_change = False

            # ✅ S3: Révoquer tous les tokens lors du changement de mot de passe
            try:
                from security.security_metrics import (
                    security_token_invalidations_total,
                )
                # revoke_all_user_tokens est déjà importé depuis security.refresh_token_service

                revoke_all_user_tokens(user.id, reason="Changement de mot de passe")
                security_token_invalidations_total.labels(
                    reason="password_change"
                ).inc()
            except Exception as revoke_error:
                # Ne pas bloquer le changement de mot de passe si la révocation échoue
                logger.warning(
                    "Échec révocation tokens lors changement mot de passe (ignoré): %s",
                    str(revoke_error),
                )

            db.session.commit()
            return {"message": "Mot de passe réinitialisé avec succès."}, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            return APIErrorHandler.handle_exception(e, logger)


# ========================
# Endpoint admin : Révoquer toutes les sessions d'un utilisateur
# ========================
@auth_ns.route("/revoke-all-sessions/<int:user_id>")
class RevokeAllSessions(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    @auth_ns.response(
        200, "Sessions révoquées avec succès", revoke_sessions_response_model
    )
    @auth_ns.response(401, "Non autorisé")
    @auth_ns.response(403, "Accès refusé (admin uniquement)")
    @auth_ns.response(404, "Utilisateur non trouvé")
    @auth_ns.response(500, "Erreur interne")
    # ✅ S2: Rate limiting strict pour revoke-all-sessions (action sensible)
    @limiter.limit("10 per hour")
    def post(self, user_id: int):
        """Révoque toutes les sessions actives d'un utilisateur (admin uniquement).

        Cette action déconnecte l'utilisateur de tous ses appareils en révoquant
        tous ses refresh tokens actifs dans la base de données.
        """
        try:
            # 1. Vérifier que l'utilisateur existe
            user = user_repo.find_by_id(user_id)
            if not user:
                return APIErrorHandler.handle_not_found(
                    "Utilisateur",
                    user_id,
                    logger,
                )

            # 2. Récupérer l'admin qui effectue l'action (pour audit logging)
            admin_public_id = get_jwt_identity()
            admin_user = user_repo.find_by_public_id(admin_public_id)

            # 3. Révoquer tous les tokens de l'utilisateur
            # ✅ S3: Métrique Prometheus pour invalidation de tokens
            try:
                from security.security_metrics import (
                    security_token_invalidations_total,
                )

                security_token_invalidations_total.labels(reason="admin_revoke").inc()
            except Exception:
                pass  # Ne pas bloquer si métriques indisponibles

            count = revoke_all_user_tokens(
                user_id=user_id, reason="Révoqué par l'admin"
            )

            # 4. Audit logging
            try:
                AuditLogger.log_action(
                    action_type="revoke_all_sessions",
                    action_category="security",
                    user_id=admin_user.id if admin_user else None,
                    user_type=admin_user.role.value
                    if admin_user and admin_user.role
                    else "admin",
                    result_status="success",
                    action_details={
                        "target_user_id": user_id,
                        "target_username": user.username,
                        "target_email": mask_email(user.email) if user.email else None,
                        "sessions_revoked": count,
                    },
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
            except Exception as audit_error:
                # Ne pas bloquer l'action si l'audit logging échoue
                logger.warning(
                    "Échec audit logging revoke_all_sessions: %s", audit_error
                )

            # 5. Retourner la réponse
            logger.info(
                "Admin %s a révoqué %d session(s) pour user_id=%d",
                admin_public_id,
                count,
                user_id,
            )
            return {
                "message": f"{count} session(s) révoquée(s)",
                "sessions_revoked": count,
            }, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            return APIErrorHandler.handle_exception(e, logger)


# ========================
# Endpoint : Lister les sessions actives de l'utilisateur
# ========================
@auth_ns.route("/sessions")
class ListSessions(Resource):
    @jwt_required()
    @auth_ns.response(
        200, "Sessions récupérées avec succès", list_sessions_response_model
    )
    @auth_ns.response(401, "Non autorisé")
    @auth_ns.response(404, "Utilisateur non trouvé")
    @auth_ns.response(500, "Erreur interne")
    # ✅ S2: Rate limiting pour sessions (endpoint utilisateur)
    @limiter.limit("50 per hour")
    def get(self):
        """Liste les sessions actives de l'utilisateur connecté.

        Retourne toutes les sessions (refresh tokens) actives de l'utilisateur
        actuellement connecté, incluant les informations sur les appareils et
        les dates de création/expiration/dernière utilisation.
        """
        try:
            # 1. Récupérer l'utilisateur connecté
            current_user_public_id = get_jwt_identity()
            user = user_repo.find_by_public_id(current_user_public_id)

            if not user:
                return APIErrorHandler.handle_not_found(
                    "Utilisateur",
                    current_user_public_id,
                    logger,
                )

            # 2. Récupérer les sessions actives
            sessions = get_user_active_sessions(user.id)

            # 3. Sérialiser les sessions
            sessions_data = [session.serialize() for session in sessions]

            # 4. Retourner la réponse
            return {
                "sessions": sessions_data,
                "count": len(sessions_data),
            }, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            return APIErrorHandler.handle_exception(e, logger)


# ========================
# Endpoint : Obtenir un token CSRF
# ========================
@auth_ns.route("/csrf-token")
@auth_ns.doc(
    security=None,  # Endpoint public (pas besoin d'authentification)
    description="Récupère un token CSRF pour protéger les requêtes mutantes",
)
class CSRFTokenResource(Resource):
    """Endpoint pour obtenir un token CSRF."""

    @auth_ns.marshal_with(csrf_token_response_model)
    @auth_ns.response(200, "Token CSRF généré avec succès")
    @auth_ns.response(500, "Erreur interne")
    # ✅ S2: Rate limiting plus strict pour endpoint CSRF (protection contre abus)
    @limiter.limit("50 per hour")
    def get(self):
        """Génère et retourne un token CSRF.

        Le token doit être inclus dans toutes les requêtes mutantes (POST/PUT/PATCH/DELETE)
        via le header `X-CSRF-Token` ou dans le body JSON avec la clé `csrf_token`.

        Returns:
            Dict avec le token CSRF et sa durée de vie
        """
        import os

        try:
            # Récupérer user_id depuis JWT si disponible (optionnel)
            user_id = None
            try:
                jwt_identity = get_jwt_identity()
                if jwt_identity:
                    if isinstance(jwt_identity, dict):
                        # Extraire user_id depuis les claims JWT
                        from flask_jwt_extended import (  # pyright: ignore[reportMissingImports]
                            get_jwt,
                        )

                        jwt_claims = get_jwt()
                        user_id = jwt_claims.get("user_id") or jwt_claims.get("id")
                        if not user_id:
                            user_id = jwt_identity.get("user_id") or jwt_identity.get(
                                "id"
                            )
                    elif isinstance(jwt_identity, (int, str)):
                        with suppress(ValueError, TypeError):
                            user_id = int(jwt_identity)
            except Exception:
                # JWT non disponible, générer token sans user_id
                pass

            # Générer le token CSRF
            csrf_token = generate_csrf_token(user_id=user_id)

            # TTL du token (en secondes)
            csrf_ttl = int(os.getenv("CSRF_TOKEN_TTL_SECONDS", "3600"))

            logger.debug("[CSRF] Token généré pour user_id=%s", user_id)

            return {
                "csrf_token": csrf_token,
                "ttl": csrf_ttl,
            }, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            return APIErrorHandler.handle_exception(e, logger)
