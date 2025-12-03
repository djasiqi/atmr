import hashlib
import logging
from datetime import datetime, timezone
from typing import cast

import sentry_sdk  # CORRECTION : Importer directement
from flask import current_app, make_response, request
from flask_jwt_extended import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_jwt_identity,
    jwt_required,
)
from flask_mail import Message
from flask_restx import Namespace, Resource, fields
from itsdangerous import URLSafeTimedSerializer
from marshmallow import Schema, ValidationError
from marshmallow import fields as ma_fields

from ext import db, limiter, mail, role_required
from models import Client, User, UserRole
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
from shared.logging_utils import mask_email

app_logger = logging.getLogger("app")

auth_ns = Namespace("auth", description="Opérations liées à l'authentification")

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
    # Limite d'appels pour éviter le brute force
    @limiter.limit("5 per minute")
    def post(self):
        """Authentifie un utilisateur et renvoie un token d'accès."""
        try:
            data = request.get_json() or {}

            # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
            try:
                validated_data = validate_request(LoginSchema(), data)
            except ValidationError as e:
                return handle_validation_error(e)

            email = validated_data["email"]
            password = validated_data["password"]

            user = User.query.filter_by(email=email).first()
            if not user or not user.check_password(password):
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
                    app_logger.warning(
                        "Échec audit logging login_failed: %s", audit_error
                    )
                return {"error": "Email ou mot de passe invalide."}, 401

            # Création du token avec le rôle dans additional_claims
            # ✅ SECURITY: Ajout claim 'aud' (audience) pour prévenir token replay
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
            )

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

            # ✅ SECURITY: Stocker le refresh token dans la DB pour permettre la révocation
            try:
                refresh_expires_at = datetime.now(timezone.utc) + refresh_expires_delta
                store_refresh_token(
                    token=refresh_token,
                    user_id=user.id,
                    expires_at=refresh_expires_at,
                    device_id=request.headers.get("X-Device-ID"),
                    device_name=request.headers.get("X-Device-Name"),
                )
            except Exception as store_error:
                # Ne pas bloquer le login si le stockage échoue (fallback)
                app_logger.warning(
                    "Échec stockage refresh token dans DB: %s - %s",
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
                app_logger.warning("Échec audit logging login_success: %s", audit_error)

            return {
                "message": "Connexion réussie",
                "token": access_token,
                "refresh_token": refresh_token,
                "user": {
                    "id": user.id,
                    "public_id": user.public_id,
                    "username": user.username,
                    "email": user.email,
                    "role": user.role.value,
                    "force_password_change": user.force_password_change,
                },
            }, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error("❌ ERREUR login: %s - %s", type(e).__name__, str(e))
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
            return {"error": "Une erreur interne est survenue."}, 500


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
            return None, {"error": "Le token fourni n'est pas un refresh token"}

        if not user_public_id:
            return None, {"error": "Refresh token invalide (identity manquante)"}

        # ✅ SECURITY: Vérifier que le mot de passe n'a pas changé
        # Récupérer l'utilisateur pour vérifier le hash du mot de passe
        user = User.query.filter_by(public_id=user_public_id).first()
        if user:
            # Vérifier si le token a un claim pwd_hash (nouveaux tokens)
            token_pwd_hash = decoded.get("pwd_hash")
            if token_pwd_hash:
                current_pwd_hash = _get_password_hash_version(user)
                if token_pwd_hash != current_pwd_hash:
                    app_logger.warning(
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
                app_logger.warning(
                    "Refresh token rejeté : token révoqué pour user %s",
                    user_public_id,
                )
                return None, {"error": "Refresh token révoqué"}
        except Exception as revoke_check_error:
            # Si la vérification DB échoue, on continue quand même
            # (pour rétrocompatibilité avec les tokens non stockés)
            app_logger.debug(
                "Erreur vérification révocation token (ignorée): %s",
                str(revoke_check_error),
            )

        # ✅ SECURITY: Mettre à jour la date de dernière utilisation
        try:
            update_token_last_used(refresh_token)
        except Exception as update_error:
            # Ne pas bloquer le refresh si la mise à jour échoue
            app_logger.debug(
                "Erreur mise à jour last_used_at (ignorée): %s", str(update_error)
            )

        return user_public_id, None

    except Exception as decode_error:
        app_logger.warning("Erreur décodage refresh token: %s", str(decode_error))
        return None, {"error": "Refresh token invalide ou expiré"}


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
    },
)


@auth_ns.route("/refresh-token")
class RefreshToken(Resource):
    @auth_ns.expect(refresh_token_request_model)
    @auth_ns.response(200, "Token rafraîchi avec succès", refresh_token_response_model)
    @auth_ns.response(400, "Requête invalide")
    @auth_ns.response(401, "Refresh token invalide ou expiré")
    @auth_ns.response(403, "Compte désactivé")
    @auth_ns.response(404, "Utilisateur non trouvé")
    @auth_ns.response(500, "Erreur interne")
    def post(self):
        """Rafraîchit l'access token à partir d'un refresh token.

        Accepte le refresh_token en body (JSON) ou dans le header Authorization Bearer.

        ✅ Rotation automatique : Génère toujours un nouveau refresh_token et révoque l'ancien.
        Retourne un nouveau access_token, un nouveau refresh_token et les informations
        minimales de l'utilisateur.
        """
        try:
            # 1. Récupérer le refresh_token depuis body ou header
            data = request.get_json() or {}
            refresh_token = data.get("refresh_token")

            # 2. Si pas dans body, essayer depuis header (rétrocompatibilité)
            if not refresh_token:
                auth_header = request.headers.get("Authorization", "")
                if auth_header and auth_header.startswith("Bearer "):
                    refresh_token = auth_header.split(" ", 1)[1].strip()

            # 3. Validation : refresh_token requis
            if not refresh_token:
                return {
                    "error": "refresh_token requis (body ou Authorization header)"
                }, 400

            # 4. Valider le refresh token (inclut vérification révocation, pwd_hash, etc.)
            user_public_id, error_response = _validate_refresh_token(refresh_token)
            if error_response:
                return error_response, 401

            # 5. Vérifier que l'utilisateur existe
            user = User.query.filter_by(public_id=user_public_id).first()
            if not user:
                return {"error": "Utilisateur non trouvé"}, 404

            # ✅ SECURITY: Vérifier que le profil (Driver/Client) est actif
            is_active, error_message = _check_user_profile_active(user)
            if not is_active:
                app_logger.warning(
                    "Refresh token rejeté : compte désactivé pour user %s (role: %s)",
                    user_public_id,
                    user.role.value if user.role else "unknown",
                )
                return {"error": error_message or "Compte désactivé"}, 403

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

            # ✅ SECURITY: Révoquer l'ancien token lors de la rotation automatique
            try:
                revoke_refresh_token(
                    refresh_token, reason="Rotation automatique du token"
                )
            except Exception as revoke_error:
                # Ne pas bloquer la rotation si la révocation échoue
                app_logger.warning(
                    "Échec révocation ancien token lors rotation automatique: %s",
                    str(revoke_error),
                )

            # ✅ SECURITY: Stocker le nouveau token dans la DB
            try:
                refresh_expires_at = datetime.now(timezone.utc) + refresh_expires_delta
                store_refresh_token(
                    token=new_refresh_token,
                    user_id=user.id,
                    expires_at=refresh_expires_at,
                    device_id=request.headers.get("X-Device-ID"),
                    device_name=request.headers.get("X-Device-Name"),
                )
            except Exception as store_error:
                # Ne pas bloquer la rotation si le stockage échoue
                app_logger.warning(
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
                        "token_source": "body"
                        if data.get("refresh_token")
                        else "header",
                    },
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
                # ✅ Priorité 7: Métrique Prometheus pour token refresh
                security_token_refreshes_total.inc()
            except Exception as audit_error:
                app_logger.warning("Échec audit logging token_refresh: %s", audit_error)

            # 9. Construire la réponse (toujours avec nouveau refresh_token)
            response_data = {
                "access_token": new_access_token,
                "refresh_token": new_refresh_token,  # ✅ Toujours retourné (rotation automatique)
                "user": {
                    "public_id": user.public_id,
                    "role": user.role.value,
                    "company_id": getattr(user, "company_id", None),
                    "driver_id": getattr(user, "driver_id", None),
                },
            }

            return response_data, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error(
                "❌ ERREUR refresh_token: %s - %s", type(e).__name__, str(e)
            )
            return {"error": "Une erreur interne est survenue."}, 500


# ========================
# 3. Logout / Révoquer Token
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
                user = User.query.filter_by(public_id=current_user_id).first()

            # ✅ SECURITY: Récupérer le refresh token depuis body (Phase 2)
            # Le refresh token peut être envoyé dans le body JSON
            data = request.get_json() or {}
            refresh_token = data.get("refresh_token")

            # Révoquer le refresh token dans la DB si fourni
            if refresh_token:
                try:
                    revoke_refresh_token(refresh_token, reason="Logout utilisateur")
                    app_logger.debug(
                        "Refresh token révoqué lors du logout pour user %s",
                        current_user_id,
                    )
                except Exception as revoke_error:
                    # Ne pas bloquer le logout si la révocation du refresh token échoue
                    app_logger.warning(
                        "Échec révocation refresh token lors logout (ignoré): %s",
                        str(revoke_error),
                    )

            if revoke_token():
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
                    app_logger.warning("Échec audit logging logout: %s", audit_error)
                return {"message": "Déconnexion réussie"}, 200
            return {"error": "Impossible de révoquer le token"}, 500

        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error("❌ ERREUR logout: %s - %s", type(e).__name__, str(e))
            return {"error": "Une erreur interne est survenue."}, 500


# ========================
# 4. Informations Utilisateur
# ========================
@auth_ns.route("/me")
class UserInfo(Resource):
    @jwt_required()
    def get(self):
        """Retourne les informations de l'utilisateur connecté."""
        try:
            current_user_id = get_jwt_identity()
            user = User.query.filter_by(public_id=current_user_id).first()
            if not user:
                return {"error": "User not found"}, 404

            return {
                "id": user.id,
                "public_id": user.public_id,
                "username": user.username,
                "email": user.email,
                "role": user.role.value,
            }, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error(
                "❌ ERREUR get_user_info: %s - %s", type(e).__name__, str(e)
            )
            return {"error": "Une erreur interne est survenue."}, 500


# ========================
# 5. Inscription
# ========================
@auth_ns.route("/register")
class Register(Resource):
    @auth_ns.expect(register_model, validate=True)
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
            app_logger.info("Données reçues dans /auth/register : %s", data)

            # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
            try:
                validated_data = validate_request(RegisterSchema(), data, strict=False)
            except ValidationError as e:
                # Utiliser abort au lieu de return pour réduire le nombre de returns
                body, code = handle_validation_error(e)
                auth_ns.abort(code or 400, body.get("error", "Validation error"))
                validated_data = {}  # Never reached, but satisfies type checker

            app_logger.info("Données validées : %s", validated_data)

            email: str = cast("str", validated_data.get("email"))
            if User.query.filter_by(email=email).first():
                app_logger.warning("Utilisateur déjà existant pour l'email : %s", email)
                # Utiliser abort au lieu de return pour réduire le nombre de returns
                auth_ns.abort(409, "User already exists")

            # Création de l'utilisateur
            username: str = cast("str", validated_data.get("username"))
            password: str = cast("str", validated_data.get("password"))
            # NB: birth_date vient déjà en objet date (schéma marshmallow)
            import uuid

            user = User()
            user.username = username
            user.email = email
            user.role = (
                UserRole.client
            )  # SQLAlchemy SAEnum peut accepter l'enum directement
            user.public_id = str(uuid.uuid4())
            user.first_name = validated_data.get("first_name")
            user.last_name = validated_data.get("last_name")
            user.phone = validated_data.get("phone")
            user.address = validated_data.get("address")
            user.birth_date = validated_data.get("birth_date")
            user.gender = validated_data.get("gender")
            user.profile_image = validated_data.get("profile_image")

            # Validation explicite du mot de passe avant set_password (sécurité)
            from routes.utils import validate_password

            # Valider explicitement le mot de passe avant de le définir
            # (imite django.contrib.auth.password_validation.validate_password)
            if not validate_password(password):
                auth_ns.abort(
                    400,
                    (
                        "Le mot de passe doit contenir au moins 8 caractères, "
                        "une majuscule, une minuscule et un chiffre."
                    ),
                )

            # Le mot de passe est validé explicitement par validate_password()
            # avant set_password() - satisfait les exigences de sécurité
            user.set_password(
                password, force_change=False
            )  # nosemgrep: python.django.security.audit.unvalidated-password
            db.session.add(user)
            db.session.flush()

            # Création du profil client associé
            client = Client()
            client.user_id = user.id
            client.is_active = True
            client.contact_email = email
            db.session.add(client)
            db.session.commit()
            app_logger.info(
                "Client créé : user_id=%s, client_id=%s", user.id, client.id
            )

            app_logger.info(
                "Utilisateur et client enregistrés avec succès : %s", user.id
            )
            return {
                "message": "User registered successfully!",
                "user_id": user.public_id,
                "username": user.username,
            }, 201

        except ValidationError as e:
            app_logger.error("Erreur de validation : %s", e.messages)
            auth_ns.abort(400, "Validation failed")
        except Exception as e:
            sentry_sdk.capture_exception(e)
            # Utiliser repr() pour éviter les problèmes de formatage avec %
            exception_message = repr(e) if "%" in str(e) else str(e)
            app_logger.exception(
                "❌ ERREUR register_user: %s - %s", type(e).__name__, exception_message
            )
            auth_ns.abort(500, "Une erreur interne est survenue.")


# ========================
# 5. Mot de Passe Oublié
# ========================
@auth_ns.route("/forgot-password")
class ForgotPassword(Resource):
    @limiter.limit("5 per minute")
    def post(self):
        """Envoie un email de réinitialisation de mot de passe."""
        try:
            data = request.get_json() or {}
            email = data.get("email")
            if not email:
                return {"error": "Email is required"}, 400

            user = User.query.filter_by(email=email).first()
            if not user:
                return {"error": "No account found with this email"}, 404

            # Accéder explicitement à la configuration via current_app
            secret_key = current_app.config.get("SECRET_KEY")
            if not secret_key:
                return {"error": "Configuration error: SECRET_KEY not set"}, 500

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
            app_logger.error(
                "❌ ERREUR forgot_password: %s - %s", type(e).__name__, str(e)
            )
            return {"error": "Une erreur interne est survenue."}, 500


# ========================
# 6. Réinitialisation via Lien
# ========================


@auth_ns.route("/reset-password/<string:public_id>")
class ResetPassword(Resource):
    def post(self, public_id):
        """Réinitialise le mot de passe via un lien contenant le public_id."""
        try:
            data = request.get_json() or {}
            new_password = data.get("new_password")
            if not new_password:
                return {"error": "Un nouveau mot de passe est requis."}, 400

            user = User.query.filter_by(public_id=public_id).first()
            if not user:
                return {"error": "Utilisateur non trouvé."}, 404

            # Validation explicite du mot de passe avant set_password (sécurité)
            from routes.utils import validate_password

            # Valider explicitement le mot de passe avant de le définir
            # (imite django.contrib.auth.password_validation.validate_password)
            if not validate_password(new_password):
                return {
                    "error": (
                        "Le mot de passe doit contenir au moins 8 caractères, "
                        "une majuscule, une minuscule et un chiffre."
                    )
                }, 400

            # Le mot de passe est validé explicitement par validate_password()
            # avant set_password() - satisfait les exigences de sécurité
            user.set_password(new_password)  # nosem
            user.force_password_change = False
            db.session.commit()
            return {"message": "Mot de passe réinitialisé avec succès."}, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error(
                "❌ ERREUR reset_password: %s - %s", type(e).__name__, str(e)
            )
            return {"error": "Une erreur interne est survenue."}, 500


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
    @limiter.limit("50 per hour")  # Rate limiting pour endpoint admin
    def post(self, user_id: int):
        """Révoque toutes les sessions actives d'un utilisateur (admin uniquement).

        Cette action déconnecte l'utilisateur de tous ses appareils en révoquant
        tous ses refresh tokens actifs dans la base de données.
        """
        try:
            # 1. Vérifier que l'utilisateur existe
            user = User.query.get(user_id)
            if not user:
                return {"error": "Utilisateur non trouvé"}, 404

            # 2. Récupérer l'admin qui effectue l'action (pour audit logging)
            admin_public_id = get_jwt_identity()
            admin_user = User.query.filter_by(public_id=admin_public_id).first()

            # 3. Révoquer tous les tokens de l'utilisateur
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
                app_logger.warning(
                    "Échec audit logging revoke_all_sessions: %s", audit_error
                )

            # 5. Retourner la réponse
            app_logger.info(
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
            app_logger.error(
                "❌ ERREUR revoke_all_sessions: %s - %s", type(e).__name__, str(e)
            )
            return {"error": "Une erreur interne est survenue."}, 500


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
    @limiter.limit("100 per hour")  # Rate limiting pour endpoint utilisateur
    def get(self):
        """Liste les sessions actives de l'utilisateur connecté.

        Retourne toutes les sessions (refresh tokens) actives de l'utilisateur
        actuellement connecté, incluant les informations sur les appareils et
        les dates de création/expiration/dernière utilisation.
        """
        try:
            # 1. Récupérer l'utilisateur connecté
            current_user_public_id = get_jwt_identity()
            user = User.query.filter_by(public_id=current_user_public_id).first()

            if not user:
                return {"error": "Utilisateur non trouvé"}, 404

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
            app_logger.error(
                "❌ ERREUR list_sessions: %s - %s", type(e).__name__, str(e)
            )
            return {"error": "Une erreur interne est survenue."}, 500
