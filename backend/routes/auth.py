import logging
from typing import cast

import sentry_sdk  # CORRECTION : Importer directement
from flask import current_app, make_response, request
from flask_jwt_extended import (
    create_access_token,
    create_refresh_token,
    get_jwt_identity,
    jwt_required,
)
from flask_mail import Message
from flask_restx import Namespace, Resource, fields
from itsdangerous import URLSafeTimedSerializer
from marshmallow import Schema, ValidationError
from marshmallow import fields as ma_fields

from ext import db, limiter, mail
from models import Client, User, UserRole
from schemas.auth_schemas import LoginSchema, RegisterSchema
from schemas.validation_utils import handle_validation_error, validate_request
from security.audit_log import AuditLogger
from security.security_metrics import (
    security_login_attempts_total,
    security_login_failures_total,
    security_logout_total,
    security_token_refreshes_total,
)
from shared.logging_utils import mask_email

app_logger = logging.getLogger("app")

auth_ns = Namespace("auth", description="Opérations liées à l'authentification")

# Modèle Swagger pour la connexion (login)
login_model = auth_ns.model(
    "Login",
    {
        "email": fields.String(required=True, description="L'adresse email de l'utilisateur (format email valide)"),
        "password": fields.String(required=True, description="Le mot de passe de l'utilisateur", min_length=6),
    },
)

# Modèle Swagger pour l'inscription (register)
register_model = auth_ns.model(
    "Register",
    {
        "username": fields.String(required=True, description="Le nom d'utilisateur", min_length=3, max_length=50),
        "email": fields.String(required=True, description="L'adresse email de l'utilisateur (format email valide)"),
        "password": fields.String(required=True, description="Le mot de passe de l'utilisateur", min_length=6),
        "first_name": fields.String(description="Prénom", default=None, max_length=100),
        "last_name": fields.String(description="Nom", default=None, max_length=100),
        "phone": fields.String(description="Numéro de téléphone", default=None, max_length=20),
        "address": fields.String(description="Adresse", default=None, max_length=500),
        "birth_date": fields.String(
            description="Date de naissance (YYYY-MM-DD)", default=None, pattern="^\\d{4}-\\d{2}-\\d{2}$"
        ),
        "gender": fields.String(
            description="Genre (male|female|other)", default=None, enum=["male", "female", "other"]
        ),
        "profile_image": fields.String(description="URL ou données base64 de l'image de profil", default=None),
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
                        action_details={"email": mask_email(email), "reason": "invalid_credentials"},
                        ip_address=request.remote_addr,
                        user_agent=request.headers.get("User-Agent"),
                    )
                    # ✅ Priorité 7: Métriques Prometheus pour login échoué
                    security_login_attempts_total.labels(type="failed").inc()
                    security_login_failures_total.inc()
                except Exception as audit_error:
                    # Ne pas bloquer la réponse si l'audit logging échoue
                    app_logger.warning("Échec audit logging login_failed: %s", audit_error)
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

            # Création du refresh token (durée configurée dans JWT_REFRESH_TOKEN_EXPIRES)
            refresh_token = create_refresh_token(
                identity=str(user.public_id), expires_delta=current_app.config["JWT_REFRESH_TOKEN_EXPIRES"]
            )

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
                    action_details={"email": mask_email(data.get("email", "")) if data else ""},
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
            except Exception:
                pass  # Ignorer les erreurs d'audit logging dans le gestionnaire d'erreurs
            return {"error": "Une erreur interne est survenue."}, 500


# ========================
# 2. Refresh Token
# ========================
@auth_ns.route("/refresh-token")
class RefreshToken(Resource):
    @jwt_required(refresh=True)
    def post(self):
        """Génère un nouveau token d'accès à partir d'un refresh token
        et inclut également le rôle si vous le désirez.
        """
        try:
            current_user_id = get_jwt_identity()
            user = User.query.filter_by(public_id=current_user_id).first()
            if not user:
                return {"error": "User not found"}, 404

            # ✅ SECURITY: Ajout claim 'aud' (audience) pour prévenir token replay
            claims = {
                "role": user.role.value,
                "company_id": getattr(user, "company_id", None),
                "driver_id": getattr(user, "driver_id", None),
                "aud": "atmr-api",  # Audience claim pour sécurité
            }
            new_token = create_access_token(
                identity=str(user.public_id),
                additional_claims=claims,
                expires_delta=current_app.config["JWT_ACCESS_TOKEN_EXPIRES"],
            )

            # ✅ Priorité 7: Audit logging pour token refresh
            try:
                AuditLogger.log_action(
                    action_type="token_refresh",
                    action_category="security",
                    user_id=user.id,
                    user_type=user.role.value if user.role else "unknown",
                    result_status="success",
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
                # ✅ Priorité 7: Métrique Prometheus pour token refresh
                security_token_refreshes_total.inc()
            except Exception as audit_error:
                app_logger.warning("Échec audit logging token_refresh: %s", audit_error)

            return {"access_token": new_token}, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error("❌ ERREUR refresh_token: %s - %s", type(e).__name__, str(e))
            return {"error": "Une erreur interne est survenue."}, 500


# ========================
# 3. Logout / Révoquer Token
# ========================
@auth_ns.route("/logout")
class Logout(Resource):
    @auth_ns.doc(
        description=(
            "Révoque le token JWT actuel et l'ajoute à la blacklist. "
            "Après la déconnexion, le token ne pourra plus être utilisé pour accéder aux endpoints protégés."
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

            if revoke_token():
                # ✅ Priorité 7: Audit logging pour logout réussi
                try:
                    AuditLogger.log_action(
                        action_type="logout",
                        action_category="security",
                        user_id=user.id if user else None,
                        user_type=user.role.value if user and user.role else "unknown",
                        result_status="success",
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
            app_logger.error("❌ ERREUR get_user_info: %s - %s", type(e).__name__, str(e))
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
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
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
            user.role = UserRole.client  # SQLAlchemy SAEnum peut accepter l'enum directement
            user.public_id = str(uuid.uuid4())
            user.first_name = validated_data.get("first_name")
            user.last_name = validated_data.get("last_name")
            user.phone = validated_data.get("phone")
            user.address = validated_data.get("address")
            user.birth_date = validated_data.get("birth_date")
            user.gender = validated_data.get("gender")
            user.profile_image = validated_data.get("profile_image")

            # Validation explicite du mot de passe avant set_password (sécurité)
            from routes.utils import validate_password_or_raise

            try:
                validate_password_or_raise(password, _user=user)
            except ValueError as e:
                # Utiliser abort au lieu de return pour réduire le nombre de returns
                auth_ns.abort(400, str(e))

            # Le mot de passe est validé explicitement par validate_password_or_raise() ci-dessus
            # nosemgrep: python.django.security.audit.unvalidated-password.unvalidated-password
            user.set_password(password, force_change=False)
            db.session.add(user)
            db.session.flush()

            # Création du profil client associé
            client = Client()
            client.user_id = user.id
            client.is_active = True
            client.contact_email = email
            db.session.add(client)
            db.session.commit()
            app_logger.info("Client créé : user_id=%s, client_id=%s", user.id, client.id)

            app_logger.info("Utilisateur et client enregistrés avec succès : %s", user.id)
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
            app_logger.exception("❌ ERREUR register_user: %s - %s", type(e).__name__, exception_message)
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
                body=f"Cliquez sur ce lien pour réinitialiser votre mot de passe : http://localhost:3000/reset-password/{reset_token}",
            )
            mail.send(msg)
            return {"message": "Password reset email sent successfully"}, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error("❌ ERREUR forgot_password: %s - %s", type(e).__name__, str(e))
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
            from routes.utils import validate_password_or_raise

            try:
                validate_password_or_raise(new_password, _user=user)
            except ValueError as e:
                return {"error": str(e)}, 400

            # Le mot de passe est validé explicitement par validate_password_or_raise() ci-dessus
            # nosemgrep: python.django.security.audit.unvalidated-password.unvalidated-password
            user.set_password(new_password)
            user.force_password_change = False
            db.session.commit()
            return {"message": "Mot de passe réinitialisé avec succès."}, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error("❌ ERREUR reset_password: %s - %s", type(e).__name__, str(e))
            return {"error": "Une erreur interne est survenue."}, 500
