import logging
from datetime import timedelta
from typing import Any, Dict, cast

import sentry_sdk  # CORRECTION : Importer directement
from flask import current_app, make_response, request
from flask_jwt_extended import create_access_token, create_refresh_token, get_jwt_identity, jwt_required
from flask_mail import Message
from flask_restx import Namespace, Resource, fields
from itsdangerous import URLSafeTimedSerializer
from marshmallow import Schema, ValidationError
from marshmallow import fields as ma_fields

from ext import db, limiter, mail
from models import Client, User, UserRole

app_logger = logging.getLogger('app')

auth_ns = Namespace('auth', description='Opérations liées à l’authentification')

# Modèle Swagger pour la connexion (login)
login_model = auth_ns.model('Login', {
    'email': fields.String(required=True, description="L'adresse email de l'utilisateur"),
    'password': fields.String(required=True, description="Le mot de passe de l'utilisateur")
})

# Modèle Swagger pour l'inscription (register)
register_model = auth_ns.model('Register', {
    'username': fields.String(required=True, description="Le nom d'utilisateur"),
    'email': fields.String(required=True, description="L'adresse email de l'utilisateur"),
    'password': fields.String(required=True, description="Le mot de passe de l'utilisateur"),
    'first_name': fields.String(description="Prénom", default=None),
    'last_name': fields.String(description="Nom", default=None),
    'phone': fields.String(description="Numéro de téléphone", default=None),
    'address': fields.String(description="Adresse", default=None),
    'birth_date': fields.String(description="Date de naissance (YYYY-MM-DD)", default=None),
    'gender': fields.String(description="Genre", default=None),
    'profile_image': fields.String(description="URL ou données base64 de l'image de profil", default=None)
})

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
@auth_ns.route('/login')
class Login(Resource):
    @auth_ns.expect(login_model)
    @limiter.limit("5 per minute")  # Limite d'appels pour éviter le brute force
    def post(self):
        """Authentifie un utilisateur et renvoie un token d'accès"""
        try:
            data = request.get_json() or {}
            email = data.get('email')
            password = data.get('password')

            if not email or not password:
                return {"error": "Email et mot de passe requis."}, 400

            user = User.query.filter_by(email=email).first()
            if not user or not user.check_password(password):
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
                identity=str(user.public_id), # ⚠️ ID numérique attendu par dispatch_routes
                additional_claims=claims,
                expires_delta=timedelta(hours=1)
            )

            # Création du refresh token (valide 30 jours)
            refresh_token = create_refresh_token(
                identity=str(user.public_id),
                expires_delta=timedelta(days=30)
            )

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
                    "force_password_change": user.force_password_change
                }
            }, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error(f"❌ ERREUR login: {type(e).__name__} - {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500


# ========================
# 2. Refresh Token
# ========================
@auth_ns.route('/refresh-token')
class RefreshToken(Resource):
    @jwt_required(refresh=True)
    def post(self):
        """
        Génère un nouveau token d'accès à partir d'un refresh token
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
                expires_delta=timedelta(hours=1)
            )
            return {"access_token": new_token}, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error(f"❌ ERREUR refresh_token: {type(e).__name__} - {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500


# ========================
# 3. Informations Utilisateur
# ========================
@auth_ns.route('/me')
class UserInfo(Resource):
    @jwt_required()
    def get(self):
        """
        Retourne les informations de l'utilisateur connecté
        """
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
                "role": user.role.value
            }, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error(f"❌ ERREUR get_user_info: {type(e).__name__} - {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500


# ========================
# 4. Inscription
# ========================
@auth_ns.route('/register')
class Register(Resource):
    @auth_ns.expect(register_model, validate=True)
    def post(self):
        """
        Inscrit un nouvel utilisateur avec le rôle 'client' par défaut
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
            data = request.get_json()
            app_logger.info(f"Données reçues dans /auth/register : {data}")

            schema = UserSchema()
            _loaded = schema.load(data)
            # 🔒 typage explicite pour Pylance
            validated_data: Dict[str, Any] = cast(Dict[str, Any], _loaded)
            app_logger.info(f"Données validées : {validated_data}")

            email: str = cast(str, validated_data.get('email'))
            if User.query.filter_by(email=email).first():
                app_logger.warning(f"Utilisateur déjà existant pour l'email : {email}")
                return {"error": "User already exists"}, 409

            # Création de l'utilisateur
            username: str = cast(str, validated_data.get('username'))
            password: str = cast(str, validated_data.get('password'))
            # NB: birth_date vient déjà en objet date (schéma marshmallow)
            user = cast(Any, User)(
                username=username,
                email=email,
                role=UserRole.client,
                first_name=validated_data.get('first_name'),
                last_name=validated_data.get('last_name'),
                phone=validated_data.get('phone'),
                address=validated_data.get('address'),
                birth_date=validated_data.get('birth_date'),
                gender=validated_data.get('gender'),
                profile_image=validated_data.get('profile_image'),
            )

            user.set_password(password, force_change=False)
            db.session.add(user)
            db.session.flush()

            # Création du profil client associé
            client = cast(Any, Client)(user_id=user.id, is_active=True)
            app_logger.info(f"Client créé : {client}")
            db.session.add(client)
            db.session.commit()

            app_logger.info(f"Utilisateur et client enregistrés avec succès : {user.id}")
            return {
                "message": "User registered successfully!",
                "user_id": user.public_id,
                "username": user.username
            }, 201

        except ValidationError as e:
            app_logger.error(f"Erreur de validation : {e.messages}")
            return {"error": "Validation failed", "details": e.messages}, 400
        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error(f"❌ ERREUR register_user: {type(e).__name__} - {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500


# ========================
# 5. Mot de Passe Oublié
# ========================
@auth_ns.route('/forgot-password')
class ForgotPassword(Resource):
    @limiter.limit("5 per minute")
    def post(self):
        """
        Envoie un email de réinitialisation de mot de passe
        """
        try:
            data = request.get_json() or {}
            email = data.get('email')
            if not email:
                return {"error": "Email is required"}, 400

            user = User.query.filter_by(email=email).first()
            if not user:
                return {"error": "No account found with this email"}, 404

            # Accéder explicitement à la configuration via current_app
            secret_key = current_app.config.get('SECRET_KEY')
            if not secret_key:
                return {"error": "Configuration error: SECRET_KEY not set"}, 500

            serializer = URLSafeTimedSerializer(secret_key)
            reset_token = serializer.dumps(user.email, salt='password-reset-salt')

            msg = Message(
                subject="Réinitialisation de votre mot de passe",
                recipients=[email],
                body=f"Cliquez sur ce lien pour réinitialiser votre mot de passe : "
                     f"http://localhost:3000/reset-password/{reset_token}"
            )
            mail.send(msg)
            return {"message": "Password reset email sent successfully"}, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error(f"❌ ERREUR forgot_password: {type(e).__name__} - {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500

# ========================
# 6. Réinitialisation via Lien
# ========================
@auth_ns.route('/reset-password/<string:public_id>')
class ResetPassword(Resource):
    def post(self, public_id):
        """
        Réinitialise le mot de passe via un lien contenant le public_id
        """
        try:
            data = request.get_json() or {}
            new_password = data.get("new_password")
            if not new_password:
                return {"error": "Un nouveau mot de passe est requis."}, 400

            user = User.query.filter_by(public_id=public_id).first()
            if not user:
                return {"error": "Utilisateur non trouvé."}, 404

            user.set_password(new_password)
            user.force_password_change = False
            db.session.commit()
            return {"message": "Mot de passe réinitialisé avec succès."}, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error(f"❌ ERREUR reset_password: {type(e).__name__} - {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500
