"""Routes d'authentification mobile pour les entreprises (dispatch)."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import jwt
from flask import request
from flask_jwt_extended import (
    create_access_token,
    create_refresh_token,
    get_jwt,
    get_jwt_identity,
    jwt_required,
)
from flask_restx import Namespace, Resource, fields
from marshmallow import Schema, ValidationError, validate
from marshmallow import fields as ma_fields

from ext import limiter, redis_client
from models import Company, User, UserRole

logger = logging.getLogger(__name__)


MOBILE_AUDIENCE = "atmr-mobile-enterprise"
MFA_CHALLENGE_TTL = 300  # 5 minutes
MFA_CHALLENGE_PREFIX = "company_mobile:mfa:challenge:"
DEFAULT_SCOPES = ["enterprise.dispatch:read", "enterprise.dispatch:write"]


company_mobile_auth_ns = Namespace(
    "company_mobile_auth", description="Authentification mobile entreprise (dispatch)"
)


# ====== Modèles Swagger ======
login_model = company_mobile_auth_ns.model(
    "EnterpriseMobileLogin",
    {
        "method": fields.String(
            required=False,
            default="password",
            enum=["password", "oidc"],
            description="Méthode d'authentification (password ou oidc)",
        ),
        "email": fields.String(description="Email entreprise (pour login/mot de passe)"),
        "password": fields.String(description="Mot de passe (si method=password)"),
        "id_token": fields.String(description="ID token OIDC (si method=oidc)"),
        "provider": fields.String(description="Identifiant fournisseur OIDC/SAML"),
        "mfa_code": fields.String(description="Code MFA TOTP"),
        "device_id": fields.String(description="Identifiant appareil (MDM)"),
    },
)

mfa_verify_model = company_mobile_auth_ns.model(
    "EnterpriseMobileMfaVerify",
    {
        "challenge_id": fields.String(required=True, description="Identifiant du challenge MFA"),
        "code": fields.String(required=True, description="Code TOTP à 6 chiffres"),
        "device_id": fields.String(description="Identifiant appareil (optionnel)"),
    },
)

refresh_model = company_mobile_auth_ns.model(
    "EnterpriseMobileRefresh",
    {
        "refresh_token": fields.String(required=True, description="Refresh token valide"),
    },
)


# ====== Schemas Marshmallow ======
class EnterpriseLoginSchema(Schema):
    method = ma_fields.String(
        load_default="password", validate=validate.OneOf(["password", "oidc"])
    )
    email = ma_fields.Email(load_default=None)
    password = ma_fields.String(load_default=None)
    id_token = ma_fields.String(load_default=None)
    provider = ma_fields.String(load_default=None)
    mfa_code = ma_fields.String(load_default=None)
    device_id = ma_fields.String(load_default=None)


class EnterpriseMfaVerifySchema(Schema):
    challenge_id = ma_fields.UUID(required=True)
    code = ma_fields.String(required=True, validate=validate.Length(min=4, max=10))
    device_id = ma_fields.String(load_default=None)


class EnterpriseRefreshSchema(Schema):
    refresh_token = ma_fields.String(required=True)


# ====== Helpers ======
def _get_company_security(company: Company | None) -> Dict[str, Any]:
    if not company or not company.autonomous_config:
        return {}
    try:
        payload = json.loads(company.autonomous_config)
    except (ValueError, TypeError):
        logger.warning(
            "[AUTH][Enterprise] Impossible de parser autonomous_config pour company_id=%s",
            getattr(company, "id", None),
        )
        return {}
    return payload.get("security", {}) or {}


def _company_requires_mfa(company: Company | None) -> bool:
    security = _get_company_security(company)
    policy = security.get("mobile_mfa") or {}
    return bool(policy.get("required", False))


def _get_totp_secret(company: Company | None) -> Optional[str]:
    security = _get_company_security(company)
    policy = security.get("mobile_mfa") or {}
    secret = policy.get("totp_secret")
    if isinstance(secret, str) and secret.strip():
        return secret.strip()
    return None


def _verify_totp_code(company: Company | None, code: str) -> bool:
    secret = _get_totp_secret(company)
    if not secret:
        logger.warning(
            "[AUTH][Enterprise] Aucun secret TOTP configuré pour company_id=%s",
            getattr(company, "id", None),
        )
        return False
    try:
        import pyotp
    except ImportError:  # pragma: no cover - dépendance optionnelle
        logger.error(
            "[AUTH][Enterprise] pyotp requis pour vérifier le code MFA (company_id=%s)",
            getattr(company, "id", None),
        )
        return False
    totp = pyotp.TOTP(secret)
    return bool(totp.verify(code, valid_window=1))


def _issue_tokens(
    user: User,
    company: Company,
    device_id: Optional[str] = None,
    extra_scopes: Optional[List[str]] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    scopes = list(DEFAULT_SCOPES)
    if user.role == UserRole.ADMIN:
        scopes.append("enterprise.dispatch:admin")
    if extra_scopes:
        for scope in extra_scopes:
            if scope not in scopes:
                scopes.append(scope)

    session_identifier = session_id or str(uuid.uuid4())

    additional_claims = {
        "role": user.role.value,
        "company_id": company.id,
        "aud": MOBILE_AUDIENCE,
        "scopes": scopes,
        "session_id": session_identifier,
    }
    if device_id:
        additional_claims["device_id"] = device_id

    access_token = create_access_token(
        identity=str(user.public_id),
        additional_claims=additional_claims,
        expires_delta=timedelta(minutes=45),
    )
    refresh_token = create_refresh_token(
        identity=str(user.public_id),
        additional_claims={
            "aud": MOBILE_AUDIENCE,
            "session_id": session_identifier,
        },
        expires_delta=timedelta(days=14),
    )

    return {
        "token": access_token,
        "refresh_token": refresh_token,
        "user": {
            "id": user.id,
            "public_id": user.public_id,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "role": user.role.value,
        },
        "company": {
            "id": company.id,
            "name": company.name,
            "dispatch_mode": company.dispatch_mode.value,
        },
        "scopes": scopes,
        "session_id": session_identifier,
    }


def _store_mfa_challenge(
    user: User,
    company: Company,
    method: str,
    device_id: Optional[str],
) -> str:
    challenge_id = str(uuid.uuid4())
    payload = {
        "user_public_id": user.public_id,
        "company_id": company.id,
        "method": method,
        "device_id": device_id,
    }
    if redis_client:
        redis_client.setex(
            f"{MFA_CHALLENGE_PREFIX}{challenge_id}",
            MFA_CHALLENGE_TTL,
            json.dumps(payload),
        )
    else:  # pragma: no cover - fallback
        logger.warning(
            "[AUTH][Enterprise] Redis indisponible, impossible de stocker le challenge MFA."
        )
    return challenge_id


def _consume_mfa_challenge(challenge_id: str) -> Optional[Dict[str, Any]]:
    key = f"{MFA_CHALLENGE_PREFIX}{challenge_id}"
    if not redis_client:
        logger.error("[AUTH][Enterprise] Redis requis pour vérifier le challenge MFA.")
        return None
    data = redis_client.get(key)
    if not data:
        return None
    redis_client.delete(key)
    try:
        return json.loads(data)
    except (ValueError, TypeError):
        logger.error("[AUTH][Enterprise] Challenge MFA corrompu (%s).", challenge_id)
        return None


def _find_company_user_by_email(email: str) -> Tuple[Optional[User], Optional[Company]]:
    user: Optional[User] = User.query.filter(User.email == email).first()
    if not user:
        return None, None
    if user.role not in (UserRole.COMPANY, UserRole.ADMIN):
        return None, None
    company = user.company
    if not company:
        return None, None
    return user, company


def _handle_oidc_login(id_token: str, provider: Optional[str]) -> Tuple[Optional[User], Optional[Company]]:
    if not id_token:
        raise ValueError("Token OIDC manquant.")
    try:
        decoded = jwt.decode(id_token, options={"verify_signature": False, "verify_aud": False})
    except jwt.PyJWTError as exc:  # pragma: no cover - dépend du token fourni
        logger.warning("[AUTH][Enterprise] Echec décodage token OIDC: %s", exc)
        raise ValueError("ID token invalide.") from exc

    email = decoded.get("email")
    if not email:
        raise ValueError("ID token ne contient pas d'email.")

    user, company = _find_company_user_by_email(email)
    if not user or not company:
        raise ValueError("Utilisateur OIDC non autorisé pour l'app entreprise.")

    logger.info(
        "[AUTH][Enterprise] Connexion OIDC réussie pour user_id=%s provider=%s",
        user.id,
        provider or decoded.get("iss"),
    )
    return user, company


# ====== Resources ======
@company_mobile_auth_ns.route("/login")
class EnterpriseMobileLogin(Resource):
    @company_mobile_auth_ns.expect(login_model, validate=True)
    @limiter.limit("10/minute")
    def post(self):
        payload = request.get_json() or {}
        try:
            data = EnterpriseLoginSchema().load(payload)
        except ValidationError as exc:
            return {"error": "Paramètres invalides", "details": exc.messages}, 400

        method = data["method"]
        email = data.get("email")
        password = data.get("password")
        id_token = data.get("id_token")
        mfa_code = data.get("mfa_code")
        device_id = data.get("device_id")

        user: Optional[User] = None
        company: Optional[Company] = None
        error_response: Optional[Tuple[Dict[str, Any], int]] = None

        if method == "password":
            if not email or not password:
                error_response = ({"error": "Email et mot de passe requis."}, 400)
            else:
                user, company = _find_company_user_by_email(email)
                if not user or not company or not user.check_password(password):
                    error_response = ({"error": "Identifiants invalides."}, 401)
        else:  # method == "oidc"
            try:
                user, company = _handle_oidc_login(id_token, data.get("provider"))
            except ValueError as exc:
                error_response = ({"error": str(exc)}, 401)

        if error_response:
            return error_response

        if not user or not company:
            return {"error": "Accès refusé."}, 403

        requires_mfa = _company_requires_mfa(company)
        if requires_mfa:
            if mfa_code:
                if not _verify_totp_code(company, mfa_code):
                    return {"error": "Code MFA invalide."}, 401
            else:
                challenge_id = _store_mfa_challenge(user, company, method, device_id)
                return {
                    "message": "MFA requis",
                    "mfa_required": True,
                    "challenge_id": challenge_id,
                    "methods": ["totp"],
                    "ttl": MFA_CHALLENGE_TTL,
                }, 202

        response = _issue_tokens(user, company, device_id)
        response["mfa_required"] = False
        return response, 200


@company_mobile_auth_ns.route("/mfa/verify")
class EnterpriseMobileMfaVerify(Resource):
    @company_mobile_auth_ns.expect(mfa_verify_model, validate=True)
    @limiter.limit("15/minute")
    def post(self):
        payload = request.get_json() or {}
        try:
            data = EnterpriseMfaVerifySchema().load(payload)
        except ValidationError as exc:
            return {"error": "Paramètres invalides", "details": exc.messages}, 400

        challenge = _consume_mfa_challenge(str(data["challenge_id"]))
        if not challenge:
            return {"error": "Challenge MFA expiré ou invalide."}, 410

        user = User.query.filter_by(public_id=challenge["user_public_id"]).first()
        if not user:
            return {"error": "Utilisateur introuvable."}, 404
        company = Company.query.get(challenge["company_id"])
        if not company:
            return {"error": "Entreprise introuvable."}, 404

        if not _verify_totp_code(company, data["code"]):
            return {"error": "Code MFA invalide."}, 401

        response = _issue_tokens(
            user,
            company,
            device_id=data.get("device_id") or challenge.get("device_id"),
        )
        response["mfa_required"] = False
        return response, 200


@company_mobile_auth_ns.route("/refresh")
class EnterpriseMobileRefresh(Resource):
    @company_mobile_auth_ns.expect(refresh_model, validate=True)
    @limiter.limit("20/minute")
    def post(self):
        payload = request.get_json() or {}
        try:
            data = EnterpriseRefreshSchema().load(payload)
        except ValidationError as exc:
            return {"error": "Paramètres invalides", "details": exc.messages}, 400

        try:
            decoded = jwt.decode(
                data["refresh_token"],
                options={"verify_signature": False, "verify_aud": False},
            )
        except jwt.PyJWTError as exc:  # pragma: no cover - dépend du token fourni
            logger.warning("[AUTH][Enterprise] Refresh token invalide: %s", exc)
            return {"error": "Refresh token invalide."}, 401

        public_id = decoded.get("sub")
        session_id = decoded.get("session_id")
        if not public_id:
            return {"error": "Refresh token invalide."}, 401

        user = User.query.filter_by(public_id=str(public_id)).first()
        if not user or user.role not in (UserRole.COMPANY, UserRole.ADMIN):
            return {"error": "Accès refusé."}, 403
        company = user.company
        if not company:
            return {"error": "Entreprise introuvable."}, 403

        response = _issue_tokens(user, company, session_id=session_id)
        response["mfa_required"] = False
        return response, 200


@company_mobile_auth_ns.route("/session")
class EnterpriseMobileSession(Resource):
    @jwt_required()
    def get(self):
        claims = get_jwt()
        identity = get_jwt_identity()
        user = User.query.filter_by(public_id=str(identity)).first()
        if not user or user.role not in (UserRole.COMPANY, UserRole.ADMIN):
            return {"error": "Accès refusé."}, 403
        company = user.company
        if not company:
            return {"error": "Entreprise introuvable."}, 404

        return {
            "user": {
                "id": user.id,
                "public_id": user.public_id,
                "email": user.email,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "role": user.role.value,
            },
            "company": {
                "id": company.id,
                "name": company.name,
                "dispatch_mode": company.dispatch_mode.value,
            },
            "scopes": claims.get("scopes", []),
            "session_id": claims.get("session_id"),
            "aud": claims.get("aud"),
        }, 200


